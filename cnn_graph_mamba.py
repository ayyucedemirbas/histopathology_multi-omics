import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import timm
from PIL import Image
import os
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, precision_recall_curve, auc, accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc

seed = 2026
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def correct_csv_format(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            for row in reader:
                new_row = []
                for cell in row:
                    new_row.extend(cell.split('\t'))
                writer.writerow(new_row)

def filter_zero_samples(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            for row in reader:
                if all(cell != '0.0' for cell in row):
                    writer.writerow(row)

def preprocess_metadata(file_path):
    """Load and preprocess metadata"""
    metadata = pd.read_excel(file_path)
    metadata = metadata[metadata['TCGA Subtype'] != 'BRCA.Normal']
    metadata['TCGA Subtype'] = metadata['TCGA Subtype'].replace({
        'BRCA.LumA': 0,
        'BRCA.LumB': 1,
        'BRCA.Basal': 2,
        'BRCA.Her2': 3
    })
    return metadata

def get_valid_patients(image_dir, metadata):
    valid_patients = []
    patient_image_paths = {}
    
    for patient_id in metadata['Sample ID'].values:
        sample_dirs = [item for item in os.listdir(image_dir) if patient_id in item]
        if sample_dirs:
            patient_path = os.path.join(image_dir, sample_dirs[0])
            if os.path.exists(patient_path):
                img_files = os.listdir(patient_path)
                if len(img_files) > 0:
                    random.shuffle(img_files)
                    selected_files = [os.path.join(patient_path, f) for f in img_files[:50]]
                    patient_image_paths[patient_id] = selected_files
                    valid_patients.append(patient_id)
    
    return valid_patients, patient_image_paths

def select_top_variant_genes(gex_data, top_n=3000):
    gene_variance = gex_data.var(axis=0)
    top_genes = gene_variance.nlargest(top_n).index
    return gex_data[top_genes]

def preprocess_gene_expression(gex_data):
    if (gex_data < 0).any().any():
        gex_data += abs(gex_data.min().min()) + 1
    
    gex_data = np.log1p(gex_data)
    
    gex_data = select_top_variant_genes(gex_data, top_n=3000)
    
    scaler = StandardScaler()
    gex_data_scaled = pd.DataFrame(
        scaler.fit_transform(gex_data),
        index=gex_data.index,
        columns=gex_data.columns
    )
    
    gex_data_scaled = gex_data_scaled.fillna(gex_data_scaled.mean())
    
    return gex_data_scaled

def build_patient_graph(gex_data, n_neighbors=15, threshold=0.4):
    similarity_matrix = cosine_similarity(gex_data)
    
    adj_matrix = np.zeros_like(similarity_matrix)
    for i in range(len(similarity_matrix)):
        similar_indices = np.argsort(similarity_matrix[i])[::-1][1:n_neighbors+1]
        for j in similar_indices:
            if similarity_matrix[i, j] > threshold:
                adj_matrix[i, j] = similarity_matrix[i, j]
    
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    
    edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
    edge_weight = torch.tensor(adj_matrix[adj_matrix > 0], dtype=torch.float32)
    
    return edge_index, edge_weight, adj_matrix

class MultiModalDataset(Dataset):  
    def __init__(self, patient_ids, patient_image_paths, gex_features, labels, 
                 edge_index, transform=None, is_training=True):
        self.patient_ids = patient_ids
        self.patient_image_paths = patient_image_paths
        self.gex_features = torch.FloatTensor(gex_features)
        self.labels = torch.LongTensor(labels)
        self.edge_index = edge_index
        self.is_training = is_training
        self.transform = transform or self.get_transform(is_training)
        
        self.id_to_idx = {pid: idx for idx, pid in enumerate(patient_ids)}
    
    def get_transform(self, is_training):
        if is_training:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        image_paths = self.patient_image_paths[patient_id]
        
        max_patches = 40 if self.is_training else 35
        images = []
        for img_path in image_paths[:max_patches]:
            try:
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                images.append(img)
            except Exception as e:
                continue
        
        if len(images) == 0:
            images = [torch.zeros(3, 224, 224)]
        
        images = torch.stack(images)
        
        return {
            'images': images,
            'gex': self.gex_features[idx],
            'label': self.labels[idx],
            'patient_idx': idx
        }


class S6(nn.Module):   
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        A = torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
    def forward(self, x):
        B, L, D = x.shape
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        
        x = F.silu(x)
        
        x_proj = self.x_proj(x)
        dt, B_ssm, C = torch.split(x_proj, [self.d_state, self.d_state, self.d_inner], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        
        y = self.selective_scan(x, dt, A, B_ssm, C)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)

        output = self.out_proj(y)
        
        return output
    
    def selective_scan(self, x, dt, A, B, C):
        B_batch, L, D = x.shape
        h = torch.zeros(B_batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        for i in range(L):
            x_i = x[:, i, :]
            dt_i = dt[:, i, :]
            
            B_i = B[:, i, :]
            C_i = C[:, i, :]
            
            dA = torch.exp(dt_i.unsqueeze(-1) * A.unsqueeze(0))
            
            dB = dt_i.unsqueeze(-1) * B_i.unsqueeze(1)
            
            h = h * dA + dB * x_i.unsqueeze(-1)
            y = h.sum(dim=-1) * C_i 
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba = S6(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class GraphMambaLayer(nn.Module):  
    def __init__(self, d_model, d_state=16, num_heads=4):
        super().__init__()
        self.node_transform = nn.Linear(d_model, d_model)

        self.mamba = MambaBlock(d_model, d_state=d_state)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, edge_index, edge_weight):
        x = self.node_transform(x)

        x_seq = x.unsqueeze(0)
        x_mamba = self.mamba(x_seq).squeeze(0)
        x = self.norm1(x + x_mamba)
        
        if edge_index.size(1) > 0:
            N = x.size(0)
            neighbor_features = []
            for i in range(N):
                mask = edge_index[1] == i
                if mask.sum() > 0:
                    neighbor_idx = edge_index[0][mask]
                    neighbor_feat = x[neighbor_idx]
                    neighbor_features.append(neighbor_feat)
                else:
                    neighbor_features.append(x[i:i+1])
            
            aggregated = []
            for i in range(N):
                query = x[i:i+1].unsqueeze(0)
                key_value = neighbor_features[i].unsqueeze(0)
                attn_out, _ = self.attention(query, key_value, key_value)
                aggregated.append(attn_out.squeeze(0))
            
            x_agg = torch.cat(aggregated, dim=0)
            x = self.norm2(x + x_agg)
        
        x = self.norm3(x + self.ffn(x))
        
        return x


class GraphMambaEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim=512, num_layers=4, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.layers = nn.ModuleList([
            GraphMambaLayer(hidden_dim, d_state=32, num_heads=8)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_weight):
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        
        x = self.output_norm(x)
        return x


class MambaImageEncoder(nn.Module):   
    def __init__(self, model_name='convnext_small', pretrained=True):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features
        
        self.mamba_blocks = nn.Sequential(
            MambaBlock(self.feature_dim, d_state=32, expand=2),
            MambaBlock(self.feature_dim, d_state=32, expand=2)
        )

        self.global_pool = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, images):
        B, N, C, H, W = images.shape
        
        images = images.view(B * N, C, H, W)
        features = self.backbone(images)
        features = features.view(B, N, -1)

        features = self.mamba_blocks(features)

        pooled = features.mean(dim=1)
        output = self.global_pool(pooled)
        
        return output


class CrossModalFusion(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()

        self.img_mamba = MambaBlock(feature_dim, d_state=32)
        self.gex_mamba = MambaBlock(feature_dim, d_state=32)
        
        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )
        
    def forward(self, img_feat, gex_feat):
        img_seq = img_feat.unsqueeze(1)
        gex_seq = gex_feat.unsqueeze(1)

        img_proc = self.img_mamba(img_seq).squeeze(1)
        gex_proc = self.gex_mamba(gex_seq).squeeze(1)
        
        img_seq = img_proc.unsqueeze(1)
        gex_seq = gex_proc.unsqueeze(1)
        cross_img, _ = self.cross_attn(img_seq, gex_seq, gex_seq)
        cross_gex, _ = self.cross_attn(gex_seq, img_seq, img_seq)
        
        cross_img = cross_img.squeeze(1)
        cross_gex = cross_gex.squeeze(1)

        combined = torch.cat([img_proc, gex_proc, cross_img + cross_gex], dim=1)
        fused = self.fusion(combined)
        
        return fused


class MultiModalModel(nn.Module):    
    def __init__(self, gex_features, num_patients, edge_index, edge_weight, 
                 image_model='convnext_small', dropout=0.4):
        super().__init__()
        
        self.image_encoder = MambaImageEncoder(model_name=image_model, pretrained=True)
        image_dim = self.image_encoder.feature_dim

        self.gex_encoder = GraphMambaEncoder(
            in_features=gex_features,
            hidden_dim=512,
            num_layers=4,
            dropout=0.2
        )
        
        self.num_patients = num_patients
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.fusion = CrossModalFusion(feature_dim=512)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 4)
        )
        
    def forward(self, images, gex_features, patient_indices):
        img_features = self.image_encoder(images)
        img_features = self.image_proj(img_features)

        gex_encoded = self.gex_encoder(gex_features, self.edge_index, self.edge_weight)
        batch_gex = gex_encoded[patient_indices]
        
        fused = self.fusion(img_features, batch_gex)

        output = self.classifier(fused)
        
        return output


def get_class_weights(labels):
    class_counts = Counter(labels)
    total = len(labels)
    weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
    return torch.FloatTensor([weights[i] for i in range(len(class_counts))])

def train_epoch(model, dataloader, all_gex, optimizer, criterion, scaler, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        images = batch['images'].to(device)
        labels = batch['label'].to(device)
        patient_idx = batch['patient_idx'].to(device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images, all_gex, patient_idx)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, all_gex, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images'].to(device)
            labels = batch['label'].to(device)
            patient_idx = batch['patient_idx'].to(device)
            
            outputs = model(images, all_gex, patient_idx)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return (total_loss / len(dataloader),
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


def main():
    GEX_FILE = "/content/drive/MyDrive/28050083/TCGA-BRCA-RNA-Seq.csv"
    METADATA_FILE = "/content/drive/MyDrive/28050083/TCGA-BRCA-A2-target_variable.xlsx"
    IMAGE_DIR = "/content/drive/MyDrive/28050083/TCGA-BRCA-A2-DEEPMED-TILES/BLOCKS_NORM_MACENKO"
    
    correct_csv_format(GEX_FILE, '/content/output.csv')
    filter_zero_samples('/content/output.csv', '/content/gene_expression.csv')
    
    metadata = preprocess_metadata(METADATA_FILE)
    
    valid_patients, patient_image_paths = get_valid_patients(IMAGE_DIR, metadata)
    metadata = metadata[metadata['Sample ID'].isin(valid_patients)].copy()
    
    print(f"Class distribution: {Counter(metadata['TCGA Subtype'].values)}")
    

    gex_data = pd.read_csv('/content/gene_expression.csv', index_col=0).T
    gex_data = gex_data[gex_data.index.isin(metadata['Sample ID'].values)]
    gex_data = gex_data.loc[metadata['Sample ID']]
    gex_data_scaled = preprocess_gene_expression(gex_data)
    
    edge_index, edge_weight, adj_matrix = build_patient_graph(
        gex_data_scaled.values, n_neighbors=15, threshold=0.4
    )
    
    patient_ids = metadata['Sample ID'].values
    gex_features = gex_data_scaled.values
    labels = metadata['TCGA Subtype'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_metrics = {
        'f1_macro': [], 'f1_weighted': [], 'mcc': [], 'pr_auc': [],
        'accuracy': [], 'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels), 1):
        print(f"\n{'='*50}")
        print(f"Fold {fold}/5")
        print(f"{'='*50}")
        
        train_patients = patient_ids[train_idx]
        val_patients = patient_ids[val_idx]
        train_labels = labels[train_idx]
        
        class_weights = get_class_weights(train_labels).to(device)
        print(f"Class weights: {class_weights}")
        
        train_dataset = MultiModalDataset(
            train_patients, patient_image_paths,
            gex_features[train_idx], train_labels,
            edge_index, is_training=True
        )
        
        val_dataset = MultiModalDataset(
            val_patients, patient_image_paths,
            gex_features[val_idx], labels[val_idx],
            edge_index, is_training=False
        )
        
        sample_weights = [class_weights[label].item() for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        train_loader = DataLoader(
            train_dataset, batch_size=3, sampler=sampler,
            num_workers=2, pin_memory=True, drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=3, shuffle=False,
            num_workers=2, pin_memory=True
        )
        
        all_gex = torch.FloatTensor(gex_features).to(device)
        model = MultiModalModel(
            gex_features=gex_features.shape[1],
            num_patients=len(gex_features),
            edge_index=edge_index.to(device),
            edge_weight=edge_weight.to(device),
            image_model='convnext_small',
            dropout=0.4
        ).to(device)
        
        pretrained_params = []
        new_params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                pretrained_params.append(param)
            else:
                new_params.append(param)
        
        optimizer = AdamW([
            {'params': pretrained_params, 'lr': 5e-6},
            {'params': new_params, 'lr': 2e-4}
        ], weight_decay=1e-4)
        
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        scaler = torch.amp.GradScaler('cuda')
        
        best_val_f1 = 0
        patience = 15
        patience_counter = 0
        fold_train_losses = []
        fold_val_losses = []
        fold_train_accs = []
        fold_val_accs = []
        
        for epoch in range(60):
            print(f"\nEpoch {epoch+1}/60")
            
            train_loss, train_acc = train_epoch(
                model, train_loader, all_gex, optimizer, criterion, scaler, device
            )
            
            val_loss, val_labels, val_preds, val_probs = evaluate(
                model, val_loader, all_gex, criterion, device
            )

            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            
            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)
            fold_train_accs.append(train_acc)
            fold_val_accs.append((val_preds == val_labels).mean())
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {fold_val_accs[-1]:.4f}, Val F1: {val_f1:.4f}")
            
            scheduler.step()

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        model.load_state_dict(best_model_state)
        
        _, val_labels, val_preds, val_probs = evaluate(
            model, val_loader, all_gex, criterion, device
        )

        f1_macro = f1_score(val_labels, val_preds, average='macro')
        f1_weighted = f1_score(val_labels, val_preds, average='weighted')
        mcc = matthews_corrcoef(val_labels, val_preds)
        accuracy = accuracy_score(val_labels, val_preds)

        n_classes = 4
        val_labels_bin = label_binarize(val_labels, classes=[0, 1, 2, 3])
        pr_auc_per_class = []
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(val_labels_bin[:, i], val_probs[:, i])
            pr_auc_per_class.append(auc(recall, precision))
        pr_auc_macro = np.mean(pr_auc_per_class)
        
        all_metrics['f1_macro'].append(f1_macro)
        all_metrics['f1_weighted'].append(f1_weighted)
        all_metrics['mcc'].append(mcc)
        all_metrics['pr_auc'].append(pr_auc_macro)
        all_metrics['accuracy'].append(accuracy)
        all_metrics['train_loss'].append(fold_train_losses)
        all_metrics['val_loss'].append(fold_val_losses)
        all_metrics['train_acc'].append(fold_train_accs)
        all_metrics['val_acc'].append(fold_val_accs)
        
        print(f"Fold {fold} Results:")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"PR-AUC (Macro): {pr_auc_macro:.4f}")
        
        class_names = ['LumA', 'LumB', 'Basal', 'Her2']
        print(f"\nClassification Report:")
        print(classification_report(val_labels, val_preds, target_names=class_names, zero_division=0))
        
        print(f"\nPer-class PR-AUC:")
        for i, class_name in enumerate(class_names):
            print(f"{class_name}: {pr_auc_per_class[i]:.4f}")
        
        cm = confusion_matrix(val_labels, val_preds)
        print(f"\nConfusion Matrix:")
        #Rows: True labels LumA=0, LumB=1, Basal=2, Her2=3
        #Cols: Predicted labels
        print(cm)
        
        del model, optimizer, train_loader, val_loader, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("5-Fold Cross-Validation Result")

    print(f"Accuracy: {np.mean(all_metrics['accuracy']):.4f} ± {np.std(all_metrics['accuracy']):.4f}")
    print(f"F1-Score (Macro): {np.mean(all_metrics['f1_macro']):.4f} ± {np.std(all_metrics['f1_macro']):.4f}")
    print(f"F1-Score (Weighted): {np.mean(all_metrics['f1_weighted']):.4f} ± {np.std(all_metrics['f1_weighted']):.4f}")
    print(f"MCC: {np.mean(all_metrics['mcc']):.4f} ± {np.std(all_metrics['mcc']):.4f}")
    print(f"PR-AUC (Macro): {np.mean(all_metrics['pr_auc']):.4f} ± {np.std(all_metrics['pr_auc']):.4f}")
    
    print("Individual Fold Results:")
    for fold in range(5):
        print(f"Fold {fold+1}: Acc={all_metrics['accuracy'][fold]:.4f}, "
              f"F1(W)={all_metrics['f1_weighted'][fold]:.4f}, "
              f"MCC={all_metrics['mcc'][fold]:.4f}, "
              f"PR-AUC={all_metrics['pr_auc'][fold]:.4f}")

if __name__ == "__main__":
    main()
