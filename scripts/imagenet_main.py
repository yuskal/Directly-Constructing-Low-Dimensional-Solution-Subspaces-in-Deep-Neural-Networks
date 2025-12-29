#!/usr/bin/env python3
import argparse
import os
import random
import math
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Hugging Face Imports
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
from datasets import load_dataset

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Vollständiger deterministischer Modus auf Seed {seed} gesetzt.")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class Cutout(object):
    """
    Cutout Implementierung für Tensor-Inputs (C, H, W).
    """
    def __init__(self, num_holes: int = 1, max_size: int = 32):
        self.num_holes = num_holes
        self.max_size = max_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(img):
            return img
        _, h, w = img.shape
        for _ in range(self.num_holes):
            hole_size = random.randint(1, self.max_size)
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            y1 = max(0, y - hole_size // 2)
            y2 = min(h, y + hole_size // 2)
            x1 = max(0, x - hole_size // 2)
            x2 = min(w, x + hole_size // 2)
            img[:, y1:y2, x1:x2] = 0.0
        return img

# ------------------------------------------------------------
# Daten Laden via Hugging Face Datasets
# ------------------------------------------------------------

def get_hf_dataloaders(model_id: str, dataset_id: str, batch_size: int, num_workers: int = 4, seed: int = 42):
    """
    Lädt ImageNet-100 via Hugging Face 'datasets' und nutzt 'transformers' Preprocessing.
    """
    print(f"⏳ Lade Dataset: {dataset_id}...")
    # Dataset laden (lädt automatisch herunter und cached es in ~/.cache/huggingface/datasets)
    dataset = load_dataset(dataset_id)
    
    # Processor laden (für Resize und Normalization passend zum Modell)
    processor = ViTImageProcessor.from_pretrained(model_id)

    # Cutout Instanz
    cutout = Cutout(num_holes=1, max_size=32)

    # Transformations-Funktion
    def transform(examples):
        # 1. Bildkonvertierung und Preprocessing (Resize + Norm)
        # Processor gibt Pixel-Values zurück
        inputs = processor([img.convert("RGB") for img in examples['image']], return_tensors='pt')
        
        # 2. Augmentations (Cutout) manuell anwenden
        # Da der Processor normierte Tensoren zurückgibt, können wir Cutout direkt darauf anwenden
        augmented_images = []
        for img_tensor in inputs['pixel_values']:
            # Hier könnten wir theoretisch auch RandomHorizontalFlip etc. einbauen, 
            # wenn wir torchvision transforms nutzen würden. 
            # Der Einfachheit halber hier nur Cutout wie im Originalcode.
            img_tensor = cutout(img_tensor)
            augmented_images.append(img_tensor)
            
        inputs['pixel_values'] = augmented_images
        inputs['labels'] = examples['label']
        return inputs

    # Einfache Val-Transform ohne Cutout
    def val_transform(examples):
        inputs = processor([img.convert("RGB") for img in examples['image']], return_tensors='pt')
        inputs['labels'] = examples['label']
        return inputs

    # Auf Dataset anwenden
    # Wir nutzen set_transform, damit es "on-the-fly" passiert und Speicher spart
    train_ds = dataset['train'].shuffle(seed=seed) 
    train_ds.set_transform(transform)

    # Validierungsset (prüfen ob 'validation' oder 'valid' key existiert)
    val_key = 'validation' if 'validation' in dataset else 'valid'
    val_ds = dataset[val_key]
    val_ds.set_transform(val_transform)

    # Collate Funktion: Wandelt die Liste von Dicts in Tensors um
    # WICHTIG: Gibt ein Tuple (images, targets) zurück, damit der alte Trainingsloop funktioniert!
    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch])
        return pixel_values, labels

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    # Klassenanzahl ermitteln
    labels_feat = dataset['train'].features['label']
    num_classes = labels_feat.num_classes if hasattr(labels_feat, 'num_classes') else 100

    return train_loader, val_loader, num_classes

# ------------------------------------------------------------
# Modelldefinitionen (ViT Wrapper)
# ------------------------------------------------------------

class ViTWrapper(nn.Module):
    """
    Wrapper um Hugging Face ViT, damit er sich wie ein Standard PyTorch Modell verhält.
    Gibt nur Logits zurück, kein HF-Output-Objekt.
    """
    def __init__(self, model_id: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.model = ViTForImageClassification.from_pretrained(
                model_id, 
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = ViTConfig.from_pretrained(model_id, num_labels=num_classes)
            self.model = ViTForImageClassification(config)
            
    def forward(self, x):
        # HF Models erwarten pixel_values argument
        outputs = self.model(pixel_values=x)
        return outputs.logits

    def get_hidden_dim(self):
        return self.model.config.hidden_size

def extract_backbone_vit_hf(wrapper_model: nn.Module) -> nn.Module:
    """
    Extrahiert den Backbone aus dem Wrapper.
    Wir nutzen den internen 'vit' Teil des HF Modells.
    """
    # Das innere Modell ist wrapper_model.model (ViTForImageClassification)
    # Dessen Backbone ist wrapper_model.model.vit
    backbone = wrapper_model.model.vit
    
    # Wir müssen sicherstellen, dass der Forward Pass des Backbones
    # den [CLS] Token (pooler_output oder last_hidden_state[:,0]) zurückgibt.
    
    class BackboneWrapper(nn.Module):
        def __init__(self, vit_module):
            super().__init__()
            self.vit = vit_module
            
        def forward(self, x):
            outputs = self.vit(pixel_values=x)
            # ViTForImageClassification nutzt standardmäßig den [CLS] token (index 0)
            # und jagt ihn durch einen Layernorm.
            # outputs.last_hidden_state ist (Batch, Seq, Hidden)
            cls_token = outputs.last_hidden_state[:, 0, :]
            return cls_token

    return BackboneWrapper(backbone)

class JLClassifier(nn.Module):
    """
    Identisch zum Original: JL Projektion + Linearer Classifier.
    """
    def __init__(self, backbone: nn.Module, in_dim: int, proj_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        # JL Projektion: Random Matrix, fixiert
        proj = torch.randn(in_dim, proj_dim) / math.sqrt(proj_dim)
        self.register_buffer("proj_matrix", proj)
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)  # (B, hidden_dim)
        projected = torch.matmul(features, self.proj_matrix)  # (B, proj_dim)
        logits = self.classifier(projected)
        return logits

# ------------------------------------------------------------
# Trainings- und Evaluations-Loops
# ------------------------------------------------------------

def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: torch.device,
                    epoch_idx: int, total_epochs: int, desc_prefix: str = "Train"):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    pbar = tqdm(dataloader, desc=f"{desc_prefix} Ep {epoch_idx+1}/{total_epochs}", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        # Dank Wrapper gibt model(images) direkt Logits zurück
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        acc = accuracy_from_logits(outputs, targets)
        running_loss += loss.item() * batch_size
        running_correct += acc * batch_size
        running_total += batch_size

        pbar.set_postfix({"loss": f"{running_loss/running_total:.4f}", "acc": f"{running_correct/running_total:.4f}"})

    return running_loss / max(running_total, 1), running_correct / max(running_total, 1)

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device: torch.device, desc_prefix: str = "Val"):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    pbar = tqdm(dataloader, desc=f"{desc_prefix}", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        acc = accuracy_from_logits(outputs, targets)
        running_loss += loss.item() * batch_size
        running_correct += acc * batch_size
        running_total += batch_size
        pbar.set_postfix({"loss": f"{running_loss/running_total:.4f}", "acc": f"{running_correct/running_total:.4f}"})

    return running_loss / max(running_total, 1), running_correct / max(running_total, 1)

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def finetune_full_vit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                      device: torch.device, epochs: int, lr: float, weight_decay: float):
    """ Phase 1: Full Finetuning """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # AdamW ist besser für Transformer als SGD
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("==== [Phase 1] Starte vollständiges ViT Finetuning ====")
    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs, desc_prefix="Full FT"
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc_prefix="Full FT Val")
        print(f"[Full FT] Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    return model, best_val_acc

def train_linear_head_frozen_backbone(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                                      device: torch.device, epochs: int, lr: float, weight_decay: float,
                                      num_classes: int, in_features: int):
    """ Phase 2: Frozen Backbone """
    print("\n==== [Phase 2] Backbone einfrieren & nur Head trainieren ====")
    
    # 1. Alles einfrieren
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Classifier (Head) neu initialisieren und aktivieren
    # Im Wrapper ist das model.model.classifier
    model.model.classifier = nn.Linear(in_features, num_classes)
    for param in model.model.classifier.parameters():
        param.requires_grad = True
        
    model.to(device)

    optimizer = optim.AdamW(model.model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs, desc_prefix="Frozen BB"
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc_prefix="Frozen BB Val")
        print(f"[Frozen BB] Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc: best_val_acc = val_acc
    return model, best_val_acc

def run_jl_experiments(backbone: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                       device: torch.device, jl_dims: List[int], epochs_head: int, lr_head: float,
                       weight_decay_head: float, num_classes: int, in_dim: int) -> Dict[int, float]:
    """ Phase 3: JL Experimente """
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    backbone.to(device)

    results = {}
    for proj_dim in jl_dims:
        print(f"\n==== [Phase 3] JL-Experiment: Dim {in_dim} -> {proj_dim} ====")
        model_jl = JLClassifier(backbone=backbone, in_dim=in_dim, proj_dim=proj_dim, num_classes=num_classes)
        model_jl.to(device)

        # Auch hier nutzen wir AdamW statt SGD, da es robuster für ViT features ist
        optimizer = optim.AdamW(model_jl.classifier.parameters(), lr=lr_head, weight_decay=weight_decay_head)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        for epoch in range(epochs_head):
            _, train_acc = train_one_epoch(model_jl, train_loader, criterion, optimizer, device, epoch, epochs_head, desc_prefix=f"JL {proj_dim}")
            _, val_acc = evaluate(model_jl, val_loader, criterion, device, desc_prefix=f"JL {proj_dim} Val")
            if val_acc > best_val_acc: best_val_acc = val_acc
        print(f"==== JL {proj_dim}: Beste Val-Acc: {best_val_acc:.4f} ====")
        results[proj_dim] = best_val_acc
    return results

def plot_results(results: Dict[int, float], baseline_acc: float, output_path: str, baseline_dim: int):
    if not results: return
    dims = sorted(results.keys())
    accs = [results[d] for d in dims]
    plt.figure(figsize=(8, 6))
    
    plt.plot(dims, accs, marker="o", linestyle="-", color="b", label="JL Projection")
    plt.axhline(y=baseline_acc, color='r', linestyle='--', label=f"Baseline ({baseline_dim}D): {baseline_acc:.4f}")
    plt.xlabel("Projektionsdimension (JL)")
    plt.ylabel("Validierungs-Accuracy")
    plt.title(f"ViT auf ImageNet-100: JL Random Projection")
    plt.grid(True)
    plt.xscale('log')
    plt.xticks(dims + [baseline_dim], labels=[str(d) for d in dims + [baseline_dim]])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot gespeichert unter: {output_path}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_jl_dims(dim_string: str) -> List[int]:
    dims = []
    for token in dim_string.split(","):
        token = token.strip()
        if not token: continue
        dims.append(int(token))
    return dims

def main():
    parser = argparse.ArgumentParser(description="ViT ImageNet-100 (HF Datasets): Full FT -> Linear -> JL")
    parser.add_argument("--model-id", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--dataset-id", type=str, default="clane9/imagenet-100")
    
    parser.add_argument("--batch-size", type=int, default=32, help="Batchgröße")
    parser.add_argument("--epochs-full", type=int, default=3, help="Epochen Phase 1")
    parser.add_argument("--epochs-head", type=int, default=3, help="Epochen Phase 2 & 3")
    
    # Lernraten (Standardwerte angepasst für ViT/AdamW)
    parser.add_argument("--lr-full", type=float, default=2e-5, help="Lernrate Full FT")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Lernrate Head Training")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    
    parser.add_argument("--jl-dims", type=str, default="512,256,128,64")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./results_vit_hf")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Benutze Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Daten Laden (HF Datasets)
    train_loader, val_loader, num_classes = get_hf_dataloaders(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        batch_size=args.batch_size,
        seed=args.seed
    )
    print(f"Klassen gefunden: {num_classes}")

    # 2. Phase 1: Full Finetuning
    # Erstellt Wrapper-Modell
    model = ViTWrapper(model_id=args.model_id, num_classes=num_classes, pretrained=True)
    vit_hidden_dim = model.get_hidden_dim() # z.B. 768

    model, acc_phase1 = finetune_full_vit(
        model, train_loader, val_loader, device,
        epochs=args.epochs_full, lr=args.lr_full, weight_decay=args.weight_decay
    )
    print(f"\n---> Ergebnis Phase 1 (Full Finetune): {acc_phase1:.4f}")

    # 3. Phase 2: Frozen Backbone Baseline
    model, acc_phase2 = train_linear_head_frozen_backbone(
        model, train_loader, val_loader, device,
        epochs=args.epochs_head, lr=args.lr_head, weight_decay=args.weight_decay,
        num_classes=num_classes, in_features=vit_hidden_dim
    )
    print(f"\n---> Ergebnis Phase 2 (Frozen Backbone): {acc_phase2:.4f}")

    # 4. Phase 3: JL Experimente
    # Backbone extrahieren (Wrapper)
    backbone = extract_backbone_vit_hf(model)

    jl_dims = parse_jl_dims(args.jl_dims)
    jl_results = run_jl_experiments(
        backbone=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        jl_dims=jl_dims,
        epochs_head=args.epochs_head,
        lr_head=args.lr_head,
        weight_decay_head=args.weight_decay,
        num_classes=num_classes,
        in_dim=vit_hidden_dim
    )

    # 5. Plot
    plot_path = os.path.join(args.output_dir, "vit_jl_projection.png")
    plot_results(jl_results, baseline_acc=acc_phase2, output_path=plot_path, baseline_dim=vit_hidden_dim)

if __name__ == "__main__":
    main()
