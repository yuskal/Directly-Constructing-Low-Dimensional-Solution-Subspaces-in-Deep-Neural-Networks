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
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def set_seed(seed: int = 42):
    # 1. Standard Python & Numpy
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 2. PyTorch CPU & GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # für Multi-GPU
    # 3. CuDNN Determinismus (WICHTIG!)
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
    Einfaches, leichtes Cutout:
    - arbeitet auf Tensoren (C, H, W)
    - num_holes: Anzahl der Löcher
    - max_size: maximale Seitenlänge des Quadrats (im 32x32-Raum)
    """
    def __init__(self, num_holes: int = 1, max_size: int = 8):
        self.num_holes = num_holes
        self.max_size = max_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Erwartet Tensor (C, H, W)
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


def get_dataloaders(batch_size: int, num_workers: int = 4):
    """
    CIFAR-100 laden, mit ResNet50-kompatiblen Transforms.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02,
        ),
        transforms.ToTensor(),
        Cutout(num_holes=1, max_size=8),
        transforms.Resize(224),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


# ------------------------------------------------------------
# Modelldefinitionen
# ------------------------------------------------------------

def create_resnet50(num_classes: int = 100, pretrained: bool = True) -> nn.Module:
    """
    ResNet50 mit neuem FC-Head für CIFAR-100.
    """
    if pretrained:
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            model = models.resnet50(weights=weights)
        except AttributeError:
            model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def extract_backbone(model: nn.Module) -> nn.Module:
    """
    Backbone extrahieren, der 2048-dim Feature-Vektor ausgibt.
    FC wird durch Identity ersetzt.
    """
    backbone = model
    backbone.fc = nn.Identity()
    return backbone


class JLClassifier(nn.Module):
    """
    Modell für JL-Experimente:
    - backbone: eingefrorenes ResNet50, das 2048-dim Features liefert
    - proj_matrix: feste JL-Random-Matrix (2048 x proj_dim), nicht trainierbar
    - classifier: trainierbares Linear-Head auf projiziertem Raum
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
            features = self.backbone(x)  # (B, 2048)
        projected = torch.matmul(features, self.proj_matrix)  # (B, proj_dim)
        logits = self.classifier(projected)
        return logits


# ------------------------------------------------------------
# Trainings- und Evaluations-Loops
# ------------------------------------------------------------

def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion,
                    optimizer,
                    device: torch.device,
                    epoch_idx: int,
                    total_epochs: int,
                    desc_prefix: str = "Train"):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    pbar = tqdm(dataloader, desc=f"{desc_prefix} Epoch {epoch_idx+1}/{total_epochs}", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        acc = accuracy_from_logits(outputs, targets)

        running_loss += loss.item() * batch_size
        running_correct += acc * batch_size
        running_total += batch_size

        pbar.set_postfix({
            "loss": f"{running_loss / max(running_total, 1):.4f}",
            "acc": f"{running_correct / max(running_total, 1):.4f}",
        })

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = running_correct / max(running_total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             criterion,
             device: torch.device,
             desc_prefix: str = "Val"):
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

        pbar.set_postfix({
            "loss": f"{running_loss / max(running_total, 1):.4f}",
            "acc": f"{running_correct / max(running_total, 1):.4f}",
        })

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = running_correct / max(running_total, 1)
    return epoch_loss, epoch_acc


def finetune_full_resnet(model: nn.Module,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         device: torch.device,
                         epochs: int,
                         lr: float,
                         weight_decay: float):
    """
    PHASE 1: Vollständiges Finetuning von ResNet-50 (Backbone + Head)
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    print("==== [Phase 1] Starte vollständiges Finetuning (Backbone + Head) ====")
    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs, desc_prefix="Full FT"
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc_prefix="Full FT Val")

        print(f"[Full FT] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    print(f"==== [Phase 1] Fertig. Beste Val-Acc: {best_val_acc:.4f} ====")
    return model, best_val_acc


def train_linear_head_frozen_backbone(model: nn.Module,
                                      train_loader: DataLoader,
                                      val_loader: DataLoader,
                                      device: torch.device,
                                      epochs: int,
                                      lr: float,
                                      weight_decay: float,
                                      num_classes: int = 100):
    """
    PHASE 2: Backbone einfrieren, Head resetten und nur Head trainieren.
    Dies dient als 'Linear Probing' Baseline für die Features.
    """
    print("\n==== [Phase 2] Backbone einfrieren & nur linearen Head trainieren ====")
    
    # 1. Backbone einfrieren
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    
    # 2. FC-Layer neu initialisieren (Reset), damit wir fair trainieren
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.to(device)

    # Optimizer nur für die FC-Parameter
    optimizer = optim.SGD(
        model.fc.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs, desc_prefix="Frozen Backbone"
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc_prefix="Frozen Backbone Val")

        print(f"[Frozen BB] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    print(f"==== [Phase 2] Fertig. Beste Val-Acc (Frozen Backbone, Linear Head): {best_val_acc:.4f} ====")
    return model, best_val_acc


def run_jl_experiments(backbone: nn.Module,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       device: torch.device,
                       jl_dims: List[int],
                       epochs_head: int,
                       lr_head: float,
                       weight_decay_head: float,
                       num_classes: int = 100,
                       in_dim: int = 2048) -> Dict[int, float]:
    """
    PHASE 3: JL-Experimente.
    Backbone ist bereits eingefroren (durch extract_backbone oder Phase 2).
    """
    # Sicherstellen, dass Backbone eingefroren ist
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    backbone.to(device)

    results = {}

    for proj_dim in jl_dims:
        print(f"\n==== [Phase 3] Starte JL-Experiment mit Projektionsdimension {proj_dim} ====")
        model_jl = JLClassifier(backbone=backbone,
                                in_dim=in_dim,
                                proj_dim=proj_dim,
                                num_classes=num_classes)
        model_jl.to(device)

        params = [p for p in model_jl.parameters() if p.requires_grad]
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            params,
            lr=lr_head,
            momentum=0.9,
            weight_decay=weight_decay_head,
        )

        best_val_acc = 0.0
        for epoch in range(epochs_head):
            train_loss, train_acc = train_one_epoch(
                model_jl, train_loader, criterion, optimizer, device, epoch, epochs_head,
                desc_prefix=f"JL {proj_dim}"
            )
            val_loss, val_acc = evaluate(
                model_jl, val_loader, criterion, device,
                desc_prefix=f"JL {proj_dim} Val"
            )

            print(f"[JL {proj_dim}] Epoch {epoch+1}/{epochs_head} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        print(f"==== JL {proj_dim}: Beste Val-Acc: {best_val_acc:.4f} ====")
        results[proj_dim] = best_val_acc

    return results


def plot_results(results: Dict[int, float], baseline_acc: float, output_path: str):
    """
    Plot Val-Accuracy vs. Projektionsdimension inkl. Baseline.
    """
    if not results:
        print("Keine Ergebnisse zum Plotten vorhanden.")
        return

    dims = sorted(results.keys())
    accs = [results[d] for d in dims]

    plt.figure()
    plt.plot(dims, accs, marker="o", label="JL Projection")
    # Baseline Linie zeichnen (Ergebnis von Phase 2)
    plt.axhline(y=baseline_acc, color='r', linestyle='--', label=f"Baseline (Frozen 2048D): {baseline_acc:.4f}")
    
    plt.xlabel("Projektionsdimension (JL)")
    plt.ylabel("Validierungs-Accuracy")
    plt.title("JL-Projektion auf CIFAR-100: Val-Acc vs. Dimension")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot gespeichert unter: {output_path}")


# ------------------------------------------------------------
# Argument Parsing und Main
# ------------------------------------------------------------

def parse_jl_dims(dim_string: str) -> List[int]:
    dims = []
    for token in dim_string.split(","):
        token = token.strip()
        if not token:
            continue
        dims.append(int(token))
    return dims


def main():
    parser = argparse.ArgumentParser(
        description="ResNet50 CIFAR-100: Full FT -> Linear Probing -> JL Experiments"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batchgröße")
    parser.add_argument("--epochs-full", type=int, default=10,
                        help="Epochen für Phase 1 (Full Finetune)")
    parser.add_argument("--epochs-head", type=int, default=5,
                        help="Epochen für Phase 2 & 3 (Linear Head Training)")
    parser.add_argument("--lr-full", type=float, default=0.01,
                        help="Lernrate für Full Finetuning")
    parser.add_argument("--lr-head", type=float, default=0.01,
                        help="Lernrate für Head-Training")
    parser.add_argument("--weight-decay-full", type=float, default=5e-4,
                        help="Weight Decay für Full Finetuning")
    parser.add_argument("--weight-decay-head", type=float, default=5e-4,
                        help="Weight Decay für Head-Training")
    parser.add_argument("--jl-dims", type=str, default="1024,512,256,128",
                        help="Komma-separierte JL-Dimensionen")
    parser.add_argument("--seed", type=int, default=42, help="Zufallsseed")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader-Worker")
    parser.add_argument("--output-dir", type=str, default="./results_cifar100_v2",
                        help="Ausgabeverzeichnis für Plot")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Benutze Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, "jl_projection_final.png")

    jl_dims = parse_jl_dims(args.jl_dims)
    print(f"JL-Projektionsdimensionen: {jl_dims}")

    # Daten laden
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = 100

    # ---------------------------------------------------------
    # 1) Phase 1: Vollständiges Finetuning von ResNet-50
    # ---------------------------------------------------------
    model = create_resnet50(num_classes=num_classes, pretrained=True)
    model, acc_phase1 = finetune_full_resnet(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs_full,
        lr=args.lr_full,
        weight_decay=args.weight_decay_full,
    )
    print(f"\n---> Ergebnis Phase 1 (Full Finetune): {acc_phase1:.4f}")

    # ---------------------------------------------------------
    # 2) Phase 2: Backbone einfrieren, Head resetten & trainieren
    #    (Linear Probing Baseline)
    # ---------------------------------------------------------
    model, acc_phase2 = train_linear_head_frozen_backbone(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs_head,
        lr=args.lr_head,
        weight_decay=args.weight_decay_head,
        num_classes=num_classes
    )
    print(f"\n---> Ergebnis Phase 2 (Frozen Backbone + Linear Head): {acc_phase2:.4f}")

    # ---------------------------------------------------------
    # 3) Phase 3: JL-Experimente
    # ---------------------------------------------------------
    # Jetzt extrahieren wir den Backbone (entfernt den FC layer komplett)
    backbone = extract_backbone(model)

    jl_results = run_jl_experiments(
        backbone=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        jl_dims=jl_dims,
        epochs_head=args.epochs_head,
        lr_head=args.lr_head,
        weight_decay_head=args.weight_decay_head,
        num_classes=num_classes,
        in_dim=2048,
    )

    print("\n==== Übersicht Ergebnisse ====")
    print(f"Phase 1 (Full FT): {acc_phase1:.4f}")
    print(f"Phase 2 (Frozen 2048D Baseline): {acc_phase2:.4f}")
    print("Phase 3 (JL Projections):")
    for d in sorted(jl_results.keys()):
        print(f"  Dim {d}: {jl_results[d]:.4f}")

    # 4) Plot erstellen (Vergleich JL vs. Phase 2 Baseline)
    plot_results(jl_results, baseline_acc=acc_phase2, output_path=plot_path)


if __name__ == "__main__":
    main()
