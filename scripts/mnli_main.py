#!/usr/bin/env python3
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import math
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Hugging Face Imports für NLP
from transformers import AutoTokenizer, BertForSequenceClassification, BertModel, DataCollatorWithPadding
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

# ------------------------------------------------------------
# Daten Laden (GLUE MNLI)
# ------------------------------------------------------------

def get_mnli_dataloaders(model_id: str, batch_size: int, num_workers: int = 2, seed: int = 42, max_length: int = 128):
    """
    Lädt den MNLI Datensatz (GLUE) und tokenisiert ihn für BERT.
    """
    print(f"⏳ Lade Dataset: glue/mnli...")
    # MNLI hat 3 Klassen: 0=Entailment, 1=Neutral, 2=Contradiction
    dataset = load_dataset("glue", "mnli")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def preprocess_function(examples):
        # BERT erwartet Paar-Input: (Premise, Hypothesis)
        return tokenizer(
            examples['premise'], 
            examples['hypothesis'], 
            truncation=True, 
            max_length=max_length
        )

    # Tokenisierung anwenden
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    # Wir müssen unnötige Spalten entfernen, damit der DataLoader nur Tensoren sieht
    tokenized_datasets = tokenized_datasets.remove_columns(['premise', 'hypothesis', 'idx'])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # Data Collator übernimmt das Padding innerhalb des Batches (effizienter als statisches Padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = tokenized_datasets['train'].shuffle(seed=seed)
    
    # MNLI hat 'validation_matched' und 'validation_mismatched'. 
    # Standard für Dev ist 'validation_matched'.
    val_ds = tokenized_datasets['validation_matched']

    # Wir nutzen eine kleine Untermenge für schnelleres Experimentieren, falls gewünscht. 
    # Um das "exakte" Experiment zu haben, nutzen wir hier alles, aber MNLI ist RIESIG (392k Beispiele).
    # Für dieses Demo-Skript begrenzen wir es optional, sonst dauert Phase 1 Tage auf einer GPU.
    # Hier nehmen wir 20% des Train-Sets als Beispiel, aber kommentiere es aus für Full Run.
    # train_ds = train_ds.select(range(int(len(train_ds)*0.2))) 

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=data_collator, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=data_collator, pin_memory=True
    )

    num_classes = 3
    return train_loader, val_loader, num_classes

# ------------------------------------------------------------
# Modelldefinitionen (BERT Wrapper)
# ------------------------------------------------------------

class BertWrapper(nn.Module):
    """
    Wrapper um Hugging Face BertForSequenceClassification.
    Gibt nur Logits zurück.
    """
    def __init__(self, model_id: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.model = BertForSequenceClassification.from_pretrained(
                model_id, 
                num_labels=num_classes
            )
        else:
            from transformers import BertConfig
            config = BertConfig.from_pretrained(model_id, num_labels=num_classes)
            self.model = BertForSequenceClassification(config)
            
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # HF Models geben ein Tuple/Object zurück. Wir wollen nur Logits.
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        return outputs.logits

    def get_hidden_dim(self):
        return self.model.config.hidden_size

def extract_backbone_bert(wrapper_model: nn.Module) -> nn.Module:
    """
    Extrahiert den BERT-Teil (Backbone).
    Wir wollen den Vektor erhalten, der normalerweise in den Classifier geht.
    Bei BERT ist das der 'Pooler Output' (CLS Token -> Dense -> Tanh).
    """
    # wrapper_model.model ist BertForSequenceClassification
    # wrapper_model.model.bert ist das BertModel
    original_bert = wrapper_model.model.bert
    
    class BackboneWrapper(nn.Module):
        def __init__(self, bert_module):
            super().__init__()
            self.bert = bert_module
            
        def forward(self, input_ids, attention_mask, token_type_ids=None):
            outputs = self.bert(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            # pooler_output ist (Batch, 768). 
            # Das ist die Repräsentation des ganzen Satzes ([CLS]), die für Klassifikation genutzt wird.
            return outputs.pooler_output

    return BackboneWrapper(original_bert)

class JLClassifier(nn.Module):
    """
    JL Projektion für NLP Embeddings.
    Identisch zum Vision-Code, da es nur auf Vektoren arbeitet.
    """
    def __init__(self, backbone: nn.Module, in_dim: int, proj_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        # JL Projektion: Random Matrix, fixiert
        proj = torch.randn(in_dim, proj_dim) / math.sqrt(proj_dim)
        self.register_buffer("proj_matrix", proj)
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        with torch.no_grad():
            # Backbone liefert (B, 768)
            features = self.backbone(input_ids, attention_mask, token_type_ids)
            
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
    for batch in pbar:
        # Batch auf Device schieben
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels") # Labels extrahieren

        optimizer.zero_grad()
        
        # Forward Pass (Die Wrapper Klasse und JL Klasse akzeptieren **batch Argumente nicht direkt sauber beim Unpacken wenn extra keys da sind,
        # daher explizit: input_ids und attention_mask sind immer da)
        outputs = model(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'],
            token_type_ids=batch.get('token_type_ids')
        )
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy_from_logits(outputs, labels)
        
        batch_size = labels.size(0)
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
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")

        outputs = model(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'],
            token_type_ids=batch.get('token_type_ids')
        )
        loss = criterion(outputs, labels)

        acc = accuracy_from_logits(outputs, labels)
        
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += acc * batch_size
        running_total += batch_size
        pbar.set_postfix({"loss": f"{running_loss/running_total:.4f}", "acc": f"{running_correct/running_total:.4f}"})

    return running_loss / max(running_total, 1), running_correct / max(running_total, 1)

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def finetune_full_bert(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                      device: torch.device, epochs: int, lr: float, weight_decay: float):
    """ Phase 1: Full Finetuning (BERT + Head) """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Für BERT Full Finetuning ist eine sehr kleine LR (2e-5) üblich
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("==== [Phase 1] Starte vollständiges BERT Finetuning ====")
    best_val_acc = 0.0
    for epoch in range(epochs):
        _, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs, desc_prefix="Full FT"
        )
        _, val_acc = evaluate(model, val_loader, criterion, device, desc_prefix="Full FT Val")
        print(f"[Full FT] Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Wir könnten hier das Modell speichern, um das beste zu behalten
    return model, best_val_acc

def train_linear_head_frozen_backbone(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                                      device: torch.device, epochs: int, lr: float, weight_decay: float,
                                      num_classes: int, in_features: int):
    """ Phase 2: Frozen BERT, neuer Head """
    print("\n==== [Phase 2] BERT einfrieren & nur Head trainieren ====")
    
    # 1. Alles einfrieren
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Classifier neu initialisieren
    # Bei BertForSequenceClassification heißt der Layer 'classifier'
    model.model.classifier = nn.Linear(in_features, num_classes)
    for param in model.model.classifier.parameters():
        param.requires_grad = True
        
    model.to(device)

    # Hier können wir eine höhere LR nehmen als beim Full FT, da wir nur den Head trainieren
    optimizer = optim.AdamW(model.model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        _, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs, desc_prefix="Frozen BB"
        )
        _, val_acc = evaluate(model, val_loader, criterion, device, desc_prefix="Frozen BB Val")
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
    plt.ylabel("Validierungs-Accuracy (MNLI)")
    plt.title(f"BERT-base auf MNLI: JL Random Projection")
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
    parser = argparse.ArgumentParser(description="BERT MNLI: Full FT -> Linear -> JL")
    parser.add_argument("--model-id", type=str, default="bert-base-uncased")
    
    # MNLI ist groß, wir nehmen größere Batches wenn VRAM reicht, oder Gradient Accumulation (hier vereinfacht)
    parser.add_argument("--batch-size", type=int, default=32, help="Batchgröße")
    
    # Standard Epochen für NLP Fine-Tuning sind meist 3-4
    parser.add_argument("--epochs-full", type=int, default=3, help="Epochen Phase 1")
    parser.add_argument("--epochs-head", type=int, default=3, help="Epochen Phase 2 & 3")
    
    # Lernraten: Full FT braucht sehr kleine LR bei BERT!
    parser.add_argument("--lr-full", type=float, default=2e-5, help="Lernrate Full FT")
    # Linear Probing verträgt höhere LR
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Lernrate Head Training")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    
    parser.add_argument("--jl-dims", type=str, default="512,256,128,64")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./results_bert_mnli")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Benutze Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Daten Laden (Hugging Face Datasets)
    train_loader, val_loader, num_classes = get_mnli_dataloaders(
        model_id=args.model_id,
        batch_size=args.batch_size,
        seed=args.seed
    )
    print(f"Klassen: {num_classes} (Entailment, Neutral, Contradiction)")

    # 2. Phase 1: Full Finetuning
    # Erstellt Wrapper-Modell
    model = BertWrapper(model_id=args.model_id, num_classes=num_classes, pretrained=True)
    bert_hidden_dim = model.get_hidden_dim() # 768 bei bert-base

    model, acc_phase1 = finetune_full_bert(
        model, train_loader, val_loader, device,
        epochs=args.epochs_full, lr=args.lr_full, weight_decay=args.weight_decay
    )
    print(f"\n---> Ergebnis Phase 1 (Full Finetune): {acc_phase1:.4f}")

    # 3. Phase 2: Frozen Backbone Baseline
    model, acc_phase2 = train_linear_head_frozen_backbone(
        model, train_loader, val_loader, device,
        epochs=args.epochs_head, lr=args.lr_head, weight_decay=args.weight_decay,
        num_classes=num_classes, in_features=bert_hidden_dim
    )
    print(f"\n---> Ergebnis Phase 2 (Frozen Backbone): {acc_phase2:.4f}")

    # 4. Phase 3: JL Experimente
    # Backbone extrahieren (Pooler Output)
    backbone = extract_backbone_bert(model)

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
        in_dim=bert_hidden_dim
    )

    # 5. Plot
    plot_path = os.path.join(args.output_dir, "bert_jl_projection.png")
    plot_results(jl_results, baseline_acc=acc_phase2, output_path=plot_path, baseline_dim=bert_hidden_dim)

if __name__ == "__main__":
    main()
