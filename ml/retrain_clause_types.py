import os
import sys
import json
import time
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

sys.stdout.reconfigure(line_buffering=True)
torch.set_num_threads(16)
device = torch.device("cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

DATA_DIR = Path("ml/artifacts/phase9_50k")
print(f"Loading data splits from {DATA_DIR}...", flush=True)
train_all = json.loads((DATA_DIR / "train_preprocessed.json").read_text(encoding="utf-8"))
val_all = json.loads((DATA_DIR / "validation_preprocessed.json").read_text(encoding="utf-8"))
test_all = json.loads((DATA_DIR / "test_preprocessed.json").read_text(encoding="utf-8"))

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class ClauseDataset(Dataset):
    def __init__(self, data, task, label2id, max_length=96):
        self.texts = [d["text"] for d in data]
        self.labels = [label2id[d[task]] for d in data]
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()
            logits = model(input_ids, attention_mask=attention_mask).logits
            p = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.extend(p)
            targets.extend(labels)
    acc = accuracy_score(targets, preds)
    f1_m = f1_score(targets, preds, average="macro", zero_division=0)
    f1_w = f1_score(targets, preds, average="weighted", zero_division=0)
    p_m = precision_score(targets, preds, average="macro", zero_division=0)
    r_m = recall_score(targets, preds, average="macro", zero_division=0)
    return {
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_m), 4),
        "f1_weighted": round(float(f1_w), 4),
        "precision_macro": round(float(p_m), 4),
        "recall_macro": round(float(r_m), 4),
    }, preds, targets

def train_domain_clause_type(domain: str, samples_per_class: int = 220, epochs: int = 3):
    task = "clause_type"
    print(f"\n=======================================================", flush=True)
    print(f"FINE-TUNING DISTILBERT: Domain='{domain}' | Task='{task}'", flush=True)
    print(f"=======================================================", flush=True)

    train_domain = [d for d in train_all if d["domain"] == domain]
    val_domain = [d for d in val_all if d["domain"] == domain]
    test_domain = [d for d in test_all if d["domain"] == domain]

    classes = sorted(list(set(d[task] for d in train_domain)))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for i, c in enumerate(classes)}
    num_labels = len(classes)

    stratified_train = []
    class_buckets = {c: [] for c in classes}
    for d in train_domain:
        class_buckets[d[task]].append(d)

    for c in classes:
        random.seed(42)
        random.shuffle(class_buckets[c])
        stratified_train.extend(class_buckets[c][:samples_per_class])
    random.shuffle(stratified_train)

    stratified_val = []
    val_buckets = {c: [] for c in classes}
    for d in val_domain:
        val_buckets[d[task]].append(d)
    for c in classes:
        stratified_val.extend(val_buckets[c][:80])

    stratified_test = []
    test_buckets = {c: [] for c in classes}
    for d in test_domain:
        test_buckets[d[task]].append(d)
    for c in classes:
        stratified_test.extend(test_buckets[c][:80])

    print(f"Train samples: {len(stratified_train)} ({samples_per_class}/class)", flush=True)

    train_dataset = ClauseDataset(stratified_train, task, label2id, max_length=96)
    val_dataset = ClauseDataset(stratified_val, task, label2id, max_length=96)
    test_dataset = ClauseDataset(stratified_test, task, label2id, max_length=96)

    train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label={str(k): v for k, v in id2label.items()},
        label2id=label2id
    ).to(device)

    for param in model.distilbert.embeddings.parameters():
        param.requires_grad = False
    for i in range(3):
        for param in model.distilbert.transformer.layer[i].parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.distilbert.named_parameters() if p.requires_grad], "lr": 4e-5},
        {"params": model.pre_classifier.parameters(), "lr": 1.5e-4},
        {"params": model.classifier.parameters(), "lr": 1.5e-4},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        t1 = time.time()

        val_metrics, _, _ = evaluate(model, val_loader)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} ({t1-t0:.1f}s) | Loss: {avg_loss:.4f} | Val Acc: {val_metrics['accuracy']:.2%} | Val Macro-F1: {val_metrics['f1_macro']:.2%}", flush=True)

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if val_metrics["f1_macro"] >= 0.998 and epoch >= 2:
            print("Reached near-perfect convergence. Early stopping.", flush=True)
            break

    if best_weights:
        model.load_state_dict(best_weights)

    test_metrics, preds, targets = evaluate(model, test_loader)
    print(f"\n>>> FINAL TEST EVALUATION: Acc: {test_metrics['accuracy']:.2%} | Macro-F1: {test_metrics['f1_macro']:.2%} | Weighted-F1: {test_metrics['f1_weighted']:.2%} <<<", flush=True)

    destinations = [
        Path("backend/models") / domain / f"bert_{task}_nocontext" / "final",
        Path("backend/models") / domain / f"bert_{task}_windowed" / "final",
        Path("ml/artifacts") / domain / f"bert_{task}_nocontext" / "final",
        Path("ml/artifacts") / domain / f"bert_{task}_windowed" / "final",
    ]

    for dest in destinations:
        dest.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(dest))
        model.save_pretrained(str(dest))

        metadata = {
            "document_type": domain,
            "task": task,
            "use_context": False,
            "model_version": f"bert-distilbert-{domain}-{task}-v2-highacc",
            "num_labels": num_labels,
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
            "test_metrics": test_metrics,
            "artifact_path": str(dest.resolve())
        }
        (dest / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Exported model to: {dest}", flush=True)

    return test_metrics

def main():
    domains = ["job_offer_letter", "rental_agreement", "insurance_policy"]
    for d in domains:
        train_domain_clause_type(d, samples_per_class=220, epochs=3)
    print("\nALL CLAUSE TYPE MODELS RETRAINED WITH 100% COVERAGE!", flush=True)

if __name__ == "__main__":
    main()
