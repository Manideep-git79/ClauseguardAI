import os
import sys
import json
import time
import random
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

# Ensure unbuffered output
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
print(f"Splits loaded: Train={len(train_all):,}, Val={len(val_all):,}, Test={len(test_all):,}", flush=True)

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

def train_and_export_task(domain: str, task: str, samples_per_class: int = 160, epochs: int = 3):
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

    print(f"Classes ({num_labels}): {classes}", flush=True)

    # Stratified balanced sampling
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
        stratified_val.extend(val_buckets[c][:120])

    stratified_test = []
    test_buckets = {c: [] for c in classes}
    for d in test_domain:
        test_buckets[d[task]].append(d)
    for c in classes:
        stratified_test.extend(test_buckets[c][:120])

    print(f"Dataset Sizes: Train={len(stratified_train)} ({samples_per_class}/class), Val={len(stratified_val)}, Test={len(stratified_test)}", flush=True)

    train_dataset = ClauseDataset(stratified_train, task, label2id, max_length=96)
    val_dataset = ClauseDataset(stratified_val, task, label2id, max_length=96)
    test_dataset = ClauseDataset(stratified_test, task, label2id, max_length=96)

    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label={str(k): v for k, v in id2label.items()},
        label2id=label2id
    ).to(device)

    # Freeze embeddings and layers 0-2; train layers 3-5 + classification head
    for param in model.distilbert.embeddings.parameters():
        param.requires_grad = False
    for i in range(3):
        for param in model.distilbert.transformer.layer[i].parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total:.1%})", flush=True)

    # Differential learning rate
    lr_bert = 4e-5 if task == "clause_type" else 5e-5
    lr_head = 1.5e-4 if task == "clause_type" else 2e-4

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.distilbert.named_parameters() if p.requires_grad], "lr": lr_bert},
        {"params": model.pre_classifier.parameters(), "lr": lr_head},
        {"params": model.classifier.parameters(), "lr": lr_head},
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

        if val_metrics["f1_macro"] >= 0.995 and epoch >= 2:
            print("Reached near-perfect convergence (Val Macro-F1 >= 99.5%). Early stopping.", flush=True)
            break

    if best_weights:
        model.load_state_dict(best_weights)

    test_metrics, preds, targets = evaluate(model, test_loader)
    print(f"\n>>> FINAL TEST EVALUATION: Acc: {test_metrics['accuracy']:.2%} | Macro-F1: {test_metrics['f1_macro']:.2%} | Weighted-F1: {test_metrics['f1_weighted']:.2%} <<<", flush=True)

    report_str = classification_report(targets, preds, target_names=classes, zero_division=0)
    print("Classification Report:\n" + report_str, flush=True)

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
    print("=" * 65, flush=True)
    print("DISTILBERT MASTER 95%+ TRAINING & DEPLOYMENT PIPELINE", flush=True)
    print("=" * 65, flush=True)

    configs = [
        # Domain, Task, samples_per_class, epochs
        ("rental_agreement", "clause_type", 600, 3),
        ("rental_agreement", "favorability", 1500, 3),
        ("job_offer_letter", "clause_type", 600, 3),
        ("job_offer_letter", "favorability", 1500, 3),
        ("insurance_policy", "clause_type", 600, 3),
        ("insurance_policy", "favorability", 1500, 3),
    ]

    results = {}
    for domain, task, n_samples, eps in configs:
        if domain not in results:
            results[domain] = {}
        m = train_and_export_task(domain, task, samples_per_class=n_samples, epochs=eps)
        results[domain][task] = m

    # Update model comparison reports
    print("\nUpdating model comparison reports...", flush=True)
    comp_paths = [
        Path("backend/models/model_comparison_report.json"),
        Path("ml/artifacts/model_comparison_report.json"),
    ]

    for cp in comp_paths:
        current_report = {}
        if cp.exists():
            try:
                current_report = json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                current_report = {}

        for domain in results:
            if domain not in current_report:
                current_report[domain] = {}
            for task in results[domain]:
                if task not in current_report[domain]:
                    current_report[domain][task] = {}
                
                distil_m = results[domain][task]
                current_report[domain][task]["selected_model"] = "distilbert"
                current_report[domain][task]["distilbert"] = distil_m
                current_report[domain][task]["delta_vs_baseline"] = round(distil_m["f1_macro"] - 1.0, 4)

        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(current_report, indent=2), encoding="utf-8")
        print(f"Updated {cp} successfully!", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("MASTER DISTILBERT TRAINING COMPLETED ACROSS ALL DOMAINS!", flush=True)
    print("=" * 65, flush=True)
    for dom in results:
        for tsk, m in results[dom].items():
            print(f"  [{dom}] {tsk}: Accuracy={m['accuracy']:.2%}, Macro-F1={m['f1_macro']:.2%}, Weighted-F1={m['f1_weighted']:.2%}", flush=True)

if __name__ == "__main__":
    main()
