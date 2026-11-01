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
from sklearn.metrics import f1_score, accuracy_score, classification_report

torch.set_num_threads(16)
device = torch.device("cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# Load data
train_all = json.loads(Path("ml/artifacts/phase9_50k/train_preprocessed.json").read_text(encoding="utf-8"))
val_all = json.loads(Path("ml/artifacts/phase9_50k/validation_preprocessed.json").read_text(encoding="utf-8"))
test_all = json.loads(Path("ml/artifacts/phase9_50k/test_preprocessed.json").read_text(encoding="utf-8"))

domain = "rental_agreement"
task = "clause_type"

train_domain = [d for d in train_all if d["domain"] == domain]
val_domain = [d for d in val_all if d["domain"] == domain]
test_domain = [d for d in test_all if d["domain"] == domain]

classes = sorted(list(set(d[task] for d in train_domain)))
label2id = {c: i for i, c in enumerate(classes)}
id2label = {i: c for i, c in enumerate(classes)}
num_labels = len(classes)

print(f"Domain: {domain}, Task: {task}, Num Labels: {num_labels}")
print("Classes:", classes)

# Select stratified subset for fast, high-quality training: 150 samples per class = 1,500 samples
samples_per_class = 150
stratified_train = []
class_buckets = {c: [] for c in classes}
for d in train_domain:
    class_buckets[d[task]].append(d)

for c in classes:
    random.shuffle(class_buckets[c])
    stratified_train.extend(class_buckets[c][:samples_per_class])

random.shuffle(stratified_train)
print(f"Selected {len(stratified_train)} stratified training samples ({samples_per_class} per class)")

# Subsample val & test to 50 per class for rapid evaluation during test run
stratified_val = []
for c in classes:
    c_val = [d for d in val_domain if d[task] == c]
    stratified_val.extend(c_val[:50])

stratified_test = []
for c in classes:
    c_test = [d for d in test_domain if d[task] == c]
    stratified_test.extend(c_test[:50])

print(f"Val samples: {len(stratified_val)}, Test samples: {len(stratified_test)}")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class ClauseDataset(Dataset):
    def __init__(self, data, max_length=96):
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

train_dataset = ClauseDataset(stratified_train, max_length=96)
val_dataset = ClauseDataset(stratified_val, max_length=96)
test_dataset = ClauseDataset(stratified_test, max_length=96)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to(device)

# Freeze lower layers (embeddings + layer 0-2) and fine-tune layers 3-5
for param in model.distilbert.embeddings.parameters():
    param.requires_grad = False
for i in range(3):
    for param in model.distilbert.transformer.layer[i].parameters():
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total:.1%})")

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=4e-5, weight_decay=0.01)
epochs = 3
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

def evaluate(loader):
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
    f1_m = f1_score(targets, preds, average="macro")
    f1_w = f1_score(targets, preds, average="weighted")
    return acc, f1_m, f1_w

print("\n=== STARTING TRAINING ===")
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
    val_acc, val_f1, _ = evaluate(val_loader)
    print(f"Epoch {epoch}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2%} | Val Macro-F1: {val_f1:.2%} | Time: {t1-t0:.1f}s")

# Evaluate on test set
test_acc, test_f1, test_w = evaluate(test_loader)
print(f"\n>>> FINAL TEST EVALUATION: Acc: {test_acc:.2%} | Macro-F1: {test_f1:.2%} | Weighted-F1: {test_w:.2%} <<<\n")
