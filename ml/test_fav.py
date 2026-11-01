import json, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score

torch.set_num_threads(16)
device = torch.device('cpu')

train_all = json.loads(Path('ml/artifacts/phase9_50k/train_preprocessed.json').read_text(encoding='utf-8'))
val_all = json.loads(Path('ml/artifacts/phase9_50k/validation_preprocessed.json').read_text(encoding='utf-8'))
test_all = json.loads(Path('ml/artifacts/phase9_50k/test_preprocessed.json').read_text(encoding='utf-8'))

dom_train = [d for d in train_all if d['domain'] == 'rental_agreement']
dom_val = [d for d in val_all if d['domain'] == 'rental_agreement']
dom_test = [d for d in test_all if d['domain'] == 'rental_agreement']

task = 'favorability'
classes = sorted(list(set(d[task] for d in dom_train)))
l2i = {c: i for i, c in enumerate(classes)}

# Sample 600 per class
train_data = []
for c in classes:
    cb = [d for d in dom_train if d[task] == c]
    random.seed(42)
    random.shuffle(cb)
    train_data.extend(cb[:600])
random.shuffle(train_data)

val_data = []
for c in classes:
    val_data.extend([d for d in dom_val if d[task] == c][:100])

test_data = []
for c in classes:
    test_data.extend([d for d in dom_test if d[task] == c][:100])

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

class DS(Dataset):
    def __init__(self, data):
        self.enc = tokenizer([d['text'] for d in data], truncation=True, padding='max_length', max_length=96, return_tensors='pt')
        self.labels = torch.tensor([l2i[d[task]] for d in data], dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return {'input_ids': self.enc['input_ids'][i], 'attention_mask': self.enc['attention_mask'][i], 'labels': self.labels[i]}

train_ld = DataLoader(DS(train_data), batch_size=48, shuffle=True)
val_ld = DataLoader(DS(val_data), batch_size=64, shuffle=False)
test_ld = DataLoader(DS(test_data), batch_size=64, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3).to(device)
for p in model.distilbert.embeddings.parameters(): p.requires_grad = False
for i in range(3):
    for p in model.distilbert.transformer.layer[i].parameters(): p.requires_grad = False

optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.distilbert.named_parameters() if p.requires_grad], 'lr': 5e-5},
    {'params': model.pre_classifier.parameters(), 'lr': 2e-4},
    {'params': model.classifier.parameters(), 'lr': 2e-4},
], weight_decay=0.01)

epochs = 3
total_steps = len(train_ld) * epochs
sched = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

for ep in range(1, epochs + 1):
    model.train()
    tot_loss = 0
    t0 = time.time()
    for b in train_ld:
        optimizer.zero_grad()
        out = model(b['input_ids'], attention_mask=b['attention_mask'], labels=b['labels'])
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sched.step()
        tot_loss += out.loss.item()
    t1 = time.time()
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for b in val_ld:
            out = model(b['input_ids'], attention_mask=b['attention_mask'])
            preds.extend(torch.argmax(out.logits, -1).numpy())
            tgts.extend(b['labels'].numpy())
    acc = accuracy_score(tgts, preds)
    f1 = f1_score(tgts, preds, average='macro')
    print(f'Epoch {ep} ({t1-t0:.1f}s): Loss={tot_loss/len(train_ld):.4f}, Val Acc={acc:.2%}, Val Macro-F1={f1:.2%}', flush=True)

model.eval()
preds, tgts = [], []
with torch.no_grad():
    for b in test_ld:
        out = model(b['input_ids'], attention_mask=b['attention_mask'])
        preds.extend(torch.argmax(out.logits, -1).numpy())
        tgts.extend(b['labels'].numpy())
test_acc = accuracy_score(tgts, preds)
test_f1 = f1_score(tgts, preds, average='macro')
print(f'FINAL TEST: Acc={test_acc:.2%}, Macro-F1={test_f1:.2%}', flush=True)
