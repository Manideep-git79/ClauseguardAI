import json
from collections import Counter, defaultdict
from pathlib import Path

data_dir = Path(r"c:\Users\DELL\Downloads\clauseguard-ai\ml\artifacts\phase9_50k")

splits = {}
for split_name in ["train", "validation", "test"]:
    p = data_dir / f"{split_name}_preprocessed.json"
    with open(p, "r", encoding="utf-8") as f:
        splits[split_name] = json.load(f)

print("=== EXACT SPLIT COUNTS ===")
total_rows = 0
for name, data in splits.items():
    print(f"{name}: {len(data)} clauses")
    total_rows += len(data)
print(f"Total clauses: {total_rows}")

domain_counts = Counter()
domain_split_counts = defaultdict(Counter)
doc_ids_by_split = defaultdict(set)
doc_ids_by_domain = defaultdict(set)
texts_by_split = defaultdict(set)
classes_by_domain = defaultdict(set)
fav_by_domain = defaultdict(set)

for name, data in splits.items():
    for item in data:
        dom = item["domain"]
        doc_id = item["doc_id"]
        txt = item["text"]
        ct = item["clause_type"]
        fav = item["favorability"]

        domain_counts[dom] += 1
        domain_split_counts[dom][name] += 1
        doc_ids_by_split[name].add(doc_id)
        doc_ids_by_domain[dom].add(doc_id)
        texts_by_split[name].add(txt)
        classes_by_domain[dom].add(ct)
        fav_by_domain[dom].add(fav)

print("\n=== DOMAIN TOTALS ===")
for dom, count in domain_counts.items():
    print(f"{dom}: {count} clauses, {len(doc_ids_by_domain[dom])} unique docs, {len(classes_by_domain[dom])} classes, favs: {fav_by_domain[dom]}")

print("\n=== DOMAIN x SPLIT COUNTS ===")
for dom in domain_counts:
    tr = domain_split_counts[dom]["train"]
    va = domain_split_counts[dom]["validation"]
    te = domain_split_counts[dom]["test"]
    print(f"{dom}: Train={tr}, Val={va}, Test={te}, Total={tr+va+te}")

print("\n=== DOCUMENT OVERLAP CHECK ===")
train_val_docs = doc_ids_by_split["train"] & doc_ids_by_split["validation"]
train_test_docs = doc_ids_by_split["train"] & doc_ids_by_split["test"]
val_test_docs = doc_ids_by_split["validation"] & doc_ids_by_split["test"]
print(f"Train & Val doc overlap: {len(train_val_docs)}")
print(f"Train & Test doc overlap: {len(train_test_docs)}")
print(f"Val & Test doc overlap: {len(val_test_docs)}")

print("\n=== EXACT TEXT OVERLAP CHECK ===")
train_val_txt = texts_by_split["train"] & texts_by_split["validation"]
train_test_txt = texts_by_split["train"] & texts_by_split["test"]
val_test_txt = texts_by_split["validation"] & texts_by_split["test"]
print(f"Train & Val verbatim text overlap: {len(train_val_txt)}")
print(f"Train & Test verbatim text overlap: {len(train_test_txt)}")
print(f"Val & Test verbatim text overlap: {len(val_test_txt)}")

total_unique_docs = len(doc_ids_by_split["train"] | doc_ids_by_split["validation"] | doc_ids_by_split["test"])
print(f"\nTotal unique document structures: {total_unique_docs}")

