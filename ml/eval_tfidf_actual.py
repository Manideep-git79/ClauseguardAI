import json
import time
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

test_path = Path("ml/artifacts/phase9_50k/test_preprocessed.json")
with open(test_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"Loaded {len(test_data)} test clauses.")

domains = ["rental_agreement", "job_offer_letter", "insurance_policy"]
tasks = ["clause_type", "favorability"]

print("\n=== EVALUATING TF-IDF BASELINE ON FROZEN TEST SET ===")
tfidf_results = {}
for dom in domains:
    tfidf_results[dom] = {}
    dom_test = [d for d in test_data if d["domain"] == dom]
    texts = [d["text"] for d in dom_test]
    for task in tasks:
        labels = [d[task] for d in dom_test]
        model_path = Path(f"backend/models/{dom}/baseline_tfidf_lr_{task}.joblib")
        if not model_path.exists():
            print(f"Missing {model_path}")
            continue
        pipe = joblib.load(model_path)
        t0 = time.time()
        preds = pipe.predict(texts)
        lat = (time.time() - t0) / len(texts)
        acc = accuracy_score(labels, preds)
        f1_m = f1_score(labels, preds, average="macro", zero_division=0)
        p_m = precision_score(labels, preds, average="macro", zero_division=0)
        r_m = recall_score(labels, preds, average="macro", zero_division=0)
        tfidf_results[dom][task] = {
            "f1_macro": round(float(f1_m), 4),
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(p_m), 4),
            "recall_macro": round(float(r_m), 4),
            "latency_ms": round(lat * 1000, 4)
        }
        print(f"TF-IDF [{dom}] [{task}]: Acc={acc:.4f}, F1={f1_m:.4f}, Prec={p_m:.4f}, Rec={r_m:.4f}, Latency={lat*1000:.3f}ms")

print("\nJSON TF-IDF Results:")
print(json.dumps(tfidf_results, indent=2))

