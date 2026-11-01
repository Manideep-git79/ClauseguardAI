"""
ClauseGuard AI: Safe DistilBERT Fine-Tuning Without Overfitting
==============================================================
Production Fine-Tuning Pipeline with:
1. Complete Document-Level Isolation & Anti-Leakage Safeguards
2. Conservative Hyperparameter Optimization (LR, Weight Decay, Warmup, Grad Clip)
3. Strict Validation-Only Checkpoint Selection & Early Stopping
4. Target-Preserving Context Windowing vs. Target-Only Ablation
5. Untouched Frozen Test Evaluation with Error Analysis & Confusion Matrix
6. Full Backward Compatibility with ClauseGuard AI Inference Engine
"""

import os
import sys
import json
import time
import math
import random
import argparse
from pathlib import Path
from copy import deepcopy
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Root directory configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "ml" / "artifacts" / "phase4" / "preprocessed_dataset"
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
BACKEND_MODELS_DIR = PROJECT_ROOT / "backend" / "models"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Canonical taxonomies per domain
CANONICAL_LABELS = {
    "rental_agreement": [
        "entry_notice_access",
        "indemnification",
        "late_fees_penalty",
        "maintenance_repairs",
        "pet_policy",
        "rent_payment_terms",
        "security_deposit",
        "subletting",
        "termination",
        "utilities",
    ],
    "job_offer_letter": [
        "at_will_employment",
        "benefits_overview",
        "bonus_incentive",
        "compensation_salary",
        "confidentiality_nda",
        "job_title_role",
        "non_compete_non_solicit",
        "start_date_contingencies",
        "termination_conditions",
        "working_hours_location",
    ],
    "insurance_policy": [
        "cancellation_non_renewal",
        "claims_process",
        "coverage_scope",
        "deductible_premium",
        "dispute_resolution_appraisal",
        "exclusions",
        "grace_period",
        "limits_of_liability",
        "policy_period_renewal",
        "subrogation",
    ],
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ClauseDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=256, use_context=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_context = use_context

        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        for r in records:
            tgt = (r.get("clean_text") or r.get("clause_text") or r.get("text") or "").strip()
            
            if use_context:
                prev_c = (r.get("prev_clause_text") or r.get("prev_text") or "").strip()
                next_c = (r.get("next_clause_text") or r.get("next_text") or "").strip()
                # Target-preserving context window formatting
                # Allocate space to target first, context fills remaining budget
                max_chars = max_length * 4
                tgt_chars = len(tgt)
                if tgt_chars + len(prev_c) + len(next_c) > max_chars:
                    rem = max(0, max_chars - tgt_chars - 30)
                    half = rem // 2
                    if len(prev_c) > half:
                        prev_c = "..." + prev_c[-half:]
                    if len(next_c) > half:
                        next_c = next_c[:half] + "..."
                text = f"{prev_c} [TARGET] {tgt} [/TARGET] {next_c}".strip()
            else:
                text = tgt

            enc = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.input_ids.append(enc["input_ids"].squeeze(0))
            self.attention_masks.append(enc["attention_mask"].squeeze(0))
            self.labels.append(r["label_id"])

        self.input_ids = torch.stack(self.input_ids)
        self.attention_masks = torch.stack(self.attention_masks)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


def load_dataset_split(split_name: str, domain: str, label2id: dict) -> list[dict]:
    path = DATA_DIR / f"{split_name}_preprocessed.jsonl"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("domain") == domain:
                    lbl = r.get("clause_type") or r.get("label")
                    if lbl in label2id:
                        r["label_id"] = label2id[lbl]
                        records.append(r)
    return records


def audit_splits(domain: str, label2id: dict):
    train_recs = load_dataset_split("train", domain, label2id)
    val_recs = load_dataset_split("validation", domain, label2id)
    test_recs = load_dataset_split("test", domain, label2id)

    train_docs = set(r["doc_id"] for r in train_recs)
    val_docs = set(r["doc_id"] for r in val_recs)
    test_docs = set(r["doc_id"] for r in test_recs)

    train_texts = set((r.get("clean_text") or r.get("clause_text")).strip() for r in train_recs)
    val_texts = set((r.get("clean_text") or r.get("clause_text")).strip() for r in val_recs)
    test_texts = set((r.get("clean_text") or r.get("clause_text")).strip() for r in test_recs)

    leakage_report = {
        "domain": domain,
        "sample_counts": {
            "train": len(train_recs),
            "val": len(val_recs),
            "test": len(test_recs),
            "total": len(train_recs) + len(val_recs) + len(test_recs),
        },
        "document_counts": {
            "train": len(train_docs),
            "val": len(val_docs),
            "test": len(test_docs),
        },
        "document_overlaps": {
            "train_val": len(train_docs & val_docs),
            "train_test": len(train_docs & test_docs),
            "val_test": len(val_docs & test_docs),
        },
        "text_overlaps": {
            "train_val": len(train_texts & val_texts),
            "train_test": len(train_texts & test_texts),
            "val_test": len(val_texts & test_texts),
        },
        "class_counts_train": dict(Counter(r["clause_type"] for r in train_recs)),
        "class_counts_val": dict(Counter(r["clause_type"] for r in val_recs)),
        "class_counts_test": dict(Counter(r["clause_type"] for r in test_recs)),
    }
    return leakage_report, train_recs, val_recs, test_recs


def evaluate_model(model, loader, device, labels_list):
    model.eval()
    all_preds, all_labels = [], []
    total_val_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / max(1, len(loader))
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    prec_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(labels_list))),
        target_names=labels_list,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(labels_list)))).tolist()

    return {
        "val_loss": round(avg_val_loss, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "raw_preds": [int(p) for p in all_preds],
        "raw_labels": [int(l) for l in all_labels],
    }


def train_single_run(
    train_ds,
    val_ds,
    labels_list,
    device,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    epochs: int = 4,
    batch_size: int = 16,
    grad_clip: float = 1.0,
    early_stopping_patience: int = 2,
    seed: int = 42,
    subsample_train: int | None = None,
):
    set_seed(seed)

    if subsample_train and subsample_train < len(train_ds):
        indices = list(range(len(train_ds)))
        random.seed(seed)
        sub_indices = random.sample(indices, subsample_train)
        train_ds_actual = torch.utils.data.Subset(train_ds, sub_indices)
    else:
        train_ds_actual = train_ds

    train_loader = DataLoader(train_ds_actual, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_labels = len(labels_list)
    label2id = {l: i for i, l in enumerate(labels_list)}
    id2label = {i: l for i, l in enumerate(labels_list)}

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        seq_classif_dropout=0.2,
    ).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_macro_f1 = 0.0
    best_epoch = 0
    best_state_dict = None
    best_val_metrics = None
    epochs_without_improvement = 0
    training_history = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        val_metrics = evaluate_model(model, val_loader, device, labels_list)
        val_f1 = val_metrics["macro_f1"]
        val_loss = val_metrics["val_loss"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "val_accuracy": val_metrics["accuracy"],
        }
        training_history.append(epoch_record)

        print(
            f"  Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
        )

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"  [Early Stopping triggered at epoch {epoch} (patience={early_stopping_patience})]")
                break

    train_runtime = round(time.time() - start_time, 2)

    # Load best checkpoint state
    model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_metrics": best_val_metrics,
        "training_history": training_history,
        "runtime_seconds": train_runtime,
        "hyperparameters": {
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "batch_size": batch_size,
            "grad_clip": grad_clip,
            "seed": seed,
            "subsample_train": subsample_train,
        },
    }


def run_full_suite(
    domain: str = "rental_agreement",
    max_length: int = 256,
    subsample_train: int | None = 600,
    seeds: list[int] = [42],
):
    print("=" * 80)
    print(f"CLAUSEGUARD AI: SAFE DISTILBERT FINE-TUNING SUITE — {domain.upper()}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} | CPU Threads: {torch.get_num_threads()}")

    labels_list = CANONICAL_LABELS[domain]
    label2id = {l: i for i, l in enumerate(labels_list)}
    id2label = {i: l for i, l in enumerate(labels_list)}

    # Phase 2: Audit and verify zero leakage
    print("\n--- PHASE 2: AUDITING SPLITS FOR DATA LEAKAGE ---")
    audit_rep, train_recs, val_recs, test_recs = audit_splits(domain, label2id)
    print(f"Sample counts: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")
    print(f"Document counts: Train={audit_rep['document_counts']['train']}, Val={audit_rep['document_counts']['val']}, Test={audit_rep['document_counts']['test']}")
    print(f"Cross-split document overlap: Train & Val={audit_rep['document_overlaps']['train_val']}, Train & Test={audit_rep['document_overlaps']['train_test']}, Val & Test={audit_rep['document_overlaps']['val_test']}")
    print(f"Cross-split exact text overlap: Train & Val={audit_rep['text_overlaps']['train_val']}, Train & Test={audit_rep['text_overlaps']['train_test']}, Val & Test={audit_rep['text_overlaps']['val_test']}")
    assert audit_rep['document_overlaps']['train_val'] == 0, "Data leakage detected: train and val overlap!"
    assert audit_rep['document_overlaps']['train_test'] == 0, "Data leakage detected: train and test overlap!"
    assert audit_rep['document_overlaps']['val_test'] == 0, "Data leakage detected: val and test overlap!"
    print(">>> VERIFICATION PASSED: ZERO DOCUMENT LEAKAGE DETECTED across splits!")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Build datasets (target-only)
    print("\nEncoding Target-Only Datasets...")
    train_ds_target = ClauseDataset(train_recs, tokenizer, max_length=max_length, use_context=False)
    val_ds_target = ClauseDataset(val_recs, tokenizer, max_length=max_length, use_context=False)
    test_ds_target = ClauseDataset(test_recs, tokenizer, max_length=max_length, use_context=False)

    # Build datasets (context-windowed)
    print("Encoding Context-Windowed Datasets...")
    train_ds_ctx = ClauseDataset(train_recs, tokenizer, max_length=max_length, use_context=True)
    val_ds_ctx = ClauseDataset(val_recs, tokenizer, max_length=max_length, use_context=True)
    test_ds_ctx = ClauseDataset(test_recs, tokenizer, max_length=max_length, use_context=True)

    experiments = [
        {
            "id": "Exp_A_Legacy_Baseline",
            "name": "Legacy Baseline (No Warmup, No GradClip, LR=5e-5, WD=0.01)",
            "use_context": False,
            "dataset": (train_ds_target, val_ds_target),
            "lr": 5e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.0,
            "grad_clip": None,
            "epochs": 4,
            "batch_size": 16,
        },
        {
            "id": "Exp_B1_Conservative_LR1e5",
            "name": "Conservative LR 1e-5 (Warmup=0.05, GradClip=1.0, WD=0.01)",
            "use_context": False,
            "dataset": (train_ds_target, val_ds_target),
            "lr": 1e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "grad_clip": 1.0,
            "epochs": 4,
            "batch_size": 16,
        },
        {
            "id": "Exp_B2_Conservative_LR2e5",
            "name": "Conservative LR 2e-5 (Warmup=0.05, GradClip=1.0, WD=0.01)",
            "use_context": False,
            "dataset": (train_ds_target, val_ds_target),
            "lr": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "grad_clip": 1.0,
            "epochs": 4,
            "batch_size": 16,
        },
        {
            "id": "Exp_B3_Tuned_LR5e5_Warmup",
            "name": "Tuned LR 5e-5 (Warmup=0.05, GradClip=1.0, WD=0.01)",
            "use_context": False,
            "dataset": (train_ds_target, val_ds_target),
            "lr": 5e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "grad_clip": 1.0,
            "epochs": 4,
            "batch_size": 16,
        },
        {
            "id": "Exp_C_Regularized_WD005",
            "name": "High Regularization (LR=3e-5, WD=0.05, Warmup=0.10, GradClip=1.0)",
            "use_context": False,
            "dataset": (train_ds_target, val_ds_target),
            "lr": 3e-5,
            "weight_decay": 0.05,
            "warmup_ratio": 0.10,
            "grad_clip": 1.0,
            "epochs": 4,
            "batch_size": 16,
        },
        {
            "id": "Exp_D_Context_Windowed",
            "name": "Target-Preserved Context Window (LR=3e-5, WD=0.01, Warmup=0.05)",
            "use_context": True,
            "dataset": (train_ds_ctx, val_ds_ctx),
            "lr": 3e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "grad_clip": 1.0,
            "epochs": 4,
            "batch_size": 16,
        },
    ]

    results_table = []
    trained_models = {}

    print("\n" + "=" * 80)
    print("PHASES 3, 4, 5: RUNNING CONTROLLED EXPERIMENTS ON VALIDATION DATA ONLY")
    print("=" * 80)

    for exp in experiments:
        exp_id = exp["id"]
        print(f"\n>>> Running {exp_id}: {exp['name']}...")
        tr_ds, va_ds = exp["dataset"]

        run_out = train_single_run(
            train_ds=tr_ds,
            val_ds=va_ds,
            labels_list=labels_list,
            device=device,
            lr=exp["lr"],
            weight_decay=exp["weight_decay"],
            warmup_ratio=exp["warmup_ratio"],
            epochs=exp["epochs"],
            batch_size=exp["batch_size"],
            grad_clip=exp["grad_clip"],
            early_stopping_patience=2,
            seed=seeds[0],
            subsample_train=subsample_train,
        )

        trained_models[exp_id] = run_out
        vm = run_out["best_val_metrics"]

        hist_str = " -> ".join([f"Ep{h['epoch']}: F1={h['val_macro_f1']:.4f} (loss={h['val_loss']:.4f})" for h in run_out["training_history"]])

        res_row = {
            "experiment_id": exp_id,
            "name": exp["name"],
            "use_context": exp["use_context"],
            "seed": seeds[0],
            "best_epoch": run_out["best_epoch"],
            "val_loss": vm["val_loss"],
            "val_macro_f1": vm["macro_f1"],
            "val_weighted_f1": vm["weighted_f1"],
            "val_accuracy": vm["accuracy"],
            "val_precision_macro": vm["precision_macro"],
            "val_recall_macro": vm["recall_macro"],
            "runtime_s": run_out["runtime_seconds"],
            "training_progression": hist_str,
        }
        results_table.append(res_row)
        print(f"    --> Best Epoch: {run_out['best_epoch']} | Val Macro-F1: {vm['macro_f1']:.4f} | Val Loss: {vm['val_loss']:.4f}")

    # Select best model based on VALIDATION Macro-F1 strictly
    best_exp_entry = max(results_table, key=lambda x: x["val_macro_f1"])
    best_exp_id = best_exp_entry["experiment_id"]
    best_run = trained_models[best_exp_id]

    print("\n" + "=" * 80)
    print("PHASE 4 SELECTION VERDICT (STRICTLY FROM VALIDATION DATA)")
    print("=" * 80)
    print(f"Selected Winner: {best_exp_id} ({best_exp_entry['name']})")
    print(f"Validation Macro-F1: {best_exp_entry['val_macro_f1']:.4f} (Accuracy: {best_exp_entry['val_accuracy']:.4f})")
    print(f"Best Checkpoint Epoch: {best_run['best_epoch']}")

    # PHASE 6: Evaluate selected winner strictly ONCE on untouched test set
    print("\n" + "=" * 80)
    print("PHASE 6: FINAL EVALUATION ON FROZEN UNTOUCHED TEST SET")
    print("=" * 80)

    test_ds = test_ds_ctx if best_exp_entry["use_context"] else test_ds_target
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    test_metrics = evaluate_model(best_run["model"], test_loader, device, labels_list)

    print(f"Frozen Test Macro-F1:     {test_metrics['macro_f1']:.4f}")
    print(f"Frozen Test Weighted-F1:  {test_metrics['weighted_f1']:.4f}")
    print(f"Frozen Test Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"Frozen Test Precision:    {test_metrics['precision_macro']:.4f}")
    print(f"Frozen Test Recall:       {test_metrics['recall_macro']:.4f}")

    print("\n--- PER-CLASS PERFORMANCE (TEST SET) ---")
    per_class_summary = {}
    for cls_name in labels_list:
        rep = test_metrics["classification_report"][cls_name]
        per_class_summary[cls_name] = {
            "precision": round(rep["precision"], 4),
            "recall": round(rep["recall"], 4),
            "f1_score": round(rep["f1-score"], 4),
            "support": int(rep["support"]),
        }
        print(f"  {cls_name:24s} | Prec: {rep['precision']:.4f} | Rec: {rep['recall']:.4f} | F1: {rep['f1-score']:.4f} | Supp: {int(rep['support'])}")

    # Multi-seed stability evaluation if requested
    multi_seed_results = {}
    if len(seeds) > 1:
        print("\n--- MULTI-SEED STABILITY EVALUATION ---")
        best_cfg = next(e for e in experiments if e["id"] == best_exp_id)
        tr_ds, va_ds = best_cfg["dataset"]
        seed_f1s = []
        for s in seeds:
            print(f"Training seed {s}...")
            s_out = train_single_run(
                train_ds=tr_ds,
                val_ds=va_ds,
                labels_list=labels_list,
                device=device,
                lr=best_cfg["lr"],
                weight_decay=best_cfg["weight_decay"],
                warmup_ratio=best_cfg["warmup_ratio"],
                epochs=best_cfg["epochs"],
                batch_size=best_cfg["batch_size"],
                grad_clip=best_cfg["grad_clip"],
                early_stopping_patience=2,
                seed=s,
                subsample_train=subsample_train,
            )
            s_test = evaluate_model(s_out["model"], test_loader, device, labels_list)
            seed_f1s.append(s_test["macro_f1"])
            print(f"  Seed {s} Test Macro-F1: {s_test['macro_f1']:.4f}")
        multi_seed_results = {
            "seeds": seeds,
            "test_macro_f1s": seed_f1s,
            "mean": round(float(np.mean(seed_f1s)), 4),
            "std": round(float(np.std(seed_f1s)), 4),
        }
        print(f"Seed Mean: {multi_seed_results['mean']:.4f} ± {multi_seed_results['std']:.4f}")

    # Concrete Error Analysis (False Positives / False Negatives)
    print("\n--- ERROR ANALYSIS: EXTRACTING MISCLASSIFIED EXAMPLES ---")
    error_examples = []
    raw_preds = test_metrics["raw_preds"]
    raw_labels = test_metrics["raw_labels"]

    for idx, (p, l) in enumerate(zip(raw_preds, raw_labels)):
        if p != l:
            rec = test_recs[idx]
            err = {
                "clause_id": rec.get("clause_id", f"sample_{idx}"),
                "doc_id": rec.get("doc_id", "unknown"),
                "clause_text": (rec.get("clean_text") or rec.get("clause_text"))[:160] + "...",
                "actual_label": labels_list[l],
                "predicted_label": labels_list[p],
            }
            error_examples.append(err)

    print(f"Total misclassifications on test set: {len(error_examples)} / {len(test_recs)} ({len(error_examples)/len(test_recs)*100:.1f}%)")
    print("Sample misclassifications:")
    for err in error_examples[:5]:
        print(f"  [ACTUAL: {err['actual_label']}] -> [PREDICTED: {err['predicted_label']}]")
        print(f"   Text: \"{err['clause_text']}\"")

    # Save artifacts and output report
    output_dir = ARTIFACTS_DIR / "safe_finetuned" / domain
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best model
    best_model = best_run["model"]
    best_model.config.id2label = id2label
    best_model.config.label2id = label2id
    best_model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "domain": domain,
        "document_type": domain,
        "task": "clause_type",
        "dataset_audit": audit_rep,
        "experiments_table": results_table,
        "selected_winner": best_exp_entry,
        "validation_metrics": best_run["best_val_metrics"],
        "frozen_test_metrics": {
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "accuracy": test_metrics["accuracy"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "per_class": per_class_summary,
        },
        "multi_seed_stability": multi_seed_results,
        "error_analysis_sample": error_examples[:10],
        "total_test_errors": len(error_examples),
        "total_test_samples": len(test_recs),
    }

    report_path = output_dir / "safe_finetuning_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    model_metadata = {
        "document_type": domain,
        "task": "clause_type",
        "use_context": best_exp_entry["use_context"],
        "model_version": "distilbert-safe-finetuned-v2",
        "num_labels": len(labels_list),
        "label2id": label2id,
        "id2label": id2label,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "f1_macro": test_metrics["macro_f1"],
            "f1_weighted": test_metrics["weighted_f1"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"],
        },
        "artifact_path": str(output_dir / "final"),
    }
    with open(output_dir / "final" / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    print(f"\nSaved fine-tuned model and reports to: {output_dir}")
    print("=" * 80)
    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClauseGuard AI Safe DistilBERT Fine-Tuning")
    parser.add_argument("--domain", type=str, default="rental_agreement", choices=["rental_agreement", "job_offer_letter", "insurance_policy"])
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--subsample_train", type=int, default=500, help="Train subsample size for rapid reproducible verification on CPU (None for full)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    subsample = None if args.subsample_train <= 0 else args.subsample_train
    run_full_suite(
        domain=args.domain,
        max_length=args.max_length,
        subsample_train=subsample,
        seeds=args.seeds,
    )

