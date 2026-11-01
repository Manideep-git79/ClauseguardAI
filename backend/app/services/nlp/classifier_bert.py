import os
import sys
import json
import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.services.nlp.classifier_baseline import predict_baseline

def build_windowed_input(prev_clause_text: str | None, clause_text: str, next_clause_text: str | None, max_length: int = 256) -> str:
    prev_part = (prev_clause_text or "").strip()
    next_part = (next_clause_text or "").strip()
    target_part = (clause_text or "").strip()

    max_chars = max_length * 4
    target_chars = len(target_part)

    if target_chars + len(prev_part) + len(next_part) > max_chars:
        remaining_budget = max(0, max_chars - target_chars - 30)
        half_budget = remaining_budget // 2
        if len(prev_part) > half_budget:
            prev_part = "..." + prev_part[-half_budget:]
        if len(next_part) > half_budget:
            next_part = next_part[:half_budget] + "..."

    return f"{prev_part} [TARGET] {target_part} [/TARGET] {next_part}".strip()

logger = logging.getLogger(__name__)

_bert_models = {}
_bert_tokenizers = {}
_bert_metadata = {}

def reload_bert_models():
    global _bert_models, _bert_tokenizers, _bert_metadata
    _bert_models.clear()
    _bert_tokenizers.clear()
    _bert_metadata.clear()

def get_models_dir() -> Path:
    env_dir = os.environ.get("MODELS_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    backend_models = Path(__file__).resolve().parents[3] / "models"
    if backend_models.exists():
        return backend_models
    repo_artifacts = Path(__file__).resolve().parents[4] / "ml" / "artifacts"
    if repo_artifacts.exists():
        return repo_artifacts
    return backend_models

def get_bert_model_and_tokenizer(document_type: str, task: str = "clause_type"):
    key = f"{document_type}_{task}"
    if key in _bert_models:
        return _bert_models[key], _bert_tokenizers[key], _bert_metadata[key]

    models_dir = get_models_dir()
    ml_dir = models_dir / document_type
    report_file = models_dir / "model_comparison_report.json"
    
    # Check environment overrides:
    # In cloud container deployments (e.g. Railway 512MB RAM limit), PyTorch DistilBERT
    # allocates >750MB of RAM and triggers an immediate OS kernel SIGKILL (OOM).
    # Default to the certified winning model from model_comparison_report.json (baseline_tfidf_lr, 2MB RAM),
    # unless FORCE_BERT=1 is explicitly requested on a high-memory host.
    force_baseline = os.environ.get("FORCE_BASELINE", "0").strip().lower() in ("1", "true", "yes")
    force_bert = os.environ.get("FORCE_BERT", "0").strip().lower() in ("1", "true", "yes")

    # In memory-constrained cloud environments (e.g. Railway), enforce lightweight model
    is_cloud_container = bool(os.environ.get("RAILWAY_SERVICE_ID") or os.environ.get("RAILWAY_ENVIRONMENT"))
    if is_cloud_container:
        force_bert = False
        force_baseline = True

    winning_model_type = None
    if not force_bert and report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as rf:
                report = json.load(rf)
                winning_model_type = report.get(document_type, {}).get(task, {}).get("selected_model")
        except Exception:
            pass

    if force_baseline or (winning_model_type == "baseline_tfidf_lr" and not force_bert):
        logger.info(f"Using certified baseline TF-IDF + LR for {document_type} {task} (force_baseline={force_baseline}, winning={winning_model_type})")
        return None, None, {"model_version": f"baseline-tfidf-lr-{task}-v1"}

    strategy = os.environ.get("BERT_STRATEGY", "nocontext").strip().lower()
    if strategy == "windowed":
        candidate_paths = [
            ml_dir / f"bert_{task}_windowed" / "final",
            ml_dir / f"bert_{task}_nocontext" / "final",
        ]
    else:
        candidate_paths = [
            ml_dir / f"bert_{task}_nocontext" / "final",
            ml_dir / f"bert_{task}_windowed" / "final",
        ]

    # Additional fallback to ml/artifacts if present
    repo_artifacts = Path(__file__).resolve().parents[4] / "ml" / "artifacts"
    if repo_artifacts.exists():
        candidate_paths.extend([
            repo_artifacts / document_type / f"bert_{task}_{strategy}" / "final",
            repo_artifacts / document_type / f"bert_{task}_nocontext" / "final",
            repo_artifacts / document_type / f"bert_{task}_windowed" / "final",
        ])

    model_dir = None
    for cp in candidate_paths:
        if (cp / "model.safetensors").exists() or (cp / "pytorch_model.bin").exists():
            model_dir = cp
            break

    if not model_dir:
        logger.warning(f"No BERT checkpoint found in candidate paths for {document_type} {task}: {candidate_paths}")
        return None, None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()

        meta_path = model_dir / "model_metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        # Ensure id2label is populated from model config if missing from metadata
        if not metadata.get("id2label") and hasattr(model.config, "id2label"):
            metadata["id2label"] = model.config.id2label

        metadata["model_version"] = "Fine-Tuned DistiBERT"

        _bert_models[key] = model
        _bert_tokenizers[key] = tokenizer
        _bert_metadata[key] = metadata
        logger.info(f"Successfully loaded BERT model for {document_type} {task} from {model_dir} (version: {metadata.get('model_version')})")
        return model, tokenizer, metadata
    except Exception as e:
        logger.error(f"Error loading BERT model from {model_dir}: {e}")
        return None, None, None

def classify_clause(
    clause_text: str,
    document_type: str,
    prev_clause_text: str = None,
    next_clause_text: str = None,
    task: str = "clause_type",
) -> tuple[str, float, str]:
    """
    Classifies a single clause using the fine-tuned BERT model (with context windowing).
    Gracefully falls back to TF-IDF+LR baseline if BERT is not yet loaded.
    """
    model, tokenizer, meta = get_bert_model_and_tokenizer(document_type, task)

    if model is None:
        label, conf = predict_baseline(clause_text, document_type, task)
        return label, conf, f"baseline-tfidf-lr-{task}"

    use_context = meta.get("use_context", True)
    if use_context:
        input_text = build_windowed_input(prev_clause_text, clause_text, next_clause_text)
    else:
        input_text = clause_text

    try:
        inputs = tokenizer(input_text, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            temperature = float(os.environ.get("BERT_TEMPERATURE", "0.58"))
            scaled_logits = outputs.logits / max(0.1, temperature)
            probs = F.softmax(scaled_logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            conf = round(float(probs[pred_idx].item()), 4)

        id2label = meta.get("id2label", {})
        pred_label = id2label.get(str(pred_idx)) or id2label.get(pred_idx, "unknown")
        model_version = meta.get("model_version", f"bert-{task}-v1")
        return pred_label, conf, model_version
    except Exception as e:
        logger.error(f"Error running BERT inference: {e}")
        label, conf = predict_baseline(clause_text, document_type, task)
        return label, conf, f"baseline-tfidf-lr-{task}"

def classify_clauses_batch(
    clause_items: list[dict],
    document_type: str,
    task: str = "clause_type",
    batch_size: int = 32
) -> list[tuple[str, float, str]]:
    """
    High-speed batched classification for an entire document's clauses.
    Runs multiple inputs through BERT in a single forward pass, providing a 10x-15x speedup.
    """
    if not clause_items:
        return []

    model, tokenizer, meta = get_bert_model_and_tokenizer(document_type, task)

    if model is None:
        results = []
        for item in clause_items:
            txt = item.get("text") or item.get("redacted_text", "")
            lbl, conf = predict_baseline(txt, document_type, task)
            results.append((lbl, conf, f"baseline-tfidf-lr-{task}"))
        return results

    use_context = meta.get("use_context", True)
    id2label = meta.get("id2label", {})
    model_version = meta.get("model_version", f"bert-{task}-v1")

    prepared_inputs = []
    for item in clause_items:
        txt = item.get("text") or item.get("redacted_text", "")
        if use_context:
            inp = build_windowed_input(item.get("prev_text"), txt, item.get("next_text"))
        else:
            inp = txt
        prepared_inputs.append(inp)

    results = []
    for i in range(0, len(prepared_inputs), batch_size):
        chunk = prepared_inputs[i : i + batch_size]
        try:
            tokens = tokenizer(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**tokens)
                temperature = float(os.environ.get("BERT_TEMPERATURE", "0.58"))
                scaled_logits = outputs.logits / max(0.1, temperature)
                probs = F.softmax(scaled_logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                for j in range(len(chunk)):
                    idx = preds[j].item()
                    conf = round(float(probs[j][idx].item()), 4)
                    lbl = id2label.get(str(idx)) or id2label.get(idx, "unknown")
                    results.append((lbl, conf, model_version))
        except Exception as e:
            logger.error(f"Error in batch inference: {e}. Falling back to baseline for chunk.")
            for j in range(i, min(i + batch_size, len(clause_items))):
                item = clause_items[j]
                txt = item.get("text") or item.get("redacted_text", "")
                lbl, conf = predict_baseline(txt, document_type, task)
                results.append((lbl, conf, f"baseline-tfidf-lr-{task}"))

    return results
