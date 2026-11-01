import os
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

_models_cache = {}

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

def get_baseline_model(document_type: str, task: str = "clause_type"):
    key = f"{document_type}_{task}"
    if key in _models_cache:
        return _models_cache[key]

    ml_dir = get_models_dir() / document_type
    model_path = ml_dir / f"baseline_tfidf_lr_{task}.joblib"

    if not model_path.exists():
        fallback_dir = get_models_dir() / "rental_agreement"
        if (fallback_dir / f"baseline_tfidf_lr_{task}.joblib").exists():
            model_path = fallback_dir / f"baseline_tfidf_lr_{task}.joblib"
        else:
            model_path = ml_dir.parent / f"baseline_tfidf_lr.joblib"

    if not model_path.exists():
        logger.warning(f"Baseline model not found at {model_path}")
        return None

    try:
        model = joblib.load(model_path)
        _models_cache[key] = model
        return model
    except Exception as e:
        logger.error(f"Error loading baseline model {model_path}: {e}")
        return None

def predict_baseline(text: str, document_type: str, task: str = "clause_type") -> tuple[str, float]:
    """
    Predicts clause_type or favorability using the TF-IDF + Logistic Regression baseline.
    Returns: (predicted_label, confidence_score)
    """
    model = get_baseline_model(document_type, task)
    if model is None:
        default_label = "other" if task == "clause_type" else "needs_review"
        return default_label, 0.5

    try:
        preds = model.predict([text])
        pred_label = preds[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            max_prob = float(max(probs))
        else:
            max_prob = 0.85
        return pred_label, round(max_prob, 4)
    except Exception as e:
        logger.error(f"Error predicting with baseline: {e}")
        default_label = "other" if task == "clause_type" else "needs_review"
        return default_label, 0.5
