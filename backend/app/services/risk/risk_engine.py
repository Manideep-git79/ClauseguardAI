import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_config_cache = None

def get_risk_config() -> dict:
    global _config_cache
    if _config_cache is None:
        config_path = Path(__file__).resolve().parent / "risk_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            _config_cache = json.load(f)
    return _config_cache

def normalize_favorability(label: str | None) -> str:
    """
    Normalizes favorability labels to a canonical lowercase representation.
    Valid canonical labels: 'fair', 'neutral', 'needs_review', 'unfavorable'.
    Missing, malformed, or unrecognized labels return 'unknown'.
    """
    if not label or not isinstance(label, str):
        return "unknown"
    clean = label.strip().lower().replace(" ", "_").replace("-", "_")
    if clean in ("fair", "standard", "favorable", "low_risk"):
        return "fair"
    if clean in ("neutral", "not_applicable", "na", "none"):
        return "neutral"
    if clean in ("needs_review", "review", "uncertain", "flagged"):
        return "needs_review"
    if clean in ("unfavorable", "adverse", "high_risk", "hazard"):
        return "unfavorable"
    # Explicit return for unrecognized labels
    return "unknown"

def normalize_clause_type(clause_type: str | None) -> str:
    """
    Normalizes clause_type to a canonical lowercase representation.
    """
    if not clause_type or not isinstance(clause_type, str):
        return "NOT_APPLICABLE"
    clean = clause_type.strip().lower().replace(" ", "_").replace("-", "_")
    if clean in ("not_applicable", "none", "n_a", ""):
        return "NOT_APPLICABLE"
    if clean in ("uncertain", "unknown"):
        return "UNCERTAIN"
    if clean in ("unclassified",):
        return "UNCLASSIFIED"
    return clean

def compute_clause_risk(
    clause_type: str,
    favorability_label: str,
    favorability_confidence: float = 1.0,
    prior_clause_type: str = None,
    config: dict = None,
    content_type: str = "SUBSTANTIVE_CLAUSE",
    is_substantive: bool = True,
) -> float:
    """
    Computes a 0-100 risk score for an individual clause based on transparent defaults.
    formula: base_weight * (severity_multiplier + sequence_bump) * confidence * 100
    
    HARD SAFETY GATE:
    Only SUBSTANTIVE_CLAUSE is allowed to enter the legal risk engine.
    For any non-substantive content or excluded metadata, risk_score is strictly 0.0.
    """
    if content_type != "SUBSTANTIVE_CLAUSE" or not is_substantive:
        return 0.0

    norm_type = normalize_clause_type(clause_type)
    if norm_type in ("NOT_APPLICABLE", "UNCERTAIN", "UNCLASSIFIED"):
        return 0.0

    if config is None:
        config = get_risk_config()

    norm_fav = normalize_favorability(favorability_label)

    # Base weight lookup with explicit zero-risk fallback for unknown categories
    unknown_type_weight = config.get("unknown_clause_type_base_weight", 0.0)
    base = config["clause_type_base_weight"].get(norm_type, unknown_type_weight)

    # Severity multiplier lookup with explicit zero-risk fallback for unknown categories
    unknown_fav_mult = config.get("unknown_favorability_multiplier", 0.0)
    multiplier = config["favorability_severity_multiplier"].get(norm_fav, unknown_fav_mult)

    if prior_clause_type:
        norm_prior = normalize_clause_type(prior_clause_type)
        pair_key = f"{norm_prior}->{norm_type}"
        seq_bump = config.get("sequence_adjustment_pairs", {}).get(pair_key, 0.0)
        multiplier += seq_bump

    multiplier = min(1.0, max(0.0, multiplier))
    conf = favorability_confidence if favorability_confidence is not None else 1.0
    conf = max(0.0, min(1.0, conf))

    score = round(base * multiplier * conf * 100, 1)
    return max(0.0, min(100.0, score))


def evaluate_clause_risk_and_review(
    clause_type: str,
    favorability_label: str,
    favorability_confidence: float = 1.0,
    content_type: str = "SUBSTANTIVE_CLAUSE",
    is_substantive: bool = True,
    prior_clause_type: str = None,
    config: dict = None,
) -> dict:
    """
    Evaluates risk score and human review flag with mandatory safety gating.
    Enforces that non-substantive blocks never trigger human review and have 0 risk.
    Guarantees that unknown or missing substantive classifications are NOT silently
    presented as safe, but are flagged for human review with an explicit explanation.
    """
    if content_type != "SUBSTANTIVE_CLAUSE" or not is_substantive:
        return {
            "risk_score": 0.0,
            "human_review": False,
            "classification": "NOT_APPLICABLE",
            "favorability_label": "neutral",
            "review_reason": None,
        }

    norm_type = normalize_clause_type(clause_type)
    if norm_type == "NOT_APPLICABLE":
        return {
            "risk_score": 0.0,
            "human_review": False,
            "classification": "NOT_APPLICABLE",
            "favorability_label": "neutral",
            "review_reason": None,
        }

    if norm_type == "UNCERTAIN":
        return {
            "risk_score": 0.0,
            "human_review": True,
            "classification": "UNCERTAIN",
            "favorability_label": "needs_review",
            "review_reason": "Low classification confidence (<0.45): clause requires manual legal review.",
        }

    if norm_type == "UNCLASSIFIED":
        return {
            "risk_score": 0.0,
            "human_review": True,
            "classification": "UNCLASSIFIED",
            "favorability_label": "needs_review",
            "review_reason": "Unsupported contract type or unclassified clause: requires manual legal review.",
        }

    norm_fav = normalize_favorability(favorability_label)
    cfg = config if config is not None else get_risk_config()
    is_unknown_type = norm_type not in cfg.get("clause_type_base_weight", {})
    is_unknown_fav = norm_fav == "unknown"

    # Human review conditions with explicit review reasons (DEF-301: symmetric confidence review)
    review_reason = None
    if is_unknown_type:
        needs_review = True
        effective_fav = "needs_review"
        review_reason = f"Unrecognized substantive clause type '{norm_type}': requires human legal assessment."
    elif is_unknown_fav:
        needs_review = True
        effective_fav = "needs_review"
        review_reason = "Unrecognized or missing favorability classification requires manual review."
    elif norm_fav == "needs_review":
        needs_review = True
        effective_fav = "needs_review"
        review_reason = "Model evaluated clause as requiring manual legal review."
    elif norm_fav == "fair" and (favorability_confidence or 0.0) < 0.6:
        needs_review = True
        effective_fav = "needs_review"
        review_reason = f"Low confidence fair prediction ({(favorability_confidence or 0.0):.2f} < 0.60): clause requires manual verification."
    elif norm_fav == "unfavorable" and (favorability_confidence or 0.0) < 0.6:
        needs_review = True
        effective_fav = "needs_review"
        review_reason = f"Low confidence unfavorable prediction ({(favorability_confidence or 0.0):.2f} < 0.60): requires verification."
    else:
        score_candidate = compute_clause_risk(
            clause_type=norm_type,
            favorability_label=norm_fav,
            favorability_confidence=favorability_confidence,
            prior_clause_type=prior_clause_type,
            config=cfg,
            content_type=content_type,
            is_substantive=is_substantive,
        )
        if score_candidate >= 40.0:
            needs_review = True
            effective_fav = norm_fav
            review_reason = f"Elevated clause risk score ({score_candidate}/100 >= 40.0 threshold): flagged for review."
        else:
            needs_review = False
            effective_fav = norm_fav

    score = compute_clause_risk(
        clause_type=norm_type,
        favorability_label=norm_fav,
        favorability_confidence=favorability_confidence,
        prior_clause_type=prior_clause_type,
        config=cfg,
        content_type=content_type,
        is_substantive=is_substantive,
    )

    return {
        "risk_score": score,
        "human_review": needs_review,
        "classification": norm_type,
        "favorability_label": effective_fav,
        "review_reason": review_reason,
    }

def compute_document_risk(
    clause_scores: list[float],
    missing_clauses: list[dict],
    config: dict = None,
) -> float:
    """
    Computes document-level risk score combining average clause risk and missing clause penalties.
    If there are no substantive clause scores (empty document or only excluded metadata),
    no fabricated risk is produced, returning 0.0.
    """
    if not clause_scores:
        return 0.0

    if config is None:
        config = get_risk_config()

    mean_clause_risk = sum(clause_scores) / len(clause_scores)
    missing_penalty = sum(
        config["missing_clause_penalty_by_severity"].get(m.get("severity", "medium"), 0)
        for m in missing_clauses
    )
    missing_penalty = min(100.0, missing_penalty)

    w = config["composite_weights"]
    overall = (
        w["mean_clause_risk_weight"] * mean_clause_risk
        + w["missing_clause_penalty_weight"] * missing_penalty
    )
    return round(min(100.0, max(0.0, overall)), 1)

def compute_risk_band(score: float, config: dict = None, has_substantive_content: bool = True) -> str:
    """
    Maps 0-100 risk score to low, medium, high, critical, or insufficient_content band.
    When a document has zero substantive clauses (e.g. heading only, salutation only),
    it returns 'insufficient_content' so that the absence of analyzed clauses is not
    silently represented as proof that a contract is safe.
    """
    if not has_substantive_content:
        return "insufficient_content"

    if config is None:
        config = get_risk_config()

    for band, (low, high) in config["risk_bands"].items():
        if low <= score <= high:
            return band
    return "critical" if score > 75 else "low"
