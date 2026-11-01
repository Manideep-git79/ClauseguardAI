import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_checklists_cache = {}

def get_checklist(document_type: str) -> dict:
    global _checklists_cache
    if document_type in _checklists_cache:
        return _checklists_cache[document_type]

    checklist_dir = Path(__file__).resolve().parent / "checklists"
    file_path = checklist_dir / f"{document_type}.json"
    if not file_path.exists():
        file_path = checklist_dir / "rental_agreement.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        _checklists_cache[document_type] = data
        return data

def detect_missing_clauses(
    classified_clauses: list[dict],
    document_type: str,
    confidence_threshold: float = 0.08,
) -> list[dict]:
    """
    Detects expected clauses that are absent from the document.
    Only counts a clause type as present if its classification confidence >= confidence_threshold.
    """
    checklist = get_checklist(document_type)
    expected_items = {
        item["type"]: item for item in checklist.get("expected_clause_types", [])
    }

    confident_types = set()
    for clause in classified_clauses:
        if isinstance(clause, str):
            c_type = clause
            c_conf = 1.0
        elif isinstance(clause, dict):
            c_type = clause.get("clause_type")
            c_conf = clause.get("clause_type_confidence", 1.0)
        else:
            continue
        if c_type and c_conf >= confidence_threshold:
            confident_types.add(c_type)

    missing = []
    for c_type, item in expected_items.items():
        if c_type not in confident_types:
            missing.append({
                "clause_type": c_type,
                "severity": item["severity_if_missing"],
                "checklist_note": item.get("note", "Expected standard clause type not identified."),
            })

    severity_order = {"high": 0, "medium": 1, "low": 2}
    missing.sort(key=lambda m: severity_order.get(m["severity"], 3))
    return missing
