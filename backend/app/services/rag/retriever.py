import logging
from app.extensions import db
from app.models.clause import Clause

logger = logging.getLogger(__name__)

def retrieve_relevant_clauses(document_id, query_vector: list[float], top_k: int = 5) -> list[dict]:
    """
    Retrieves the top-k most relevant substantive clauses for a document using pgvector cosine distance.
    Filters out Phase 6 non-substantive headers/footers/TOC when classified.
    """
    substantive_filter = Clause.is_substantive.is_(True) | Clause.is_substantive.is_(None)

    if not query_vector:
        clauses = (
            db.session.query(Clause)
            .filter(Clause.document_id == document_id)
            .filter(substantive_filter)
            .order_by(Clause.clause_index)
            .limit(top_k)
            .all()
        )
        return [
            {
                "clause_id": str(c.id),
                "clause_index": c.clause_index,
                "clause_type": c.clause_type,
                "favorability_label": c.favorability_label,
                "risk_score": c.risk_score,
                "page_number": getattr(c, "page_number", 1) or 1,
                "redacted_text": c.redacted_text,
                "similarity": 0.5,
            }
            for c in clauses
        ]

    try:
        distance_col = Clause.embedding.cosine_distance(query_vector)

        results = (
            db.session.query(Clause, distance_col.label("distance"))
            .filter(Clause.document_id == document_id)
            .filter(Clause.embedding.isnot(None))
            .filter(substantive_filter)
            .order_by(distance_col)
            .limit(top_k)
            .all()
        )

        retrieved = []
        for clause, dist in results:
            sim = 1.0 - float(dist) if dist is not None else 0.0
            retrieved.append({
                "clause_id": str(clause.id),
                "clause_index": clause.clause_index,
                "clause_type": clause.clause_type,
                "favorability_label": clause.favorability_label,
                "risk_score": clause.risk_score,
                "page_number": getattr(clause, "page_number", 1) or 1,
                "redacted_text": clause.redacted_text,
                "similarity": round(sim, 4),
            })

        return retrieved
    except Exception as e:
        logger.error(f"Error querying pgvector for document {document_id}: {e}")
        clauses = (
            db.session.query(Clause)
            .filter(Clause.document_id == document_id)
            .filter(substantive_filter)
            .order_by(Clause.clause_index)
            .limit(top_k)
            .all()
        )
        return [
            {
                "clause_id": str(c.id),
                "clause_index": c.clause_index,
                "clause_type": c.clause_type,
                "favorability_label": c.favorability_label,
                "risk_score": c.risk_score,
                "page_number": getattr(c, "page_number", 1) or 1,
                "redacted_text": c.redacted_text,
                "similarity": 0.5,
            }
            for c in clauses
        ]
