import logging
from app.extensions import db
from app.models.clause import Clause

logger = logging.getLogger(__name__)

def retrieve_relevant_clauses(document_id, query_vector: list[float], top_k: int = 5) -> list[dict]:
    """
    Retrieves the top-k most relevant clauses for a document using pgvector cosine distance.
    Returns list of dicts with clause info and similarity score.
    """
    if not query_vector:
        return []

    try:
        distance_col = Clause.embedding.cosine_distance(query_vector)
        
        results = (
            db.session.query(Clause, distance_col.label("distance"))
            .filter(Clause.document_id == document_id)
            .filter(Clause.embedding.isnot(None))
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
                "redacted_text": clause.redacted_text,
                "similarity": round(sim, 4),
            })

        return retrieved
    except Exception as e:
        logger.error(f"Error querying pgvector for document {document_id}: {e}")
        clauses = (
            db.session.query(Clause)
            .filter(Clause.document_id == document_id)
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
                "redacted_text": c.redacted_text,
                "similarity": 0.5,
            }
            for c in clauses
        ]
