import uuid
from datetime import datetime, timezone
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.extensions import db

class Clause(db.Model):
    __tablename__ = "clauses"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    clause_index = db.Column(db.Integer, nullable=False)
    clause_type = db.Column(db.String(64))
    clause_type_confidence = db.Column(db.Float)
    favorability_label = db.Column(db.String(16))
    favorability_confidence = db.Column(db.Float)
    risk_score = db.Column(db.Float)
    redacted_text = db.Column(db.Text, nullable=False)
    raw_text_encrypted = db.Column(db.LargeBinary)
    start_offset = db.Column(db.Integer)
    end_offset = db.Column(db.Integer)
    page_number = db.Column(db.Integer, nullable=True)
    embedding = db.Column(Vector(768))
    content_type = db.Column(db.String(64), nullable=True, default="SUBSTANTIVE_CLAUSE")
    is_substantive = db.Column(db.Boolean, nullable=True, default=True)
    filter_reason = db.Column(db.Text, nullable=True)
    human_review = db.Column(db.Boolean, nullable=True, default=False)
    clause_eligibility_score = db.Column(db.Float, nullable=True, default=1.0)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    document = db.relationship("Document", back_populates="clauses")
    deadlines = db.relationship("Deadline", back_populates="clause")

    def to_dict(self):
        is_sub = self.is_substantive if self.is_substantive is not None else True
        effective_risk = 0.0 if not is_sub else (self.risk_score or 0.0)
        effective_clause_type = "NOT_APPLICABLE" if not is_sub else (self.clause_type or "NOT_APPLICABLE")
        effective_fav = "neutral" if not is_sub else (self.favorability_label or "neutral")
        effective_human_review = False if not is_sub else bool(self.human_review)

        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "clause_index": self.clause_index,
            "clause_type": effective_clause_type,
            "classification": effective_clause_type,
            "clause_type_confidence": 1.0 if not is_sub else self.clause_type_confidence,
            "favorability_label": effective_fav,
            "favorability_confidence": 1.0 if not is_sub else self.favorability_confidence,
            "risk_score": effective_risk,
            "content_type": self.content_type or ("SUBSTANTIVE_CLAUSE" if is_sub else "OTHER_NON_SUBSTANTIVE"),
            "is_substantive": is_sub,
            "human_review": effective_human_review,
            "filter_reason": self.filter_reason,
            "clause_eligibility_score": 0.0 if not is_sub else (self.clause_eligibility_score or 1.0),
            "redacted_text": self.redacted_text,
            "text": self.redacted_text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "page_number": self.page_number,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
