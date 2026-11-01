import uuid
from datetime import datetime, timezone
from sqlalchemy.dialects.postgresql import UUID
from app.extensions import db

class IntegrityFinding(db.Model):
    __tablename__ = "integrity_findings"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    category = db.Column(db.String(64), nullable=False)
    severity = db.Column(db.String(16), nullable=False, default="medium")  # low, medium, high, critical
    confidence = db.Column(db.Float, nullable=False, default=1.0)
    detection_method = db.Column(db.String(64), nullable=False)
    page_number = db.Column(db.Integer, nullable=True)
    bbox = db.Column(db.JSON, nullable=True)
    explanation = db.Column(db.Text, nullable=False)
    evidence_snippet = db.Column(db.Text, nullable=True)
    review_status = db.Column(db.String(32), nullable=False, default="Flagged for Review")
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    document = db.relationship("Document", back_populates="integrity_findings")

    def to_dict(self):
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "category": self.category,
            "severity": self.severity,
            "confidence": round(self.confidence, 2) if self.confidence is not None else 1.0,
            "detection_method": self.detection_method,
            "page_number": self.page_number,
            "bbox": self.bbox,
            "explanation": self.explanation,
            "evidence_snippet": self.evidence_snippet,
            "review_status": self.review_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
