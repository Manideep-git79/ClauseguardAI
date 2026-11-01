import uuid
from sqlalchemy.dialects.postgresql import UUID
from app.extensions import db

class Deadline(db.Model):
    __tablename__ = "deadlines"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    clause_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("clauses.id", ondelete="SET NULL"),
        nullable=True,
    )
    deadline_type = db.Column(db.String(64))
    raw_text = db.Column(db.Text, nullable=False)
    parsed_date = db.Column(db.Date)
    relative_days = db.Column(db.Integer)
    confidence = db.Column(db.String(16), nullable=False, default="needs_review")

    # Enriched obligation & temporal reasoning fields (Phase 7)
    obligation_title = db.Column(db.String(256))
    responsible_party = db.Column(db.String(128))
    required_action = db.Column(db.Text)
    trigger_event = db.Column(db.String(256))
    trigger_date = db.Column(db.Date)
    duration_count = db.Column(db.Integer)
    duration_unit = db.Column(db.String(32))
    day_count_convention = db.Column(db.String(32))
    recurrence = db.Column(db.String(64))
    conditions = db.Column(db.Text)
    page_number = db.Column(db.Integer)
    start_offset = db.Column(db.Integer)
    end_offset = db.Column(db.Integer)
    status = db.Column(db.String(32), default="RESOLVED_EXPLICIT")
    review_reason = db.Column(db.Text)
    calculation_note = db.Column(db.Text)

    document = db.relationship("Document", back_populates="deadlines")
    clause = db.relationship("Clause", back_populates="deadlines")

    def to_dict(self):
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "clause_id": str(self.clause_id) if self.clause_id else None,
            "deadline_type": self.deadline_type,
            "raw_text": self.raw_text,
            "parsed_date": self.parsed_date.isoformat() if self.parsed_date else None,
            "relative_days": self.relative_days,
            "confidence": self.confidence,
            "obligation_title": self.obligation_title,
            "responsible_party": self.responsible_party,
            "required_action": self.required_action,
            "trigger_event": self.trigger_event,
            "trigger_date": self.trigger_date.isoformat() if self.trigger_date else None,
            "duration_count": self.duration_count,
            "duration_unit": self.duration_unit,
            "day_count_convention": self.day_count_convention,
            "recurrence": self.recurrence,
            "conditions": self.conditions,
            "page_number": self.page_number,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "status": self.status or ("RESOLVED_EXPLICIT" if self.parsed_date else "UNRESOLVED_EVENT_BASED"),
            "review_reason": self.review_reason,
            "calculation_note": self.calculation_note,
        }
