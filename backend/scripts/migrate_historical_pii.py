"""
Phase 2 Historical PII Compatibility and Migration Script.

Audits and safely recomputes/invalidates historical PII findings across the database:
- Documents with valid encrypted raw text are safely recomputed using Phase 2 DocumentEntityResolver
  and exact redacted_text offset tracking.
- Documents without valid encrypted raw text have their historical raw-text offsets safely invalidated
  (start_offset=None, end_offset=None) rather than being silently reinterpreted against redacted_text.
- Preserves all authentication, tenant isolation, and encrypted source data.
"""

import sys
import logging
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app
from app.extensions import db
from app.models.document import Document
from app.models.pii_finding import PIIFinding
from app.utils.encryption import decrypt_text
from app.services.privacy.pii_redactor import redact

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pii_migrator")

def check_findings_compatibility(doc: Document, findings: list[PIIFinding]) -> bool:
    if not findings:
        return True
    redacted = doc.redacted_text or ""
    for f in findings:
        if f.start_offset is None or f.end_offset is None:
            return False
        if not (0 <= f.start_offset < f.end_offset <= len(redacted)):
            return False
        slice_text = redacted[f.start_offset:f.end_offset]
        if not (slice_text.startswith("[") and slice_text.endswith("]")):
            return False
    return True

def migrate_documents(dry_run: bool = False):
    app = create_app()
    with app.app_context():
        docs = Document.query.all()
        logger.info(f"Auditing {len(docs)} documents for PII offset compatibility...")

        compatible_count = 0
        recomputed_count = 0
        invalidated_count = 0
        no_findings_count = 0

        for doc in docs:
            findings = PIIFinding.query.filter_by(document_id=doc.id).order_by(PIIFinding.start_offset.asc()).all()
            if not findings:
                no_findings_count += 1
                continue

            if check_findings_compatibility(doc, findings):
                compatible_count += 1
                logger.info(f"[COMPATIBLE] Doc {doc.id} ({doc.filename}): {len(findings)} findings already aligned.")
                continue

            # Incompatible historical findings detected
            logger.info(f"[INCOMPATIBLE] Doc {doc.id} ({doc.filename}): {len(findings)} findings have legacy offsets.")

            # Attempt safe recomputation
            recomputed = False
            if doc.raw_text_encrypted:
                try:
                    raw_text = decrypt_text(doc.raw_text_encrypted)
                    if raw_text and raw_text.strip():
                        clean_redacted, new_findings = redact(raw_text)
                        if not dry_run:
                            doc.redacted_text = clean_redacted
                            PIIFinding.query.filter_by(document_id=doc.id).delete()
                            for nf in new_findings:
                                db.session.add(PIIFinding(
                                    document_id=doc.id,
                                    entity_type=nf["entity_type"],
                                    start_offset=nf["start"],
                                    end_offset=nf["end"],
                                    confidence=nf["confidence"],
                                ))
                            db.session.commit()
                        recomputed = True
                        recomputed_count += 1
                        logger.info(f"  -> Recomputed successfully: {len(new_findings)} neutral pseudonyms aligned.")
                except Exception as e:
                    logger.warning(f"  -> Decryption/recomputation failed: {e}")

            if not recomputed:
                # Safe Invalidation Strategy:
                # Set offsets to None to prevent erroneous highlights on mismatched text
                if not dry_run:
                    for f in findings:
                        f.start_offset = None
                        f.end_offset = None
                    db.session.commit()
                invalidated_count += 1
                logger.info("  -> Offsets safely invalidated (marked None).")

        logger.info("=" * 60)
        logger.info(f"Migration Summary (Dry Run: {dry_run}):")
        logger.info(f"  Total Documents:       {len(docs)}")
        logger.info(f"  Already Compatible:    {compatible_count}")
        logger.info(f"  No Findings/Empty:     {no_findings_count}")
        logger.info(f"  Safely Recomputed:     {recomputed_count}")
        logger.info(f"  Safely Invalidated:    {invalidated_count}")
        logger.info("=" * 60)

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    migrate_documents(dry_run=dry_run)
