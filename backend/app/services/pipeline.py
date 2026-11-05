import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from app.extensions import db
from app.models.document import Document
from app.models.clause import Clause
from app.models.missing_clause import MissingClause
from app.models.deadline import Deadline
from app.models.pii_finding import PIIFinding
from app.utils.encryption import encrypt_text
from app.services.ingestion.extractor import extract_text, extract_text_and_metadata
from app.services.privacy.pii_redactor import redact
from app.services.nlp.segmenter import segment_clauses
from app.services.nlp.classifier_bert import classify_clause, classify_clauses_batch
from app.services.risk.risk_engine import compute_clause_risk, compute_document_risk, compute_risk_band
from app.services.missing_clauses.detector import detect_missing_clauses
from app.services.deadlines.deadline_extractor import extract_all_deadlines
from app.services.rag.embedder import embed_text, batch_embed_texts
from app.services.audit_service import log_audit_event

logger = logging.getLogger(__name__)

def process_document_pipeline(document_id: str, file_path: str, ext: str) -> Document:
    """
    Executes the full end-to-end processing pipeline for a legal document:
    1. Extract text (PyMuPDF / OCR / python-docx / txt)
    2. Redact PII (Presidio, DATE_TIME excluded)
    3. Segment into clauses (Rule-based + spaCy)
    4. Classify clause types and favorability (Context-windowed BERT / Baseline)
    5. Compute deterministic risk score & missing clause penalties
    6. Extract notice periods and deadlines
    7. Generate 768-dim embeddings for RAG
    8. Store results & update stage status
    """
    doc = db.session.get(Document, document_id)
    if not doc:
        raise ValueError(f"Document {document_id} not found in database.")

    db.session.expire_on_commit = False

    try:
        doc.status = "processing"
        doc.processing_stage = "extracting_text"
        db.session.commit()
        logger.info(f"[{doc.id}] Stage 1: Extracting text from {file_path}")

        raw_text, page_count, page_metadata = extract_text_and_metadata(file_path, ext)
        if not raw_text or not raw_text.strip():
            raise ValueError("Document contains no readable text.")

        doc.page_count = page_count
        doc.raw_text_encrypted = encrypt_text(raw_text)

        # Extract contracting parties, entity types, and signatories
        from app.services.nlp.contract_parties import extract_contract_parties
        try:
            parties_meta = extract_contract_parties(raw_text, document_type=doc.document_type)
            doc.parties_metadata = parties_meta
            logger.info(f"[{doc.id}] Extracted {len(parties_meta.get('parties', []))} party(ies) and {len(parties_meta.get('signatories', []))} signatory(ies)")
        except Exception as p_err:
            logger.warning(f"[{doc.id}] Non-fatal error during party extraction: {p_err}")
            doc.parties_metadata = {}

        # Ensure idempotence on repeat document analysis
        from app.models.integrity_finding import IntegrityFinding
        Deadline.query.filter_by(document_id=doc.id).delete()
        MissingClause.query.filter_by(document_id=doc.id).delete()
        Clause.query.filter_by(document_id=doc.id).delete()
        PIIFinding.query.filter_by(document_id=doc.id).delete()
        IntegrityFinding.query.filter_by(document_id=doc.id).delete()
        db.session.commit()

        doc.processing_stage = "redacting_pii"
        db.session.commit()
        logger.info(f"[{doc.id}] Stage 2: Redacting PII")

        redacted_text, pii_findings = redact(raw_text)
        doc.redacted_text = redacted_text

        import gc
        pii_objs = [
            PIIFinding(
                document_id=doc.id,
                entity_type=f["entity_type"],
                start_offset=f["start"],
                end_offset=f["end"],
                confidence=f["confidence"],
            )
            for f in pii_findings
        ]
        db.session.add_all(pii_objs)
        db.session.commit()
        gc.collect()

        # Phase 5: Document Integrity, Hidden-Text, and Prompt-Injection Defense Scan
        from app.models.integrity_finding import IntegrityFinding
        from app.services.integrity.detector import analyze_document_integrity

        try:
            detected_signals = analyze_document_integrity(
                file_path=file_path,
                ext=ext,
                raw_text=raw_text,
                redacted_text=redacted_text,
                page_metadata=page_metadata,
                document_type=doc.document_type,
            )
            integrity_objs = [
                IntegrityFinding(
                    document_id=doc.id,
                    category=sig["category"],
                    severity=sig["severity"],
                    confidence=sig["confidence"],
                    detection_method=sig["detection_method"],
                    page_number=sig.get("page_number"),
                    bbox=sig.get("bbox"),
                    explanation=sig["explanation"],
                    evidence_snippet=sig.get("evidence_snippet"),
                    review_status=sig.get("review_status", "Flagged for Review"),
                )
                for sig in detected_signals
            ]
            db.session.add_all(integrity_objs)
            db.session.commit()
            gc.collect()
            logger.info(f"[{doc.id}] Document integrity scan stored {len(detected_signals)} signal(s).")
        except Exception as sig_err:
            logger.warning(f"[{doc.id}] Non-fatal integrity scan error: {sig_err}")

        doc.processing_stage = "segmenting_clauses"
        db.session.commit()
        logger.info(f"[{doc.id}] Stage 3: Segmenting logical blocks & gating non-substantive content for {doc.document_type}")

        from app.services.nlp.block_segmenter import segment_blocks
        from app.services.risk.risk_engine import evaluate_clause_risk_and_review

        raw_segments = segment_blocks(redacted_text, document_type=doc.document_type, page_metadata=page_metadata)
        if not raw_segments:
            raw_segments = [{
                "clause_index": 0,
                "text": redacted_text.strip(),
                "start_offset": 0,
                "end_offset": len(redacted_text.strip()),
                "page_number": 1,
                "content_type": "SUBSTANTIVE_CLAUSE",
                "is_substantive": True,
                "classification": "PENDING_DISTILBERT",
                "risk_score": 0.0,
                "human_review": False,
                "clause_eligibility_score": 1.0,
                "filter_reason": None,
            }]

        import uuid
        clause_objs = []
        for seg in raw_segments:
            cid = uuid.uuid4()
            c = Clause(
                id=cid,
                document_id=doc.id,
                clause_index=seg["clause_index"],
                redacted_text=seg["text"],
                start_offset=seg["start_offset"],
                end_offset=seg["end_offset"],
                page_number=seg.get("page_number", 1),
                content_type=seg.get("content_type", "SUBSTANTIVE_CLAUSE"),
                is_substantive=seg.get("is_substantive", True),
                filter_reason=seg.get("filter_reason"),
                clause_eligibility_score=seg.get("clause_eligibility_score", 1.0),
                human_review=seg.get("human_review", False),
                clause_type=seg.get("classification", "NOT_APPLICABLE"),
                risk_score=seg.get("risk_score") or 0.0,
            )
            clause_objs.append(c)

        doc.processing_stage = "classifying_clauses"
        db.session.commit()

        substantive_clauses = [c for c in clause_objs if c.is_substantive]
        logger.info(
            f"[{doc.id}] Stage 4: Found {len(clause_objs)} total blocks "
            f"({len(substantive_clauses)} substantive, {len(clause_objs) - len(substantive_clauses)} non-substantive excluded)"
        )

        # Build context-windowed items ONLY for substantive clauses
        clause_items = []
        for i, c in enumerate(substantive_clauses):
            prev_t = substantive_clauses[i - 1].redacted_text if i > 0 else None
            next_t = substantive_clauses[i + 1].redacted_text if i + 1 < len(substantive_clauses) else None
            clause_items.append({
                "text": c.redacted_text,
                "prev_text": prev_t,
                "next_text": next_t,
            })

        model_version_used = "Fine-Tuned DistiBERT"
        if clause_items:
            type_results = classify_clauses_batch(clause_items, document_type=doc.document_type, task="clause_type")
            fav_results = classify_clauses_batch(clause_items, document_type=doc.document_type, task="favorability")

            for i, c in enumerate(substantive_clauses):
                c_type, c_conf, m_ver = type_results[i]
                model_version_used = m_ver

                # Section 8: Classifier Abstention - if confidence is inadequate (< 0.45), classify as UNCERTAIN to void false predictions
                if c_conf is not None and c_conf < 0.45:
                    c.clause_type = "UNCERTAIN"
                    c.clause_type_confidence = c_conf
                    c.human_review = True
                else:
                    c.clause_type = c_type
                    c.clause_type_confidence = c_conf

                fav_label, fav_conf, _ = fav_results[i]
                # Section 8: Favorability Abstention - if confidence is inadequate (< 0.45), abstain to needs_review
                if fav_conf is not None and fav_conf < 0.45:
                    c.favorability_label = "needs_review"
                    c.favorability_confidence = fav_conf
                    c.human_review = True
                else:
                    c.favorability_label = fav_label.strip().lower() if fav_label else "neutral"
                    c.favorability_confidence = fav_conf

        # Non-substantive clauses strictly have NOT_APPLICABLE and 0 risk
        for c in clause_objs:
            if not c.is_substantive:
                c.clause_type = "NOT_APPLICABLE"
                c.clause_type_confidence = 1.0
                c.favorability_label = "neutral"
                c.favorability_confidence = 1.0
                c.risk_score = 0.0
                c.human_review = False

        doc.model_version = model_version_used
        doc.processing_stage = "scoring_risk"
        db.session.commit()
        logger.info(f"[{doc.id}] Stage 5: Scoring risk with mandatory non-substantive safety gate")

        clause_scores = []
        for i, c in enumerate(clause_objs):
            prior_type = clause_objs[i - 1].clause_type if i > 0 else None
            eval_res = evaluate_clause_risk_and_review(
                clause_type=c.clause_type,
                favorability_label=c.favorability_label,
                favorability_confidence=c.favorability_confidence if c.favorability_confidence is not None else 1.0,
                content_type=c.content_type,
                is_substantive=c.is_substantive,
                prior_clause_type=prior_type,
            )
            c.risk_score = eval_res["risk_score"]
            c.human_review = eval_res["human_review"]
            if c.is_substantive:
                c.favorability_label = eval_res.get("favorability_label", c.favorability_label or "needs_review")
                if eval_res.get("review_reason"):
                    c.filter_reason = eval_res["review_reason"]
                clause_scores.append(c.risk_score)

        if substantive_clauses:
            classified_dicts = [
                {"clause_type": c.clause_type, "clause_type_confidence": c.clause_type_confidence}
                for c in substantive_clauses
                if c.clause_type not in ("NOT_APPLICABLE", "UNCERTAIN")
            ]
            missing_list = detect_missing_clauses(classified_dicts, document_type=doc.document_type)
        else:
            missing_list = []

        has_substantive = bool(substantive_clauses)
        doc_risk = compute_document_risk(clause_scores, missing_list) if has_substantive else 0.0
        band = compute_risk_band(doc_risk, has_substantive_content=has_substantive)

        doc.overall_risk_score = doc_risk
        doc.risk_band = band
        doc.processing_stage = "extracting_deadlines"
        db.session.commit()
        logger.info(f"[{doc.id}] Stage 6: Extracting deadlines & obligations from substantive clauses")

        clause_dict_list = [
            {
                "id": c.id,
                "redacted_text": c.redacted_text,
                "clause_index": c.clause_index,
                "page_number": getattr(c, "page_number", 1) or 1,
                "start_offset": getattr(c, "start_offset", 0) or 0,
                "end_offset": getattr(c, "end_offset", len(c.redacted_text or "")) or len(c.redacted_text or ""),
            }
            for c in substantive_clauses
        ]
        extracted_deadlines = extract_all_deadlines(clause_dict_list)

        doc.processing_stage = "generating_embeddings"
        db.session.commit()
        logger.info(f"[{doc.id}] Stage 7: Generating Gemini embeddings for {len(clause_objs)} total clauses in batch")

        texts_to_embed = [c.redacted_text for c in clause_objs if c.redacted_text and c.redacted_text.strip()]
        if texts_to_embed:
            batch_embs = batch_embed_texts(texts_to_embed, task_type="RETRIEVAL_DOCUMENT", batch_size=100)
            for c, emb in zip(clause_objs, batch_embs):
                c.embedding = emb

        # Single bulk insertion of all clauses, missing clauses, and deadlines
        db.session.add_all(clause_objs)

        for m in missing_list:
            mc = MissingClause(
                document_id=doc.id,
                clause_type=m["clause_type"],
                severity=m["severity"],
                checklist_note=m["checklist_note"],
            )
            db.session.add(mc)

        for dl in extracted_deadlines:
            d_obj = Deadline(
                document_id=doc.id,
                clause_id=dl.get("clause_id"),
                deadline_type=dl.get("deadline_type"),
                raw_text=dl.get("raw_text"),
                parsed_date=dl.get("parsed_date"),
                relative_days=dl.get("relative_days"),
                confidence=dl.get("confidence", "needs_review"),
                obligation_title=dl.get("obligation_title"),
                responsible_party=dl.get("responsible_party"),
                required_action=dl.get("required_action"),
                trigger_event=dl.get("trigger_event"),
                trigger_date=dl.get("trigger_date"),
                duration_count=dl.get("duration_count"),
                duration_unit=dl.get("duration_unit"),
                day_count_convention=dl.get("day_count_convention"),
                recurrence=dl.get("recurrence"),
                conditions=dl.get("conditions"),
                page_number=dl.get("page_number"),
                start_offset=dl.get("start_offset"),
                end_offset=dl.get("end_offset"),
                status=dl.get("status"),
                review_reason=dl.get("review_reason"),
                calculation_note=dl.get("calculation_note"),
            )
            db.session.add(d_obj)

        doc.status = "analyzed"
        doc.processing_stage = "completed"
        doc.analyzed_at = datetime.now(timezone.utc)
        db.session.commit()
        logger.info(f"[{doc.id}] Document processing completed successfully! Risk Score: {doc.overall_risk_score} ({doc.risk_band})")

        log_audit_event(
            action="DOC_ANALYZED",
            resource_type="document",
            resource_id=str(doc.id),
            details={
                "filename": doc.filename,
                "overall_risk_score": doc.overall_risk_score,
                "risk_band": doc.risk_band,
                "clauses_count": len(clause_objs),
                "deadlines_count": len(extracted_deadlines),
                "missing_clauses_count": len(missing_list)
            }
        )

        return doc

    except Exception as e:
        logger.exception(f"Pipeline error for document {document_id}: {e}")
        try:
            db.session.rollback()
        except Exception:
            try:
                db.session.remove()
            except Exception:
                pass
        try:
            failed_doc = db.session.get(Document, document_id)
            if failed_doc:
                failed_doc.status = "failed"
                failed_doc.processing_stage = "failed"
                failed_doc.error_message = str(e)
                db.session.commit()
        except Exception as rec_err:
            logger.error(f"Failed to record error state for {document_id}: {rec_err}")
            try:
                db.session.remove()
                failed_doc = db.session.get(Document, document_id)
                if failed_doc:
                    failed_doc.status = "failed"
                    failed_doc.processing_stage = "failed"
                    failed_doc.error_message = str(e)
                    db.session.commit()
            except Exception:
                pass
        raise e
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
