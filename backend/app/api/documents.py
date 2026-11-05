import os
import tempfile
import threading
import logging
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, g
from werkzeug.utils import secure_filename
from app.extensions import db
from app.models.document import Document
from app.utils.validators import (
    allowed_file,
    validate_document_type,
    validate_file_content,
    EncryptedFileError,
    CorruptedFileError,
    EmptyFileError,
    FileLimitExceededError,
    FileValidationError,
)
from app.services.pipeline import process_document_pipeline
from app.services.audit_service import log_audit_event

logger = logging.getLogger(__name__)

documents_bp = Blueprint("documents", __name__)

from app.middleware.auth import require_auth
import uuid

def run_pipeline_async(app, document_id, file_path, ext):
    with app.app_context():
        try:
            process_document_pipeline(document_id, file_path, ext)
        except Exception as e:
            logger.error(f"Async pipeline processing failed for {document_id}: {e}")
            try:
                db.session.rollback()
                failed_doc = db.session.get(Document, document_id)
                if failed_doc and failed_doc.status != "analyzed":
                    failed_doc.status = "failed"
                    failed_doc.processing_stage = "failed"
                    failed_doc.error_message = str(e)
                    db.session.commit()
            except Exception as inner_err:
                logger.error(f"Failed updating failed status in run_pipeline_async: {inner_err}")

@documents_bp.route("/documents", methods=["POST"])
@require_auth
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded in 'file' field."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Selected file has an empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format. Supported formats: .pdf, .docx, .txt, .png, .jpg, .jpeg"}), 400

    doc_type = request.form.get("document_type", "").strip()
    if not validate_document_type(doc_type):
        return jsonify({
            "error": "Invalid document_type. Must be one of: 'rental_agreement', 'job_offer_letter', 'insurance_policy'"
        }), 400

    # Validate that uploaded file is not 0 bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size == 0:
        return jsonify({"error": "Uploaded file is empty (0 bytes)."}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename."}), 400
    ext = filename.rsplit(".", 1)[1].lower()

    # Prevent concurrent upload collisions using a unique, unguessable file path
    unique_upload_id = uuid.uuid4().hex
    temp_dir = tempfile.gettempdir()
    temp_filename = f"cg_{unique_upload_id}_{filename}"
    temp_path = os.path.join(temp_dir, temp_filename)
    file.save(temp_path)

    # Validate file integrity, signature, and encryption
    try:
        validate_file_content(temp_path, ext)
    except EncryptedFileError as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e), "code": "ENCRYPTED_FILE"}), 400
    except EmptyFileError as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e), "code": "EMPTY_FILE"}), 400
    except (CorruptedFileError, FileValidationError) as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e), "code": "INVALID_FILE"}), 400
    except FileLimitExceededError as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e), "code": "FILE_LIMIT_EXCEEDED"}), 400

    user = g.user

    doc = Document(
        filename=filename,
        original_format=ext,
        document_type=doc_type,
        status="uploaded",
        processing_stage="queued",
        user_id=user.id,
    )
    db.session.add(doc)
    db.session.commit()

    app = current_app._get_current_object()
    thread = threading.Thread(target=run_pipeline_async, args=(app, str(doc.id), temp_path, ext))
    thread.daemon = True
    thread.start()

    log_audit_event(
        action="DOC_UPLOADED",
        user_id=user.id,
        user_email=user.email,
        resource_type="document",
        resource_id=str(doc.id),
        details={"filename": doc.filename, "document_type": doc.document_type, "format": ext}
    )

    return jsonify({
        "id": str(doc.id),
        "document_id": str(doc.id),
        "status": doc.status,
        "processing_stage": doc.processing_stage,
        "filename": doc.filename,
        "document_type": doc.document_type,
    }), 201

def _check_and_update_watchdog(doc):
    if doc.status in ("uploaded", "processing") and doc.uploaded_at:
        now = datetime.now(timezone.utc)
        uploaded = doc.uploaded_at
        if uploaded.tzinfo is None:
            uploaded = uploaded.replace(tzinfo=timezone.utc)
        if (now - uploaded).total_seconds() > 600:
            doc.status = "failed"
            doc.processing_stage = "failed"
            doc.error_message = "Document processing timed out or server restarted. Please re-upload the document."
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()

@documents_bp.route("/documents", methods=["GET"])
@require_auth
def list_documents():
    user = g.user
    if getattr(user, "role", "") == "admin":
        docs = Document.query.order_by(Document.uploaded_at.desc()).all()
    else:
        docs = Document.query.filter_by(user_id=user.id).order_by(Document.uploaded_at.desc()).all()
    for d in docs:
        _check_and_update_watchdog(d)
    return jsonify([d.to_dict(include_text=False) for d in docs]), 200

@documents_bp.route("/documents/<uuid:doc_id>", methods=["GET"])
@require_auth
def get_document(doc_id):
    doc = Document.query.get_or_404(doc_id)
    user = g.user
    if doc.user_id != user.id and getattr(user, "role", "") != "admin":
        return jsonify({"error": "Unauthorized access to this document."}), 403
    _check_and_update_watchdog(doc)
    return jsonify(doc.to_dict(include_text=True)), 200

@documents_bp.route("/documents/<uuid:doc_id>/status", methods=["GET"])
@require_auth
def get_document_status(doc_id):
    doc = Document.query.get_or_404(doc_id)
    user = g.user
    if doc.user_id != user.id and getattr(user, "role", "") != "admin":
        return jsonify({"error": "Unauthorized access to this document."}), 403

    _check_and_update_watchdog(doc)

    return jsonify({
        "document_id": str(doc.id),
        "status": doc.status,
        "processing_stage": doc.processing_stage,
        "error_message": doc.error_message,
        "overall_risk_score": doc.overall_risk_score,
        "risk_band": doc.risk_band,
    }), 200

@documents_bp.route("/documents/<uuid:doc_id>", methods=["DELETE"])
@require_auth
def delete_document(doc_id):
    doc = db.session.get(Document, doc_id)
    if not doc:
        return jsonify({"message": f"Document {doc_id} already deleted."}), 200

    user = g.user
    if doc.user_id != user.id and getattr(user, "role", "") != "admin":
        return jsonify({"error": "Unauthorized. You can only delete your own documents."}), 403

    try:
        from app.models.chat import ChatMessage, ChatSession
        from app.models.clause import Clause
        from app.models.missing_clause import MissingClause
        from app.models.deadline import Deadline
        from app.models.pii_finding import PIIFinding

        filename = doc.filename
        doc_id_str = str(doc.id)

        sessions = ChatSession.query.filter_by(document_id=doc.id).all()
        session_ids = [s.id for s in sessions]
        if session_ids:
            ChatMessage.query.filter(ChatMessage.session_id.in_(session_ids)).delete(synchronize_session=False)

        ChatMessage.query.filter_by(document_id=doc.id).delete(synchronize_session=False)
        ChatSession.query.filter_by(document_id=doc.id).delete(synchronize_session=False)
        Clause.query.filter_by(document_id=doc.id).delete(synchronize_session=False)
        MissingClause.query.filter_by(document_id=doc.id).delete(synchronize_session=False)
        Deadline.query.filter_by(document_id=doc.id).delete(synchronize_session=False)
        PIIFinding.query.filter_by(document_id=doc.id).delete(synchronize_session=False)

        db.session.delete(doc)
        db.session.commit()

        log_audit_event(
            action="DOC_DELETED",
            user_id=user.id,
            user_email=user.email,
            resource_type="document",
            resource_id=doc_id_str,
            details={"filename": filename}
        )

        return jsonify({"message": f"Document {doc_id} deleted successfully."}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to delete document {doc_id}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to delete document: {str(e)}"}), 500

@documents_bp.route("/documents/<uuid:doc_id>/reanalyze", methods=["POST"])
@require_auth
def reanalyze_document(doc_id):
    doc = Document.query.get_or_404(doc_id)
    user = g.user
    if doc.user_id != user.id and getattr(user, "role", "") != "admin":
        return jsonify({"error": "Unauthorized access to this document."}), 403


    from app.models.clause import Clause
    from app.models.missing_clause import MissingClause
    from app.models.deadline import Deadline
    from app.services.nlp.block_segmenter import segment_blocks
    from app.services.nlp.classifier_bert import classify_clauses_batch
    from app.services.risk.risk_engine import compute_document_risk, compute_risk_band, evaluate_clause_risk_and_review
    from app.services.missing_clauses.detector import detect_missing_clauses
    from datetime import datetime, timezone

    if not doc.redacted_text or not doc.redacted_text.strip():
        return jsonify({"error": "Document has no redacted text to reanalyze."}), 400

    try:
        logger.info(f"Re-analyzing document {doc_id} using block_segmenter in optimized memory mode")
        raw_segments = segment_blocks(doc.redacted_text, document_type=doc.document_type)
        if not raw_segments:
            return jsonify({"error": "No logical blocks extracted from document."}), 400

        substantive_indices = [i for i, s in enumerate(raw_segments) if s.get("is_substantive")]
        substantive_segments = [raw_segments[i] for i in substantive_indices]
        excluded_count = len(raw_segments) - len(substantive_segments)

        logger.info(
            f"Document {doc_id}: {len(raw_segments)} total blocks "
            f"({len(substantive_segments)} substantive, {excluded_count} excluded)"
        )

        # Prepare inputs for substantive segments
        clause_items = []
        for i, s in enumerate(substantive_segments):
            prev_t = substantive_segments[i - 1]["text"] if i > 0 else None
            next_t = substantive_segments[i + 1]["text"] if i + 1 < len(substantive_segments) else None
            clause_items.append({
                "text": s["text"],
                "prev_text": prev_t,
                "next_text": next_t,
            })

        model_version_used = "Fine-Tuned DistiBERT"
        substantive_classifications = {}
        if clause_items:
            type_results = classify_clauses_batch(clause_items, document_type=doc.document_type, task="clause_type")
            fav_results = classify_clauses_batch(clause_items, document_type=doc.document_type, task="favorability")

            for idx_in_sub, orig_idx in enumerate(substantive_indices):
                c_type, c_conf, m_ver = type_results[idx_in_sub]
                model_version_used = m_ver
                fav_label, fav_conf, _ = fav_results[idx_in_sub]

                # Classifier abstention for low confidence
                if c_conf is not None and c_conf < 0.25:
                    final_type = "UNCERTAIN"
                    needs_review = True
                else:
                    final_type = c_type
                    needs_review = False

                substantive_classifications[orig_idx] = {
                    "clause_type": final_type,
                    "clause_type_confidence": c_conf,
                    "favorability_label": fav_label,
                    "favorability_confidence": fav_conf,
                    "human_review": needs_review,
                }

        doc.model_version = model_version_used

        # Build Clause objects in memory and evaluate risk with safety gate
        clause_objs = []
        clause_scores = []
        prior_clause_type = None

        for i, seg in enumerate(raw_segments):
            is_sub = seg.get("is_substantive", True)
            if is_sub and i in substantive_classifications:
                cls_info = substantive_classifications[i]
                c_type = cls_info["clause_type"]
                c_conf = cls_info["clause_type_confidence"]
                fav_label = cls_info["favorability_label"]
                fav_conf = cls_info["favorability_confidence"]
                initial_review = cls_info["human_review"]
            else:
                c_type = "NOT_APPLICABLE"
                c_conf = 1.0
                fav_label = "neutral"
                fav_conf = 1.0
                initial_review = False

            eval_res = evaluate_clause_risk_and_review(
                clause_type=c_type,
                favorability_label=fav_label,
                favorability_confidence=fav_conf,
                content_type=seg.get("content_type", "SUBSTANTIVE_CLAUSE"),
                is_substantive=is_sub,
                prior_clause_type=prior_clause_type,
            )

            r_score = eval_res["risk_score"]
            h_review = initial_review or eval_res["human_review"]
            effective_fav = eval_res.get("favorability_label", fav_label)
            f_reason = eval_res.get("review_reason") if is_sub and eval_res.get("review_reason") else seg.get("filter_reason")

            if is_sub:
                clause_scores.append(r_score)
                prior_clause_type = c_type

            c = Clause(
                document_id=doc.id,
                clause_index=seg["clause_index"],
                redacted_text=seg["text"],
                start_offset=seg["start_offset"],
                end_offset=seg["end_offset"],
                content_type=seg.get("content_type", "SUBSTANTIVE_CLAUSE"),
                is_substantive=is_sub,
                filter_reason=f_reason,
                clause_eligibility_score=seg.get("clause_eligibility_score", 1.0),
                human_review=h_review,
                clause_type=eval_res["classification"],
                clause_type_confidence=c_conf,
                favorability_label=effective_fav,
                favorability_confidence=fav_conf,
                risk_score=r_score,
            )
            clause_objs.append(c)

        # Detect missing clauses only from substantive classifications
        if substantive_indices:
            classified_dicts = [
                {"clause_type": c.clause_type, "clause_type_confidence": c.clause_type_confidence}
                for c in clause_objs
                if c.is_substantive and c.clause_type not in ("NOT_APPLICABLE", "UNCERTAIN")
            ]
            missing_list = detect_missing_clauses(classified_dicts, document_type=doc.document_type)
        else:
            missing_list = []

        has_substantive = bool(substantive_indices)
        doc_risk = compute_document_risk(clause_scores, missing_list) if has_substantive else 0.0
        band = compute_risk_band(doc_risk, has_substantive_content=has_substantive)
        doc.overall_risk_score = doc_risk
        doc.risk_band = band
        doc.analyzed_at = datetime.now(timezone.utc)

        # Perform atomic replace in DB
        Deadline.query.filter_by(document_id=doc.id).delete(synchronize_session=False)
        Clause.query.filter_by(document_id=doc.id).delete(synchronize_session=False)
        MissingClause.query.filter_by(document_id=doc.id).delete(synchronize_session=False)

        for c in clause_objs:
            db.session.add(c)
        db.session.flush()

        for m in missing_list:
            mc = MissingClause(
                document_id=doc.id,
                clause_type=m["clause_type"],
                severity=m["severity"],
                checklist_note=m["checklist_note"],
            )
            db.session.add(mc)

        # Enriched obligation & deadline extraction for substantive clauses during re-analysis
        from app.services.deadlines.deadline_extractor import extract_all_deadlines
        substantive_db_clauses = [c for c in clause_objs if c.is_substantive]
        clause_dict_list = [
            {
                "id": c.id,
                "redacted_text": c.redacted_text,
                "clause_index": c.clause_index,
                "page_number": getattr(c, "page_number", 1) or 1,
                "start_offset": getattr(c, "start_offset", 0) or 0,
                "end_offset": getattr(c, "end_offset", len(c.redacted_text or "")) or len(c.redacted_text or ""),
            }
            for c in substantive_db_clauses
        ]
        extracted_deadlines = extract_all_deadlines(clause_dict_list)
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

        # Refresh prompt injection signals on redacted text
        from app.models.integrity_finding import IntegrityFinding
        from app.services.integrity.detector import detect_prompt_injection
        try:
            inj_findings = detect_prompt_injection(doc.redacted_text)
            IntegrityFinding.query.filter_by(document_id=doc.id, detection_method="heuristic_rule_scan").delete(synchronize_session=False)
            for inf in inj_findings:
                db.session.add(IntegrityFinding(
                    document_id=doc.id,
                    category=inf["category"],
                    severity=inf["severity"],
                    confidence=inf["confidence"],
                    detection_method=inf["detection_method"],
                    page_number=inf.get("page_number"),
                    bbox=inf.get("bbox"),
                    explanation=inf["explanation"],
                    evidence_snippet=inf.get("evidence_snippet"),
                    review_status="Flagged for Review",
                ))
        except Exception as inj_err:
            logger.debug(f"Integrity scan error during reanalysis: {inj_err}")

        db.session.commit()

        log_audit_event(
            action="DOC_ANALYZED",
            user_id=user.id if user else None,
            user_email=user.email if user else None,
            resource_type="document",
            resource_id=str(doc.id),
            details={"model_version": doc.model_version, "reanalyzed": True}
        )

        return jsonify({
            "message": "Document re-analyzed successfully.",
            "model_version": doc.model_version,
            "overall_risk_score": doc.overall_risk_score,
            "risk_band": doc.risk_band,
            "document": doc.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Re-analysis failed for {doc_id}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to reanalyze document: {str(e)}"}), 500


