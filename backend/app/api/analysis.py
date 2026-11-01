import io
from flask import Blueprint, jsonify, send_file, current_app, g, request
from app.models.document import Document
from app.models.clause import Clause
from app.models.missing_clause import MissingClause
from app.models.deadline import Deadline
from app.models.pii_finding import PIIFinding
from app.reports.pdf_report import generate_pdf_report
from app.middleware.auth import require_auth

analysis_bp = Blueprint("analysis", __name__)

def get_authorized_doc_or_403(doc_id):
    user = getattr(g, "user", None)
    if not user:
        return None, (jsonify({"error": "Authentication required. No token provided."}), 401)
    doc = Document.query.get_or_404(doc_id)
    if doc.user_id != user.id and getattr(user, "role", "") != "admin":
        return None, (jsonify({"error": "Unauthorized access to document analysis."}), 403)
    return doc, None

@analysis_bp.route("/documents/<uuid:doc_id>/clauses", methods=["GET"])
@require_auth
def get_document_clauses(doc_id):
    doc, err = get_authorized_doc_or_403(doc_id)
    if err:
        return err
    clauses = Clause.query.filter_by(document_id=doc_id).order_by(Clause.clause_index.asc()).all()
    substantive_only = request.args.get("substantive_only", "false").lower() == "true"
    if substantive_only:
        clauses = [c for c in clauses if c.is_substantive]
    return jsonify([c.to_dict() for c in clauses]), 200

@analysis_bp.route("/documents/<uuid:doc_id>/exclusions-summary", methods=["GET"])
@require_auth
def get_document_exclusions_summary(doc_id):
    doc, err = get_authorized_doc_or_403(doc_id)
    if err:
        return err
    clauses = Clause.query.filter_by(document_id=doc_id).order_by(Clause.clause_index.asc()).all()
    substantive = [c for c in clauses if c.is_substantive]
    excluded = [c for c in clauses if not c.is_substantive]

    breakdown = {}
    for c in excluded:
        ct = c.content_type or "OTHER_NON_SUBSTANTIVE"
        breakdown[ct] = breakdown.get(ct, 0) + 1

    structural_count = sum(v for k, v in breakdown.items() if k not in ("EXTRACTION_DUPLICATE", "OCR_NOISE"))

    return jsonify({
        "document_id": str(doc_id),
        "total_blocks": len(clauses),
        "substantive_count": len(substantive),
        "excluded_count": len(excluded),
        "structural_metadata_count": structural_count,
        "insufficient_substantive_content": len(substantive) == 0,
        "breakdown": breakdown,
    }), 200

@analysis_bp.route("/documents/<uuid:doc_id>/missing-clauses", methods=["GET"])
@require_auth
def get_missing_clauses(doc_id):
    doc, err = get_authorized_doc_or_403(doc_id)
    if err:
        return err
    missing = MissingClause.query.filter_by(document_id=doc_id).all()
    return jsonify([m.to_dict() for m in missing]), 200

@analysis_bp.route("/documents/<uuid:doc_id>/deadlines", methods=["GET"])
@require_auth
def get_deadlines(doc_id):
    doc, err = get_authorized_doc_or_403(doc_id)
    if err:
        return err
    deadlines = Deadline.query.filter_by(document_id=doc_id).order_by(
        Deadline.relative_days.asc().nulls_last(),
        Deadline.parsed_date.asc().nulls_last()
    ).all()
    return jsonify([d.to_dict() for d in deadlines]), 200

@analysis_bp.route("/documents/<uuid:doc_id>/pii-summary", methods=["GET"])
@require_auth
def get_pii_summary(doc_id):
    doc, err = get_authorized_doc_or_403(doc_id)
    if err:
        return err

    findings = PIIFinding.query.filter_by(document_id=doc_id).order_by(PIIFinding.start_offset.asc()).all()
    clauses = Clause.query.filter_by(document_id=doc_id).order_by(Clause.clause_index.asc()).all()

    redacted_text = doc.redacted_text or ""

    # Phase 2 Offset Compatibility Verification:
    # Verify whether persisted findings align exactly with neutral pseudonym slices in redacted_text
    offsets_compatible = True
    if findings:
        for f in findings:
            if f.start_offset is None or f.end_offset is None:
                offsets_compatible = False
                break
            if not (0 <= f.start_offset < f.end_offset <= len(redacted_text)):
                offsets_compatible = False
                break
            target_slice = redacted_text[f.start_offset:f.end_offset]
            if not (target_slice.startswith("[") and target_slice.endswith("]")):
                offsets_compatible = False
                break

    # If historical offsets are incompatible, attempt safe recomputation from encrypted raw text
    if not offsets_compatible and doc.raw_text_encrypted:
        try:
            from app.utils.encryption import decrypt_text
            from app.services.privacy.pii_redactor import redact
            raw_text = decrypt_text(doc.raw_text_encrypted)
            if raw_text and raw_text.strip():
                clean_redacted_text, new_findings = redact(raw_text)
                doc.redacted_text = clean_redacted_text
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
                # Reload updated findings and text
                findings = PIIFinding.query.filter_by(document_id=doc_id).order_by(PIIFinding.start_offset.asc()).all()
                redacted_text = doc.redacted_text
                offsets_compatible = True
                current_app.logger.info(f"Safely recomputed historical PII findings for document {doc_id}")
        except Exception as e:
            current_app.logger.warning(f"Could not recompute historical PII for document {doc_id}: {e}")

    counts = {}
    label_map = {}
    details = []

    for f in findings:
        counts[f.entity_type] = counts.get(f.entity_type, 0) + 1

        if offsets_compatible:
            # Verified Phase 2 offset alignment
            label = None
            if f.start_offset is not None and f.end_offset is not None:
                if 0 <= f.start_offset < f.end_offset <= len(redacted_text):
                    candidate = redacted_text[f.start_offset:f.end_offset]
                    if candidate.startswith("[") and candidate.endswith("]"):
                        label = candidate
            if not label:
                label = f"[{f.entity_type.replace('_', ' ').title()}]"

            clause_refs = []
            for c in clauses:
                if c.start_offset is not None and c.end_offset is not None:
                    if not (f.end_offset <= c.start_offset or f.start_offset >= c.end_offset):
                        clause_refs.append(f"Clause {c.clause_index + 1}")
            if not clause_refs:
                clause_refs = ["General Text"]

            clause_ref_str = ", ".join(clause_refs)
            review_status = "Auto-redacted" if (f.confidence or 0) >= 0.75 else "Needs Review"
            start_off = f.start_offset
            end_off = f.end_offset
            offsets_valid = True
        else:
            # Invalidation Strategy for historical records:
            # Do NOT silently reinterpret old raw-text offsets against redacted_text
            label = f"[{f.entity_type.replace('_', ' ').title()}]"
            clause_refs = ["Legacy Format - Offsets Invalidated"]
            clause_ref_str = "Legacy Format - Offsets Invalidated"
            review_status = "Needs Re-analysis"
            start_off = None
            end_off = None
            offsets_valid = False

        details.append({
            "id": str(f.id),
            "document_id": str(doc_id),
            "entity_type": f.entity_type,
            "replacement_label": label,
            "start_offset": start_off,
            "end_offset": end_off,
            "offsets_valid": offsets_valid,
            "confidence": f.confidence,
            "clause_references": clause_refs,
            "clause_reference": clause_ref_str,
            "review_status": review_status,
        })

        if label not in label_map:
            label_map[label] = {
                "replacement_label": label,
                "entity_type": f.entity_type,
                "occurrence_count": 0,
                "clause_references": set(),
                "review_status": review_status,
            }
        label_map[label]["occurrence_count"] += 1
        label_map[label]["clause_references"].update(clause_refs)
        if review_status in ("Needs Review", "Needs Re-analysis"):
            label_map[label]["review_status"] = review_status

    mapping_table = []
    for label, item in label_map.items():
        mapping_table.append({
            "replacement_label": item["replacement_label"],
            "entity_type": item["entity_type"],
            "occurrence_count": item["occurrence_count"],
            "clause_references": sorted(list(item["clause_references"])),
            "review_status": item["review_status"],
        })

    return jsonify({
        "document_id": str(doc_id),
        "total_redactions": len(findings),
        "total_findings": len(findings),
        "entity_counts": counts,
        "mapping_table": mapping_table,
        "findings": details,
        "offset_compatibility": "compatible" if offsets_compatible else "invalidated_legacy",
        "historical_offsets_invalidated": not offsets_compatible,
    }), 200

@analysis_bp.route("/documents/<uuid:doc_id>/integrity-findings", methods=["GET"])
@require_auth
def get_integrity_findings(doc_id):
    doc, err = get_authorized_doc_or_403(doc_id)
    if err:
        return err

    from app.models.integrity_finding import IntegrityFinding
    findings = IntegrityFinding.query.filter_by(document_id=doc_id).order_by(IntegrityFinding.created_at.asc()).all()

    category_counts = {}
    severity_counts = {}
    for f in findings:
        category_counts[f.category] = category_counts.get(f.category, 0) + 1
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

    return jsonify({
        "document_id": str(doc_id),
        "total_findings": len(findings),
        "has_integrity_signals": len(findings) > 0,
        "category_counts": category_counts,
        "severity_counts": severity_counts,
        "findings": [f.to_dict() for f in findings],
    }), 200

@analysis_bp.route("/documents/<uuid:doc_id>/report.pdf", methods=["GET"])
@require_auth
def download_pdf_report(doc_id):
    doc, err = get_authorized_doc_or_403(doc_id)
    if err:
        return err
    clauses = Clause.query.filter_by(document_id=doc_id).order_by(Clause.clause_index.asc()).all()
    missing = MissingClause.query.filter_by(document_id=doc_id).all()
    deadlines = Deadline.query.filter_by(document_id=doc_id).order_by(
        Deadline.relative_days.asc().nulls_last()
    ).all()
    findings = PIIFinding.query.filter_by(document_id=doc_id).all()

    from app.models.integrity_finding import IntegrityFinding
    integrity_findings = IntegrityFinding.query.filter_by(document_id=doc_id).order_by(IntegrityFinding.created_at.asc()).all()

    pdf_bytes = generate_pdf_report(doc, clauses, missing, deadlines, findings, integrity_findings=integrity_findings)

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"clauseguard_report_{doc.filename.rsplit('.', 1)[0]}.pdf",
    )

