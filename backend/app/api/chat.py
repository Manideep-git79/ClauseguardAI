import uuid
import logging
from flask import Blueprint, request, jsonify, g
from app.extensions import db
from app.models.document import Document
from app.models.chat import ChatSession, ChatMessage
from app.middleware.auth import require_auth
from app.utils.gemini_chat import ask_gemini_or_fallback
from app.services.rag.chat_engine import answer_document_question
from app.services.rag.embedder import embed_query
from app.services.rag.retriever import retrieve_relevant_clauses

logger = logging.getLogger(__name__)
chat_bp = Blueprint("chat", __name__)


def _build_document_context(doc: Document, question: str) -> str:
    """
    Build grounded document context combining pgvector-retrieved substantive clauses
    (covering long multi-page documents) with the redacted full-text excerpt.
    """
    raw_text = (doc.redacted_text or "").strip()
    rag_sections = []
    try:
        q_vec = embed_query(question)
        hits = retrieve_relevant_clauses(doc.id, q_vec, top_k=6)
        for h in hits:
            txt = (h.get("redacted_text") or "").strip()
            if txt:
                rag_sections.append(f"[Clause {h['clause_index'] + 1}]: {txt}")
    except Exception as rag_err:
        logger.debug(f"Vector retrieval fallback to redacted_text for {doc.id}: {rag_err}")

    if rag_sections and len(raw_text) > 10000:
        return "\n".join(rag_sections) + "\n\n---\n" + raw_text[:10000]
    if raw_text:
        return raw_text[:14000]
    return "\n".join(rag_sections)


@chat_bp.route("/chat", methods=["POST"])
@require_auth
def post_chat_message():
    """
    POST /api/chat
    Authenticates user, verifies document ownership if documentId is provided,
    invokes Gemini Legal Copilot with grounded document context / general legal fallback,
    and records chat history with citations and grounding status.
    """
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    document_id = data.get("documentId")

    if not question:
        return jsonify({"error": "Question is required and cannot be empty"}), 400

    user = g.user

    document_text = ""
    doc_uuid = None
    if document_id:
        try:
            doc_uuid = uuid.UUID(str(document_id))
        except Exception:
            return jsonify({"error": f"Invalid documentId format: {document_id}"}), 400

        doc = db.session.get(Document, doc_uuid)
        if not doc:
            return jsonify({"error": "Document not found."}), 404
        if doc.user_id != user.id and getattr(user, "role", "") != "admin":
            return jsonify({"error": "Unauthorized access to this document."}), 403
        document_text = _build_document_context(doc, question)

    ai_result = ask_gemini_or_fallback(question, document_text)

    user_msg_id = uuid.uuid4()
    user_msg = ChatMessage(
        id=user_msg_id,
        document_id=doc_uuid,
        user_id=user.id,
        role="USER",
        content=question,
        created_at=None,
    )
    db.session.add(user_msg)

    assistant_msg_id = uuid.uuid4()
    asst_msg = ChatMessage(
        id=assistant_msg_id,
        document_id=doc_uuid,
        user_id=user.id,
        role="ASSISTANT",
        content=ai_result["answer"],
        confidence=ai_result.get("confidence", 0.94),
        grounded=ai_result.get("grounded", True),
        sources=ai_result.get("sources", []),
    )
    db.session.add(asst_msg)
    db.session.commit()

    from app.services.audit_service import log_audit_event
    log_audit_event(
        action="CHAT_QUERY",
        user_id=user.id,
        user_email=user.email,
        resource_type="chat_message",
        resource_id=str(assistant_msg_id),
        details={
            "question": question[:120],
            "documentId": str(doc_uuid) if doc_uuid else None,
            "grounded": ai_result.get("grounded", True),
            "confidence": ai_result.get("confidence", 0.94),
        },
    )

    return jsonify({
        "id": str(assistant_msg_id),
        "answer": ai_result["answer"],
        "grounded": ai_result.get("grounded", True),
        "confidence": ai_result.get("confidence", 0.94),
        "sources": ai_result.get("sources", []),
    }), 200


@chat_bp.route("/chat/history", methods=["GET"])
@require_auth
def get_conversation_history():
    """
    GET /api/chat/history?documentId=...
    Retrieves chronological conversation history strictly scoped to documentId
    or the standalone general copilot thread (document_id IS NULL).
    """
    document_id = request.args.get("documentId")
    user = g.user

    query = ChatMessage.query.filter_by(user_id=user.id)

    if document_id and str(document_id).strip():
        try:
            doc_uuid = uuid.UUID(str(document_id).strip())
        except Exception:
            return jsonify({"error": f"Invalid documentId format: {document_id}"}), 400

        doc = db.session.get(Document, doc_uuid)
        if not doc:
            return jsonify({"error": "Document not found."}), 404
        if doc.user_id != user.id and getattr(user, "role", "") != "admin":
            return jsonify({"error": "Unauthorized access to document chat history."}), 403
        query = query.filter_by(document_id=doc_uuid)
    else:
        query = query.filter(ChatMessage.document_id.is_(None))

    messages = query.order_by(ChatMessage.created_at.asc()).limit(100).all()
    return jsonify({
        "messages": [m.to_dict() for m in messages],
    }), 200


@chat_bp.route("/documents/<uuid:doc_id>/chat", methods=["POST"])
@require_auth
def chat_with_document_legacy(doc_id):
    doc = Document.query.get_or_404(doc_id)
    user = g.user
    if doc.user_id != user.id and getattr(user, "role", "") != "admin":
        return jsonify({"error": "Unauthorized access to document chat."}), 403

    data = request.get_json() or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question field cannot be empty."}), 400

    session_id = data.get("session_id")
    try:
        result = answer_document_question(
            document_id=doc_id,
            question=question,
            session_id=session_id,
            user_id=user.id,
        )
    except ValueError as sess_err:
        return jsonify({"error": str(sess_err)}), 403

    return jsonify(result), 200


@chat_bp.route("/documents/<uuid:doc_id>/chat/history", methods=["GET"])
@require_auth
def get_chat_history_legacy(doc_id):
    doc = Document.query.get_or_404(doc_id)
    user = g.user
    if doc.user_id != user.id and getattr(user, "role", "") != "admin":
        return jsonify({"error": "Unauthorized access to document chat history."}), 403

    sessions = (
        ChatSession.query.filter_by(document_id=doc_id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    return jsonify([s.to_dict() for s in sessions]), 200
