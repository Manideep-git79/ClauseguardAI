import uuid
import logging
from flask import Blueprint, request, jsonify, g
from app.extensions import db
from app.models.document import Document
from app.models.chat import ChatSession, ChatMessage
from app.models.user import User
from app.middleware.auth import require_auth, optional_auth
from app.utils.gemini_chat import ask_gemini_or_fallback
from app.services.rag.chat_engine import answer_document_question

logger = logging.getLogger(__name__)
chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["POST"])
@optional_auth
def post_chat_message():
    """
    POST /api/chat
    Authenticates user (or creates guest user context if unauthenticated),
    retrieves document context if documentId is provided,
    invokes Gemini LLM with grounded prompt / deterministic heuristic fallback,
    records chat history with citations and grounding status.
    """
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    document_id = data.get("documentId")

    if not question:
        return jsonify({"error": "Question is required and cannot be empty"}), 400

    user = getattr(g, "user", None)
    if not user:
        guest_email = "guest@clauseguard.ai"
        user = User.query.filter_by(email=guest_email).first()
        if not user:
            from app.utils.password_policy import hash_password
            user = User(
                name="Guest User",
                email=guest_email,
                password_hash=hash_password("GuestPass123!"),
                role="guest",
            )
            db.session.add(user)
            db.session.commit()

    document_text = ""
    doc_uuid = None
    if document_id:
        try:
            doc_uuid = uuid.UUID(str(document_id))
            doc = db.session.get(Document, doc_uuid)
            if doc:
                if doc.user_id and (not user or doc.user_id != user.id):
                    return jsonify({"error": "Unauthorized access to this document."}), 403
                document_text = doc.redacted_text or ""
        except Exception as e:
            logger.warning(f"Could not parse document_id {document_id}: {e}")

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
            "confidence": ai_result.get("confidence", 0.94)
        }
    )

    return jsonify({
        "id": str(assistant_msg_id),
        "answer": ai_result["answer"],
        "grounded": ai_result.get("grounded", True),
        "confidence": ai_result.get("confidence", 0.94),
        "sources": ai_result.get("sources", []),
    }), 200

@chat_bp.route("/chat/history", methods=["GET"])
@optional_auth
def get_conversation_history():
    """
    GET /api/chat/history?documentId=...
    Retrieves chronological conversation history for user and/or document.
    """
    document_id = request.args.get("documentId")
    user = getattr(g, "user", None)
    if not user:
        return jsonify({"messages": []}), 200

    query = ChatMessage.query.filter_by(user_id=user.id)
    
    if document_id:
        try:
            doc_uuid = uuid.UUID(str(document_id))
            query = query.filter_by(document_id=doc_uuid)
        except Exception:
            pass

    messages = query.order_by(ChatMessage.created_at.asc()).limit(100).all()
    return jsonify({
        "messages": [m.to_dict() for m in messages]
    }), 200

@chat_bp.route("/documents/<uuid:doc_id>/chat", methods=["POST"])
@optional_auth
def chat_with_document_legacy(doc_id):
    doc = Document.query.get_or_404(doc_id)
    user = getattr(g, "user", None)
    if doc.user_id and (not user or doc.user_id != user.id):
        return jsonify({"error": "Unauthorized access to document chat."}), 403

    data = request.get_json() or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question field cannot be empty."}), 400

    session_id = data.get("session_id")
    result = answer_document_question(
        document_id=doc_id,
        question=question,
        session_id=session_id,
    )
    return jsonify(result), 200

@chat_bp.route("/documents/<uuid:doc_id>/chat/history", methods=["GET"])
@optional_auth
def get_chat_history_legacy(doc_id):
    doc = Document.query.get_or_404(doc_id)
    user = getattr(g, "user", None)
    if doc.user_id and (not user or doc.user_id != user.id):
        return jsonify({"error": "Unauthorized access to document chat history."}), 403

    sessions = (
        ChatSession.query.filter_by(document_id=doc_id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    return jsonify([s.to_dict() for s in sessions]), 200
