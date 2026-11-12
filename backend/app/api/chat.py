import uuid
import logging
from flask import Blueprint, request, jsonify, g
from app.extensions import db
from app.models.document import Document
from app.models.chat import ChatSession, ChatMessage
from app.models.user import User
from app.middleware.auth import require_auth, optional_auth
from app.utils.gemini_chat import ask_gemini_or_fallback
from app.services.rag.chat_engine import answer_document_question, answer_account_question

logger = logging.getLogger(__name__)
chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["POST"])
@require_auth
def post_chat_message():
    """
    POST /api/chat
    Authenticates user, verifies document ownership if documentId is provided,
    invokes Gemini LLM with grounded prompt / deterministic heuristic fallback,
    and records chat history with citations and grounding status.
    """
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    document_id = data.get("documentId")
    scope = (data.get("scope") or "").strip().lower()

    if not question:
        return jsonify({"error": "Question is required and cannot be empty"}), 400

    user = g.user

    if document_id:
        try:
            doc_uuid = uuid.UUID(str(document_id))
            doc = db.session.get(Document, doc_uuid)
            if not doc:
                return jsonify({"error": "Document not found."}), 404
            if doc.user_id != user.id and getattr(user, "role", "") != "admin":
                return jsonify({"error": "Unauthorized access to this document."}), 403

            result = answer_document_question(
                document_id=doc_uuid,
                question=question,
                session_id=None,
                user_id=user.id,
            )

            from app.services.audit_service import log_audit_event
            log_audit_event(
                action="CHAT_QUERY",
                user_id=user.id,
                user_email=user.email,
                resource_type="chat_message",
                resource_id=result.get("id"),
                details={
                    "question": question[:120],
                    "documentId": str(doc_uuid),
                    "grounded": result.get("grounded", True),
                    "confidence": result.get("confidence", 0.94)
                }
            )
            return jsonify(result), 200
        except ValueError:
            return jsonify({"error": f"Invalid documentId format: {document_id}"}), 400
        except Exception as e:
            logger.warning(f"Could not process document chat for {document_id}: {e}")
            return jsonify({"error": "Failed to load document context."}), 500

    # Account-wide multi-document RAG (default when documentId is omitted unless scope='general')
    if scope != "general":
        try:
            result = answer_account_question(user_id=user.id, question=question)
            from app.services.audit_service import log_audit_event
            log_audit_event(
                action="CHAT_QUERY",
                user_id=user.id,
                user_email=user.email,
                resource_type="chat_message",
                resource_id=result.get("id"),
                details={
                    "question": question[:120],
                    "documentId": None,
                    "scope": "account",
                    "grounded": result.get("grounded", False),
                    "confidence": result.get("confidence"),
                }
            )
            return jsonify(result), 200
        except Exception as acc_err:
            logger.warning(f"Account-wide chat failed for user {user.id}: {acc_err}")
            return jsonify({"error": "Failed to process account-wide question."}), 500

    # Pure General Mode when explicitly requested (scope='general')
    recent_history = []
    try:
        prior_msgs = (
            ChatMessage.query.filter_by(user_id=user.id, document_id=None)
            .order_by(ChatMessage.created_at.desc())
            .limit(6)
            .all()
        )
        recent_history = [{"role": m.role, "content": m.content} for m in reversed(prior_msgs)]
    except Exception:
        recent_history = []

    ai_result = ask_gemini_or_fallback(question, "", history=recent_history)

    user_msg_id = uuid.uuid4()
    user_msg = ChatMessage(
        id=user_msg_id,
        document_id=None,
        user_id=user.id,
        role="USER",
        content=question,
        created_at=None,
    )
    db.session.add(user_msg)

    assistant_msg_id = uuid.uuid4()
    asst_msg = ChatMessage(
        id=assistant_msg_id,
        document_id=None,
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
            "documentId": None,
            "scope": "general",
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
@require_auth
def get_conversation_history():
    """
    GET /api/chat/history?documentId=...
    Retrieves chronological conversation history strictly scoped to either documentId
    or the standalone account-wide thread (document_id IS NULL).
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
        # Strictly isolate standalone/account-wide thread from individual document threads
        query = query.filter(ChatMessage.document_id.is_(None))

    messages = query.order_by(ChatMessage.created_at.asc()).limit(100).all()
    return jsonify({
        "messages": [m.to_dict() for m in messages]
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

    from app.services.audit_service import log_audit_event
    log_audit_event(
        action="CHAT_QUERY",
        user_id=user.id,
        user_email=user.email,
        resource_type="chat_message",
        resource_id=result.get("id"),
        details={
            "question": question[:120],
            "documentId": str(doc_id),
            "grounded": result.get("grounded", True),
            "confidence": result.get("confidence", 0.94)
        }
    )
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

