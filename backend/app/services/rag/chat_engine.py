import re
import os
import logging
from flask import current_app
from app.extensions import db
from app.models.chat import ChatSession, ChatMessage
from app.services.rag.embedder import embed_query, get_genai_client
from app.services.rag.retriever import retrieve_relevant_clauses
from app.utils.gemini_chat import GEMINI_CANDIDATE_MODELS, local_heuristic_search
from app.utils.validators import sanitize_chat_input

logger = logging.getLogger(__name__)

_STALE_MODELS = {"gemini-3.8-flash", "gemini-3.7-flash", "gemini-flash-latest"}

SYSTEM_INSTRUCTION = """You are ClauseGuard AI's document analysis assistant.
Answer the user's question STRICTLY AND SOLELY using the retrieved document clauses provided below.

CRITICAL GROUNDING RULES:
1. ONLY use facts and terms explicitly stated in the provided clauses. Never hallucinate, invent details, or assume terms not in the text.
2. ALWAYS cite the specific clause you rely on using the format [Clause X] (e.g., "[Clause 2]").
3. If the retrieved clauses DO NOT contain the answer, you MUST say plainly:
   "The provided document does not contain information addressing this question."
   Do NOT guess, hypothesize, or introduce external legal principles.
4. Keep answers factual, concise, and objective.
5. Formatting & Structure: Respond using clean Markdown structure suitable for rendering in a professional chatbot. Use headings, paragraphs, bullet lists, numbered lists, bold emphasis, tables, and blockquotes when appropriate. Do not use decorative Markdown, excessive symbols, or unnecessary formatting. Never place asterisks around entire paragraphs. Use bold only for important labels or short phrases. Keep responses concise, structured, and readable.
6. Reminder: ClauseGuard AI provides automated, non-expert analysis for informational purposes only. It is not legal advice.
"""


def answer_document_question(document_id, question: str, session_id=None, user_id=None) -> dict:
    """
    RAG pipeline:
    1. Scan question for prompt injection.
    2. Validate chat session ownership against document_id.
    3. Embed question with Gemini embedding model & retrieve top clauses via pgvector.
    4. Generate grounded answer via fast Gemini multi-model cascade (with local fallback).
    5. Track citations and verify grounding.
    """
    sanitized_question, was_flagged, patterns = sanitize_chat_input(question)
    if was_flagged:
        logger.warning(f"Sanitized injection patterns {patterns} in user question")

    if session_id:
        session = db.session.get(ChatSession, session_id)
        if session and str(session.document_id) != str(document_id):
            raise ValueError("Chat session does not belong to the specified document.")
    else:
        session = None

    if not session:
        session = ChatSession(document_id=document_id)
        db.session.add(session)
        db.session.commit()

    user_msg = ChatMessage(
        session_id=session.id,
        document_id=document_id,
        user_id=user_id,
        role="user",
        content=question,
        retrieved_clause_ids=[],
        grounded=True,
    )
    db.session.add(user_msg)
    db.session.commit()

    query_vector = embed_query(sanitized_question)
    retrieved_clauses = retrieve_relevant_clauses(document_id, query_vector, top_k=6)

    if not retrieved_clauses:
        fallback_answer = "No clauses found for this document to answer your question."
        asst_msg = ChatMessage(
            session_id=session.id,
            document_id=document_id,
            user_id=user_id,
            role="assistant",
            content=fallback_answer,
            retrieved_clause_ids=[],
            grounded=False,
            sources=[],
        )
        db.session.add(asst_msg)
        db.session.commit()
        return {
            "session_id": str(session.id),
            "message_id": str(asst_msg.id),
            "answer": fallback_answer,
            "cited_clause_ids": [],
            "grounded": False,
            "sources": [],
            "retrieved_clauses": [],
        }

    context_blocks = []
    index_to_id = {}
    index_to_clause = {}
    for c in retrieved_clauses:
        idx = c["clause_index"]
        c_id = c["clause_id"]
        index_to_id[idx] = c_id
        index_to_clause[idx] = c
        context_blocks.append(
            f"[Clause {idx}] (Type: {c['clause_type']}, Risk: {c['risk_score']}):\n{c['redacted_text']}"
        )

    context_str = "\n\n".join(context_blocks)
    full_prompt = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"--- RETRIEVED CLAUSES FROM THIS DOCUMENT ---\n"
        f"{context_str}\n\n"
        f"--- USER QUESTION ---\n"
        f"{sanitized_question}\n\n"
        f"ANSWER (cite [Clause X] or declare ungrounded):"
    )

    client = get_genai_client()
    preferred = (
        os.environ.get("GEMINI_CHAT_MODEL")
        or (current_app.config.get("GEMINI_CHAT_MODEL") if current_app else None)
    )
    candidate_models = []
    for m in ([preferred] if preferred and preferred not in _STALE_MODELS else []) + list(GEMINI_CANDIDATE_MODELS):
        if m and m not in candidate_models and m not in _STALE_MODELS:
            candidate_models.append(m)

    answer_text = ""
    if client:
        for chat_model in candidate_models:
            try:
                response = client.models.generate_content(
                    model=chat_model,
                    contents=full_prompt,
                )
                if response and getattr(response, "text", None) and response.text.strip():
                    answer_text = response.text.strip()
                    break
            except Exception as e:
                logger.warning(f"Gemini model {chat_model} failed for document {document_id}: {e}")
                continue

    if not answer_text:
        answer_text = local_heuristic_search(sanitized_question, context_str)["answer"]

    ungrounded_phrases = [
        "does not contain information",
        "no information addressing",
        "not mentioned in the provided",
        "cannot answer",
        "not addressed in the clauses",
    ]
    is_unsupported = any(phrase in answer_text.lower() for phrase in ungrounded_phrases)

    cited_indices = re.findall(r"\[Clause\s+(\d+)(?:[^\]]*)\]", answer_text, re.IGNORECASE)
    cited_clause_ids = []
    sources = []
    seen_indices = set()
    for idx_str in cited_indices:
        idx_int = int(idx_str)
        if idx_int in index_to_id and idx_int not in seen_indices:
            seen_indices.add(idx_int)
            cited_clause_ids.append(index_to_id[idx_int])
            c_obj = index_to_clause[idx_int]
            c_type = (c_obj.get("clause_type") or "Clause").replace("_", " ").title()
            sources.append({
                "section": f"Clause #{idx_int + 1} ({c_type})",
                "clause_id": index_to_id[idx_int],
                "clause_index": idx_int,
                "page_number": c_obj.get("page_number", 1),
                "excerpt": (c_obj.get("redacted_text") or "")[:200],
            })

    grounded = (not is_unsupported) and (len(cited_clause_ids) > 0)
    if not cited_clause_ids and not is_unsupported:
        grounded = False

    asst_msg = ChatMessage(
        session_id=session.id,
        document_id=document_id,
        user_id=user_id,
        role="assistant",
        content=answer_text,
        retrieved_clause_ids=[c["clause_id"] for c in retrieved_clauses],
        grounded=grounded,
        sources=sources if grounded else [],
    )
    db.session.add(asst_msg)
    db.session.commit()

    return {
        "session_id": str(session.id),
        "message_id": str(asst_msg.id),
        "answer": answer_text,
        "cited_clause_ids": cited_clause_ids if grounded else [],
        "grounded": grounded,
        "sources": sources if grounded else [],
        "retrieved_clauses": retrieved_clauses,
    }
