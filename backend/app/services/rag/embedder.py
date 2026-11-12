import os
import logging
from typing import Optional
from google import genai
from flask import current_app

logger = logging.getLogger(__name__)

_genai_client = None

def get_genai_client():
    global _genai_client
    if _genai_client is None:
        api_key = current_app.config.get("GEMINI_API_KEY") if current_app else os.environ.get("GEMINI_API_KEY")
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY is not configured for embeddings.")
            return None
        try:
            _genai_client = genai.Client(api_key=api_key, http_options={"timeout": 15000})
        except Exception as e:
            logger.warning(f"Failed to initialize GenAI client: {e}")
            _genai_client = None
    return _genai_client

def get_embed_model_name() -> str:
    if current_app:
        return current_app.config.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    return os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")

def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> Optional[list[float]]:
    """
    Generates a 768-dimensional embedding vector using Gemini API via google-genai SDK.
    Returns None on failure so pgvector never stores all-zero vectors.
    """
    if not text or not text.strip():
        return None

    client = get_genai_client()
    if not client:
        return None

    model = get_embed_model_name()
    try:
        result = client.models.embed_content(
            model=model,
            contents=[text[:2000]],
            config={
                "output_dimensionality": 768,
                "task_type": task_type,
            },
        )
        if result.embeddings and len(result.embeddings) > 0:
            vals = result.embeddings[0].values
            if vals and len(vals) == 768 and any(v != 0.0 for v in vals):
                return vals
    except Exception as e:
        logger.warning(f"Error generating embedding with model {model}: {e}")

    return None

def embed_query(query_text: str) -> Optional[list[float]]:
    """
    Convenience method to embed a search query.
    """
    return embed_text(query_text, task_type="RETRIEVAL_QUERY")

def batch_embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT", batch_size: int = 50) -> list[Optional[list[float]]]:
    """
    High-performance batch embedder: sends chunks of up to batch_size texts per API call.
    Returns None for failed items so pgvector never stores NaN-inducing zero vectors.
    """
    if not texts:
        return []

    client = get_genai_client()
    if not client:
        return [None for _ in texts]

    model = get_embed_model_name()
    all_embeddings: list[Optional[list[float]]] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        cleaned_chunk = [t[:2000] if t and t.strip() else "empty clause" for t in chunk]
        try:
            result = client.models.embed_content(
                model=model,
                contents=cleaned_chunk,
                config={
                    "output_dimensionality": 768,
                    "task_type": task_type,
                },
            )
            if result.embeddings and len(result.embeddings) == len(chunk):
                for idx, emb in enumerate(result.embeddings):
                    if not chunk[idx] or not chunk[idx].strip():
                        all_embeddings.append(None)
                    else:
                        vals = emb.values
                        all_embeddings.append(vals if (vals and len(vals) == 768 and any(v != 0.0 for v in vals)) else None)
            else:
                all_embeddings.extend([None for _ in chunk])
        except Exception as e:
            logger.warning(f"Batch embedding failed for chunk {i}: {e}.")
            all_embeddings.extend([None for _ in chunk])

    return all_embeddings

def embed_document_clauses(clauses: list[dict]) -> list[list[float]]:
    """
    Embeds multiple clauses efficiently in batch mode.
    """
    texts = [c.get("redacted_text") or c.get("text", "") for c in clauses]
    return batch_embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
