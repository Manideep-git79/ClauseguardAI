import os
import math
import time
import random
import logging
from typing import Optional, List, Any
from google import genai
from flask import current_app

logger = logging.getLogger(__name__)

_genai_client = None


def get_genai_client():
    global _genai_client
    if _genai_client is None:
        api_key = (
            (current_app.config.get("GEMINI_API_KEY") if current_app else None)
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            logger.warning("GEMINI_API_KEY is not configured for embeddings.")
            return None
        try:
            _genai_client = genai.Client(api_key=api_key, http_options={"timeout": 15000})
        except Exception as e:
            logger.warning(f"Failed to initialize GenAI client: {type(e).__name__}")
            _genai_client = None
    return _genai_client


def get_embed_model_name() -> str:
    return (
        (current_app.config.get("GEMINI_EMBED_MODEL") if current_app else None)
        or os.environ.get("GEMINI_EMBED_MODEL")
        or "gemini-embedding-001"
    )


def validate_embedding_vector(vector: Any, expected_dim: int = 768) -> bool:
    """
    Validates that an embedding vector is a list of exact expected dimension,
    with all values finite numeric floats (rejecting NaN, +Inf, -Inf).
    """
    if not isinstance(vector, (list, tuple)):
        return False
    if len(vector) != expected_dim:
        return False
    for val in vector:
        if not isinstance(val, (int, float)):
            return False
        if not math.isfinite(val):
            return False
    return True


def is_transient_error(e: Exception) -> bool:
    """
    Determines if an embedding API error is transient and eligible for retry.
    Transient errors include:
    - HTTP 429 (Rate Limit / Quota Exceeded / ResourceExhausted)
    - HTTP 500, 502, 503, 504 (Server error, Gateway timeout, Unavailable)
    - Network/socket timeouts and connection resets
    Non-transient errors (e.g. 400 Bad Request, 401/403 Auth/Permission, ValueError) are not retried.
    """
    if isinstance(e, (TimeoutError, ConnectionError)):
        return True

    # Check for HTTP status code on exception if present
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if isinstance(code, int) and (code == 429 or code in (500, 502, 503, 504)):
        return True

    err_msg = str(e).lower()
    err_type = type(e).__name__.lower()
    transient_indicators = (
        "timeout", "timed out", "deadline", "rate", "quota", "resource_exhausted",
        "429", "503", "504", "temporarily unavailable", "connection reset", "broken pipe"
    )
    if any(ind in err_msg or ind in err_type for ind in transient_indicators):
        # Ensure it is not an authentication or client error masked with similar text
        if any(term in err_msg for term in ("400", "bad request", "401", "403", "unauthorized", "permission denied")):
            return False
        return True

    return False


def embed_text(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 4.0,
) -> Optional[List[float]]:
    """
    Generates a 768-dimensional embedding vector using Gemini API via google-genai SDK.

    Returns:
        List[float] (768 finite floats) if successful.
        None if embedding generation fails, times out, is rate-limited, or receives invalid vector.
        NEVER silently converts an error into a fake zero-vector [0.0] * 768.
    """
    if not text or not text.strip():
        logger.debug("embed_text received empty text; returning None.")
        return None

    client = get_genai_client()
    if not client:
        logger.warning("GenAI client unavailable; cannot generate embeddings.")
        return None

    model = get_embed_model_name()
    sanitized_len = len(text)

    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=model,
                contents=[text[:2000]],
                config={
                    "output_dimensionality": 768,
                    "task_type": task_type,
                },
            )
            if not result or not getattr(result, "embeddings", None) or len(result.embeddings) == 0:
                logger.warning(
                    f"Empty embedding response from model={model}, text_length={sanitized_len} (attempt {attempt+1}/{max_retries})."
                )
                if attempt < max_retries - 1:
                    continue
                return None

            raw_vec = result.embeddings[0].values
            if not validate_embedding_vector(raw_vec, expected_dim=768):
                dim = len(raw_vec) if isinstance(raw_vec, (list, tuple)) else "not_list"
                has_finite = all(math.isfinite(x) for x in raw_vec) if isinstance(raw_vec, (list, tuple)) else False
                logger.error(
                    f"Invalid embedding vector returned from model={model}: dim={dim}, all_finite={has_finite}."
                )
                return None

            return raw_vec

        except Exception as e:
            is_transient = is_transient_error(e)
            logger.warning(
                f"Embedding error (attempt {attempt+1}/{max_retries}): "
                f"error_type={type(e).__name__}, transient={is_transient}, "
                f"model={model}, text_length={sanitized_len}."
            )
            if not is_transient or attempt >= max_retries - 1:
                logger.error(
                    f"Exhausted retries or non-transient error for model={model}: {type(e).__name__}."
                )
                return None

            # Exponential backoff with jitter
            raw_delay = initial_delay * (backoff_factor ** attempt)
            delay = min(max_delay, raw_delay) + random.uniform(0.0, 0.25 * raw_delay)
            time.sleep(delay)

    return None


def embed_query(query_text: str, max_retries: int = 3) -> Optional[List[float]]:
    """
    Convenience method to embed a search query.
    Returns Optional[List[float]]: 768-dim float vector on success, None on failure.
    """
    return embed_text(query_text, task_type="RETRIEVAL_QUERY", max_retries=max_retries)


def _split_text_with_overlap(text: str, max_chars: int = 1600, overlap: int = 200, max_slices: int = 6) -> list[str]:
    """Split oversized clauses into overlapping slices so tail text is represented in embeddings."""
    clean = (text or "").strip()
    if not clean:
        return ["empty clause"]
    if len(clean) <= max_chars:
        return [clean]

    slices: list[str] = []
    start = 0
    step = max(200, max_chars - overlap)
    while start < len(clean) and len(slices) < max_slices:
        end = min(len(clean), start + max_chars)
        if end < len(clean):
            ws = clean.rfind(" ", start + int(max_chars * 0.6), end)
            if ws > start:
                end = ws
        piece = clean[start:end].strip()
        if piece:
            slices.append(piece)
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
    return slices or [clean[:max_chars]]


def _average_and_normalize_vectors(vectors: list[list[float]]) -> Optional[list[float]]:
    """Average slice vectors and L2-normalize to unit length for cosine distance retrieval."""
    if not vectors:
        return None
    if len(vectors) == 1:
        return vectors[0]
    dim = len(vectors[0])
    mean_vec = [sum(v[d] for v in vectors) / len(vectors) for d in range(dim)]
    norm = math.sqrt(sum(x * x for x in mean_vec))
    if norm <= 1e-12 or not math.isfinite(norm):
        return vectors[0]
    return [x / norm for x in mean_vec]


def batch_embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 50,
    max_retries: int = 3,
) -> list[Optional[list[float]]]:
    """
    High-performance batch embedder: sends chunks of up to batch_size texts per API call.
    Splits oversized clauses (>1600 chars) with overlap and averages normalized slice vectors.
    Returns a list of Optional[list[float]], where each element is a 768-dim vector or None.
    NEVER silently injects fake zero-vectors on failure.
    """
    if not texts:
        return []

    client = get_genai_client()
    if not client:
        logger.warning("GenAI client unavailable; returning None for all batch texts.")
        return [None for _ in texts]

    # Expand oversized items into overlapping slices while preserving 1:1 fast path for standard clauses
    flat_slices: list[str] = []
    slice_ranges: list[tuple[int, int]] = []
    for t in texts:
        s_list = _split_text_with_overlap(t) if (t and t.strip()) else ["empty clause"]
        start_idx = len(flat_slices)
        flat_slices.extend(s_list)
        slice_ranges.append((start_idx, len(flat_slices)))

    model = get_embed_model_name()
    flat_embeddings: list[Optional[list[float]]] = []

    for i in range(0, len(flat_slices), batch_size):
        chunk = flat_slices[i : i + batch_size]
        chunk_success = False

        for attempt in range(max_retries):
            try:
                result = client.models.embed_content(
                    model=model,
                    contents=chunk,
                    config={
                        "output_dimensionality": 768,
                        "task_type": task_type,
                    },
                )
                if result and getattr(result, "embeddings", None) and len(result.embeddings) == len(chunk):
                    chunk_vectors: list[Optional[list[float]]] = []
                    for emb_obj in result.embeddings:
                        vec = emb_obj.values
                        if validate_embedding_vector(vec, 768):
                            chunk_vectors.append(vec)
                        else:
                            chunk_vectors.append(None)
                    flat_embeddings.extend(chunk_vectors)
                    chunk_success = True
                    break
            except Exception as e:
                is_transient = is_transient_error(e)
                logger.warning(
                    f"Batch embedding error for chunk {i} (attempt {attempt+1}/{max_retries}): "
                    f"error_type={type(e).__name__}, transient={is_transient}."
                )
                if not is_transient or attempt >= max_retries - 1:
                    break
                delay = 0.5 * (2 ** attempt) + random.uniform(0.0, 0.2)
                time.sleep(delay)

        if not chunk_success:
            logger.error(
                f"Failed to generate embeddings for chunk {i} of size {len(chunk)}. Returning None for items."
            )
            flat_embeddings.extend([None for _ in chunk])

    pooled_embeddings: list[Optional[list[float]]] = []
    for idx, (s_idx, e_idx) in enumerate(slice_ranges):
        if not texts[idx] or not texts[idx].strip():
            pooled_embeddings.append(None)
            continue
        valid_vecs = [v for v in flat_embeddings[s_idx:e_idx] if v is not None]
        pooled_embeddings.append(_average_and_normalize_vectors(valid_vecs))

    return pooled_embeddings


def embed_document_clauses(clauses: list[dict]) -> list[Optional[list[float]]]:
    """
    Embeds multiple clauses efficiently in batch mode.
    """
    texts = [c.get("redacted_text") or c.get("text", "") for c in clauses]
    return batch_embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
