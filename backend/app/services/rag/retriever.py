import re
import math
import logging
from typing import List, Dict, Any, Optional
from collections import Counter
from app.extensions import db
from app.models.clause import Clause
from app.models.document import Document
from app.services.rag.intent import classify_query_intent, normalize_query_text, QueryIntent

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

def extract_query_keywords(query_text: str) -> list[str]:
    """Extract meaningful alphanumeric keywords from query, normalizing spellings and excluding stopwords."""
    if not query_text:
        return []
    normalized = normalize_query_text(query_text)
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\[\]]{2,}\b", normalized.lower())
    return [t for t in tokens if t not in STOP_WORDS]


def compute_bm25_scores(
    clauses: List[Clause],
    query_keywords: List[str],
    expansion_keywords: List[str] = None,
    k1: float = 1.5,
    b: float = 0.75,
    type_match_boost: float = 0.5,
) -> List[tuple[Clause, float]]:
    """
    Computes mathematically sound, token-level BM25 relevance scores across candidate clauses.

    Key Principles:
    1. Exact token matching: eliminates substring inflation (e.g. 'term' matching 'determination').
    2. Robertson-Spärck Jones IDF: ubiquitous words across the document (like 'agreement' or 'party')
       naturally receive near-zero IDF, preventing boilerplate dominance without manual word blacklisting.
    3. Metadata isolation: favorability_label and risk_score are strictly excluded from lexical BM25 scoring.
    4. Field-aware clause_type matching: normalized clause_type tokens receive a controlled boost
       scaled by the term's IDF, preventing unweighted metadata hits from overpowering body text.
    5. Term frequency saturation: prevents repetitive boilerplate terms from dominating the score.
    """
    if not clauses or not (query_keywords or expansion_keywords):
        return [(c, 0.0) for c in clauses]

    all_query_terms = list(dict.fromkeys(query_keywords or []))

    # Tokenize multi-word expansion phrases into distinct non-stopword terms
    expanded_tokens = []
    for exp in (expansion_keywords or []):
        for tok in re.findall(r"\b[a-zA-Z0-9_\-\[\]]{2,}\b", exp.lower()):
            if tok not in STOP_WORDS and tok not in expanded_tokens and tok not in all_query_terms:
                expanded_tokens.append(tok)
    all_expansion_terms = expanded_tokens
    target_terms = all_query_terms + all_expansion_terms

    # Tokenize each clause's substantive text and clause_type separately
    clause_body_tokens: List[List[str]] = []
    clause_type_token_sets: List[set] = []
    doc_lens: List[int] = []

    for c in clauses:
        # Body text tokens (strict token boundary, lowercase)
        body_text = c.redacted_text or ""
        body_toks = re.findall(r"\b[a-zA-Z0-9_\-\[\]]{2,}\b", body_text.lower())
        clause_body_tokens.append(body_toks)
        doc_lens.append(len(body_toks))

        # Normalized clause_type tokens (e.g., 'termination_notice' -> {'termination', 'notice'})
        c_type_raw = (c.clause_type or "").replace("_", " ").replace("-", " ").lower()
        type_toks = set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", c_type_raw))
        clause_type_token_sets.append(type_toks)

    N = len(clauses)
    avgdl = sum(doc_lens) / N if N > 0 else 1.0
    if avgdl == 0:
        avgdl = 1.0

    # Calculate Document Frequency (DF) strictly based on exact token occurrences
    df = Counter()
    for i in range(N):
        clause_unique_tokens = set(clause_body_tokens[i]) | clause_type_token_sets[i]
        for term in target_terms:
            if term in clause_unique_tokens:
                df[term] += 1

    # Precompute standard Robertson-Spärck Jones IDF for target terms
    idf_map = {}
    for term in target_terms:
        n_q = df[term]
        # Standard non-negative BM25 IDF formulation: ln(1 + (N - n_q + 0.5) / (n_q + 0.5))
        idf_map[term] = math.log(1.0 + (N - n_q + 0.5) / (n_q + 0.5)) if N > 0 else 1.0

    scored_clauses = []
    for i, c in enumerate(clauses):
        body_tokens = clause_body_tokens[i]
        token_counts = Counter(body_tokens)
        doc_len = doc_lens[i]
        type_tokens = clause_type_token_sets[i]

        score = 0.0

        # Primary query keywords (weight: 2.0x)
        for term in all_query_terms:
            idf = idf_map.get(term, 0.0)
            if idf <= 0.0:
                continue

            # Strict token count: NO substring count fallback
            tf = token_counts.get(term, 0)
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
            body_score = idf * (numerator / denominator) if denominator > 0 else 0.0

            # Field-aware clause_type boost scaled by IDF (exact token match only)
            field_boost = 0.0
            if term in type_tokens:
                field_boost = idf * type_match_boost

            score += (body_score + field_boost) * 2.0

        # Expansion keywords from intent classification (weight: 1.0x)
        for term in all_expansion_terms:
            idf = idf_map.get(term, 0.0)
            if idf <= 0.0:
                continue

            tf = token_counts.get(term, 0)
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
            body_score = idf * (numerator / denominator) if denominator > 0 else 0.0

            field_boost = 0.0
            if term in type_tokens:
                field_boost = idf * (type_match_boost * 0.75)

            score += (body_score + field_boost) * 1.0

        scored_clauses.append((c, round(score, 4)))

    return scored_clauses


def _is_valid_vector(vec: Any) -> bool:
    return (
        vec is not None
        and isinstance(vec, (list, tuple))
        and len(vec) == 768
        and any(abs(v) > 1e-6 for v in vec)
        and all(math.isfinite(v) for v in vec)
    )


def _format_clause_hit(
    c: Clause,
    similarity: float,
    intent: str,
    doc_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hit = {
        "clause_id": str(c.id),
        "document_id": str(c.document_id),
        "clause_index": c.clause_index,
        "clause_type": c.clause_type,
        "favorability_label": c.favorability_label,
        "risk_score": c.risk_score,
        "page_number": getattr(c, "page_number", 1) or 1,
        "start_offset": getattr(c, "start_offset", 0),
        "end_offset": getattr(c, "end_offset", 0),
        "redacted_text": c.redacted_text,
        "similarity": similarity,
        "intent": intent,
        "retrieval_status": "matched",
    }
    if doc_info:
        hit["filename"] = doc_info.get("filename")
        hit["document_type"] = doc_info.get("document_type")
    return hit


def retrieve_relevant_clauses(
    document_id,
    query_vector: Optional[list[float]] = None,
    query_text: str = "",
    top_k: int = 5,
    intent_info: Optional[QueryIntent] = None,
    candidate_pool_size: int = 25,
    rrf_weights: Optional[Dict[str, float]] = None,
    extra_queries: Optional[List[str]] = None,
) -> list[dict]:
    """
    Hybrid intent-aware retrieval for ClauseGuard AI RAG:
    1. Classifies query intent and normalizes British/American spellings.
    2. Performs structured retrieval using existing clause-analysis and risk results.
    3. Performs token-level BM25 lexical ranking using query terms and intent expansion terms.
    4. Performs semantic vector retrieval via pgvector cosine distance (when available).
    5. Fuses ranked candidates using weighted Reciprocal Rank Fusion (RRF).
    6. Separates broad candidate generation (candidate_pool_size) from final bounded top_k.
    7. Deduplicates identical or overlapping passages and returns top-k clauses with exact provenance.
    Strictly isolated to document_id.
    """
    if intent_info is None:
        intent_info = classify_query_intent(query_text)

    intent = intent_info.intent
    norm_query = intent_info.normalized_query
    q_lower = norm_query.lower()

    # Scope filter: only substantive clauses for legal analysis,
    # or include title/signatory blocks for contracting party queries
    is_party_query = any(k in q_lower for k in [
        "who", "party", "parties", "name", "names", "customer", "provider", "tenant", "landlord",
        "employer", "employee", "sign", "signed", "signatory", "signatories", "ceo", "director", "insured", "between"
    ])

    if is_party_query:
        clause_filter = (
            (Clause.is_substantive.is_(True))
            | (Clause.content_type.in_(["DOCUMENT_TITLE", "DOCUMENT_FRONT_MATTER", "SIGNATURE_BLOCK", "SUBSTANTIVE_CLAUSE"]))
            | (Clause.content_type.is_(None))
        )
    else:
        clause_filter = (
            (Clause.is_substantive.is_(True))
            | (Clause.content_type == "SUBSTANTIVE_CLAUSE")
        )

    # Fetch candidate clauses for document
    try:
        all_candidate_clauses = (
            db.session.query(Clause)
            .filter(Clause.document_id == document_id)
            .filter(clause_filter)
            .order_by(Clause.clause_index.asc())
            .all()
        )
    except Exception as err:
        logger.error(f"Failed to fetch candidate clauses for document {document_id}: {err}")
        return []

    if not all_candidate_clauses:
        return []

    # Map candidate clauses by ID for fusion
    clause_map = {str(c.id): c for c in all_candidate_clauses}
    pool_limit = max(candidate_pool_size, top_k * 2)

    # -------------------------------------------------------------
    # Strategy A: Semantic Vector Retrieval
    # -------------------------------------------------------------
    vector_ranked_ids = []
    vector_sim_map = {}
    has_valid_vector = (
        query_vector is not None
        and isinstance(query_vector, (list, tuple))
        and len(query_vector) == 768
        and any(abs(v) > 1e-6 for v in query_vector)
        and all(math.isfinite(v) for v in query_vector)
    )

    if has_valid_vector:
        try:
            distance_col = Clause.embedding.cosine_distance(query_vector)
            v_query = (
                db.session.query(Clause, distance_col.label("distance"))
                .filter(Clause.document_id == document_id)
                .filter(Clause.embedding.isnot(None))
                .filter(clause_filter)
                .order_by(distance_col)
                .limit(pool_limit)
            )
            v_results = v_query.all()
            for c_obj, dist in v_results:
                cid = str(c_obj.id)
                vector_ranked_ids.append(cid)
                sim = 1.0 - float(dist) if dist is not None else 0.0
                vector_sim_map[cid] = max(0.0, min(1.0, sim))
        except Exception as v_err:
            logger.warning(f"pgvector query for document {document_id} failed or unavailable: {v_err}")
    else:
        logger.info(
            f"Vector search unavailable for document {document_id}; proceeding with lexical (BM25) and structured retrieval."
        )

    # -------------------------------------------------------------
    # Strategy B: Lexical BM25 Retrieval
    # -------------------------------------------------------------
    keywords = extract_query_keywords(norm_query)
    extra_expansion = list(intent_info.expansion_terms or [])
    if extra_queries:
        for eq in extra_queries:
            for tok in extract_query_keywords(eq):
                if tok not in keywords and tok not in extra_expansion:
                    extra_expansion.append(tok)
    bm25_scored = compute_bm25_scores(
        all_candidate_clauses,
        query_keywords=keywords,
        expansion_keywords=extra_expansion
    )
    # Sort by BM25 score descending, then natural clause order
    bm25_scored.sort(key=lambda item: (-item[1], item[0].clause_index))
    bm25_ranked_ids = [str(c.id) for c, score in bm25_scored if score > 0][:pool_limit]
    bm25_score_map = {str(c.id): score for c, score in bm25_scored}

    # -------------------------------------------------------------
    # Strategy C: Structured Retrieval from Existing Risk Analysis
    # -------------------------------------------------------------
    structured_ranked_ids = []

    if intent in ("DOCUMENT_RISK_SUMMARY", "RECOMMENDATION_QUERY"):
        # Prioritize clauses with high risk score, unfavorable label, and key risk types
        risk_sorted = sorted(
            all_candidate_clauses,
            key=lambda c: (
                1 if (c.favorability_label or "").lower() == "unfavorable" else 0,
                c.risk_score or 0.0,
                -(c.clause_index or 0)
            ),
            reverse=True
        )
        structured_ranked_ids = [str(c.id) for c in risk_sorted][:pool_limit]

    elif intent == "UNFAVORABLE_CLAUSE_QUERY":
        # Target explicitly unfavorable clauses first
        unfavorable_clauses = [
            c for c in all_candidate_clauses
            if (c.favorability_label or "").lower() == "unfavorable" or (c.risk_score or 0.0) >= 40.0
        ]
        unfavorable_clauses.sort(key=lambda c: (c.risk_score or 0.0), reverse=True)
        if unfavorable_clauses:
            structured_ranked_ids = [str(c.id) for c in unfavorable_clauses][:pool_limit]
        else:
            # If no unfavorable clauses exist, get highest-risk clauses for transparency
            risk_sorted = sorted(all_candidate_clauses, key=lambda c: (c.risk_score or 0.0), reverse=True)
            structured_ranked_ids = [str(c.id) for c in risk_sorted[:pool_limit]]

    elif intent_info.target_clause_index is not None:
        target_idx = intent_info.target_clause_index
        exact = [c for c in all_candidate_clauses if c.clause_index == target_idx]
        zero_based = [c for c in all_candidate_clauses if c.clause_index == (target_idx - 1)]
        matched = exact if exact else zero_based
        structured_ranked_ids = [str(c.id) for c in matched][:pool_limit]

    elif intent in (
        "TERMINATION_QUERY", "PENALTY_QUERY", "PAYMENT_QUERY",
        "PRICE_ESCALATION_QUERY", "OBLIGATION_QUERY", "RIGHTS_QUERY", "DEADLINE_QUERY"
    ):
        target_types = [t.lower() for t in intent_info.target_clause_types]
        matched = [
            c for c in all_candidate_clauses
            if (c.clause_type and any(t in c.clause_type.lower() for t in target_types))
        ]
        matched.sort(key=lambda c: (c.risk_score or 0.0), reverse=True)
        structured_ranked_ids = [str(c.id) for c in matched][:pool_limit]

    # -------------------------------------------------------------
    # Strategy D: Reciprocal Rank Fusion (RRF)
    # -------------------------------------------------------------
    k_rrf = 60.0
    if rrf_weights is not None:
        w_vec = float(rrf_weights.get("w_vec", 1.5))
        w_bm25 = float(rrf_weights.get("w_bm25", 1.5))
        w_struct = float(rrf_weights.get("w_struct", 1.0))
        k_rrf = float(rrf_weights.get("k_rrf", 60.0))
    elif not has_valid_vector:
        # Rebalance weights when vector retrieval is unavailable: emphasize BM25 and structured signals
        if intent in ("DOCUMENT_RISK_SUMMARY", "UNFAVORABLE_CLAUSE_QUERY", "RECOMMENDATION_QUERY"):
            w_vec = 0.0
            w_bm25 = 1.5
            w_struct = 2.5
        elif intent in ("CLAUSE_LOOKUP", "CLAUSE_EXPLANATION") or intent_info.target_clause_index is not None:
            w_vec = 0.0
            w_bm25 = 1.0
            w_struct = 3.0
        else:
            w_vec = 0.0
            w_bm25 = 2.0
            w_struct = 1.0
    elif intent in ("DOCUMENT_RISK_SUMMARY", "UNFAVORABLE_CLAUSE_QUERY", "RECOMMENDATION_QUERY"):
        w_vec = 1.0
        w_bm25 = 1.2
        w_struct = 2.5
    elif intent in ("CLAUSE_LOOKUP", "CLAUSE_EXPLANATION") or intent_info.target_clause_index is not None:
        w_vec = 0.5
        w_bm25 = 1.0
        w_struct = 3.0
    else:
        w_vec = 1.5
        w_bm25 = 1.5
        w_struct = 1.0

    rrf_scores = Counter()

    for rank, cid in enumerate(vector_ranked_ids):
        rrf_scores[cid] += w_vec / (k_rrf + rank + 1)

    for rank, cid in enumerate(bm25_ranked_ids):
        rrf_scores[cid] += w_bm25 / (k_rrf + rank + 1)

    for rank, cid in enumerate(structured_ranked_ids):
        rrf_scores[cid] += w_struct / (k_rrf + rank + 1)

    # If no strategy scored any clause above threshold, do NOT select arbitrary first-five clauses.
    # Return empty list so downstream components can accurately declare ungrounded/no evidence.
    if not rrf_scores:
        logger.info(
            f"No relevant clauses found across vector, lexical, or structured retrieval for document {document_id}."
        )
        return []

    # Sort candidates by RRF score descending
    sorted_cids = [cid for cid, _ in rrf_scores.most_common()]
    top_candidates = [clause_map[cid] for cid in sorted_cids if cid in clause_map]

    # Deduplicate passages while preserving provenance
    seen_texts = set()
    retrieved = []

    for c in top_candidates:
        norm_text = (c.redacted_text or "").strip().lower()
        if not norm_text or norm_text in seen_texts:
            continue
        seen_texts.add(norm_text)

        cid = str(c.id)
        # Compute calibrated similarity
        vec_sim = vector_sim_map.get(cid)
        bm25_val = bm25_score_map.get(cid, 0.0)

        if vec_sim is not None:
            similarity = round(vec_sim, 4)
        elif bm25_val > 0:
            similarity = round(min(0.95, 0.55 + (bm25_val * 0.08)), 4)
        elif cid in structured_ranked_ids:
            similarity = 0.85
        else:
            similarity = 0.50

        retrieved.append(_format_clause_hit(c, similarity=similarity, intent=intent))

        if len(retrieved) >= top_k:
            break

    return retrieved


def retrieve_relevant_clauses_for_user(
    user_id,
    query_vectors: Optional[List[list[float]]] = None,
    query_texts: Optional[List[str]] = None,
    top_k: int = 12,
    per_doc_cap: int = 4,
    intent_info: Optional[QueryIntent] = None,
    target_doc_hints: Optional[List[str]] = None,
    candidate_pool_size: int = 60,
) -> list[dict]:
    """
    Strictly account-scoped multi-document hybrid retriever.
    Searches ONLY analyzed documents belonging to `user_id` (Document.user_id == user_id),
    fuses pgvector + multi-query BM25 + structured risk/clause-type signals + doc hints via RRF,
    and balances per-document diversity so multi-document comparisons represent every matching agreement.
    """
    if not user_id:
        return []

    clean_texts = [t.strip() for t in (query_texts or []) if t and t.strip()]
    primary_text = clean_texts[0] if clean_texts else ""
    if intent_info is None:
        intent_info = classify_query_intent(primary_text)
    intent = intent_info.intent

    try:
        user_docs = (
            db.session.query(Document)
            .filter(Document.user_id == user_id)
            .filter(Document.status == "analyzed")
            .order_by(Document.uploaded_at.desc())
            .all()
        )
    except Exception as err:
        logger.error(f"Failed to fetch documents for user {user_id}: {err}")
        return []

    if not user_docs:
        return []

    doc_ids = [d.id for d in user_docs]
    doc_meta_map = {
        str(d.id): {
            "id": str(d.id),
            "filename": d.filename,
            "document_type": d.document_type,
            "overall_risk_score": d.overall_risk_score,
            "risk_band": d.risk_band,
        }
        for d in user_docs
    }

    # Include substantive clauses + title/front-matter/signatories while excluding headers/footers/TOC
    clause_filter = (
        (Clause.is_substantive.is_(True))
        | (Clause.content_type.in_(["SUBSTANTIVE_CLAUSE", "DOCUMENT_TITLE", "DOCUMENT_FRONT_MATTER", "SIGNATURE_BLOCK"]))
        | (Clause.content_type.is_(None))
    )

    try:
        all_candidates = (
            db.session.query(Clause)
            .filter(Clause.document_id.in_(doc_ids))
            .filter(clause_filter)
            .order_by(Clause.document_id, Clause.clause_index.asc())
            .all()
        )
    except Exception as err:
        logger.error(f"Failed to fetch candidate clauses for user {user_id}: {err}")
        return []

    if not all_candidates:
        return []

    clause_map = {str(c.id): c for c in all_candidates}
    pool_limit = max(candidate_pool_size, top_k * 3)

    # 1. Multi-query Semantic Vector Search (strictly bounded to user's doc_ids)
    vector_sim_map: Dict[str, float] = {}
    valid_vecs = [v for v in (query_vectors or []) if _is_valid_vector(v)]
    for qvec in valid_vecs:
        try:
            distance_col = Clause.embedding.cosine_distance(qvec)
            v_results = (
                db.session.query(Clause, distance_col.label("distance"))
                .filter(Clause.document_id.in_(doc_ids))
                .filter(Clause.embedding.isnot(None))
                .filter(clause_filter)
                .order_by(distance_col)
                .limit(pool_limit)
                .all()
            )
            for c_obj, dist in v_results:
                cid = str(c_obj.id)
                sim = max(0.0, min(1.0, 1.0 - float(dist))) if dist is not None else 0.0
                if sim > vector_sim_map.get(cid, -1.0):
                    vector_sim_map[cid] = sim
        except Exception as v_err:
            logger.warning(f"Account-wide pgvector search failed for user {user_id}: {v_err}")
            break

    vector_ranked_ids = [
        cid for cid, _ in sorted(vector_sim_map.items(), key=lambda x: -x[1])
    ][:pool_limit]

    # 2. Multi-query Lexical BM25 across all owned documents
    primary_keywords: List[str] = []
    expansion_terms: List[str] = list(intent_info.expansion_terms or [])
    for idx, qt in enumerate(clean_texts):
        toks = extract_query_keywords(qt)
        if idx == 0:
            primary_keywords = toks
        else:
            for t in toks:
                if t not in primary_keywords and t not in expansion_terms:
                    expansion_terms.append(t)

    bm25_scored = compute_bm25_scores(
        all_candidates,
        query_keywords=primary_keywords,
        expansion_keywords=expansion_terms,
    )
    bm25_scored.sort(key=lambda item: (-item[1], str(item[0].document_id), item[0].clause_index))
    bm25_ranked_ids = [str(c.id) for c, score in bm25_scored if score > 0][:pool_limit]
    bm25_score_map = {str(c.id): score for c, score in bm25_scored}

    # 3. Structured & Target-Document Hint Ranking
    structured_ranked_ids: List[str] = []
    hint_terms = [h.lower().strip() for h in (target_doc_hints or []) if h and h.strip()]
    matched_doc_ids_by_hint = set()
    if hint_terms or primary_text:
        q_low = primary_text.lower()
        for d_id, dmeta in doc_meta_map.items():
            fname_low = (dmeta["filename"] or "").lower()
            dtype_low = (dmeta["document_type"] or "").replace("_", " ").lower()
            fname_stem = re.sub(r"\.[a-z0-9]+$", "", fname_low).replace("_", " ").replace("-", " ")
            if any(h in fname_low or h in dtype_low for h in hint_terms) or (
                len(fname_stem) >= 4 and fname_stem in q_low
            ):
                matched_doc_ids_by_hint.add(d_id)

    if intent in ("DOCUMENT_RISK_SUMMARY", "RECOMMENDATION_QUERY", "UNFAVORABLE_CLAUSE_QUERY"):
        risk_sorted = sorted(
            all_candidates,
            key=lambda c: (
                1 if (c.favorability_label or "").lower() == "unfavorable" else 0,
                c.risk_score or 0.0,
            ),
            reverse=True,
        )
        structured_ranked_ids = [str(c.id) for c in risk_sorted][:pool_limit]
    elif intent_info.target_clause_types:
        target_types = [t.lower() for t in intent_info.target_clause_types]
        matched = [
            c for c in all_candidates
            if c.clause_type and any(t in c.clause_type.lower() for t in target_types)
        ]
        matched.sort(key=lambda c: (c.risk_score or 0.0), reverse=True)
        structured_ranked_ids = [str(c.id) for c in matched][:pool_limit]

    # 4. Balanced RRF Fusion
    k_rrf = 60.0
    w_vec = 1.6 if vector_ranked_ids else 0.0
    w_bm25 = 1.6
    w_struct = 1.2
    w_hint = 1.5

    rrf_scores = Counter()
    for rank, cid in enumerate(vector_ranked_ids):
        rrf_scores[cid] += w_vec / (k_rrf + rank + 1)
    for rank, cid in enumerate(bm25_ranked_ids):
        rrf_scores[cid] += w_bm25 / (k_rrf + rank + 1)
    for rank, cid in enumerate(structured_ranked_ids):
        rrf_scores[cid] += w_struct / (k_rrf + rank + 1)
    if matched_doc_ids_by_hint:
        for c in all_candidates:
            if str(c.document_id) in matched_doc_ids_by_hint and str(c.id) in rrf_scores:
                rrf_scores[str(c.id)] += w_hint / k_rrf

    if not rrf_scores:
        return []

    sorted_cids = [cid for cid, _ in rrf_scores.most_common()]

    # Diverse multi-document selection (respect per_doc_cap first, then fill any remaining slots)
    per_doc_counts: Dict[str, int] = Counter()
    seen_texts_per_doc: Dict[str, set] = {}
    selected: List[tuple[Clause, float]] = []
    overflow: List[tuple[Clause, float]] = []

    for cid in sorted_cids:
        c = clause_map.get(cid)
        if not c:
            continue
        did = str(c.document_id)
        norm_text = (c.redacted_text or "").strip().lower()
        if not norm_text:
            continue
        doc_seen = seen_texts_per_doc.setdefault(did, set())
        if norm_text in doc_seen:
            continue
        doc_seen.add(norm_text)

        vec_sim = vector_sim_map.get(cid)
        bm25_val = bm25_score_map.get(cid, 0.0)
        if vec_sim is not None:
            sim = round(vec_sim, 4)
        elif bm25_val > 0:
            sim = round(min(0.95, 0.55 + (bm25_val * 0.08)), 4)
        elif cid in structured_ranked_ids:
            sim = 0.85
        else:
            sim = 0.55

        if per_doc_counts[did] < per_doc_cap:
            per_doc_counts[did] += 1
            selected.append((c, sim))
        else:
            overflow.append((c, sim))

        if len(selected) >= top_k:
            break

    if len(selected) < top_k and overflow:
        for c, sim in overflow:
            selected.append((c, sim))
            if len(selected) >= top_k:
                break

    return [
        _format_clause_hit(c, similarity=sim, intent=intent, doc_info=doc_meta_map.get(str(c.document_id)))
        for c, sim in selected
    ]
