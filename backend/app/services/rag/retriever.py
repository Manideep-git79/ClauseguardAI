import re
import math
import logging
from typing import List, Dict, Any, Optional
from collections import Counter
from app.extensions import db
from app.models.clause import Clause
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


def retrieve_relevant_clauses(
    document_id,
    query_vector: Optional[list[float]] = None,
    query_text: str = "",
    top_k: int = 5,
    intent_info: Optional[QueryIntent] = None,
    candidate_pool_size: int = 25,
    rrf_weights: Optional[Dict[str, float]] = None,
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
    bm25_scored = compute_bm25_scores(
        all_candidate_clauses,
        query_keywords=keywords,
        expansion_keywords=intent_info.expansion_terms
    )
    # Sort by BM25 score descending, then natural clause order
    bm25_scored.sort(key=lambda item: (-item[1], item[0].clause_index))
    bm25_ranked_ids = [str(c.id) for c, score in bm25_scored if score > 0][:pool_limit]
    bm25_score_map = {str(c.id): score for c, score in bm25_scored}

    # -------------------------------------------------------------
    # Strategy C: Structured Retrieval from Existing Risk Analysis
    # -------------------------------------------------------------
    structured_ranked_ids = []

    if intent == "DOCUMENT_RISK_SUMMARY":
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
        if intent in ("DOCUMENT_RISK_SUMMARY", "UNFAVORABLE_CLAUSE_QUERY"):
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
    elif intent in ("DOCUMENT_RISK_SUMMARY", "UNFAVORABLE_CLAUSE_QUERY"):
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

        retrieved.append({
            "clause_id": str(c.id),
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
        })

        if len(retrieved) >= top_k:
            break

    return retrieved
