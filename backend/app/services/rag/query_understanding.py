import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from app.services.rag.intent import classify_query_intent, normalize_query_text, QueryIntent
from app.utils.gemini_chat import generate_content_with_cascade, STALE_CHAT_MODELS

logger = logging.getLogger(__name__)

# Fast, lightweight models prioritized specifically for sub-second JSON query rewriting
_REWRITE_CANDIDATE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
]

# Common typo repairs (exact word-boundary replacement)
_TYPO_CORRECTIONS = {
    r"\bterminat(?:on|ionn|eion)\b": "termination",
    r"\btermiantion\b": "termination",
    r"\bindem(?:inity|infy|nityy)\b": "indemnity",
    r"\bnotise\b": "notice",
    r"\bliabilty\b": "liability",
    r"\bconfidentialty\b": "confidentiality",
    r"\brenwal\b": "renewal",
    r"\barbitraton\b": "arbitration",
    r"\bcompenstion\b": "compensation",
    r"\bsalery\b": "salary",
    r"\bpenality\b": "penalty",
    r"\beligibilty\b": "eligibility",
    r"\bagreemnt\b": "agreement",
    r"\binsurence\b": "insurance",
    r"\bdeductable\b": "deductible",
    r"\bsevrance\b": "severance",
    r"\bexlusiv(?:e|ity)\b": "exclusivity",
    r"\bsubrogaton\b": "subrogation",
    r"\bforfit(?:ure)?\b": "forfeiture",
    r"\bguarntee\b": "guarantee",
}

# Abbreviations & informal phrasing semantic expansions (appended for retrieval, never alters intent)
_ABBREVIATION_AND_SLANG_EXPANSIONS = [
    (r"\bctc\b", "cost to company annual salary compensation package"),
    (r"\blpa\b", "lakhs per annum annual base salary compensation"),
    (r"\bpto\b", "paid time off vacation leave holiday"),
    (r"\bnda\b", "non-disclosure confidentiality proprietary information"),
    (r"\bsla\b", "service level agreement uptime response time"),
    (r"\bidv\b", "insured declared value coverage amount limit"),
    (r"\bncb\b", "no claim bonus discount"),
    (r"\bhoa\b", "homeowners association rules assessment"),
    (r"\bemi\b", "monthly installment payment due"),
    (r"\b(?:money\s+held\s+by\s+(?:the\s+)?landlord|security\s+money|deposit\s+money)\b", "security deposit refund deduction"),
    (r"\b(?:time\s+before\s+leaving|before\s+quitting|advance\s+warning)\b", "notice period written notice termination"),
    (r"\b(?:extra\s+time\s+to\s+pay|extra\s+days\s+to\s+pay|late\s+window)\b", "grace period late fee rent due"),
    (r"\b(?:coverage\s+amount|max\s+payout|cover\s+limit|policy\s+limit)\b", "sum insured coverage limit liability"),
    (r"\b(?:side\s+work|side\s+gigs?|freelanc(?:e|ing)|weekend\s+work)\b", "outside employment moonlighting exclusivity"),
    (r"\b(?:break\s+(?:the\s+)?contract|walk\s+away|get\s+out\s+of|exit\s+conditions?)\b", "early termination penalty liquidated damages notice"),
    (r"\b(?:company\s+remove\s+me|let\s+me\s+go|kick\s+me\s+out)\b", "dismissal discharge termination eviction"),
]

_FOLLOWUP_ANAPHORA_PATTERN = re.compile(
    r"\b(?:it|its|that|this|they|them|those|these|such|same|both|neither|either|"
    r"first\s+one|second\s+one|last\s+one|previous|above|earlier|"
    r"what\s+about|how\s+about|and\s+if|what\s+if\s+i\s+don'?t|"
    r"explain\s+(?:more|further|simpler|simply)|tell\s+me\s+more|why\s+is\s+that|"
    r"which\s+(?:doc|document|agreement|contract|policy|one))\b",
    re.IGNORECASE,
)

_COMPARATIVE_PATTERN = re.compile(
    r"\b(?:compare|comparison|difference|differ(?:s|ent)?|conflict(?:ing|s)?|"
    r"contradict(?:s|ion)?|across\s+(?:all|my)|between\s+my|all\s+(?:my\s+)?documents?|"
    r"all\s+(?:my\s+)?contracts?|vs\.?|versus)\b",
    re.IGNORECASE,
)

_CONVERSATIONAL_PATTERN = re.compile(
    r"^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening)|how\s+are\s+you|"
    r"thanks?|thank\s+you|ok(?:ay)?|cool|great|bye|goodbye|"
    r"who\s+are\s+you|what\s+can\s+you\s+do|help(?:\s+me)?)\W*$",
    re.IGNORECASE,
)

_UNSUPPORTED_REQUEST_PATTERN = re.compile(
    r"\b(?:hack\s+(?:into|the)|sql\s+inject|drop\s+table|execute\s+(?:shell|bash|command)|"
    r"scrape\s+(?:live|external|this\s+website)|browse\s+the\s+internet|send\s+an?\s+email\s+for\s+me|"
    r"delete\s+(?:all\s+)?(?:user|database)\s+records|other\s+users?'?\s+(?:account|documents?|data)|"
    r"represent\s+me\s+in\s+court|file\s+a\s+lawsuit\s+on\s+my\s+behalf|notarize\s+this|"
    r"\[DEFANGED_INSTRUCTION\])\b",
    re.IGNORECASE,
)


@dataclass
class QueryUnderstandingResult:
    raw_query: str
    standalone_query: str
    semantic_queries: List[str] = field(default_factory=list)
    query_category: str = "DOCUMENT_SPECIFIC"
    target_doc_hints: List[str] = field(default_factory=list)
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    intent_info: Optional[QueryIntent] = None
    used_llm_rewrite: bool = False


def repair_typos_and_normalize(text: str) -> str:
    """Fix common legal typos and UK/US spelling variants without altering intent."""
    cleaned = normalize_query_text(text or "")
    for pat, repl in _TYPO_CORRECTIONS.items():
        cleaned = re.sub(pat, repl, cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def expand_slang_and_abbreviations(text: str) -> list[str]:
    """Generate clean semantic expansion search queries for informal phrasing or abbreviations."""
    q_norm = repair_typos_and_normalize(text)
    queries = [q_norm] if q_norm else []
    extra_terms = []
    for pat, exp in _ABBREVIATION_AND_SLANG_EXPANSIONS:
        if re.search(pat, q_norm, re.IGNORECASE):
            extra_terms.append(exp)
    if extra_terms:
        queries.append(f"{q_norm} {' '.join(extra_terms)}".strip())
    return queries


def is_conversational_query(query: str) -> bool:
    """Detect pure greetings, acknowledgements, or assistant meta-questions."""
    q = (query or "").strip().lower()
    if not q or len(q.split()) > 8:
        return False
    return bool(_CONVERSATIONAL_PATTERN.match(q))


def is_unsupported_request(query: str) -> bool:
    """Detect out-of-scope action requests, cross-tenant/system abuse, or defanged injections."""
    q = (query or "").strip()
    if not q:
        return False
    return bool(_UNSUPPORTED_REQUEST_PATTERN.search(q))


def detect_other_document_reference(
    query: str,
    active_doc_id: Any,
    user_docs_meta: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """
    Metadata-only check to detect if a document-specific question explicitly names a *different*
    document belonging to the user's account (never reads or exposes that other document's content).
    """
    if not query or not user_docs_meta:
        return None
    q_lower = query.lower()
    active_str = str(active_doc_id) if active_doc_id else ""

    active_doc = next((d for d in user_docs_meta if str(d.get("id")) == active_str), None)
    active_type = (active_doc.get("document_type") or "").lower() if active_doc else ""

    # 1. Exact or stem filename match on another document
    for d in user_docs_meta:
        if str(d.get("id")) == active_str:
            continue
        fname = (d.get("filename") or "").strip()
        if not fname:
            continue
        stem = re.sub(r"\.[a-z0-9]+$", "", fname, flags=re.IGNORECASE).replace("_", " ").replace("-", " ").strip().lower()
        if fname.lower() in q_lower or (len(stem) >= 5 and stem in q_lower):
            return {"filename": fname, "document_type": d.get("document_type")}

    # 2. Explicit cross-domain inquiry when user owns a matching other-domain document
    domain_phrases = {
        "rental_agreement": ["rental agreement", "lease agreement", "my lease", "landlord", "apartment lease"],
        "job_offer_letter": ["offer letter", "employment agreement", "employment contract", "my job offer"],
        "insurance_policy": ["insurance policy", "health insurance", "vehicle insurance", "car insurance", "auto insurance"],
    }
    for dtype, phrases in domain_phrases.items():
        if dtype == active_type:
            continue
        if any(p in q_lower for p in phrases):
            matched_other = next((d for d in user_docs_meta if (d.get("document_type") or "").lower() == dtype and str(d.get("id")) != active_str), None)
            if matched_other:
                return {"filename": matched_other.get("filename"), "document_type": dtype}

    return None


def _needs_llm_rewrite(query: str, history: Optional[List[Dict[str, Any]]], mode: str) -> bool:
    """
    Decide if an ultra-fast LLM semantic rewrite step is warranted.
    Skips unnecessary latency on clear single-turn queries while resolving follow-ups,
    abbreviations, ambiguities, informal wording, and comparative cross-doc queries.
    """
    if is_conversational_query(query):
        return False

    q_clean = repair_typos_and_normalize(query)
    words = q_clean.split()

    # Always rewrite follow-ups that reference prior turns with pronouns/ellipsis
    if history and (_FOLLOWUP_ANAPHORA_PATTERN.search(q_clean) or len(words) <= 5):
        return True

    # Comparative / multi-doc account queries benefit from structured multi-query decomposition
    if mode == "account" and _COMPARATIVE_PATTERN.search(q_clean):
        return True

    # Very short or informal/ambiguous queries that regex intent didn't cleanly classify
    intent_probe = classify_query_intent(q_clean)
    if intent_probe.intent == "GENERAL_DOCUMENT_QUESTION" and len(words) <= 9:
        # Check if slang/typo terms exist or query could be general vs ambiguous
        if any(re.search(pat, q_clean, re.IGNORECASE) for pat, _ in _ABBREVIATION_AND_SLANG_EXPANSIONS):
            return False  # Already cleanly expanded deterministically in 0ms!
        return True

    return False


def understand_and_rewrite_query(
    question: str,
    history: Optional[List[Dict[str, Any]]] = None,
    mode: str = "document",
    available_docs: Optional[List[Dict[str, Any]]] = None,
    client: Any = None,
) -> QueryUnderstandingResult:
    """
    Fast semantic query understanding & conversational rewriting shared across both chatbots.
    - Resolves pronouns/ellipsis using conversation history BEFORE retrieval.
    - Normalizes typos, abbreviations, and informal phrases without inventing facts.
    - Triggers a fast LLM JSON rewrite (~0.8-1.4s) only when context/ambiguity requires it.
    """
    raw_q = (question or "").strip()
    repaired = repair_typos_and_normalize(raw_q)

    if is_conversational_query(repaired):
        intent_info = classify_query_intent(repaired)
        return QueryUnderstandingResult(
            raw_query=raw_q,
            standalone_query=repaired,
            semantic_queries=[],
            query_category="CONVERSATIONAL",
            intent_info=intent_info,
            used_llm_rewrite=False,
        )

    if is_unsupported_request(raw_q) or is_unsupported_request(repaired):
        intent_info = classify_query_intent(repaired)
        return QueryUnderstandingResult(
            raw_query=raw_q,
            standalone_query=repaired,
            semantic_queries=[],
            query_category="UNSUPPORTED_REQUEST",
            intent_info=intent_info,
            used_llm_rewrite=False,
        )

    semantic_queries = expand_slang_and_abbreviations(repaired)
    standalone_query = semantic_queries[-1] if semantic_queries else repaired
    query_category = "ACCOUNT_WIDE" if mode == "account" else "DOCUMENT_SPECIFIC"
    if _COMPARATIVE_PATTERN.search(repaired):
        query_category = "CROSS_DOC_COMPARISON"

    used_llm = False
    target_doc_hints: list[str] = []
    clarification_needed = False
    clarification_question = None

    if client and _needs_llm_rewrite(repaired, history, mode):
        hist_lines = []
        for m in (history or [])[-4:]:
            role = (m.get("role") or "user").upper()
            content = (m.get("content") or "").strip()[:300]
            docs_tag = ""
            doc_list = [str(x).strip() for x in (m.get("documents") or []) if str(x).strip()]
            if doc_list:
                docs_tag = f" [Docs: {', '.join(doc_list)}]"
            if content:
                hist_lines.append(f"{role}{docs_tag}: {content}")
        hist_text = "\n".join(hist_lines) if hist_lines else "None"

        doc_list_text = ""
        if available_docs:
            doc_names = [f"{d.get('filename')} ({d.get('document_type')})" for d in available_docs[:15]]
            doc_list_text = f"\nUSER DOCUMENTS: {', '.join(doc_names)}"

        rewrite_prompt = (
            "Rewrite the user's latest query for legal contract retrieval & classify its intent. Return STRICT JSON only.\n"
            f"MODE: {mode}{doc_list_text}\n"
            f"HISTORY:\n{hist_text}\n"
            f"LATEST QUERY: {repaired}\n\n"
            'JSON schema: {"standalone_query": "...", "semantic_queries": ["...", "..."], '
            '"category": "DOCUMENT_SPECIFIC|CROSS_DOC_COMPARISON|GENERAL_LEGAL_TERM|GENERAL_KNOWLEDGE|AMBIGUOUS|CONVERSATIONAL|UNSUPPORTED_REQUEST", '
            '"target_doc_hints": ["optional filename"], "clarification_needed": false, "clarification_question": null}\n'
            "Rules:\n"
            "- Resolve pronouns ('it', 'that', 'the second one', 'both parties') using HISTORY without inventing facts.\n"
            "- Expand informal phrases/abbreviations into clean legal search concepts.\n"
            "- Set clarification_needed=true ONLY if the query is genuinely too vague to search or answer without guessing.\n"
            "- Keep JSON concise."
        )

        # Call lightweight rewrite models cleanly
        for m_name in _REWRITE_CANDIDATE_MODELS:
            try:
                resp = client.models.generate_content(model=m_name, contents=rewrite_prompt)
                raw_resp = (getattr(resp, "text", None) or "").strip()
                if not raw_resp:
                    continue
                json_match = re.search(r"\{.*\}", raw_resp, re.DOTALL)
                if not json_match:
                    continue
                parsed = json.loads(json_match.group(0))
                sq = (parsed.get("standalone_query") or "").strip()
                if sq:
                    standalone_query = repair_typos_and_normalize(sq)
                extra_qs = [
                    repair_typos_and_normalize(x)
                    for x in (parsed.get("semantic_queries") or [])
                    if isinstance(x, str) and x.strip()
                ]
                merged = []
                for cand in [standalone_query] + extra_qs + semantic_queries:
                    if cand and cand not in merged:
                        merged.append(cand)
                semantic_queries = merged[:3]
                cat = (parsed.get("category") or "").strip().upper()
                if cat in {
                    "DOCUMENT_SPECIFIC", "CROSS_DOC_COMPARISON", "GENERAL_LEGAL_TERM",
                    "GENERAL_KNOWLEDGE", "AMBIGUOUS", "CONVERSATIONAL", "ACCOUNT_WIDE", "UNSUPPORTED_REQUEST",
                }:
                    query_category = cat
                hints = parsed.get("target_doc_hints") or []
                if isinstance(hints, list):
                    target_doc_hints = [str(h).strip() for h in hints if str(h).strip()]
                clarification_needed = bool(parsed.get("clarification_needed", False))
                cq = parsed.get("clarification_question")
                clarification_question = str(cq).strip() if cq else None
                used_llm = True
                break
            except Exception as err:
                logger.debug(f"[QueryRewrite] {m_name} failed: {err}")
                continue

    intent_info = classify_query_intent(standalone_query or repaired)
    return QueryUnderstandingResult(
        raw_query=raw_q,
        standalone_query=standalone_query or repaired,
        semantic_queries=semantic_queries or [repaired],
        query_category=query_category,
        target_doc_hints=target_doc_hints,
        clarification_needed=clarification_needed,
        clarification_question=clarification_question,
        intent_info=intent_info,
        used_llm_rewrite=used_llm,
    )

