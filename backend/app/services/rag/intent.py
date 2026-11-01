"""
Intent-aware query classification and normalization layer for ClauseGuard AI RAG.

Recognizes contractual and document-level intents:
- DOCUMENT_RISK_SUMMARY
- UNFAVORABLE_CLAUSE_QUERY
- CLAUSE_EXPLANATION
- TERMINATION_QUERY (with distinct actions: resignation, dismissal, lease surrender, general termination)
- PENALTY_QUERY
- PRICE_ESCALATION_QUERY
- PAYMENT_QUERY
- OBLIGATION_QUERY
- RIGHTS_QUERY
- DEADLINE_QUERY
- DOCUMENT_SUMMARY
- CLAUSE_LOOKUP
- GENERAL_DOCUMENT_QUESTION

Normalizes British/American legal spellings, expands relevant keywords,
and provides structured signals for hybrid retrieval.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# British to American spelling normalization map for legal terms
SPELLING_NORMALIZATIONS = {
    r"\bunfavourable\b": "unfavorable",
    r"\bfavour\b": "favor",
    r"\bfavourable\b": "favorable",
    r"\bunfavourably\b": "unfavorably",
    r"\bfavourably\b": "favorably",
    r"\bcancelled\b": "canceled",
    r"\bcancelling\b": "canceling",
    r"\bindemnise\b": "indemnify",
    r"\bindemnised\b": "indemnified",
    r"\bindemnisation\b": "indemnification",
    r"\bpenalise\b": "penalize",
    r"\bpenalised\b": "penalized",
    r"\bpenalisation\b": "penalization",
    r"\blicence\b": "license",
    r"\blicenced\b": "licensed",
    r"\bjudgement\b": "judgment",
    r"\banalyse\b": "analyze",
    r"\banalysed\b": "analyzed",
    r"\banalysing\b": "analyzing",
    r"\blabour\b": "labor",
    r"\bdefence\b": "defense",
    r"\boffence\b": "offense",
    r"\bpractise\b": "practice",
}


@dataclass
class QueryIntent:
    intent: str
    confidence: float
    raw_query: str
    normalized_query: str
    target_clause_types: List[str] = field(default_factory=list)
    expansion_terms: List[str] = field(default_factory=list)
    requires_risk_summary: bool = False
    requires_unfavorable_clauses: bool = False
    target_clause_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_negation(self) -> bool:
        return bool(self.metadata.get("negation_detected", False))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "target_clause_types": self.target_clause_types,
            "expansion_terms": self.expansion_terms,
            "requires_risk_summary": self.requires_risk_summary,
            "requires_unfavorable_clauses": self.requires_unfavorable_clauses,
            "target_clause_index": self.target_clause_index,
            "metadata": self.metadata,
        }


def normalize_query_text(query: str) -> str:
    """Normalize whitespace, punctuation, and British/American spelling variations."""
    if not query:
        return ""
    normalized = query.strip()
    for pattern, replacement in SPELLING_NORMALIZATIONS.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    # Collapse multiple whitespaces
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


NEGATION_PATTERNS = [
    r"\bwithout\s+(?:any\s+)?(?:prior\s+|advance\s+)?(?:cause|notice|penalty|fault|reason|consent|prejudice|delay|warning)\b",
    r"\bnot\s+required\b",
    r"\bno\s+(?:penalty|cause|notice|increase|escalation|fee|liability)\b",
    r"\bnon[- ]refundable\b",
    r"\bwithout\s+paying\b",
    r"\bcannot\b",
    r"\bcan\s+not\b",
    r"\bnever\b",
]


def extract_negated_conditions(query: str) -> List[str]:
    """Find specific negated contractual conditions in the query."""
    q_lower = query.lower()
    matches = []
    for pat in NEGATION_PATTERNS:
        for m in re.finditer(pat, q_lower):
            matches.append(m.group(0).strip())
    return matches


def classify_query_intent(query_text: str) -> QueryIntent:
    """
    Classifies the user's question into a legal/contractual intent.
    Uses deterministic rule matching and semantic feature extraction.
    Preserves original query and extracts expansion terms for hybrid retrieval.
    Never treats legally distinct actions (resignation, dismissal, surrender, termination)
    as interchangeable.
    """
    raw_query = query_text or ""
    norm = normalize_query_text(raw_query)
    q_lower = norm.lower()

    # Extract negated contractual conditions if present
    negated_conditions = extract_negated_conditions(q_lower)
    negation_detected = len(negated_conditions) > 0

    # 1. CLAUSE_LOOKUP: Explicit request for a numbered clause or section
    # e.g. "clause 1", "clause #2", "section 3", "show me clause 4"
    lookup_match = re.search(r"\b(?:clause|section|article|paragraph)\s*(?:#|\b)?\s*(\d+)\b", q_lower)
    is_explain = any(w in q_lower for w in ["explain", "what does", "clarify", "mean", "meaning", "interpret"])
    
    if lookup_match and not is_explain:
        clause_idx = int(lookup_match.group(1))
        # Note: Users often say "Clause 1" meaning the 1st clause (index 0) or explicit Clause #1.
        # We store both 1-based and 0-based in metadata for retrieval flexibility
        return QueryIntent(
            intent="CLAUSE_LOOKUP",
            confidence=0.98,
            raw_query=raw_query,
            normalized_query=norm,
            target_clause_index=clause_idx,
            expansion_terms=[f"clause {clause_idx}", f"section {clause_idx}"],
            metadata={
                "extracted_number": clause_idx,
                "negation_detected": negation_detected,
                "negated_conditions": negated_conditions,
            },
        )

    # 2. CLAUSE_EXPLANATION: Request to explain, clarify or interpret a specific clause
    if is_explain and (lookup_match or any(k in q_lower for k in [
        "clause", "section", "provision", "indemnity", "confidentiality", "non-compete",
        "arbitration", "severability", "force majeure", "governing law"
    ])):
        target_idx = int(lookup_match.group(1)) if lookup_match else None
        return QueryIntent(
            intent="CLAUSE_EXPLANATION",
            confidence=0.92,
            raw_query=raw_query,
            normalized_query=norm,
            target_clause_index=target_idx,
            expansion_terms=["mean", "meaning", "obligation", "scope", "effect", "interpretation"],
            metadata={
                "has_clause_number": target_idx is not None,
                "negation_detected": negation_detected,
                "negated_conditions": negated_conditions,
            },
        )

    # 3. UNFAVORABLE_CLAUSE_QUERY: Specific inquiry about unfavorable, one-sided, unfair or disadvantageous terms
    # e.g. "any unfavourable clauses detected", "unfavourable clauses", "unfavorable terms",
    # "unfair conditions", "clauses against me", "provisions that disadvantage me"
    unfavorable_patterns = [
        r"\bunfavorable\b",
        r"\bunfair\b",
        r"\bone[- ]sided\b",
        r"\bharsh\s+terms\b",
        r"\bdisadvantage(?:ous|s)?\b",
        r"\bagainst\s+me\b",
        r"\bclauses?\s+against\b",
        r"\bprovisions?\s+that\s+disadvantage\b",
        r"\bbad\s+terms\b",
        r"\bharmful\s+clauses?\b",
        r"\boppressive\b",
        r"\bpredatory\b",
        r"\bdetrimental\b",
        r"\bis\s+there\s+anything\s+bad\b",
        r"\bstacked\s+against\b",
    ]
    if any(re.search(pat, q_lower) for pat in unfavorable_patterns):
        return QueryIntent(
            intent="UNFAVORABLE_CLAUSE_QUERY",
            confidence=0.95,
            raw_query=raw_query,
            normalized_query=norm,
            requires_unfavorable_clauses=True,
            target_clause_types=[
                "indemnity", "limitation_of_liability", "termination_notice",
                "liquidated_damages", "non_compete", "remedies", "penalty"
            ],
            expansion_terms=[
                "unfavorable", "unfair", "penalty", "forfeit", "disadvantage",
                "strict", "one-sided", "liability", "damages", "indemnify"
            ],
            metadata={
                "source_rule": "unfavorable_pattern_match",
                "negation_detected": negation_detected,
                "negated_conditions": negated_conditions,
            },
        )

    # 4. DOCUMENT_RISK_SUMMARY: Broad or general inquiry about document risks, dangers, hazards, or pitfalls
    # e.g. "any risks", "what are the risks", "are there risks", "is there anything dangerous",
    # "what are the risky terms?", "any problematic conditions?", "what should I worry about?",
    # "red flags", "is this safe", "pitfalls", "concerns"
    risk_patterns = [
        r"\b(?:any|what|are\s+there|is\s+there|what\s+are\s+the|what\s+is\s+the)\s+risks?\b",
        r"\bwhat\s+are\s+the\s+risks\b",
        r"\bwhat\s+is\s+the\s+risk\b",
        r"\brisk\s+summary\b",
        r"\boverall\s+risk\b",
        r"\brisky\s+terms?\b",
        r"\brisks?\s+(?:in|of)\s+this\b",
        r"\banything\s+dangerous\b",
        r"\bdangerous\s+clauses?\b",
        r"\bproblematic\s+(?:conditions?|terms?|clauses?)\b",
        r"\bwhat\s+should\s+i\s+worry\s+about\b",
        r"\bred\s+flags?\b",
        r"\bhidden\s+traps?\b",
        r"\bpitfalls?\b",
        r"\bmajor\s+hazards?\b",
        r"\bthreats?\b",
        r"\bis\s+this\s+(?:safe|dangerous|risky)\b",
        r"\bhigh\s+risk\b",
        r"\bmain\s+(?:liabilities|risks)\b",
        r"\bsummarize\s+(?:the\s+)?(?:main\s+)?(?:liabilities|risks)\b",
        r"^risks?$",
        r"^any\s+risks?$",
    ]
    if any(re.search(pat, q_lower) for pat in risk_patterns):
        return QueryIntent(
            intent="DOCUMENT_RISK_SUMMARY",
            confidence=0.95,
            raw_query=raw_query,
            normalized_query=norm,
            requires_risk_summary=True,
            target_clause_types=[
                "indemnity", "limitation_of_liability", "termination_notice",
                "remedies", "non_compete", "penalty", "dispute_resolution"
            ],
            expansion_terms=[
                "risk", "liability", "breach", "penalty", "indemnity", "damages",
                "default", "termination", "forfeiture", "remedies", "governing law"
            ],
            metadata={
                "source_rule": "risk_pattern_match",
                "negation_detected": negation_detected,
                "negated_conditions": negated_conditions,
            },
        )

    # 5. TERMINATION_QUERY: Legally distinct actions
    # NEVER treat resignation, dismissal, lease surrender, and general termination as interchangeable!

    # 5a. Resignation: Employee/worker voluntary departure
    resignation_patterns = [
        r"\bcan\s+i\s+resign\b",
        r"\bcan\s+i\s+quit\b",
        r"\bresign(?:ation)?\b",
        r"\bgiving\s+(?:my\s+)?notice\b",
        r"\bhanding\s+in\s+my\s+notice\b",
        r"\bstep\s+down\b",
        r"\bvoluntary\s+(?:departure|quitting|resignation|termination)\b",
        r"\bquit\b",
    ]
    if any(re.search(pat, q_lower) for pat in resignation_patterns):
        exp_terms = [
            "resignation", "resign", "voluntary termination", "notice period", "quit", "hand in notice"
        ]
        if negation_detected:
            for cond in negated_conditions:
                exp_terms.insert(0, cond)
            exp_terms.append("immediate resignation")
            exp_terms.append("without notice")
        return QueryIntent(
            intent="TERMINATION_QUERY",
            confidence=0.96,
            raw_query=raw_query,
            normalized_query=norm,
            target_clause_types=["resignation", "termination_notice", "notice_period"],
            expansion_terms=exp_terms,
            metadata={
                "source_rule": "resignation_pattern_match",
                "action_type": "resignation",
                "negation_detected": negation_detected,
                "negated_conditions": negated_conditions,
            },
        )

    # 5b. Involuntary Termination: Dismissal (employment) vs Eviction / Lease Termination (tenancy)
    involuntary_patterns = [
        r"\bfire\s+me\b",
        r"\bcan\s+they\s+fire\b",
        r"\bget\s+fired\b",
        r"\bsack(?:ed)?\b",
        r"\blet\s+go\b",
        r"\bdischarge\b",
        r"\bdismiss(?:al)?\b",
        r"\bterminated\s+by\s+(?:employer|company|client)\b",
        r"\bcan\s+(?:the\s+)?company\s+terminate\b",
        r"\bcan\s+(?:the\s+)?(?:landlord|lessor)\s+terminate\b",
        r"\bevict(?:ion)?\b",
        r"\bterminated\s+for\s+cause\b",
    ]
    if any(re.search(pat, q_lower) for pat in involuntary_patterns):
        is_tenancy = any(k in q_lower for k in ["landlord", "lessor", "tenant", "tenancy", "lease", "evict", "eviction", "premises", "rent"])
        if is_tenancy:
            exp_terms = [
                "eviction", "terminate tenancy", "notice period", "cause", "hazardous waste", "unauthorized", "lease termination"
            ]
            if negation_detected:
                for cond in negated_conditions:
                    exp_terms.insert(0, cond)
                exp_terms.append("without notice")
                exp_terms.append("without prior notice")
                exp_terms.append("without advance notice")
            return QueryIntent(
                intent="TERMINATION_QUERY",
                confidence=0.96,
                raw_query=raw_query,
                normalized_query=norm,
                target_clause_types=["termination", "termination_notice", "default"],
                expansion_terms=exp_terms,
                metadata={
                    "source_rule": "eviction_pattern_match",
                    "action_type": "eviction",
                    "is_tenancy": True,
                    "negation_detected": negation_detected,
                    "negated_conditions": negated_conditions,
                },
            )
        else:
            exp_terms = [
                "dismissal", "discharge", "termination by employer", "notice period", "cause", "summary dismissal", "severance"
            ]
            if negation_detected:
                for cond in negated_conditions:
                    exp_terms.insert(0, cond)
                exp_terms.append("without cause")
                exp_terms.append("summary dismissal")
                exp_terms.append("immediate dismissal")
            return QueryIntent(
                intent="TERMINATION_QUERY",
                confidence=0.96,
                raw_query=raw_query,
                normalized_query=norm,
                target_clause_types=["termination_notice", "dismissal", "termination"],
                expansion_terms=exp_terms,
                metadata={
                    "source_rule": "dismissal_pattern_match",
                    "action_type": "dismissal",
                    "is_employment": True,
                    "negation_detected": negation_detected,
                    "negated_conditions": negated_conditions,
                },
            )

    # 5c. Lease Surrender: Tenant early vacation or surrender of lease / premises
    surrender_patterns = [
        r"\bcan\s+i\s+vacate\s+early\b",
        r"\bsurrender\s+(?:the\s+)?lease\b",
        r"\bmove\s+out\s+early\b",
        r"\bhand\s+back\s+(?:the\s+)?keys\b",
        r"\bearly\s+vacat(?:e|ion)\b",
        r"\bsurrender\s+of\s+(?:tenancy|premises)\b",
        r"\bvacat(?:e|ing)\s+(?:the\s+)?premises\b",
        r"\bbreak\s+(?:the\s+)?lease\b",
        r"\bcan\s+i\s+move\s+out\b",
        r"\bleave\s+the\s+apartment\b",
    ]
    if any(re.search(pat, q_lower) for pat in surrender_patterns):
        exp_terms = [
            "surrender", "vacate", "hand over keys", "early termination", "premise surrender", "move out", "break lease"
        ]
        if negation_detected:
            for cond in negated_conditions:
                exp_terms.insert(0, cond)
            exp_terms.append("without penalty")
            exp_terms.append("early surrender")
        return QueryIntent(
            intent="TERMINATION_QUERY",
            confidence=0.95,
            raw_query=raw_query,
            normalized_query=norm,
            target_clause_types=["surrender", "early_termination", "vacating", "termination_notice"],
            expansion_terms=exp_terms,
            metadata={
                "source_rule": "lease_surrender_match",
                "action_type": "lease_surrender",
                "negation_detected": negation_detected,
                "negated_conditions": negated_conditions,
            },
        )

    # 5d. PENALTY_QUERY: Penalties, liquidated damages, fines, service bonds, exit fees
    # Prioritized before general termination so inquiries about penalties for leaving early target penalty clauses
    # e.g. "what are the penalties", "liquidated damages", "do I need to pay if I leave", "service bond"
    penalty_patterns = [
        r"\bpenalt(?:y|ies)\b",
        r"\bliquidated\s+damages\b",
        r"\bdo\s+i\s+(?:have|need)\s+to\s+pay\s+if\s+i\s+leave\b",
        r"\bservice\s+bond\b",
        r"\brepayment\b",
        r"\blate\s+fees?\b",
        r"\bforfeit(?:ure)?\b",
        r"\bfines?\b",
        r"\bdamages\s+for\s+breach\b",
    ]
    if any(re.search(pat, q_lower) for pat in penalty_patterns):
        exp_terms = [
            "penalty", "liquidated damages", "forfeiture", "fee", "fine",
            "damages", "bond", "repayment", "interest", "default"
        ]
        if any(k in q_lower for k in ["leave", "early", "surrender", "depart", "vacate"]):
            exp_terms.extend(["termination", "early termination", "notice period", "surrender", "leave early", "terminate"])
        return QueryIntent(
            intent="PENALTY_QUERY",
            confidence=0.94,
            raw_query=raw_query,
            normalized_query=norm,
            target_clause_types=["liquidated_damages", "penalty", "termination_notice", "remedies", "surrender"],
            expansion_terms=exp_terms,
            metadata={
                "source_rule": "penalty_pattern_match",
                "negation_detected": negation_detected,
                "negated_conditions": negated_conditions,
            },
        )

    # 5e. General Termination: Contract cancellation, mutual ending, default, early termination
    general_termination_patterns = [
        r"\bleave\s+early\b",
        r"\bterminat[a-z]*\b",
        r"\bend\s+(?:the\s+)?(?:contract|agreement|lease|employment)\b",
        r"\bcancel\s+(?:the\s+)?(?:contract|agreement|lease)\b",
        r"\bwalk\s+away\b",
        r"\bdiscontinue\s+(?:the\s+)?appointment\b",
        r"\bhow\s+to\s+terminate\b",
        r"\bearly\s+termination\b",
        r"\btermination\s+clause\b",
    ]
    if any(re.search(pat, q_lower) for pat in general_termination_patterns):
        exp_terms = [
            "terminate", "termination", "cancel", "notice period", "leave early", "cure", "cause", "convenience"
        ]
        if negation_detected:
            for cond in negated_conditions:
                exp_terms.insert(0, cond)
        return QueryIntent(
            intent="TERMINATION_QUERY",
            confidence=0.93,
            raw_query=raw_query,
            normalized_query=norm,
            target_clause_types=["termination_notice", "termination", "remedies"],
            expansion_terms=exp_terms,
            metadata={
                "source_rule": "general_termination_match",
                "action_type": "general_termination",
                "negation_detected": negation_detected,
                "negated_conditions": negated_conditions,
            },
        )

    # 7. PRICE_ESCALATION_QUERY: Rent increase, price escalation, fee adjustments, CPI/inflation adjustments
    # e.g. "can the landlord raise rent?", "can prices go up?", "rent increase", "annual increase"
    price_escalation_patterns = [
        r"\brais(?:e|ing)\s+(?:the\s+)?(?:rent|prices?|rates?|fees?)\b",
        r"\b(?:rent|price|rate|fee)\s+(?:increase|escalation|hike|adjustment)\b",
        r"\bcan\s+(?:the\s+)?(?:landlord|lessor|vendor|company|provider)\s+(?:raise|increase|hike)\b",
        r"\bcan\s+prices?\s+(?:go\s+up|increase)\b",
        r"\binflation\s+(?:adjustment|increase)\b",
        r"\bcpi\s+(?:adjustment|increase)\b",
        r"\bannual\s+(?:rent\s+)?(?:increase|escalation)\b",
        r"\brent\s+revision\b",
    ]
    if any(re.search(pat, q_lower) for pat in price_escalation_patterns):
        exp_terms = [
            "rent increase", "escalation", "raise rent", "adjustment",
            "percentage", "index", "cpi", "annual increase"
        ]
        if negation_detected:
            for cond in negated_conditions:
                exp_terms.insert(0, cond)
        return QueryIntent(
            intent="PRICE_ESCALATION_QUERY",
            confidence=0.95,
            raw_query=raw_query,
            normalized_query=norm,
            target_clause_types=["rent_payment", "payment_terms", "compensation", "escalation"],
            expansion_terms=exp_terms,
            metadata={
                "source_rule": "price_escalation_match",
                "negation_detected": negation_detected,
                "negated_conditions": negated_conditions,
            },
        )

    # 8a. SEVERANCE_QUERY: Severance compensation, redundancy pay, separation packages
    # Legally distinct from ordinary ongoing monthly rent or base salary payments
    severance_patterns = [
        r"\bseverance\b",
        r"\bredundancy\b",
        r"\bseparation\s+pay\b",
        r"\bdeparture\s+package\b",
    ]
    if any(re.search(pat, q_lower) for pat in severance_patterns):
        return QueryIntent(
            intent="SEVERANCE_QUERY",
            confidence=0.96,
            raw_query=raw_query,
            normalized_query=norm,
            target_clause_types=["severance"],
            expansion_terms=[
                "severance", "redundancy", "severance pay", "separation pay"
            ],
            metadata={
                "source_rule": "severance_pattern_match",
                "negation_detected": negation_detected,
                "negated_conditions": negated_conditions,
            },
        )

    # 8b. PAYMENT_QUERY: Financial obligations, rent, salary, fees, invoices, deposits
    # e.g. "what do I have to pay", "how much is the rent", "payment schedule", "salary", "invoicing"
    payment_patterns = [
        r"\bwhat\s+do\s+i\s+have\s+to\s+pay\b",
        r"\bhow\s+much\s+(?:is|do\s+i)\s+(?:the\s+)?(?:rent|pay|cost|fee)\b",
        r"\bpayment\s+(?:terms?|schedule|due)\b",
        r"\brent\s+payment\b",
        r"\bsecurity\s+deposit\b",
        r"\bcompensation\b",
        r"\bsalary\b",
        r"\binvoic(?:e|ing)\b",
        r"\bremuneration\b",
        r"\bfees?\b",
    ]
    if any(re.search(pat, q_lower) for pat in payment_patterns):
        return QueryIntent(
            intent="PAYMENT_QUERY",
            confidence=0.92,
            raw_query=raw_query,
            normalized_query=norm,
            target_clause_types=["rent_payment", "payment_terms", "compensation"],
            expansion_terms=[
                "pay", "payment", "rent", "fee", "cost", "deposit", "invoice",
                "monthly", "due", "advance", "compensation"
            ],
            metadata={
                "source_rule": "payment_pattern_match",
                "negation_detected": negation_detected,
                "negated_conditions": negated_conditions,
            },
        )

    # 8. DEADLINE_QUERY: Dates, notice periods, calendar durations, renewal windows
    # e.g. "when is rent due", "what is the notice period", "how many days notice", "due date"
    deadline_patterns = [
        r"\bwhen\s+is\b",
        r"\bdeadline\b",
        r"\bdue\s+date\b",
        r"\bnotice\s+period\b",
        r"\bhow\s+many\s+days\b",
        r"\brenewal\s+(?:date|window|notice)\b",
        r"\bgrace\s+period\b",
        r"\bcure\s+period\b",
        r"\bexpir(?:y|ation)\b",
        r"\btimeline\b",
    ]
    if any(re.search(pat, q_lower) for pat in deadline_patterns):
        return QueryIntent(
            intent="DEADLINE_QUERY",
            confidence=0.92,
            raw_query=raw_query,
            normalized_query=norm,
            target_clause_types=["notice_period", "renewal", "cure_period", "payment_terms"],
            expansion_terms=[
                "deadline", "notice", "days", "due", "calendar", "business days",
                "expiration", "renewal", "date", "trigger", "cure"
            ],
            metadata={
                "source_rule": "deadline_pattern_match",
                "negation_detected": negation_detected,
                "negated_conditions": negated_conditions,
            },
        )

    # 10. OBLIGATION_QUERY: Duties, responsibilities, commitments, covenants
    # e.g. "what are my obligations", "what do I have to do", "duties", "responsibilities"
    obligation_patterns = [
        r"\bwhat\s+are\s+my\s+obligations\b",
        r"\bwhat\s+do\s+i\s+have\s+to\s+do\b",
        r"\bwhat\s+am\s+i\s+required\s+to\b",
        r"\bduties\b",
        r"\bresponsibilit(?:y|ies)\b",
        r"\bcovenants?\b",
        r"\bcommitments?\b",
    ]
    if any(re.search(pat, q_lower) for pat in obligation_patterns):
        return QueryIntent(
            intent="OBLIGATION_QUERY",
            confidence=0.90,
            raw_query=raw_query,
            normalized_query=norm,
            target_clause_types=["covenants", "maintenance", "performance", "delivery"],
            expansion_terms=[
                "obligation", "duty", "shall", "must", "required", "responsible",
                "covenant", "compliance", "maintain", "perform"
            ],
            metadata={
                "source_rule": "obligation_pattern_match",
                "negation_detected": negation_detected,
                "negated_conditions": negated_conditions,
            },
        )

    # 11. RIGHTS_QUERY: Entitlements, permissions, audit rights, IP ownership
    # e.g. "what are my rights", "can I inspect", "audit rights", "ip ownership"
    rights_patterns = [
        r"\bwhat\s+are\s+my\s+rights\b",
        r"\bcan\s+i\s+(?:inspect|audit|access)\b",
        r"\baudit\s+rights?\b",
        r"\bintellectual\s+property\b",
        r"\bip\s+ownership\b",
        r"\bentitle(?:d|ment)?\b",
        r"\bownership\s+of\b",
    ]
    if any(re.search(pat, q_lower) for pat in rights_patterns):
        return QueryIntent(
            intent="RIGHTS_QUERY",
            confidence=0.90,
            raw_query=raw_query,
            normalized_query=norm,
            target_clause_types=["intellectual_property", "audit_rights", "inspection"],
            expansion_terms=[
                "right", "entitled", "intellectual property", "ownership",
                "audit", "inspection", "license", "remedy"
            ],
            metadata={
                "source_rule": "rights_pattern_match",
                "negation_detected": negation_detected,
                "negated_conditions": negated_conditions,
            },
        )

    # 12. DOCUMENT_SUMMARY: General overview or summarization of the entire document
    # e.g. "summarize the document", "give me an overview", "what is this contract about"
    summary_patterns = [
        r"\bsummar(?:y|ize)\s+(?:the\s+)?(?:document|contract|agreement|lease)\b",
        r"\boverview\s+of\s+(?:the\s+)?(?:document|contract|agreement)\b",
        r"\bwhat\s+is\s+this\s+(?:contract|agreement|document|lease)\s+about\b",
        r"\bexecutive\s+summary\b",
        r"\bbrief\s+me\b",
    ]
    if any(re.search(pat, q_lower) for pat in summary_patterns):
        return QueryIntent(
            intent="DOCUMENT_SUMMARY",
            confidence=0.91,
            raw_query=raw_query,
            normalized_query=norm,
            target_clause_types=["governing_law", "rent_payment", "termination_notice"],
            expansion_terms=[
                "agreement", "contract", "parties", "purpose", "scope",
                "effective date", "term", "summary"
            ],
            metadata={
                "source_rule": "summary_pattern_match",
                "negation_detected": negation_detected,
                "negated_conditions": negated_conditions,
            },
        )

    # 13. GENERAL_DOCUMENT_QUESTION: Fallback for all other specific inquiries
    # e.g. "what is the governing law?", "who is paying the rent?", "what is the jurisdiction?"
    return QueryIntent(
        intent="GENERAL_DOCUMENT_QUESTION",
        confidence=0.75,
        raw_query=raw_query,
        normalized_query=norm,
        target_clause_types=[],
        expansion_terms=[],
        metadata={
            "source_rule": "general_fallback",
            "negation_detected": negation_detected,
            "negated_conditions": negated_conditions,
        },
    )
