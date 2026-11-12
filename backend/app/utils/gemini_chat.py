import os
import re
import logging
from google import genai

logger = logging.getLogger(__name__)

GEMINI_CANDIDATE_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]

_genai_client = None

def get_genai_client():
    global _genai_client
    if _genai_client is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            _genai_client = genai.Client(api_key=api_key)
    return _genai_client

def ask_gemini_or_fallback(question: str, context_text: str = "") -> dict:
    """
    Elite Legal AI Copilot Integration with Grounded RAG, Gemini Multi-Model Cascade,
    and Comprehensive Local Legal Knowledge Base Fallback.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = get_genai_client() if api_key else None

    clean_markdown_rule = (
        "Respond using clean Markdown structure suitable for rendering in a professional chatbot. "
        "Use headings, paragraphs, bullet lists, numbered lists, bold emphasis, tables, and blockquotes when appropriate. "
        "Do not use decorative Markdown, excessive symbols, or unnecessary formatting. "
        "Never place asterisks around entire paragraphs. Use bold only for important labels or short phrases. "
        "Keep responses concise, structured, and readable."
    )

    if context_text and context_text.strip():
        has_doc = True
        prompt = f"""You are ClauseGuard AI, an elite corporate and legal AI copilot.
You are assisting a client in analyzing a legal agreement.

CONTEXT DOCUMENT EXCERPT:
\"\"\"
{context_text[:14000]}
\"\"\"

USER'S QUESTION:
{question}

INSTRUCTIONS:
1. Grounded Analysis: If the user's question relates to the document, base your answer primarily on the provided text. Cite relevant sections or clauses where possible.
2. Comprehensive Legal Guidance: If the document does not explicitly state an answer, clearly explain what IS and IS NOT in the text, and then provide standard legal principles, statutory protections (e.g. state tenant codes, standard employment norms, insurance industry standards), and actionable negotiation tips.
3. Structure & Formatting: {clean_markdown_rule}
4. Transparency: Provide a direct summary in the first paragraph.
"""
    else:
        has_doc = False
        prompt = f"""You are ClauseGuard AI, an elite corporate and legal AI copilot.
The user is asking a general legal, contract, or regulatory question.

USER'S QUESTION:
{question}

INSTRUCTIONS:
1. Authoritative Advice: Provide a thorough, practical, and highly informative answer covering standard contract clauses, risk exposures, legal definitions, and best practices.
2. Practical Context: Explain common legal standards for residential leases, employment offer letters, and insurance policies where applicable.
3. Structure & Formatting: {clean_markdown_rule}
4. Disclaimer: Emphasize that while this guidance offers standard legal analysis, it does not constitute formal attorney-client representation.
"""

    if client:
        for model_name in GEMINI_CANDIDATE_MODELS:
            try:
                logger.info(f"[AI Chat] Attempting Gemini inference with model: {model_name}")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and response.text:
                    answer_text = response.text.strip()
                    is_grounded = has_doc and ("not mentioned in the provided text" not in answer_text.lower())
                    
                    sources = []
                    if has_doc:
                        q_words = [w.lower() for w in re.split(r"\W+", question) if len(w) > 3]
                        sentences = [s.strip() for s in re.split(r"[.!?\n]+", context_text) if len(s.strip()) > 20]
                        matched_sents = [s for s in sentences if any(w in s.lower() for w in q_words)]
                        
                        citations = matched_sents[:2] if matched_sents else sentences[:1]
                        for idx, c in enumerate(citations):
                            sources.append({
                                "section": f"Clause Excerpt {idx + 1}",
                                "excerpt": c[:160] + "..." if len(c) > 160 else c
                            })

                    return {
                        "answer": answer_text,
                        "engine": "neural_rag",
                        "provider": "ai_copilot",
                        "model": "legal-copilot-engine",
                        "grounded": is_grounded,
                        "confidence": 0.96 if is_grounded else 0.88,
                        "sources": sources,
                        "fallbackUsed": False,
                    }
            except Exception as e:
                logger.warning(f"[Gemini API Model {model_name} failed]: {e}")
                continue

    logger.info("[AI Chat] Using enhanced local legal knowledge engine.")
    return local_heuristic_search(question, context_text)

LEGAL_KNOWLEDGE_BASE = [
    {
        "keywords": ["termination", "cancel", "vacate", "break lease", "quit"],
        "answer": "### Lease & Contract Termination Principles\n\n- **Notice Requirement:** Most jurisdictions and residential leases require written notice (typically **30 to 60 days**) prior to the expiration or intended move-out date.\n- **Early Termination Penalties:** Breaking a lease without statutory cause (such as military deployment, uninhabitable conditions, or domestic violence protections) usually incurs liquidated damages—commonly 1 to 2 months' rent plus forfeiture of security deposits.\n- **Mitigation of Damages:** In most US states, landlords have an affirmative legal duty to make reasonable efforts to re-rent the premises rather than simply charging the departing tenant for the entire remaining lease term.\n- **Action Tip:** Always send termination notices via Certified Mail with Return Receipt or tracked digital delivery as stipulated in the contract's Notice clause."
    },
    {
        "keywords": ["deposit", "security deposit", "refund", "withhold"],
        "answer": "### Security Deposit Protections & Timelines\n\n- **Statutory Return Windows:** State laws strictly mandate the return of security deposits (commonly **14 to 30 days** post-tenancy, e.g., 21 days in California, 30 days in Texas and New York).\n- **Itemized Deductions:** Landlords must provide an itemized statement detailing any deductions for damage beyond normal wear and tear, accompanied by receipts or cost estimates.\n- **Normal Wear and Tear:** Routine wall scuffs, faded paint, and carpet aging cannot legally be deducted from a tenant's deposit.\n- **Remedies for Bad Faith Withholding:** If a landlord wrongfully retains a deposit, many jurisdictions award the tenant up to **2x to 3x the deposit amount** in statutory punitive damages."
    },
    {
        "keywords": ["non-compete", "noncompete", "restrictive covenant", "competition"],
        "answer": "### Non-Compete Agreements & Enforceability\n\n- **Federal & State Trends:** The FTC and numerous states (California, Minnesota, North Dakota, Oklahoma) have enacted bans or severe limits on non-compete agreements for workers.\n- **Reasonableness Standards:** Where permitted, courts enforce non-competes only if narrowly tailored in three dimensions: **Duration** (typically 6–12 months maximum), **Geographic Scope** (limited to actual market territory), and **Scope of Activity** (only directly competing roles).\n- **Protectable Interest:** Employers must demonstrate a legitimate business interest—such as trade secrets or specialized goodwill—rather than merely suppressing labor mobility.\n- **Blue-Pencil Doctrine:** In some states, courts can modify overbroad restrictions, while other states strike down the entire agreement if any part is unreasonable."
    },
    {
        "keywords": ["at-will", "at will", "probation", "fire", "termination of employment"],
        "answer": "### At-Will Employment & Wrongful Termination\n\n- **Definition:** At-will employment means either the employer or employee may terminate the employment relationship at any time, for any lawful reason or no reason at all, with or without notice.\n- **Exceptions to At-Will:**\n  1. **Public Policy Violation:** Retaliation for whistleblowing, filing workers' compensation, or refusing to perform illegal acts.\n  2. **Statutory Discrimination:** Terminations based on race, sex, age, disability, religion, or other protected characteristics under Title VII/EEOC.\n  3. **Implied Contract:** Statements in employee handbooks promising progressive discipline before discharge can create enforceable implied contract rights in certain jurisdictions.\n- **Negotiation Tip:** Request mutual notice periods (e.g., 2 weeks or 30 days) or guaranteed severance packages upon separation without cause."
    },
    {
        "keywords": ["late fee", "grace period", "rent due", "payment"],
        "answer": "### Payment Due Dates & Late Fee Limits\n\n- **Grace Periods:** While rent is legally due on the date specified in the agreement (typically the 1st of the month), many state laws or contract terms mandate a **3 to 5-day grace period** before late penalties can be assessed.\n- **Reasonableness Caps:** Courts and statutes generally cap late fees at **5% to 8% of the monthly rent** (or a reasonable administrative cost). Unconscionable fees exceeding 10% are routinely struck down as unenforceable punitive damages.\n- **Notice of Default:** Before initiating formal summary ejectment or eviction proceedings, landlords must issue a statutory Notice to Pay or Quit (typically 3 to 10 days depending on the state)."
    },
    {
        "keywords": ["subrogation", "deductible", "loss", "insurance", "claim"],
        "answer": "### Insurance Policy Terms & Claim Rights\n\n- **Prompt Notice of Loss:** Insureds have a contractual duty to report damage or loss promptly. Failure to notify within the prescribed window (often 60 days) can provide grounds for denial if prejudice to the insurer is shown.\n- **Subrogation Clause:** Allows the insurance company to 'step into your shoes' to sue a responsible third party to recover claim payouts after indemnifying you.\n- **Replacement Cost vs. Actual Cash Value (ACV):** ACV deducts depreciation from the payout, whereas Replacement Cost covers the actual cost to repair or replace the item with new materials of like kind and quality."
    },
    {
        "keywords": ["governing law", "jurisdiction", "dispute", "arbitration"],
        "answer": "### Governing Law, Jurisdiction & Mandatory Arbitration\n\n- **Governing Law:** Establishes which state's substantive statutes and case law interpret the contract.\n- **Forum Selection:** Dictates the physical court or county where litigation must be filed.\n- **Mandatory Binding Arbitration:** Waives the right to a jury trial and class-action participation. While arbitration is often faster and private, it offers limited appeal rights and can be costly unless the drafting party covers administrative fees."
    },
]

def local_heuristic_search(question: str, doc_text: str = "") -> dict:
    """
    Intelligent local legal heuristic search engine.
    Matches against both document context and deep legal domain knowledge.
    """
    q_lower = (question or "").lower().strip()

    if re.search(r"^(hi|hello|hey|greetings|who are you|what can you do|help)\b", q_lower, re.IGNORECASE):
        return {
            "answer": "### Welcome to ClauseGuard AI Copilot\n\nI am your intelligent legal and document assistant. Here is what I can do for you:\n\n- **Document Deep-Dive:** Audit clauses, verify notice periods, analyze fee structures, and check compliance.\n- **Legal Rights & Protections:** Answer questions on residential leases, employment offer letters, and insurance policies.\n- **Risk & Obligation Warnings:** Identify unfair indemnification, aggressive non-competes, and hidden penalties.\n- **Negotiation Strategy:** Provide recommended counter-proposals and standard clause modifications.\n\n*Feel free to ask any question about your document or legal principles!*",
            "engine": "deterministic-legal-kb",
            "grounded": True,
            "confidence": 1.0,
            "sources": []
        }

    if doc_text and doc_text.strip():
        words = [w for w in re.split(r"\W+", q_lower) if len(w) > 3]
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", doc_text) if len(s.strip()) > 15]
        matches = [s for s in sentences if any(w in s.lower() for w in words)]

        if matches:
            top_matches = matches[:3]
            answer = f"### Document Findings\n\nBased on your document, the following relevant provisions were identified:\n\n"
            for idx, m in enumerate(top_matches, 1):
                answer += f"{idx}. *\"{m}\"*\n\n"
            
            for item in LEGAL_KNOWLEDGE_BASE:
                if any(kw in q_lower for kw in item["keywords"]):
                    answer += f"\n---\n{item['answer']}"
                    break

            return {
                "answer": answer.strip(),
                "engine": "deterministic-rag",
                "grounded": True,
                "confidence": 0.88,
                "sources": [
                    {"section": f"Clause Segment {i + 1}", "excerpt": m[:160]}
                    for i, m in enumerate(top_matches[:2])
                ]
            }

    for item in LEGAL_KNOWLEDGE_BASE:
        if any(kw in q_lower for kw in item["keywords"]):
            return {
                "answer": item["answer"],
                "engine": "deterministic-legal-kb",
                "grounded": False,
                "confidence": 0.85,
                "sources": []
            }

    return {
        "answer": f"### Legal Copilot Analysis: '{question.strip()}'\n\n"
                  f"While your uploaded document does not contain an explicit exact clause matching this phrase, here is the standard legal framework:\n\n"
                  f"- **Contractual Intent:** Contract interpretation prioritizes the plain meaning of words and the four corners of the agreement.\n"
                  f"- **Statutory Baselines:** State and federal laws (such as housing codes, labor regulations, and consumer protection acts) override contradictory contract provisions.\n"
                  f"- **Recommendation:** To verify your specific rights, inspect the **Clauses Tab** or **Deadlines Timeline** on this document, or ask about specific terms like *notice periods*, *penalties*, *obligations*, or *dispute resolution*.",
        "engine": "deterministic-legal-kb",
        "grounded": False,
        "confidence": 0.75,
        "sources": []
    }
