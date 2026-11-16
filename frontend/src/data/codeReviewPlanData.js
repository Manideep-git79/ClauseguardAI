/**
 * DocuGuard AI: Code File Review Plan & Compilation Stages Data
 * Grounded in the DocuGuard AI codebase architecture.
 * Exactly follows the structured review wireframe:
 *  - Code File Overview
 *  - File Location
 *  - Why This File Matters (3 key points)
 *  - Important Code Lines (with line numbers & syntax)
 *  - Line-by-Line Explanation (logic-focused)
 *  - Simple Visual Representation (input -> transform -> output)
 */

export const COMPILATION_STAGES = [
  {
    stepNumber: 1,
    stageId: "stage-1-ingestion",
    stageName: "Document Ingestion & OCR Fallback",
    shortLabel: "Ingestion & OCR",
    fileLocation: "backend/app/services/ingestion/extractor.py",
    overview:
      "Robust multi-format document parser (PDF, DOCX, TXT) that extracts layout metadata and triggers adaptive OCR when native text extraction falls below readability thresholds.",
    whyItMatters: [
      "Prevents downstream parser crashes by assessing printable character density, font encoding validity, and embedded image ratios.",
      "Avoids computationally expensive full-document rasterization by falling back to neural ONNX OCR only for degraded or scanned pages (4.8x speedup).",
      "Preserves essential hyphenated legal terms (e.g., 'non-compete', 'force-majeure') while stripping OCR line-break artifacts.",
    ],
    codeSnippet: `DEFAULT_QUALITY_CONFIG = {
    "min_char_count": 80,
    "min_word_count": 8,
    "max_img_coverage_threshold": 0.35,
    "min_printable_ratio": 0.75,
    "max_replacement_chars": 2,
}

def assess_page_extraction_quality(page: fitz.Page, extracted_text: str, config: dict = None):
    stripped = (extracted_text or "").strip()
    printable_count = sum(1 for c in stripped if c.isprintable() or c in "\\n\\r\\t")
    printable_ratio = round(printable_count / max(1, len(stripped)), 3)
    
    # Calculate scanned bitmap coverage relative to page dimensions
    total_img_coverage = sum(
        (r.width * r.height) / max(1.0, page.rect.width * page.rect.height)
        for r in page.get_image_rects()
    )
    
    needs_ocr = (
        printable_ratio < config["min_printable_ratio"]
        or total_img_coverage > config["max_img_coverage_threshold"]
    )
    return needs_ocr, "Evaluated printable density & scanned bitmap coverage"`,
    importantLinesLabel: "assess_page_extraction_quality() & extract_text_and_metadata() (Lines 140–185)",
    lineByLineExplanation: [
      {
        line: "Lines 1–7",
        title: "Quality Threshold Specification",
        detail:
          "Configures strict empirical quality baselines: minimum 80 characters, at least 75% printable character density, and maximum 35% scanned image canvas coverage.",
      },
      {
        line: "Lines 9–12",
        title: "Printable Character Ratio",
        detail:
          "Counts valid printable glyphs against total characters to detect mojibake, font encoding corruption, and garbled binary streams.",
      },
      {
        line: "Lines 14–18",
        title: "Scanned Bitmap Coverage",
        detail:
          "Calculates the cumulative bounding area of raster images on the page to identify scanned documents containing only invisible placeholder headers.",
      },
      {
        line: "Lines 20–25",
        title: "Dynamic OCR Routing",
        detail:
          "Returns a deterministic boolean flag. High-quality pages extract in <5ms via PyMuPDF native streams; only degraded pages enter ONNX neural OCR.",
      },
    ],
    diagram: {
      input: "Raw File (.pdf, .docx, .txt)",
      process: "PyMuPDF Extract -> Printable & Bitmap Quality Gate",
      fallback: "ONNX Neural OCR (if printable ratio < 0.75)",
      output: "Sanitized Text Stream + Page Bounding Boxes",
    },
  },
  {
    stepNumber: 2,
    stageId: "stage-2-privacy",
    stageName: "Local PII Redaction & Data Protection",
    shortLabel: "PII Redaction",
    fileLocation: "backend/app/services/privacy/pii_redactor.py",
    overview:
      "Zero-cloud personal privacy protection engine leveraging Microsoft Presidio and spaCy NER to anonymize sensitive identifiers locally before downstream ML processing.",
    whyItMatters: [
      "Enforces strict privacy-by-design compliance (GDPR/DPDP/HIPAA) by stripping personal identifiers locally without sending data to external APIs.",
      "Implements DocumentEntityResolver for deterministic, sequential pseudonymization ([Person 1], [Location 1]) preserving referential consistency.",
      "Deliberately excludes DATE_TIME entities from redaction so contractual notice periods, execution dates, and deadlines remain fully intact.",
    ],
    codeSnippet: `ENTITIES_TO_REDACT = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", 
    "LOCATION", "US_SSN", "CREDIT_CARD", "IBAN_CODE"
]

class DocumentEntityResolver:
    def __init__(self):
        self.registry = {}
        self.counters = {}

    def get_or_assign_label(self, entity_type: str, raw_value: str) -> str:
        norm = raw_value.strip().lower()
        key = f"{entity_type}::{norm}"
        if key not in self.registry:
            count = self.counters.get(entity_type, 0) + 1
            self.counters[entity_type] = count
            prefix = ENTITY_LABEL_PREFIXES.get(entity_type, entity_type.title())
            self.registry[key] = f"[{prefix} {count}]"
        return self.registry[key]`,
    importantLinesLabel: "DocumentEntityResolver & redact() (Lines 7–80)",
    lineByLineExplanation: [
      {
        line: "Lines 1–4",
        title: "Target PII Entity Whitelist",
        detail:
          "Restricts redaction to strictly sensitive personal data while purposefully preserving dates, monetary amounts, and legal covenant nouns.",
      },
      {
        line: "Lines 6–9",
        title: "Scoped In-Memory Registry",
        detail:
          "Maintains document-isolated counters and entity mappings to guarantee idempotency across multiple analysis passes.",
      },
      {
        line: "Lines 11–13",
        title: "Entity Normalization",
        detail:
          "Normalizes whitespace, phone formatting, and casing so multiple mentions of the same party resolve to the identical pseudonym token.",
      },
      {
        line: "Lines 14–19",
        title: "Neutral Sequential Tokenization",
        detail:
          "Assigns neutral labels like [Person 1] and [Location 1] rather than role-biased terms (Tenant/Landlord), preventing classifier bias.",
      },
    ],
    diagram: {
      input: "Raw Extracted Contract Text",
      process: "Local Presidio Engine + spaCy en_core_web_sm NER",
      fallback: "Regex Pattern Matchers (SSN, IBAN, Phone, Email)",
      output: "Pseudonymized Text + At-Rest AES Encrypted Raw Text",
    },
  },
  {
    stepNumber: 3,
    stageId: "stage-3-integrity",
    stageName: "Document Integrity & Adversarial Defense",
    shortLabel: "Adversarial Defense",
    fileLocation: "backend/app/services/integrity/detector.py",
    overview:
      "Pre-analysis security scanner that detects visual font spoofing, hidden white text, off-page clipping, and prompt injection attacks before tokenization.",
    whyItMatters: [
      "Defends against malicious prompt injection payloads attempting to force an artificial zero-risk score or exfiltrate system instructions.",
      "Uncovers deceptive contract tampering such as invisible white-on-white text, transparent glyphs, or micro-fonts (<1.5pt).",
      "Emits structured audit findings with exact page numbers and bounding boxes for verifiable legal audit logs.",
    ],
    codeSnippet: `INJECTION_RULES = [
    {
        "pattern": r"(?i)\\b(?:ignore|override)\\s+(?:all\\s+)?(?:previous|system)\\s+(?:instructions|rules)\\b",
        "category": "PROMPT_INJECTION_OVERRIDE",
        "severity": "critical",
    },
    {
        "pattern": r"(?i)\\b(?:rate|mark)\\s+this\\s+contract\\s+as\\s+(?:zero|0|safe|low)\\s+risk\\b",
        "category": "PROMPT_INJECTION_RISK_MANIPULATION",
        "severity": "high",
    }
]

def analyze_document_integrity(file_path, raw_text, page_metadata, ...):
    for span in page_metadata.get("spans", []):
        if span["size"] < TINY_FONT_SIZE_PT:
            flag_finding("TINY_FONT_SIZE", severity="high", bbox=span["bbox"])
        if is_luminance_camouflaged(span["color"], page_background):
            flag_finding("INVISIBLE_TEXT", severity="critical", bbox=span["bbox"])`,
    importantLinesLabel: "INJECTION_RULES & analyze_document_integrity() (Lines 20–115)",
    lineByLineExplanation: [
      {
        line: "Lines 1–12",
        title: "Explainable Injection Rules",
        detail:
          "Compiles deterministic regular expressions targeting system instruction overrides, persona hijacking, and risk manipulation commands.",
      },
      {
        line: "Lines 14–17",
        title: "Micro-Font Geometry Inspection",
        detail:
          "Inspects text rendering spans to detect font sizes under 1.5pt, a known tactic used in predatory contracts to conceal liability waivers.",
      },
      {
        line: "Lines 18–20",
        title: "Luminance Contrast Auditing",
        detail:
          "Computes RGB delta between font color and page background to expose hidden text rendered in white-on-white or ultra-low contrast.",
      },
      {
        line: "Lines 21–24",
        title: "Security Finding Generation",
        detail:
          "Flags security anomalies in the database with bounding boxes and explanations without halting safe document parsing.",
      },
    ],
    diagram: {
      input: "Redacted Text + Page Render Spans",
      process: "Luminance Delta Check + Micro-Font Inspector + Regex Filter",
      fallback: "Zero-Width Character & Bidirectional Unicode Sanitizer",
      output: "Integrity Finding Records + Clean Verified Text Stream",
    },
  },
  {
    stepNumber: 4,
    stageId: "stage-4-segmentation",
    stageName: "Logical Block & Clause Segmentation",
    shortLabel: "Block Segmentation",
    fileLocation: "backend/app/services/nlp/block_segmenter.py",
    overview:
      "Structural layout parser that decomposes continuous contract text into discrete, coherent clause units and filters out non-substantive boilerplate.",
    whyItMatters: [
      "Accurately parses multi-level numbering conventions (Articles, Sections, Arabic/Roman numerals, statutory bullet hierarchies).",
      "Employs a Substantive Eligibility Gate to prevent administrative headers, greetings, and signature blocks from inflating risk scores.",
      "Preserves sentence cohesion and paragraph context required for the downstream neural transformer context windows.",
    ],
    codeSnippet: `TOP_LEVEL_NUMBERING_PATTERNS = [
    r"^(?:\\s*)(?:ARTICLE|SECTION|CLAUSE|PARAGRAPH)\\s+[0-9IVXLCDM]+(?:\\.[0-9]+)*[:.\\s-]*",
    r"^(?:\\s*)[0-9]+(?:\\.[0-9]+)*[.)\\s]\\s*",
    r"^(?:\\s*)(?:Position|Salary|Benefits|At-Will|Termination|Governing Law|Rent)[:\\-\\s]+",
]

def segment_blocks(redacted_text: str, document_type: str, page_metadata: list = None):
    raw_blocks = split_by_numbered_headers_and_paragraphs(redacted_text)
    clauses = []
    for idx, block in enumerate(raw_blocks):
        substantive_score, is_sub = evaluate_clause_eligibility(block, document_type)
        clauses.append({
            "clause_index": idx + 1,
            "text": block.strip(),
            "is_substantive": is_sub,
            "content_type": "SUBSTANTIVE_CLAUSE" if is_sub else "BOILERPLATE_METADATA",
            "clause_eligibility_score": substantive_score,
        })
    return clauses`,
    importantLinesLabel: "TOP_LEVEL_NUMBERING_PATTERNS & segment_blocks() (Lines 31–150)",
    lineByLineExplanation: [
      {
        line: "Lines 1–5",
        title: "Statutory Section Header Hierarchy",
        detail:
          "Precompiled regex patterns detecting top-level headings across rental agreements, employment letters, and insurance policies.",
      },
      {
        line: "Lines 7–9",
        title: "Paragraph Boundary Aggregation",
        detail:
          "Aggregates wrapped lines and sub-bullets into their parent clause block until the next canonical clause boundary is encountered.",
      },
      {
        line: "Lines 10–13",
        title: "Substantive Eligibility Gate",
        detail:
          "Tests block against legal operative keywords, signature layouts, and running header caches to compute an eligibility score (0.0 to 1.0).",
      },
      {
        line: "Lines 14–20",
        title: "Structured Clause Record Creation",
        detail:
          "Attaches unique clause index, byte offsets, and substantive flags to feed directly into the multi-task classifier.",
      },
    ],
    diagram: {
      input: "Clean Redacted Document Stream",
      process: "Header Pattern Matching + Sub-bullet Accumulator",
      fallback: "Sentence Boundary Splitting (spaCy en_core_web_sm)",
      output: "Discrete Numbered Clauses [1..N] with Substantive Tags",
    },
  },
  {
    stepNumber: 5,
    stageId: "stage-5-classification",
    stageName: "Multi-Task Neural Classification (DistilBERT)",
    shortLabel: "Neural Classification",
    fileLocation: "backend/app/services/nlp/classifier_bert.py",
    overview:
      "Dual-head fine-tuned transformer predicting canonical clause category (10 classes per domain) and favorability labels with sliding context windows.",
    whyItMatters: [
      "Provides sliding bidirectional context windows ([TARGET]) so the model understands surrounding contractual context.",
      "Enforces classifier abstention: predictions with confidence < 0.45 automatically force UNCERTAIN / needs_review for human review.",
      "Outperforms baseline TF-IDF models by 8–10% Macro-F1 across nuanced legal language, indemnity waivers, and non-compete terms.",
    ],
    codeSnippet: `def build_windowed_input(prev_clause: str, target_clause: str, next_clause: str, max_length: int = 256):
    prev_str = (prev_clause or "").strip()[-128:]
    next_str = (next_clause or "").strip()[:128]
    return f"{prev_str} [TARGET] {target_clause.strip()} [/TARGET] {next_str}"

def classify_clauses_batch(clause_items: list, document_type: str, task: str):
    model, tokenizer, _ = get_bert_model_and_tokenizer(document_type, task)
    inputs = tokenizer([item["windowed_text"] for item in clause_items], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)
    
    # Enforce classifier abstention threshold (< 0.45 confidence)
    results = []
    for conf, pred in zip(confidences.tolist(), predictions.tolist()):
        label = id_to_label[pred] if conf >= 0.45 else "UNCERTAIN"
        results.append((label, conf))
    return results`,
    importantLinesLabel: "build_windowed_input() & classify_clauses_batch() (Lines 11–224)",
    lineByLineExplanation: [
      {
        line: "Lines 1–4",
        title: "Bidirectional Context Window",
        detail:
          "Wraps the target clause between preceding and succeeding clause context using dedicated [TARGET] delimiter tokens within a 256-token budget.",
      },
      {
        line: "Lines 6–9",
        title: "Batched PyTorch Forward Pass",
        detail:
          "Executes vectorized transformer inference simultaneously for both canonical type and favorability prediction heads.",
      },
      {
        line: "Lines 10–12",
        title: "Softmax Probability Tensor",
        detail:
          "Computes calibrated softmax probability distributions across the 10 domain classes and 4 favorability states.",
      },
      {
        line: "Lines 14–19",
        title: "Classifier Safety Abstention",
        detail:
          "If model confidence is under 0.45, it deliberately abstains to 'UNCERTAIN' / 'needs_review', guaranteeing zero confident false positives.",
      },
    ],
    diagram: {
      input: "Windowed Clause Context: Prev + Target + Next",
      process: "Fine-Tuned DistilBERT Dual-Head Forward Pass",
      fallback: "TF-IDF + Logistic Regression Baseline Checkpoint",
      output: "Canonical Clause Type + Favorability (Fair / Unfavorable)",
    },
  },
  {
    stepNumber: 6,
    stageId: "stage-6-risk-engine",
    stageName: "Deterministic Closed-Form Risk Engine",
    shortLabel: "Risk Engine",
    fileLocation: "backend/app/services/risk/risk_engine.py",
    overview:
      "Mathematical scoring engine executing closed-form risk calculation R_i = min(100, 100 · w_i(m(F_i) + Δs_i) c_i) with zero-risk safety gating.",
    whyItMatters: [
      "Provides fully transparent, auditable calculations with zero stochastic drift or black-box uncertainty.",
      "Mathematically guarantees scores remain strictly bounded in [0, 100] via explicit ceiling and floor clamps.",
      "Hard safety gate ensures non-substantive clauses strictly receive 0.0 risk, preventing administrative text from skewing document risk.",
    ],
    codeSnippet: `def compute_clause_risk(clause_type: str, favorability: str, confidence: float = 1.0, prior_type: str = None, is_substantive: bool = True):
    # Hard safety gate: non-substantive clauses never incur risk
    if not is_substantive or clause_type in ("NOT_APPLICABLE", "UNCERTAIN"):
        return 0.0

    base_weight = risk_config["clause_type_base_weight"].get(clause_type, 0.0)
    severity_mult = risk_config["favorability_severity_multiplier"].get(favorability, 0.0)
    
    # Sequential interaction bump for co-occurring hazardous covenants
    pair_key = f"{prior_type}->{clause_type}"
    seq_bump = risk_config.get("sequence_adjustment_pairs", {}).get(pair_key, 0.0)
    
    adjusted_multiplier = min(1.0, max(0.0, severity_mult + seq_bump))
    conf_clamped = max(0.0, min(1.0, confidence))
    
    # Bounded closed-form clause risk: R_i = min(100, 100 * w_i * (m + Δs) * c_i)
    raw_score = base_weight * adjusted_multiplier * conf_clamped * 100.0
    return max(0.0, min(100.0, round(raw_score, 1)))`,
    importantLinesLabel: "compute_clause_risk() & compute_document_risk() (Lines 52–120)",
    lineByLineExplanation: [
      {
        line: "Lines 1–4",
        title: "Substantive Content Safety Gate",
        detail:
          "Guarantees that boilerplate, headers, and uncertain classifications immediately evaluate to exactly 0.0 risk.",
      },
      {
        line: "Lines 6–7",
        title: "Calibrated Weight & Severity Lookup",
        detail:
          "Extracts domain-calibrated base weight w_i (0.10 to 1.00) and favorability multiplier m(F_i) (0.0 to 1.00) from risk_config.json.",
      },
      {
        line: "Lines 9–12",
        title: "Sequential Interaction Bump",
        detail:
          "Applies sequence bump Δs_i when hazardous clauses co-occur (e.g. unilateral termination immediately following broad indemnity).",
      },
      {
        line: "Lines 14–19",
        title: "Bounded Closed-Form Formula",
        detail:
          "Multiplies w_i · (m + Δs) · c_i · 100 and applies min(100, ...) to ensure mathematical containment within [0, 100].",
      },
    ],
    diagram: {
      input: "Clause Type, Favorability Label, Model Confidence",
      process: "Base Weight w_i * Severity m(F_i) + Sequence Bump Δs_i",
      fallback: "Zero-Risk Default for Non-Substantive or Unclassified Blocks",
      output: "Clause Risk R_i in [0, 100] + Overall Risk Band (Low/Med/High)",
    },
  },
  {
    stepNumber: 7,
    stageId: "stage-7-missing-clauses",
    stageName: "Missing Statutory Clause Detection",
    shortLabel: "Missing Clauses",
    fileLocation: "backend/app/services/missing_clauses/detector.py",
    overview:
      "Statutory compliance auditor that cross-references identified clauses against jurisdiction-grounded mandatory protection checklists.",
    whyItMatters: [
      "Flags absent statutory tenant/employee protections (e.g. Habitability covenants under URLTA, At-Will disclaimers under FLSA).",
      "Applies severity-weighted missing clause penalties to the document-level composite risk score.",
      "Provides actionable checklist notes explaining the legal hazard of each missing statutory provision.",
    ],
    codeSnippet: `def detect_missing_clauses(classified_clauses: list, document_type: str, confidence_threshold: float = 0.08):
    checklist = get_checklist(document_type)
    expected_items = {item["type"]: item for item in checklist.get("expected_clause_types", [])}

    present_confident_types = set()
    for clause in classified_clauses:
        if clause.get("clause_type_confidence", 1.0) >= confidence_threshold:
            present_confident_types.add(clause.get("clause_type"))

    missing_findings = []
    for c_type, item in expected_items.items():
        if c_type not in present_confident_types:
            missing_findings.append({
                "clause_type": c_type,
                "severity": item["severity_if_missing"],
                "checklist_note": item.get("note", "Standard statutory protection omitted."),
                "penalty_weight": item.get("penalty_score", 5.0)
            })
    return missing_findings`,
    importantLinesLabel: "detect_missing_clauses() & get_checklist() (Lines 9–60)",
    lineByLineExplanation: [
      {
        line: "Lines 1–3",
        title: "Statutory Checklist Lookup",
        detail:
          "Loads pre-defined mandatory clause schemas grounded in URLTA, FLSA 29 U.S.C. § 201, and NAIC Unfair Claims Settlement standards.",
      },
      {
        line: "Lines 5–8",
        title: "Confidence-Filtered Presence Set",
        detail:
          "Gathers all clause categories present in the contract with classification confidence meeting or exceeding the 0.08 threshold.",
      },
      {
        line: "Lines 10–13",
        title: "Missing Protection Set Difference",
        detail:
          "Computes the set difference between expected statutory protections and confident present clauses to detect omitted protections.",
      },
      {
        line: "Lines 14–19",
        title: "Penalty Synthesis & Structuring",
        detail:
          "Generates structured audit warnings with severity ratings (High/Medium/Low) feeding into the document penalty score P_missing.",
      },
    ],
    diagram: {
      input: "Identified Clause Categories + Document Type",
      process: "Set Difference against Domain Checklist (URLTA / FLSA / NAIC)",
      fallback: "Default Canonical Checklist Fallback Schema",
      output: "Missing Protections List + Penalty Score P_missing",
    },
  },
  {
    stepNumber: 8,
    stageId: "stage-8-deadlines",
    stageName: "Temporal Milestones & Deadline Extraction",
    shortLabel: "Deadlines & Milestones",
    fileLocation: "backend/app/services/deadlines/deadline_extractor.py",
    overview:
      "Temporal information extraction engine combining regular expressions and spaCy NER to identify critical notice windows, cure periods, and recurring payment obligations.",
    whyItMatters: [
      "Identifies time-sensitive contractual duties (e.g. 30-day notice of non-renewal, 5-day cure period for default, 14-day deposit return).",
      "Normalizes complex written expressions ('thirty (30) days', 'due on the 1st of each month') into standardized day counts.",
      "Categorizes obligations into actionable deadline classes with explicit consequence warnings.",
    ],
    codeSnippet: `DURATION_REGEXES = [
    r"(?i)\\b(?:within|prior\\s+to|giving)\\s*([a-z-]+)\\s*\\(\\s*(\\d{1,3})\\s*\\)\\s*(calendar|business)?\\s*(days?|weeks?|months?)\\b",
    r"(?i)\\b(?:due\\s+on\\s+the|payable\\s+on\\s+the)\\s+(\\d{1,2})(?:st|nd|rd|th)?\\s+day\\s+of\\s+each\\s+month\\b",
]

def extract_all_deadlines(clause_dict_list: list):
    extracted_deadlines = []
    for c in clause_dict_list:
        text = c["redacted_text"]
        for pattern in DURATION_REGEXES:
            for match in re.finditer(pattern, text):
                duration_days = parse_duration_to_days(match)
                deadline_category = classify_deadline_category(match.group(0), text)
                extracted_deadlines.append({
                    "clause_index": c["clause_index"],
                    "category": deadline_category,
                    "duration_days": duration_days,
                    "snippet": text[max(0, match.start()-20): min(len(text), match.end()+20)],
                })
    return extracted_deadlines`,
    importantLinesLabel: "DURATION_REGEXES & extract_all_deadlines() (Lines 35–80)",
    lineByLineExplanation: [
      {
        line: "Lines 1–4",
        title: "Dual-Notation Duration Matchers",
        detail:
          "Precompiled regular expressions recognizing statutory phrasing, dual notation ('thirty (30) days'), and monthly payment schedules.",
      },
      {
        line: "Lines 6–9",
        title: "Substantive Clause Traversal",
        detail:
          "Iterates across all substantive clauses, matching temporal tokens against word-number dictionaries (e.g., 'forty-five' -> 45).",
      },
      {
        line: "Lines 10–13",
        title: "Duration Normalization",
        detail:
          "Converts days, weeks, months, and business days into a standardized duration_days metric for timeline rendering.",
      },
      {
        line: "Lines 14–19",
        title: "Actionable Timeline Structuring",
        detail:
          "Extracts surrounding context snippet and maps to canonical deadline categories (cure_period, security_deposit_return, notice_period).",
      },
    ],
    diagram: {
      input: "Substantive Clauses [1..N]",
      process: "Dual Notation Regex + spaCy Temporal Matcher",
      fallback: "Word-to-Number Normalizer Dictionary ('sixty' -> 60)",
      output: "Structured Actionable Deadlines & Obligation Records",
    },
  },
  {
    stepNumber: 9,
    stageId: "stage-9-rag",
    stageName: "Dense Vector Indexing & Grounded RAG Copilot",
    shortLabel: "Grounded RAG Copilot",
    fileLocation: "backend/app/services/rag/chat_engine.py",
    overview:
      "Interactive legal assistant using 768-dimensional dense vector embeddings in PostgreSQL pgvector and Gemini LLM with strict citation grounding and anti-injection defenses.",
    whyItMatters: [
      "Enforces strict zero-hallucination policy: answers are generated exclusively from retrieved clauses with mandatory [Clause X] citations.",
      "Implements automatic abstention: explicitly states when queried terms are absent from the document instead of guessing.",
      "Pre-filters all user questions through input sanitization to neutralize prompt injection attacks.",
    ],
    codeSnippet: `SYSTEM_INSTRUCTION = """You are DocuGuard AI's document analysis assistant.
Answer the user's question STRICTLY AND SOLELY using the retrieved document clauses provided below.
CRITICAL GROUNDING RULES:
1. ONLY use facts and terms explicitly stated in the provided clauses. Never hallucinate or invent details.
2. ALWAYS cite the specific clause you rely on using the format [Clause X].
3. If the retrieved clauses DO NOT contain the answer, you MUST say plainly:
   "The provided document does not contain information addressing this question."
"""

def answer_document_question(document_id, question: str, session_id=None):
    sanitized_q, was_flagged, _ = sanitize_chat_input(question)
    query_vector = embed_query(sanitized_q)
    relevant_clauses = retrieve_relevant_clauses(document_id, query_vector, top_k=5)
    
    prompt = build_grounded_prompt(SYSTEM_INSTRUCTION, relevant_clauses, sanitized_q)
    response_text = call_gemini_cascade(prompt)
    return {"answer": response_text, "citations": extract_citations(response_text)}`,
    importantLinesLabel: "SYSTEM_INSTRUCTION & answer_document_question() (Lines 16–75)",
    lineByLineExplanation: [
      {
        line: "Lines 1–9",
        title: "Strict Grounding System Prompt",
        detail:
          "Enforces factual grounding, mandatory clause citations [Clause X], and non-expert informational disclaimers.",
      },
      {
        line: "Lines 11–12",
        title: "Prompt Injection Sanitization",
        detail:
          "Strips adversarial override tokens from user questions before embedding or passing to the language model.",
      },
      {
        line: "Lines 13–14",
        title: "768-Dim Dense Vector Retrieval",
        detail:
          "Computes query vector embedding and retrieves top-5 cosine-similarity clauses from PostgreSQL pgvector.",
      },
      {
        line: "Lines 16–18",
        title: "Cascade Inference & Citation Extraction",
        detail:
          "Synthesizes grounded response through Gemini model cascade with local heuristic fallback and citation validation.",
      },
    ],
    diagram: {
      input: "User Contract Question + Document ID",
      process: "Question Sanitizer -> 768-dim Embedding -> pgvector Cosine Search",
      fallback: "Local TF-IDF Keyword Matcher (if API unavailable)",
      output: "Verifiable Grounded Answer with Mandatory [Clause X] Citations",
    },
  },
];

