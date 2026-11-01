import re
import logging
from datetime import date, datetime, timedelta
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
import spacy

logger = logging.getLogger(__name__)

_spacy_ner = None

def get_spacy_ner():
    global _spacy_ner
    if _spacy_ner is None:
        try:
            _spacy_ner = spacy.load("en_core_web_sm")
        except Exception:
            _spacy_ner = spacy.blank("en")
    return _spacy_ner

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "twenty-one": 21, "twenty one": 21,
    "twenty-four": 24, "twenty four": 24, "twenty-five": 25, "twenty five": 25,
    "thirty": 30, "forty": 40, "forty-five": 45, "forty five": 45, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "one hundred twenty": 120, "one-hundred twenty": 120,
    "one hundred eighty": 180, "one-hundred eighty": 180,
    "three hundred sixty-five": 365, "three-hundred sixty-five": 365,
    "365": 365,
}

DURATION_REGEXES = [
    # Recurring monthly: in advance on or before the 1st day of each month, due on the 5th of every month
    r"(?i)\b(?:on\s+or\s+before\s+the|due\s+on\s+the|payable\s+on\s+the|in\s+advance\s+on\s+(?:or\s+before\s+)?the|on\s+the)\s+(\d{1,2})(?:st|nd|rd|th)?\s+(?:day\s+of\s+(?:each|every|the)\s+month|of\s+(?:each|every)\s+month)\b",
    # Dual notation: thirty (30) days, 30 (thirty) days
    r"(?i)\b(?:at\s+least|not\s+less\s+than|within|no\s+later\s+than|prior\s+to|giving|upon)?\s*([a-z-]+)\s*\(\s*(\d{1,3})\s*\)\s*(calendar|business|working)?\s*(days?|weeks?|months?|years?|hours?|hrs?)\b",
    r"(?i)\b(?:at\s+least|not\s+less\s+than|within|no\s+later\s+than|prior\s+to|giving|upon)?\s*(\d{1,3})\s*\(\s*([a-z-]+)\s*\)\s*(calendar|business|working)?\s*(days?|weeks?|months?|years?|hours?|hrs?)\b",
    # Single numeric: 30 calendar days, within 14 business days, at least 60 days
    r"(?i)\b(?:at\s+least|not\s+less\s+than|within|no\s+later\s+than|prior\s+to|giving|upon)?\s*\(?(\d{1,3})\)?[\s-]*(calendar|business|working)?\s*(days?|weeks?|months?|years?|hours?|hrs?)\b",
    # Single written: thirty days, fifteen business days
    r"(?i)\b(?:at\s+least|not\s+less\s+than|within|no\s+later\s+than|prior\s+to|giving|upon)?\s*(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fourteen|fifteen|twenty|twenty-one|twenty-four|thirty|forty-five|sixty|ninety|one\s+hundred\s+twenty|one\s+hundred\s+eighty)\s*(calendar|business|working)?\s*(days?|weeks?|months?|years?|hours?|hrs?)\b",
    # Hyphenated: 30-day, 10-day notice, 5-day grace period
    r"(?i)\b(\d{1,3})-day\b",
]

DEADLINE_TYPE_PATTERNS = [
    ("offer_acceptance_deadline", [
        r"accept\s+(?:this\s+)?offer", r"valid\s+(?:through|until)", r"acceptance\s+deadline",
        r"respond\s+by", r"offer\s+expires", r"return\s+signed\s+copy"
    ]),
    ("security_deposit_return", [
        r"security\s+deposit", r"return\s+(?:the\s+)?deposit", r"deposit\s+refund",
        r"after\s+(?:vacating|surrender|move-out|termination\s+of\s+tenancy)", r"itemized\s+deductions"
    ]),
    ("cure_period", [
        r"cure\s+such\s+default", r"remedy\s+(?:the\s+)?breach", r"notice\s+to\s+cure",
        r"pay\s+or\s+quit", r"correct\s+the\s+violation", r"cure\s+period", r"opportunity\s+to\s+cure",
        r"cure\s+the\s+delinquency"
    ]),
    ("claims_filing_deadline", [
        r"proof\s+of\s+loss", r"file\s+(?:a\s+)?claim", r"notice\s+of\s+(?:loss|claim)",
        r"report\s+(?:the\s+)?loss", r"report\s+theft", r"sworn\s+statement\s+in\s+proof"
    ]),
    ("grace_period_end", [
        r"grace\s+period", r"before\s+a\s+late\s+fee", r"grace\s+window",
        r"delinquent\s+after", r"late\s+charge\s+assessed"
    ]),
    ("payment_due", [
        r"rent\s+(?:is\s+)?(?:due|payable)", r"payment\s+due", r"payable\s+on",
        r"on\s+or\s+before\s+the", r"first\s+day\s+of\s+each\s+month", r"monthly\s+installment",
        r"recurring\s+payment", r"due\s+date", r"day\s+of\s+each\s+month"
    ]),
    ("renewal_opt_out", [
        r"auto(?:matic)?\s+renew", r"opt[- ]out", r"non-renewal", r"notice\s+of\s+non-renewal",
        r"prior\s+to\s+(?:the\s+)?expiration\s+date\s+to\s+(?:cancel|prevent|avoid|opt)"
    ]),
    ("renewal", [
        r"renew", r"renewal\s+window", r"extension", r"option\s+to\s+extend"
    ]),
    ("termination_notice", [
        r"terminat", r"vacat", r"surrender", r"quit", r"cancel",
        r"notice\s+to\s+vacate", r"move-out\s+notice", r"end\s+of\s+term",
        r"terminate\s+this\s+agreement", r"notice\s+to\s+quit"
    ]),
    ("breach_notification", [
        r"notify\s+of\s+(?:any\s+)?breach", r"notice\s+of\s+breach", r"discovering\s+(?:any\s+)?breach",
        r"written\s+notice\s+of\s+default", r"dispute\s+notice"
    ]),
    ("inspection_notice", [
        r"landlord\s+entry", r"inspect\s+the\s+premises", r"entry\s+notice",
        r"inspection\s+period", r"property\s+inspection", r"right\s+of\s+entry"
    ]),
    ("survival_period", [
        r"survive\s+(?:for\s+a\s+period\s+of|the\s+termination)", r"survival\s+period",
        r"following\s+termination\s+of\s+this\s+agreement"
    ]),
    ("commencement_date", [
        r"commence\s+on", r"effective\s+date", r"commencement\s+date", r"term\s+shall\s+begin"
    ]),
    ("expiration_date", [
        r"shall\s+expire\s+on", r"expiration\s+date", r"end\s+date", r"term\s+shall\s+end"
    ]),
    ("notice_period", [
        r"written\s+notice", r"advance\s+notice", r"prior\s+notice",
        r"calendar\s+days\s+notice", r"business\s+days\s+notice"
    ]),
]

PARTIES_PATTERN = re.compile(
    r"(?:^|[\s,;.(])(Tenant\b|Landlord\b|Employer\b|Employee\b|Company\b|Contractor\b|Client\b|Buyer\b|Seller\b|"
    r"Receiving\s+Party\b|Disclosing\s+Party\b|Borrower\b|Lender\b|Executive\b|Consultant\b|"
    r"Licensee\b|Licensor\b|Insured\b|Insurer\b|\[Person\s+\d+\])",
    re.IGNORECASE
)

ACTIONS_PATTERNS = [
    (r"provide\s+(?:written\s+)?notice(?:\s+to\s+opt[- ]out)?", "provide written notice"),
    (r"opt[- ]out(?:\s+of\s+(?:automatic\s+)?renewal)?", "opt out of renewal"),
    (r"pay(?:\s+monthly)?\s+rent", "pay rent"),
    (r"cure\s+(?:such\s+)?(?:default|breach|the\s+delinquency)", "cure default/breach"),
    (r"return\s+(?:the\s+)?(?:security\s+)?deposit", "return security deposit"),
    (r"file\s+(?:a\s+)?claim", "file claim"),
    (r"report\s+(?:the\s+)?loss", "report loss"),
    (r"vacate\s+(?:the\s+)?premises", "vacate premises"),
    (r"complete\s+(?:property\s+)?inspections?", "complete inspections"),
    (r"notify\s+(?:the\s+other\s+party\s+of\s+)?(?:any\s+)?breach", "notify of breach"),
    (r"submit\s+(?:written\s+)?(?:report|documentation|confirmation)", "submit documentation"),
    (r"terminate\s+this\s+agreement", "terminate agreement"),
    (r"accept\s+(?:this\s+)?offer", "accept offer"),
]

TRIGGER_EVENT_PATTERNS = [
    (r"\b(?:after|upon|from|following)\s+(?:the\s+)?receipt\s+of\s+(?:written\s+)?notice(?:\s+to\s+cure\s+such\s+breach)?", "receipt of written notice"),
    (r"\b(?:after|upon|from|following)\s+(?:the\s+)?receipt\s+of\s+(?:the\s+)?([^,.;]+)", "receipt of \\1"),
    (r"\b(?:after|upon|from|following)\s+([^,.;]+?\b(?:delivery|surrender|vacating|move-out|discovery|breach|default|termination|closing|execution))\b", "\\1"),
    (r"\b(?:prior\s+to|before)\s+(?:the\s+)?([^,.;]+?\b(?:expiration|expiration\s+date|renewal|termination|end\s+of\s+term|closing))\b", "prior to \\1"),
    (r"\b(?:following|after)\s+(?:the\s+)?(Effective\s+Date|Commencement\s+Date|Execution\s+Date|Closing\s+Date)\b", "\\1"),
    (r"\b(?:prior\s+to|before)\s+(?:the\s+)?(Expiration\s+Date|End\s+Date)\b", "\\1"),
]

RECURRENCE_PATTERNS = [
    (r"\b(?:first|1st)\s+day\s+of\s+each\s+month\b", "monthly on the 1st"),
    (r"\bon\s+or\s+before\s+the\s+(\d{1,2})(?:st|nd|rd|th)?\s+day\s+of\s+each\s+month\b", "monthly on the \\1th"),
    (r"\bmonthly\b", "monthly"),
    (r"\bannually|each\s+year|annual\s+basis\b", "annually"),
    (r"\bquarterly|each\s+quarter\b", "quarterly"),
]

CONDITION_PATTERNS = [
    r"\b(?:unless\s+excused\s+by\s+force\s+majeure)\b",
    r"\b(?:provided\s+that\s+[^,.;]+)\b",
    r"\b(?:unless\s+[^,.;]+)\b",
    r"\b(?:subject\s+to\s+[^,.;]+)\b",
    r"\b(?:in\s+the\s+event\s+that\s+[^,.;]+)\b",
    r"\b(?:in\s+writing\s+by\s+certified\s+mail)\b",
]

def classify_deadline_type(sentence_text: str) -> str:
    text_lower = sentence_text.lower()
    for d_type, patterns in DEADLINE_TYPE_PATTERNS:
        for p in patterns:
            if re.search(p, text_lower):
                return d_type
    return "notice_period"

def is_historical_or_example(sentence_text: str) -> bool:
    """Detects historical recitals and illustrative examples that must not become actionable deadlines."""
    s = sentence_text.strip()
    s_lower = s.lower()

    if re.search(r"^(?:whereas|recitals?|background)\b", s_lower):
        return True
    if re.search(r"\b(?:for\s+example|e\.g\.|by\s+way\s+of\s+illustration|hypothetically|illustrative)\b", s_lower):
        return True
    if re.search(r"\b(?:prior\s+agreement\s+dated|previously\s+entered\s+into\s+on|was\s+formed\s+in|was\s+paid\s+on)\b", s_lower):
        return True
    return False

def is_non_operative_definition(sentence_text: str) -> bool:
    """Detects standalone term definitions without operative obligations."""
    s = sentence_text.strip()
    if re.search(r'^["\']?[A-Z][A-Za-z\s]+["\']?\s+(?:means|shall\s+mean)\s+[^.]+?\.$', s):
        if not re.search(r"\b(?:shall|must|is\s+required\s+to|agrees\s+to|will)\s+[a-z]+", s, re.I):
            return True
    return False

def extract_document_anchor_dates(clauses: list[dict]) -> dict:
    """
    Extracts key anchor dates from document clauses:
    effective_date, commencement_date, expiration_date, execution_date.
    """
    anchors = {}

    for c in clauses:
        text = c.get("redacted_text") or c.get("text", "")
        if not text:
            continue

        # Effective date (e.g. effective date is ..., effective as of ...)
        m_eff = re.search(
            r"(?i)\b(?:effective(?:\s+date)?(?:\s+of\s+this\s+agreement)?\s+(?:is|shall\s+be|as\s+of)?|dated\s+as\s+of)\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            text
        )
        if m_eff and "effective_date" not in anchors:
            try:
                anchors["effective_date"] = date_parser.parse(m_eff.group(1), fuzzy=False).date()
            except Exception:
                pass

        # Commencement date
        m_comm = re.search(
            r"(?i)\b(?:commence(?:s|ment)?|start\s+date|term\s+shall\s+begin)\s+(?:on\s+)?([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            text
        )
        if m_comm and "commencement_date" not in anchors:
            try:
                anchors["commencement_date"] = date_parser.parse(m_comm.group(1), fuzzy=False).date()
            except Exception:
                pass

        # Expiration date
        m_exp = re.search(
            r"(?i)\b(?:expire(?:s)?|expiration\s+date|end\s+date|term\s+shall\s+end)\s+(?:on\s+)?([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            text
        )
        if m_exp and "expiration_date" not in anchors:
            try:
                anchors["expiration_date"] = date_parser.parse(m_exp.group(1), fuzzy=False).date()
            except Exception:
                pass

        # Execution date
        m_exec = re.search(
            r"(?i)\b(?:executed\s+(?:this\s+agreement\s+)?on|signed\s+on)\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            text
        )
        if m_exec and "execution_date" not in anchors:
            try:
                anchors["execution_date"] = date_parser.parse(m_exec.group(1), fuzzy=False).date()
            except Exception:
                pass

    return anchors

def extract_party_and_action(sentence_text: str) -> tuple[str | None, str | None]:
    """Extracts responsible party and required action strictly supported by clause text."""
    party = None
    action = None

    m_party = PARTIES_PATTERN.search(sentence_text)
    if m_party:
        matched = m_party.group(1).strip()
        if matched.startswith("[") and matched.endswith("]"):
            party = matched
        else:
            party = matched.title()

    for pattern, normalized in ACTIONS_PATTERNS:
        if re.search(pattern, sentence_text, re.IGNORECASE):
            action = normalized
            break

    return party, action

def extract_trigger_event(sentence_text: str) -> str | None:
    """Extracts normalized triggering event from clause text."""
    for pattern, template in TRIGGER_EVENT_PATTERNS:
        m = re.search(pattern, sentence_text, re.IGNORECASE)
        if m:
            if "\\1" in template and m.groups():
                raw_match = m.group(1).strip()
                cleaned = re.sub(r"\s+(?:and|or|by|with)\s*$", "", raw_match, flags=re.I).strip()
                if len(cleaned) >= 2:
                    return template.replace("\\1", cleaned)
            return template
    return None

def extract_recurrence(sentence_text: str) -> str | None:
    for pattern, rec_val in RECURRENCE_PATTERNS:
        m = re.search(pattern, sentence_text, re.IGNORECASE)
        if m:
            if "\\1" in rec_val:
                return rec_val.replace("\\1", m.group(1))
            return rec_val
    return None

def extract_conditions(sentence_text: str) -> str | None:
    found_conditions = []
    for p in CONDITION_PATTERNS:
        m = re.search(p, sentence_text, re.IGNORECASE)
        if m:
            found_conditions.append(m.group(0).strip())
    if found_conditions:
        return "; ".join(found_conditions)
    return None

def parse_duration_details(matched_text: str) -> dict:
    """
    Parses duration count, duration unit, day-count convention, and relative days.
    Detects numerical conflicts between word and digits (e.g. thirty (60)).
    """
    text_clean = matched_text.lower().strip()
    res = {
        "count": None,
        "unit": "days",
        "convention": "unknown",
        "relative_days": None,
        "conflict": False,
        "conflict_reason": None,
    }

    # Recurring monthly check
    if re.search(r"(?:day\s+of\s+(?:each|every|the)\s+month|of\s+(?:each|every)\s+month)", text_clean):
        m_day = re.search(r"\b(\d{1,2})\b", text_clean)
        day_val = int(m_day.group(1)) if m_day else 1
        res["count"] = day_val
        res["unit"] = "day_of_month"
        res["convention"] = "monthly_recurring"
        res["relative_days"] = day_val
        return res

    # Detect dual notation with potential conflict: thirty (60) or 30 (sixty)
    m_dual1 = re.search(r"([a-z-]+)\s*\(\s*(\d{1,3})\s*\)", text_clean)
    if m_dual1:
        word = m_dual1.group(1)
        digit = int(m_dual1.group(2))
        word_val = WORD_NUMBERS.get(word)
        if word_val is not None and word_val != digit:
            res["conflict"] = True
            res["conflict_reason"] = f"Conflicting duration numbers in text: '{word}' ({word_val}) vs '({digit})'"
            res["count"] = digit
        else:
            res["count"] = digit
    else:
        m_dual2 = re.search(r"(\d{1,3})\s*\(\s*([a-z-]+)\s*\)", text_clean)
        if m_dual2:
            digit = int(m_dual2.group(1))
            word = m_dual2.group(2)
            word_val = WORD_NUMBERS.get(word)
            if word_val is not None and word_val != digit:
                res["conflict"] = True
                res["conflict_reason"] = f"Conflicting duration numbers in text: '{digit}' vs '({word})'"
                res["count"] = digit
            else:
                res["count"] = digit

    # If count not yet parsed, check digits or words
    if res["count"] is None:
        m_num = re.search(r"\b(\d{1,3})\b", text_clean)
        if m_num:
            res["count"] = int(m_num.group(1))
        else:
            for word, val in WORD_NUMBERS.items():
                if re.search(rf"\b{word}\b", text_clean):
                    res["count"] = val
                    break

    # Determine unit and convention
    if re.search(r"\bbusiness|working\b", text_clean):
        res["unit"] = "business_days"
        res["convention"] = "business_days"
        res["relative_days"] = res["count"]
    elif re.search(r"\bcalendar\b", text_clean):
        res["unit"] = "calendar_days"
        res["convention"] = "calendar_days"
        res["relative_days"] = res["count"]
    elif re.search(r"\bweeks?\b", text_clean):
        res["unit"] = "weeks"
        res["convention"] = "calendar_weeks"
        res["relative_days"] = (res["count"] or 1) * 7
    elif re.search(r"\bmonths?\b", text_clean):
        res["unit"] = "months"
        res["convention"] = "calendar_months"
        res["relative_days"] = (res["count"] or 1) * 30
    elif re.search(r"\byears?\b", text_clean):
        res["unit"] = "years"
        res["convention"] = "calendar_years"
        res["relative_days"] = (res["count"] or 1) * 365
    elif re.search(r"\bhours?|hrs?\b", text_clean):
        res["unit"] = "hours"
        res["convention"] = "hours"
        res["relative_days"] = max(1, (res["count"] or 24) // 24)
    else:
        res["unit"] = "days"
        res["convention"] = "unknown"
        res["relative_days"] = res["count"]

    return res

def calculate_deterministic_due_date(
    trigger_date: date,
    duration_count: int,
    duration_unit: str,
    is_prior: bool = False
) -> tuple[date | None, str]:
    """
    Deterministically computes a calendar due date when the trigger date is known.
    Uses dateutil.relativedelta for accurate month/year addition handling leap years.
    """
    if not trigger_date or duration_count is None:
        return None, "Trigger date or duration unavailable"

    # Strict rule: Do not silently treat business days as calendar days!
    if duration_unit == "business_days":
        return None, "Business day calculation requires applicable jurisdiction holiday calendar and weekend rules"

    if is_prior:
        if duration_unit == "months":
            res_date = trigger_date - relativedelta(months=duration_count)
        elif duration_unit == "years":
            res_date = trigger_date - relativedelta(years=duration_count)
        elif duration_unit == "weeks":
            res_date = trigger_date - timedelta(weeks=duration_count)
        elif duration_unit in ("calendar_days", "days"):
            res_date = trigger_date - timedelta(days=duration_count)
        else:
            res_date = trigger_date - timedelta(days=duration_count)
        note = f"Calculated: {trigger_date.isoformat()} - {duration_count} {duration_unit} = {res_date.isoformat()}"
    else:
        if duration_unit == "months":
            res_date = trigger_date + relativedelta(months=duration_count)
        elif duration_unit == "years":
            res_date = trigger_date + relativedelta(years=duration_count)
        elif duration_unit == "weeks":
            res_date = trigger_date + timedelta(weeks=duration_count)
        elif duration_unit in ("calendar_days", "days"):
            res_date = trigger_date + timedelta(days=duration_count)
        else:
            res_date = trigger_date + timedelta(days=duration_count)
        note = f"Calculated: {trigger_date.isoformat()} + {duration_count} {duration_unit} = {res_date.isoformat()}"

    return res_date, note

def extract_deadlines_from_clause(
    clause_text: str,
    clause_id=None,
    clause_meta: dict | None = None,
    doc_anchors: dict | None = None
) -> list[dict]:
    """
    Extracts structured deadlines and date-bound obligations from clause text.
    Preserves exact provenance, temporal anchoring, day-count conventions, and uncertainty.
    """
    if not clause_text or not clause_text.strip():
        return []

    nlp = get_spacy_ner()
    doc = nlp(clause_text)
    deadlines = []
    seen_keys = set()
    doc_anchors = doc_anchors or {}
    meta = clause_meta or {}

    page_number = meta.get("page_number", 1)
    base_start = meta.get("start_offset", 0)

    for sent in doc.sents:
        sent_str = sent.text.strip()
        if len(sent_str) < 10:
            continue

        # 1. Filter out historical recitals, examples, and non-operative definitions
        if is_historical_or_example(sent_str):
            continue
        if is_non_operative_definition(sent_str):
            continue

        d_type = classify_deadline_type(sent_str)
        party, action = extract_party_and_action(sent_str)
        trigger_event = extract_trigger_event(sent_str)
        recurrence = extract_recurrence(sent_str)
        conditions = extract_conditions(sent_str)

        if d_type == "renewal_opt_out" and not action:
            action = "opt out of renewal"

        # 2. Relative Duration & Recurring Extraction
        for p in DURATION_REGEXES:
            for match in re.finditer(p, sent_str):
                matched_phrase = match.group(0).strip()
                dur_info = parse_duration_details(matched_phrase)
                if not dur_info["count"]:
                    continue

                days = dur_info["relative_days"]
                dur_count = dur_info["count"]
                dur_unit = dur_info["unit"]
                convention = dur_info["convention"]

                # Determine if this relative duration is 'prior to'
                is_prior = bool(
                    re.search(r"\b(?:prior\s+to|before)\b", matched_phrase, re.I) or
                    re.search(r"\b(?:prior\s+to|before)\s+(?:the\s+)?(?:expiration|end)", sent_str, re.I)
                )

                # Attempt anchor resolution
                trigger_date = None
                calc_due_date = None
                status = "UNRESOLVED_EVENT_BASED"
                review_reason = None
                calculation_note = None

                # Anchor matching against document anchors
                matched_anchor_name = None
                if trigger_event:
                    te_lower = trigger_event.lower()
                    if "effective" in te_lower and "effective_date" in doc_anchors:
                        trigger_date = doc_anchors["effective_date"]
                        matched_anchor_name = "Effective Date"
                    elif "commence" in te_lower and "commencement_date" in doc_anchors:
                        trigger_date = doc_anchors["commencement_date"]
                        matched_anchor_name = "Commencement Date"
                    elif ("expiration" in te_lower or "end of term" in te_lower) and "expiration_date" in doc_anchors:
                        trigger_date = doc_anchors["expiration_date"]
                        matched_anchor_name = "Expiration Date"
                    elif "execution" in te_lower and "execution_date" in doc_anchors:
                        trigger_date = doc_anchors["execution_date"]
                        matched_anchor_name = "Execution Date"

                if dur_info["conflict"]:
                    status = "NEEDS_REVIEW"
                    review_reason = dur_info["conflict_reason"]
                    confidence = "needs_review"
                elif trigger_date:
                    if dur_unit == "business_days":
                        status = "NEEDS_REVIEW"
                        review_reason = "Business day computation requires applicable jurisdiction holiday calendar and weekend rules"
                        confidence = "needs_review"
                        calculation_note = f"Trigger date known ({matched_anchor_name}: {trigger_date.isoformat()}), but business days cannot be converted to calendar days without local statutory rules."
                    else:
                        calc_due_date, calc_note = calculate_deterministic_due_date(
                            trigger_date=trigger_date,
                            duration_count=dur_count,
                            duration_unit=dur_unit,
                            is_prior=is_prior
                        )
                        status = "RESOLVED_CALCULATED"
                        confidence = "confident"
                        calculation_note = calc_note
                elif dur_unit == "day_of_month":
                    status = "RESOLVED_EXPLICIT"
                    confidence = "confident"
                    calculation_note = f"Recurring payment obligation: {recurrence or 'monthly on day ' + str(dur_count)}"
                else:
                    status = "UNRESOLVED_EVENT_BASED"
                    confidence = "confident" if trigger_event and d_type != "notice_period" else "needs_review"
                    if not trigger_event:
                        review_reason = "Triggering event is undefined or requires external factual occurrence"
                        calculation_note = "Relative duration without explicit triggering event anchor in text."
                    else:
                        calculation_note = f"Trigger date for '{trigger_event}' is unresolved in contract text; calendar due date cannot be determined."

                key = f"{clause_id}_{d_type}_{dur_count}_{dur_unit}_{trigger_event}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                # Normalized title
                title = d_type.replace("_", " ").title()
                if action and party:
                    title = f"{party}: {action.title()} ({dur_count} {dur_unit.replace('_', ' ')})"
                elif action:
                    title = f"{action.title()} ({dur_count} {dur_unit.replace('_', ' ')})"

                snippet = sent_str[:160] + "..." if len(sent_str) > 160 else sent_str

                deadlines.append({
                    "clause_id": clause_id,
                    "deadline_type": d_type,
                    "raw_text": f"{matched_phrase} — {snippet}",
                    "obligation_title": title,
                    "responsible_party": party,
                    "required_action": action,
                    "trigger_event": trigger_event,
                    "trigger_date": trigger_date,
                    "parsed_date": calc_due_date,
                    "relative_days": days,
                    "duration_count": dur_count,
                    "duration_unit": dur_unit,
                    "day_count_convention": convention,
                    "recurrence": recurrence,
                    "conditions": conditions,
                    "page_number": page_number,
                    "start_offset": base_start,
                    "end_offset": base_start + len(sent_str),
                    "confidence": confidence,
                    "status": status,
                    "review_reason": review_reason,
                    "calculation_note": calculation_note,
                })

        # 3. Explicit Calendar Dates Extraction
        for ent in sent.ents:
            if ent.label_ == "DATE":
                phrase = ent.text.strip()
                if len(phrase) < 5:
                    continue

                if re.search(r"^(?:days?|weeks?|months?|hours?|calendar\s+days|business\s+days)$", phrase, re.I):
                    continue

                is_calendar_candidate = bool(
                    re.search(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b", phrase, re.I)
                    or re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", phrase)
                    or re.search(r"\b202[0-9]|203[0-9]\b", phrase)
                )
                if not is_calendar_candidate:
                    continue

                parsed_dt = None
                confidence = "needs_review"
                try:
                    parsed_candidate = date_parser.parse(phrase, fuzzy=False, default=None)
                    if parsed_candidate:
                        parsed_dt = parsed_candidate.date()
                        if re.search(r"\b202[4-9]|203[0-9]\b", phrase):
                            confidence = "confident"
                except Exception:
                    parsed_dt = None

                if not parsed_dt:
                    continue

                date_key = f"{clause_id}_{d_type}_{parsed_dt}"
                if date_key in seen_keys:
                    continue
                seen_keys.add(date_key)

                snippet = sent_str[:160] + "..." if len(sent_str) > 160 else sent_str
                title = d_type.replace("_", " ").title()
                if action and party:
                    title = f"{party}: {action.title()} (by {parsed_dt.isoformat()})"
                elif action:
                    title = f"{action.title()} (by {parsed_dt.isoformat()})"

                deadlines.append({
                    "clause_id": clause_id,
                    "deadline_type": d_type,
                    "raw_text": f"{phrase} — {snippet}",
                    "obligation_title": title,
                    "responsible_party": party,
                    "required_action": action,
                    "trigger_event": "explicit_calendar_date",
                    "trigger_date": parsed_dt,
                    "parsed_date": parsed_dt,
                    "relative_days": None,
                    "duration_count": None,
                    "duration_unit": "calendar_date",
                    "day_count_convention": "calendar_date",
                    "recurrence": recurrence,
                    "conditions": conditions,
                    "page_number": page_number,
                    "start_offset": base_start,
                    "end_offset": base_start + len(sent_str),
                    "confidence": confidence,
                    "status": "RESOLVED_EXPLICIT",
                    "review_reason": None,
                    "calculation_note": f"Explicit calendar date extracted from text: {parsed_dt.isoformat()}",
                })

    return deadlines

def extract_all_deadlines(clauses: list[dict]) -> list[dict]:
    """
    Extracts deadlines across all clauses in a document.
    First resolves document-level anchors (effective date, expiration date, etc.),
    then processes each clause, deduplicating and sorting soonest-first.
    """
    all_deadlines = []
    seen = set()

    # Step 1: Detect document-level anchor dates
    doc_anchors = extract_document_anchor_dates(clauses)

    # Step 2: Extract deadlines for each clause with anchor awareness
    for c in clauses:
        c_id = c.get("id")
        c_text = c.get("redacted_text") or c.get("text", "")
        if not c_text:
            continue

        c_meta = {
            "page_number": c.get("page_number", 1),
            "start_offset": c.get("start_offset", 0),
            "end_offset": c.get("end_offset", len(c_text)),
        }

        found = extract_deadlines_from_clause(
            clause_text=c_text,
            clause_id=c_id,
            clause_meta=c_meta,
            doc_anchors=doc_anchors
        )

        for d in found:
            sig = (
                c_id,
                d.get("deadline_type"),
                d.get("duration_count"),
                d.get("duration_unit"),
                str(d.get("parsed_date")),
                d.get("trigger_event"),
            )
            if sig not in seen:
                seen.add(sig)
                all_deadlines.append(d)

    def sort_key(d):
        p_date = d.get("parsed_date")
        if p_date:
            return (0, p_date.isoformat())
        rel = d.get("relative_days")
        if rel is not None:
            return (1, f"{rel:06d}")
        return (2, "999999")

    all_deadlines.sort(key=sort_key)
    return all_deadlines
