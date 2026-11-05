import re
from typing import Dict, Any, Tuple, Optional, Set, List
from app.services.nlp.ocr_quality import assess_ocr_quality
from app.services.nlp.header_footer_detector import is_page_metadata, is_running_header_footer

# ---------------------------------------------------------------------------
# Phase 2 Standard Content Type Constants
# ---------------------------------------------------------------------------
CONTENT_TYPE_DOCUMENT_TITLE = "DOCUMENT_TITLE"
CONTENT_TYPE_HEADING = "HEADING"
CONTENT_TYPE_SUBHEADING = "SUBHEADING"
CONTENT_TYPE_SALUTATION = "SALUTATION"
CONTENT_TYPE_PARTY_METADATA = "PARTY_METADATA"
CONTENT_TYPE_PARTY_NAME = "PARTY_NAME"
CONTENT_TYPE_ADDRESS = "ADDRESS"
CONTENT_TYPE_CONTACT_METADATA = "CONTACT_METADATA"
CONTENT_TYPE_DATE_METADATA = "DATE_METADATA"
CONTENT_TYPE_DOCUMENT_METADATA = "DOCUMENT_METADATA"
CONTENT_TYPE_TABLE_OF_CONTENTS_ENTRY = "TABLE_OF_CONTENTS_ENTRY"
CONTENT_TYPE_TABLE_OF_CONTENTS = "TABLE_OF_CONTENTS"  # Backwards compatibility alias
CONTENT_TYPE_PAGE_HEADER = "PAGE_HEADER"
CONTENT_TYPE_PAGE_FOOTER = "PAGE_FOOTER"
CONTENT_TYPE_PAGE_NUMBER = "PAGE_NUMBER"
CONTENT_TYPE_PAGE_METADATA = "PAGE_METADATA"          # Backwards compatibility alias
CONTENT_TYPE_SIGNATURE_BLOCK = "SIGNATURE_BLOCK"
CONTENT_TYPE_OCR_ARTIFACT = "OCR_ARTIFACT"
CONTENT_TYPE_OCR_NOISE = "OCR_NOISE"                  # Backwards compatibility alias
CONTENT_TYPE_EMPTY_OR_WHITESPACE = "EMPTY_OR_WHITESPACE"
CONTENT_TYPE_SUBSTANTIVE_CLAUSE = "SUBSTANTIVE_CLAUSE"
CONTENT_TYPE_DEFINITION = "DEFINITION"
CONTENT_TYPE_MIXED_CONTENT = "MIXED_CONTENT"
CONTENT_TYPE_UNCERTAIN = "UNCERTAIN"
CONTENT_TYPE_FORM_FIELD = "FORM_FIELD"
CONTENT_TYPE_CORPORATE_INFORMATION = "CORPORATE_INFORMATION"
CONTENT_TYPE_PRODUCT_DESCRIPTION = "PRODUCT_DESCRIPTION"
CONTENT_TYPE_DOCUMENT_FRONT_MATTER = "DOCUMENT_FRONT_MATTER"
CONTENT_TYPE_PREAMBLE = "PREAMBLE"
CONTENT_TYPE_WITNESS_CLAUSE = "WITNESS_CLAUSE"
CONTENT_TYPE_EXTRACTION_DUPLICATE = "EXTRACTION_DUPLICATE"
CONTENT_TYPE_OTHER_NON_SUBSTANTIVE = "OTHER_NON_SUBSTANTIVE"

SUBSTANTIVE_CONTENT_TYPES: Set[str] = {
    CONTENT_TYPE_SUBSTANTIVE_CLAUSE,
    CONTENT_TYPE_DEFINITION,
}

NON_SUBSTANTIVE_CONTENT_TYPES: Set[str] = {
    CONTENT_TYPE_DOCUMENT_TITLE,
    CONTENT_TYPE_HEADING,
    CONTENT_TYPE_SUBHEADING,
    "SECTION_HEADING",
    CONTENT_TYPE_SALUTATION,
    "GREETING",
    CONTENT_TYPE_PARTY_METADATA,
    CONTENT_TYPE_PARTY_NAME,
    CONTENT_TYPE_ADDRESS,
    "ADDRESS_BLOCK",
    CONTENT_TYPE_CONTACT_METADATA,
    CONTENT_TYPE_DATE_METADATA,
    CONTENT_TYPE_DOCUMENT_METADATA,
    CONTENT_TYPE_TABLE_OF_CONTENTS_ENTRY,
    CONTENT_TYPE_TABLE_OF_CONTENTS,
    CONTENT_TYPE_PAGE_HEADER,
    "HEADER",
    CONTENT_TYPE_PAGE_FOOTER,
    "FOOTER",
    CONTENT_TYPE_PAGE_NUMBER,
    CONTENT_TYPE_PAGE_METADATA,
    CONTENT_TYPE_SIGNATURE_BLOCK,
    CONTENT_TYPE_OCR_ARTIFACT,
    CONTENT_TYPE_OCR_NOISE,
    CONTENT_TYPE_EMPTY_OR_WHITESPACE,
    CONTENT_TYPE_FORM_FIELD,
    CONTENT_TYPE_CORPORATE_INFORMATION,
    CONTENT_TYPE_PRODUCT_DESCRIPTION,
    CONTENT_TYPE_DOCUMENT_FRONT_MATTER,
    CONTENT_TYPE_PREAMBLE,
    CONTENT_TYPE_WITNESS_CLAUSE,
    CONTENT_TYPE_EXTRACTION_DUPLICATE,
    CONTENT_TYPE_OTHER_NON_SUBSTANTIVE,
}

# ---------------------------------------------------------------------------
# Operative Legal Commitments Patterns
# ---------------------------------------------------------------------------
OPERATIVE_LEGAL_PATTERNS = [
    r"\bshall\b",
    r"\bmust\b",
    r"\bagrees?\s+to\b",
    r"\bwill\s+pay\b",
    r"\bmay\s+not\b",
    r"\bis\s+entitled\s+to\b",
    r"\bis\s+non-assignable\b",
    r"\bis\s+not\s+assignable\b",
    r"\bnon-assignable\b",
    r"\bis\s+responsible\s+for\b",
    r"\bshall\s+be\s+governed\b",
    r"\bshall\s+terminate\b",
    r"\bshall\s+pay\b",
    r"\bshall\s+provide\b",
    r"\bshall\s+notify\b",
    r"\bshall\s+keep\b",
    r"\bshall\s+maintain\b",
    r"\bshall\s+not\b",
    r"\bshall\s+deposit\b",
    r"\bshall\s+refund\b",
    r"\bshall\s+cancel\b",
    r"\bunder\s+no\s+circumstances\b",
    r"\b(?:shall|agrees?\s+to|will)\s+indemnif(?:y|ies)\b",
    r"\bindemnif(?:ies|y)\b",
    r"\bindemnification\s+(?:against|for|of|obligations?)\b",
    r"\brepresents?\s+and\s+warrants?\b",
    r"\bholds?\s+harmless\b",
    r"\bliable\s+for\b",
    r"\bgrace\s+period\s*:\s*\d+",
    r"\bgrace\s+period\b",
    r"\bfree\s+look\s+period\b",
    r"\bnotice\s+period\b",
    r"\bhereby\s+(?:agrees?|acknowledges?|leases?|grants?)\b",
    r"\bin\s+the\s+event\s+(?:of|that)\b",
    r"\bsubject\s+to\s+the\s+terms\b",
    r"\bprovided\s+(?:that|however)\b",
    r"\bforfeited\b",
    r"\b(?:shall|may|will|can)\s+terminate\b",
    r"\bterminat(?:es|ed)\b",
    r"\btermination\s+(?:upon|if|for|of\s+this|with|by|fee|penalty|charge)\b",
    r"\bvalid\s+termination\b",
    r"\bright\s+to\s+terminate\b",
    r"\bnotice\s+of\s+termination\b",
    r"\bupon\s+termination\b",
    r"\bpenalty\s+of\b",
    r"\bevent\s+of\s+default\b",
    r"\bremedies\s+(?:under|available|available\s+to)\b",
    r"\bcovenants?\s+and\s+agrees?\b",
    r"\bwarrant(?:ies|y)\s+(?:that|to)\b",
    r"\bwaive(?:s|d|r)?\s+(?:any|all|the\s+right)\b",
    r"\bobligation\s+to\b",
    r"\bentitled\s+to\b",
    # Employment offer operative patterns
    r"\b(?:delighted|pleased)\s+to\s+offer\b",
    r"\boffer\s+of\s+employment\b",
    r"\bextend\s+to\s+you\s+this\s+(?:formal\s+)?offer\b",
    r"\bposition\s+of\b",
    r"\bstarting\s+(?:annual\s+)?(?:base\s+)?salary\s+(?:of|is)\b",
    r"\bbase\s+(?:salary|compensation|pay)\s+(?:will\s+be|of|is)\b",
    r"\breport\s+directly\s+to\b",
    r"\breporting\s+(?:directly\s+)?to\b",
    r"\b(?:target\s+bonus|annual\s+bonus|incentive\s+program|incentive\s+plan)\b",
    r"\brestricted\s+stock\s+units?\b",
    r"\bRSUs?\b",
    r"\binitial\s+grant\s+of\b",
    r"\bvesting\s+(?:schedule|over|period)\b",
    r"\bat-will\b",
    r"\bcontingent\s+upon\b",
    r"\bbackground\s+(?:check|screening)\b",
    r"\bemployee\s+benefit\s+plans?\b",
    r"\b401\(k\)\s+retirement\b",
    r"\bhybrid\s+work\s+policy\b",
    r"\bprimary\s+work\s+location\b",
    r"\bsemi-monthly\s+installments\b",
]

COMPILED_OPERATIVE = [re.compile(p, re.IGNORECASE) for p in OPERATIVE_LEGAL_PATTERNS]

CORPORATE_PATTERNS = [
    r"^(?:LIFE\s+INSURANCE\s+CORPORATION\s+OF\s+INDIA|LIC\s+OF\s+INDIA)$",
    r"^\(?Established\s+by\s+the\s+.*Act(?:,\s*\d{4})?\)?$",
    r"^\(?Registration\s+Number\s*[:\-]\s*\w+\)?$",
    r"^\(?(?:CIN|GSTIN|PAN|TIN|EIN|UIN|IRDAI\s+Regn?\s+No\.?)\s*[:\-]\s*[A-Z0-9/.\-]+\)?$",
    r"^(?:Registered|Corporate|Branch|Head)\s+Office\s*[:\-].*$",
    r"^\(?A\s+Government\s+of\s+India\s+Enterprise\)?$",
    r"^©\s*\d{4}\s+.*All\s+rights?\s+reserved.*$",
]
COMPILED_CORPORATE = [re.compile(p, re.IGNORECASE) for p in CORPORATE_PATTERNS]

PRODUCT_DESC_PATTERNS = [
    r"^\(?A\s+Non-Linked,\s*Non-Participating[^.)\n]*Plan\)?$",
    r"^\(?[A-Z][a-zA-Z\s,\-]+(?:Plan|Policy|Annuity|Rider|Scheme)\)?$",
    r"^Plan\s+(?:Name|Code|Number)\s*:\s*.*$",
]
COMPILED_PRODUCT_DESC = [re.compile(p, re.IGNORECASE) for p in PRODUCT_DESC_PATTERNS]

# ---------------------------------------------------------------------------
# Salutations & Greetings
# ---------------------------------------------------------------------------
GREETING_PATTERNS = [
    r"^(?:Dear\s+(?:Policyholder|Tenant|Employee|Associate|Sir|Madam|Mr\.|Ms\.|Mrs\.|Dr\.|Lessor|Lessee|Client|Customer|Colleague)[^,\n]*[,:]?)$",
    r"^(?:Dear\s+[A-Za-z\s.\-]+[,:]?)$",
    r"^(?:To\s+Whom\s+It\s+May\s+Concern\s*[,:]?)$",
    r"^(?:Greetings\s*[,:]?)$",
    r"^(?:Attention\s*[:\-].*)$",
]
COMPILED_GREETINGS = [re.compile(p, re.IGNORECASE) for p in GREETING_PATTERNS]

# ---------------------------------------------------------------------------
# Signatures & Attestation Blocks
# ---------------------------------------------------------------------------
SIGNATURE_PATTERNS = [
    r"^(?:Thanking\s+you[.,\s]*Yours\s+(?:faithfully|sincerely|truly)[.,\s]*)$",
    r"^(?:Yours\s+(?:faithfully|sincerely|truly)[.,\s]*)$",
    r"^(?:Sincerely[.,\s]*)$",
    r"^(?:Warm\s+regards[.,\s]*)$",
    r"^(?:Best\s+regards[.,\s]*)$",
    r"^(?:IN\s+WITNESS\s+WHEREOF[^,\n]*[,.]?)$",
    r"^(?:Signed,\s*sealed\s*and\s*delivered[^,\n]*[,.]?)$",
    r"^(?:p\.\s+Chief/Sr\./Branch\s+Manager|Authorized\s+Signatory|Authorized\s+Representative)$",
    r"^(?:(?:LANDLORD|TENANT|EMPLOYER|EMPLOYEE|AUTHORIZED)\s+)?SIGNATURE(?:\s+OF\s+.*)?\s*[:\-_\s]*$",
    r"^(?:Signature\s*[:\-_\s]+Date\s*[:\-_\s]+)$",
]
COMPILED_SIGNATURES = [re.compile(p, re.IGNORECASE) for p in SIGNATURE_PATTERNS]

PREAMBLE_PATTERNS = [
    r"^(?:This\s+)?(?:Master\s+Services\s+Agreement|Residential\s+Lease\s+Agreement|Employment\s+Agreement|Executive\s+Employment\s+Agreement|Non-Disclosure\s+Agreement|Mutual\s+Non-Disclosure\s+Agreement|Agreement|Lease|Contract|Policy|Letter|Addendum|Memorandum)[\s\S]*?(?:made\s+and\s+entered\s+into|by\s+and\s+between)\b",
    r"^(?:This\s+Agreement\s+is\s+(?:made\s+and\s+)?entered\s+into\s+by\s+and\s+between)\b",
    r"^PARTIES\s+AND\s+PROPERTY\s+IDENTIFICATION\b",
    r"^(?:WHEREAS|RECITALS)\b",
]
COMPILED_PREAMBLES = [re.compile(p, re.IGNORECASE) for p in PREAMBLE_PATTERNS]

WITNESS_PATTERNS = [
    r"^IN\s+WITNESS\s+WHEREOF\b",
    r"^NOW,\s*THEREFORE,\s*in\s+consideration\b",
    r"^WHEREFORE,\s*the\s*parties\b",
    r"^Signed,\s*sealed\s*and\s*delivered\b",
]
COMPILED_WITNESS = [re.compile(p, re.IGNORECASE) for p in WITNESS_PATTERNS]

# ---------------------------------------------------------------------------
# Party Metadata & Contact Patterns
# ---------------------------------------------------------------------------
PARTY_LINE_PATTERNS = [
    r"^(?:LANDLORD|TENANT|LESSOR|LESSEE|EMPLOYER|EMPLOYEE|BORROWER|LENDER|DISCLOSING\s+PARTY|RECEIVING\s+PARTY|BUYER|SELLER|CONTRACTOR|CLIENT|COMPANY|PARTY\s+[AB12]|PARTIES)\s*[:\-]\s*.+$",
    r"^(?:BETWEEN|BY\s+AND\s+BETWEEN|BETWEEN\s+THE\s+FOLLOWING\s+PARTIES|PARTIES\s+AND\s+PROPERTY\s+IDENTIFICATION)\s*[:\-]?.*$",
]
COMPILED_PARTY_LINES = [re.compile(p, re.IGNORECASE) for p in PARTY_LINE_PATTERNS]

ADDRESS_BLOCK_PATTERNS = [
    r"^\(?Address\s+and\s+e-mail\s+id\s+of\s+Branch\s+Office\)?[:\-]*$",
    r"^\(?Branch\s+Office\s+Address\)?[:\-]*$",
    r"^\(?Registered\s+Office\s+Address\)?[:\-]*$",
    r"^\(?Address\s+of\s+(?:Branch\s+Office|Grievance\s+Redressal\s+Officer|Insurance\s+Ombudsman)\)?[:\-]*$",
    r"^\(?Address\s+and\s+contact\s+details\s+of\s+.*[:\-]*$",
    r"^(?:Address|Premises\s+Address|Property\s+Address|Registered\s+Address|Office\s+Address|Branch\s+Office\s+Address)\s*[:\-]\s*.+$",
    r"^(?:Email|E-mail|Phone|Tel|Telephone|Fax|Mobile|Website|Contact(?:\s+Details|\s+Person)?)\s*[:\-]\s*.+$",
    r"^Space\s+for\s+(?:Name|Address|Signature|e-mail).*$",
]
COMPILED_ADDRESS_BLOCKS = [re.compile(p, re.IGNORECASE) for p in ADDRESS_BLOCK_PATTERNS]

DATE_PATTERNS = [
    r"^(?:Dated?|Date\s+of\s+(?:Agreement|Execution)|Effective\s+Date|Commencement\s+Date|Execution\s+Date|Expiration\s+Date)\s*[:\-]\s*.+$",
    r"^Date\s*:\s*.+$",
]
COMPILED_DATE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in DATE_PATTERNS]

ADMIN_TRANSMITTAL_PATTERNS = [
    r"^We\s+have\s+pleasure\s+in\s+forwarding\s+herewith.*$",
    r"^We\s+would\s+also\s+like\s+to\s+draw\s+your\s+kind\s+attention\s+to.*$",
    r"^Please\s+find\s+enclosed\s+(?:herewith\s+)?(?:the\s+)?policy\s+document.*$",
    r"^In\s+case\s+of\s+any\s+query\s+or\s+grievance,\s+you\s+may\s+approach.*$",
    r"^Thanking\s+you[.,\s]*$",
]
COMPILED_ADMIN_TRANSMITTAL = [re.compile(p, re.IGNORECASE) for p in ADMIN_TRANSMITTAL_PATTERNS]

FORM_FIELD_PATTERNS = [
    r"^Agent[’']s/(?:Intermediary[’']s)?\s*(?:Code|Name|Mobile|Number|License).*$",
    r"^Space\s+for\s+(?:Name|Address|Signature|e-mail).*$",
    r"^Ref\s*:\s*.*\(Address\s+and\s+e-mail\s+id\s+of\s+Branch\s+Office\)\s*[:\-]*$",
    r"^Ref\s*:\s*\w+.*$",
    r"^Date\s*:\s*.*$",
    r"^Re\s*:\s*(?:Your\s+)?Policy\s+No\.?.*$",
    r"^Address\s+of\s+(?:Grievance\s+Redressal\s+Officer|Insurance\s+Ombudsman)\s*[:\-]*$",
    r"^Address\s+and\s+contact\s+details\s+of\s+.*[:\-]*$",
    r"^_{4,}$",  # Blank fill-in-the-blank line
    r"^[_\s]{6,}$",
]
COMPILED_FORM_FIELDS = [re.compile(p, re.IGNORECASE) for p in FORM_FIELD_PATTERNS]

DOCUMENT_TITLE_PATTERNS = [
    r"^(?:Sample\s+Policy\s+Document\s*[-–—]\s*)?LIC[’']S\s+NEW\s+JEEVAN\s+SHANTI(?:\s*\(UIN:\s*[A-Z0-9]+\))?$",
    r"^(?:RESIDENTIAL\s+LEASE\s+AGREEMENT|LEASE\s+AGREEMENT|RENTAL\s+AGREEMENT|STANDARD\s+LEASE\s+AGREEMENT)$",
    r"^(?:COMMERCIAL\s+LEASE\s+AGREEMENT|OFFICE\s+LEASE\s+AGREEMENT)$",
    r"^(?:EMPLOYMENT\s+AGREEMENT|EXECUTIVE\s+EMPLOYMENT\s+AGREEMENT|JOB\s+OFFER\s+LETTER|OFFER\s+OF\s+EMPLOYMENT|CONFIDENTIAL\s+OFFER\s+LETTER)$",
    r"^(?:NON-DISCLOSURE\s+AGREEMENT|MUTUAL\s+NON-DISCLOSURE\s+AGREEMENT|CONFIDENTIALITY\s+AGREEMENT|NDA)$",
    r"^(?:MASTER\s+SERVICES\s+AGREEMENT|SERVICES\s+AGREEMENT|CONSULTING\s+AGREEMENT)$",
    r"^(?:HOMEOWNERS\s+POLICY|HO-4\s+POLICY|RENTERS\s+INSURANCE\s+POLICY|INSURANCE\s+POLICY)$",
    r"^(?:SAMPLE\s+POLICY\s+DOCUMENT.*)$",
    r"^(?:THIS\s+AGREEMENT)$",
    r"^(?:CONFIDENTIAL(?:\s+AND\s+PROPRIETARY)?)$",
]
COMPILED_DOC_TITLES = [re.compile(p, re.IGNORECASE) for p in DOCUMENT_TITLE_PATTERNS]

TOC_HEADING_PATTERNS = [
    r"^(?:TABLE\s+OF\s+CONTENTS|CONTENTS|INDEX|TABLE\s+DES\s+MATI[EÈ]RES)$",
    r"^(?:TABLE\s+OF\s+CONTENTS\s*[-–—:]?.*)$",
]
COMPILED_TOC_HEADINGS = [re.compile(p, re.IGNORECASE) for p in TOC_HEADING_PATTERNS]

TOC_ENTRY_PATTERNS = [
    r"^.+?(?:\.{2,}|[·•]{2,}|\s{3,}\.{2,}|\s*[-–—_]{3,})\s*(?:page\s*)?(?:[0-9]+|[ivxlcdm]+)\s*$",
    r"^.+?\s+(?:—|-|–)\s*Page\s+(?:[0-9]+|[ivxlcdm]+)\s*$",
    r"^(?:(?:Article|Section|Part|Clause|\d+\.|\([0-9]+\))\s+.+?)\s{4,}(?:page\s*)?(?:[0-9]+|[ivxlcdm]+)\s*$",
    r"^.+?\s*\((?:page\s*)?(?:[0-9]+|[ivxlcdm]+)\)\s*$",
]
COMPILED_TOC_ENTRIES = [re.compile(p, re.IGNORECASE) for p in TOC_ENTRY_PATTERNS]

# ---------------------------------------------------------------------------
# Section Headings & Canonical Legal Headings
# ---------------------------------------------------------------------------
COMMON_HEADING_KEYWORDS: Set[str] = {
    "TERMINATION", "INDEMNIFICATION", "INDEMNITY", "RECITALS", "TERMS AND CONDITIONS",
    "GENERAL TERMS AND CONDITIONS", "TERMS & CONDITIONS",
    "TABLE OF CONTENTS", "RENT AND PAYMENT", "RENT AND SECURITY DEPOSIT",
    "RENT", "PAYMENT", "PAYMENT TERMS", "GOVERNING LAW", "GOVERNING LAW AND JURISDICTION",
    "NOTICES", "DEFINITIONS", "DEFINITIONS AND INTERPRETATION", "CONFIDENTIALITY",
    "NON-DISCLOSURE", "SEVERABILITY", "REMEDIES", "DEFAULT", "EVENTS OF DEFAULT",
    "COVENANTS", "MISCELLANEOUS", "GENERAL PROVISIONS", "FORCE MAJEURE",
    "LIMITATION OF LIABILITY", "LIMITS OF LIABILITY", "ENTIRE AGREEMENT",
    "DISPUTE RESOLUTION", "ARBITRATION", "REPRESENTATIONS AND WARRANTIES",
    "SECURITY DEPOSIT", "MAINTENANCE AND REPAIRS", "MAINTENANCE", "REPAIRS",
    "ALTERATIONS", "UTILITIES", "ASSIGNMENT AND SUBLETTING", "ASSIGNMENT",
    "SUBLETTING", "SURRENDER OF PREMISES", "RULES AND REGULATIONS", "INSURANCE",
    "TAXES", "WAIVER", "AMENDMENT", "HEADINGS", "COUNTERPARTS", "EXECUTION",
    "BACKGROUND", "PREAMBLE", "WITNESSETH", "NOW THEREFORE", "CLOSING CONDITIONS",
    "CONDITIONS PRECEDENT", "COMPENSATION AND BENEFITS", "DUTIES AND RESPONSIBILITIES",
    "AT-WILL EMPLOYMENT", "NON-COMPETITION", "NON-SOLICITATION", "INTELLECTUAL PROPERTY",
    "OWNERSHIP OF WORK PRODUCT", "SCHEDULE", "EXHIBIT", "ANNEXURE", "APPENDIX",
}

STANDALONE_SECTION_HEADING = [
    r"^(?:PART\s*[-–—]\s*[A-G](?:[:\-\s]+[A-Z0-9\s,\-]+)?)$",
    r"^(?:SECTION\s+[0-9IVXLCDM]+(?:\.[0-9]+)*(?:[:\-\s]+[A-Z0-9\s,\-]+)?)$",
    r"^(?:ARTICLE\s+[0-9IVXLCDM]+(?:\.[0-9]+)*(?:[:\-\s]+[A-Z0-9\s,\-]+)?)$",
    r"^(?:SCHEDULE(?:\s*[-–—]\s*[A-Z0-9]+)?(?:[:\-\s]+[A-Z0-9\s,\-]+)?)$",
    r"^(?:ANNEXURE\s*[-–—]\s*[I|V|X|0-9]+(?:[:\-\s]+[A-Z0-9\s,\-]+)?)$",
    r"^(?:GENERAL\s+PROVISIONS|MISCELLANEOUS|DEFINITIONS|RECITALS)$",
]
COMPILED_SECTION_HEADINGS = [re.compile(p, re.IGNORECASE) for p in STANDALONE_SECTION_HEADING]

# ---------------------------------------------------------------------------
# Operative Definition Patterns (Phase 2 & Test F)
# ---------------------------------------------------------------------------
DEFINITION_PATTERN = re.compile(
    r'^(?:(?:As\s+used\s+in\s+this\s+Agreement,\s*)?(?:["\'“‘][A-Za-z0-9\s,\-_]+["\'”’]|[A-Z][a-zA-Z0-9\-_]+)\s+(?:means|shall\s+mean|refers\s+to|includes|has\s+the\s+meaning)\b.*)$',
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Structural Role Identification Functions
# ---------------------------------------------------------------------------
def is_preamble_block(text: str) -> bool:
    if not text:
        return False
    raw = text.strip()
    return any(p.search(raw) for p in COMPILED_PREAMBLES)


def is_witness_block(text: str) -> bool:
    if not text:
        return False
    raw = text.strip()
    return any(p.search(raw) for p in COMPILED_WITNESS)


def is_signature_block(text: str) -> bool:
    if not text:
        return False
    raw = text.strip()
    for p in COMPILED_SIGNATURES:
        if p.match(raw) or p.search(raw):
            return True
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    sig_line_indicators = sum(
        1 for ln in lines
        if re.search(r"^(?:By|Title|Date|Signature|Printed\s+Name)\s*:\s*", ln, re.IGNORECASE)
        or re.search(r"\b(?:Chief\s+Executive\s+Officer|Managing\s+Director|Authorized\s+Signatory|Authorized\s+Representative)\b", ln, re.IGNORECASE)
        or re.search(r"\b(?:SIGNATURE\s+OF\s+.*|LANDLORD\s+SIGNATURE|TENANT\s+SIGNATURE)\b", ln, re.IGNORECASE)
    )
    if sig_line_indicators >= 1 and (len(lines) <= 6 or sig_line_indicators / len(lines) >= 0.3):
        return True
    return False


def is_salutation(text: str) -> bool:
    if not text:
        return False
    raw = text.strip()
    return any(p.match(raw) for p in COMPILED_GREETINGS)


def is_party_metadata(text: str) -> bool:
    if not text:
        return False
    raw = text.strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return False
    all_party_patterns = COMPILED_PARTY_LINES + COMPILED_ADDRESS_BLOCKS
    matches = sum(1 for ln in lines if any(p.match(ln) for p in all_party_patterns))
    return matches >= 1 and (matches / len(lines) >= 0.5)


def is_date_metadata(text: str) -> bool:
    if not text:
        return False
    raw = text.strip()
    return any(p.match(raw) for p in COMPILED_DATE_PATTERNS)


def is_address_metadata(text: str) -> bool:
    if not text:
        return False
    raw = text.strip()
    return any(p.match(raw) for p in COMPILED_ADDRESS_BLOCKS)


def is_definition_clause(text: str) -> bool:
    if not text:
        return False
    return bool(DEFINITION_PATTERN.match(text.strip()))


def is_document_title(text: str) -> bool:
    if not text:
        return False
    raw = text.strip()
    for p in COMPILED_DOC_TITLES:
        if p.match(raw):
            return True
    # Generalized uppercase title banner without terminal sentence punctuation
    words = raw.split()
    if 2 <= len(words) <= 8 and raw.isupper() and not re.search(r"[.?!]$", raw):
        if any(term in raw for term in ("AGREEMENT", "CONTRACT", "POLICY", "LEASE", "DEED", "ADDENDUM", "AMENDMENT", "MEMORANDUM")):
            return True
    return False


def is_heading_or_subheading(text: str) -> bool:
    """
    Robustly identifies headings and subheadings without legal obligations.
    Checks section numbering, keywords, case patterns, and absence of sentence structure.
    """
    if not text:
        return False
    raw = text.strip()
    if not raw:
        return False

    # A genuine clause ends with sentence punctuation; headings do not (unless it is a section number like '1.')
    if re.search(r"[.?!]$", raw) and not re.search(r"^\d+\.$", raw):
        return False

    clean = re.sub(r"[:\-–—]$", "", raw).strip()
    # Strip leading section numbering (e.g., '1. ', 'Section 2: ', 'Article III - ', '§ 4. ')
    clean_no_num = re.sub(
        r"^(?:(?:SECTION|ARTICLE|CLAUSE|PART|PARAGRAPH|[§\u00A7])\s*[0-9IVXLCDM]+(?:\.[0-9]+)*[:.\s\-–—]*|[0-9]+(?:\.[0-9]+)*[.)\s]\s*|(?:\([a-zA-Z0-9]+\)|[a-zA-Z][.)]\s+|[ivxlcdm]+[.)]\s+))",
        "", clean, flags=re.IGNORECASE
    ).strip()

    words = clean.split()
    if len(words) > 10:
        return False

    # Check known heading keywords
    if clean_no_num.upper() in COMMON_HEADING_KEYWORDS:
        return True

    # Standalone uppercase short phrase
    if clean.isupper() and len(words) <= 7:
        return True

    # Explicit section heading numbering
    if re.match(r"^(?:(?:SECTION|ARTICLE|CLAUSE|PART|PARAGRAPH|[§\u00A7])\s*[0-9IVXLCDM]+(?:\.[0-9]+)*[:.\s\-–—]*|[0-9]+(?:\.[0-9]+)*[.)\s]\s*)[A-Za-z0-9\s,\-–—/&'’\"()]{1,60}$", clean, re.IGNORECASE):
        # Ensure not an operative commitment
        if not re.search(r"\b(?:shall\s+pay|must\s+pay|agrees?\s+to|hereby\s+leases?)\b", clean, re.IGNORECASE):
            return True

    # Check compiled section headings patterns
    for p in COMPILED_SECTION_HEADINGS:
        if p.match(clean) or p.match(raw):
            return True

    return False


def is_table_of_contents_entry(text: str) -> bool:
    """
    Checks if a line or block represents a table-of-contents heading or entry with page reference.
    """
    if not text:
        return False
    s = text.strip()
    for p in COMPILED_TOC_HEADINGS:
        if p.match(s):
            return True
    has_op = has_operative_legal_language(s)
    for p in COMPILED_TOC_ENTRIES:
        if p.match(s):
            if not has_op or re.search(r"(?:\.{3,}|[-–—]{3,})\s*(?:page\s*)?(?:[0-9]+|[ivxlcdm]+)\s*$", s, re.IGNORECASE):
                return True
    return False


def is_table_of_contents_block(text: str) -> bool:
    """
    Checks if a multi-line block constitutes a Table of Contents section.
    """
    if not text:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    if any(p.match(lines[0]) for p in COMPILED_TOC_HEADINGS):
        return True
    toc_matches = sum(1 for ln in lines if is_table_of_contents_entry(ln))
    if len(lines) >= 2 and (toc_matches / len(lines)) >= 0.5:
        return True
    if toc_matches >= 2:
        return True
    return False


def has_operative_legal_language(text: str) -> bool:
    """
    Checks if text contains strong operative contractual/legal language.
    Guarantees that headings, titles, salutations, and party metadata are NOT treated as operative language.
    """
    if not text or not text.strip():
        return False

    raw = text.strip()

    # Priority Gate: Structural text can NEVER be operative contractual language
    if is_heading_or_subheading(raw) or is_document_title(raw) or is_salutation(raw) or is_party_metadata(raw):
        return False

    for pattern in COMPILED_OPERATIVE:
        if pattern.search(raw):
            return True
    return False


def calculate_clause_eligibility_score(text: str) -> float:
    """
    Computes a continuous clause eligibility score from 0.0 to 1.0.
    Considers sentence structure, modal verbs, legal terminology, and length.
    """
    if not text or not text.strip():
        return 0.0

    raw = text.strip()

    # Structural blocks have 0.0 eligibility score
    if is_heading_or_subheading(raw) or is_document_title(raw) or is_salutation(raw) or is_party_metadata(raw):
        return 0.0

    words = raw.split()
    word_count = len(words)

    # 1. Operative verb presence (highest weight)
    operative_matches = sum(1 for p in COMPILED_OPERATIVE if p.search(raw))
    operative_factor = min(1.0, operative_matches * 0.4)

    # 2. Complete sentence structure (starts with Capital, ends with punctuation)
    has_sentence_structure = 0.2 if re.match(r"^[A-Z0-9\"'\(].*[.?!;:]$", raw, re.DOTALL) else 0.0

    # 3. Reasonable clause word count
    length_factor = 0.2 if 4 <= word_count <= 250 else (0.1 if word_count > 250 else 0.05)

    # 4. Absence of form lines/blank underscores
    form_penalty = -0.5 if "____" in raw else 0.0

    score = operative_factor + has_sentence_structure + length_factor + form_penalty
    return max(0.0, min(1.0, round(score, 3)))


# ---------------------------------------------------------------------------
# Authoritative Eligibility Gate (Phase 3)
# ---------------------------------------------------------------------------
def classify_content_type(
    text: str,
    document_type: str = "rental_agreement",
    document_repeated_headers: Optional[set] = None,
    page_number: Optional[int] = None,
    is_duplicate: bool = False,
) -> Dict[str, Any]:
    """
    Authoritative single eligibility gate:
    Classifies an extracted document block into substantive or non-substantive role.
    
    For any excluded non-substantive block, enforces:
    - is_substantive = False
    - classification = "NOT_APPLICABLE"
    - risk_score = 0.0
    - human_review = False
    - clause_eligibility_score = 0.0
    - filter_reason = <specific auditable reason>
    """
    raw = text.strip() if text else ""

    # 1. Empty / Whitespace
    if not raw:
        return {
            "content_type": CONTENT_TYPE_EMPTY_OR_WHITESPACE,
            "is_substantive": False,
            "classification": "NOT_APPLICABLE",
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.0,
            "filter_reason": "Empty text block"
        }

    # 2. Duplicate Extraction Artifact
    if is_duplicate:
        return {
            "content_type": CONTENT_TYPE_EXTRACTION_DUPLICATE,
            "is_substantive": False,
            "classification": "NOT_APPLICABLE",
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.0,
            "filter_reason": "Duplicate extraction artifact"
        }

    # 3. OCR Corruption / Noise Artifact
    ocr_result = assess_ocr_quality(raw)
    if ocr_result["is_ocr_noise"]:
        return {
            "content_type": CONTENT_TYPE_OCR_NOISE,
            "is_substantive": False,
            "classification": "NOT_APPLICABLE",
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.0,
            "filter_reason": ocr_result["reason"] or "OCR corruption detected"
        }

    # 4. Page Number & Pagination Metadata
    if is_page_metadata(raw):
        return {
            "content_type": CONTENT_TYPE_PAGE_METADATA,
            "is_substantive": False,
            "classification": "NOT_APPLICABLE",
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.0,
            "filter_reason": "Page number or pagination metadata"
        }

    # 5. Running Headers / Footers
    if is_running_header_footer(raw) or (document_repeated_headers and raw in document_repeated_headers):
        is_hdr = (page_number and page_number <= 2) or ("header" in raw.lower())
        return {
            "content_type": CONTENT_TYPE_PAGE_HEADER if is_hdr else CONTENT_TYPE_PAGE_FOOTER,
            "is_substantive": False,
            "classification": "NOT_APPLICABLE",
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.0,
            "filter_reason": "Running header or footer text"
        }

    # 6. Table of Contents Detection
    if is_table_of_contents_block(raw) or is_table_of_contents_entry(raw):
        return {
            "content_type": CONTENT_TYPE_TABLE_OF_CONTENTS,
            "is_substantive": False,
            "classification": "NOT_APPLICABLE",
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.0,
            "filter_reason": "Table of contents entry or section outline"
        }

    # 7. Salutations & Correspondence Greetings
    if is_salutation(raw):
        return {
            "content_type": "GREETING",  # Preserves existing label convention
            "is_substantive": False,
            "classification": "NOT_APPLICABLE",
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.0,
            "filter_reason": "Correspondence greeting or salutation"
        }

    # 8. Date Metadata
    if is_date_metadata(raw):
        return {
            "content_type": CONTENT_TYPE_DATE_METADATA,
            "is_substantive": False,
            "classification": "NOT_APPLICABLE",
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.0,
            "filter_reason": "Document date metadata"
        }

    # 9. Address Blocks & Contact Placeholders
    if is_address_metadata(raw):
        return {
            "content_type": "ADDRESS_BLOCK",
            "is_substantive": False,
            "classification": "NOT_APPLICABLE",
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.0,
            "filter_reason": "Branch or office address block metadata"
        }

    # 10. Party Metadata & Contracting Parties
    if is_party_metadata(raw) or raw.upper() in (
        "LANDLORD", "TENANT", "LESSOR", "LESSEE", "EMPLOYER", "EMPLOYEE",
        "BUYER", "SELLER", "BORROWER", "LENDER", "CLIENT", "CONTRACTOR",
        "COMPANY", "PARTY A", "PARTY B", "DISCLOSING PARTY", "RECEIVING PARTY"
    ):
        return {
            "content_type": CONTENT_TYPE_PARTY_METADATA,
            "is_substantive": False,
            "classification": "NOT_APPLICABLE",
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.0,
            "filter_reason": "Contracting party identification metadata"
        }

    # 11. Form Fields & Blank Input Lines
    for pattern in COMPILED_FORM_FIELDS:
        if pattern.match(raw):
            return {
                "content_type": CONTENT_TYPE_FORM_FIELD,
                "is_substantive": False,
                "classification": "NOT_APPLICABLE",
                "risk_score": 0.0,
                "human_review": False,
                "clause_eligibility_score": 0.0,
                "filter_reason": "Administrative form field or placeholder"
            }

    # 12. Corporate Information & Statutory Registration
    for pattern in COMPILED_CORPORATE:
        if pattern.match(raw):
            return {
                "content_type": CONTENT_TYPE_CORPORATE_INFORMATION,
                "is_substantive": False,
                "classification": "NOT_APPLICABLE",
                "risk_score": 0.0,
                "human_review": False,
                "clause_eligibility_score": 0.0,
                "filter_reason": "Corporate identity or statutory registration metadata"
            }

    # 13. Product Description / Marketing Plan
    for pattern in COMPILED_PRODUCT_DESC:
        if pattern.match(raw) and not has_operative_legal_language(raw):
            return {
                "content_type": CONTENT_TYPE_PRODUCT_DESCRIPTION,
                "is_substantive": False,
                "classification": "NOT_APPLICABLE",
                "risk_score": 0.0,
                "human_review": False,
                "clause_eligibility_score": 0.0,
                "filter_reason": "Product plan identification or marketing description"
            }

    # 14. Administrative Transmittal Boilerplate
    for pattern in COMPILED_ADMIN_TRANSMITTAL:
        if pattern.match(raw) and not has_operative_legal_language(raw):
            return {
                "content_type": CONTENT_TYPE_DOCUMENT_FRONT_MATTER,
                "is_substantive": False,
                "classification": "NOT_APPLICABLE",
                "risk_score": 0.0,
                "human_review": False,
                "clause_eligibility_score": 0.0,
                "filter_reason": "Administrative correspondence / transmittal boilerplate"
            }

    # 15. Contract Preambles & Opening Recitals
    if is_preamble_block(raw) and not has_operative_legal_language(raw):
        return {
            "content_type": CONTENT_TYPE_PREAMBLE,
            "is_substantive": False,
            "classification": "NOT_APPLICABLE",
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.0,
            "filter_reason": "Contract preamble or opening party definition"
        }

    # 16. Witness and Execution Attestation
    if is_witness_block(raw) and not has_operative_legal_language(raw):
        return {
            "content_type": CONTENT_TYPE_WITNESS_CLAUSE,
            "is_substantive": False,
            "classification": "NOT_APPLICABLE",
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.0,
            "filter_reason": "Contract execution witness or attestation clause"
        }

    # 17. Signature Blocks & Closings
    if is_signature_block(raw):
        return {
            "content_type": CONTENT_TYPE_SIGNATURE_BLOCK,
            "is_substantive": False,
            "classification": "NOT_APPLICABLE",
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.0,
            "filter_reason": "Correspondence closing or signature block"
        }

    # 18. Document Titles
    if is_document_title(raw) and not has_operative_legal_language(raw):
        return {
            "content_type": CONTENT_TYPE_DOCUMENT_TITLE,
            "is_substantive": False,
            "classification": "NOT_APPLICABLE",
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.0,
            "filter_reason": "Document title banner"
        }

    # 19. Standalone Headings & Subheadings (e.g. 'TERMINATION', '1. DEFINITIONS', 'RENT AND PAYMENT')
    if is_heading_or_subheading(raw):
        return {
            "content_type": "SECTION_HEADING",  # Preserves existing label convention
            "is_substantive": False,
            "classification": "NOT_APPLICABLE",
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.0,
            "filter_reason": "Section heading without operative legal text"
        }

    # 20. Multi-line Document Front Matter Block
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) >= 3 and not has_operative_legal_language(raw):
        front_matter_line_count = 0
        for ln in lines:
            if (
                any(p.match(ln) for p in COMPILED_DOC_TITLES)
                or any(p.match(ln) for p in COMPILED_CORPORATE)
                or any(p.match(ln) for p in COMPILED_PRODUCT_DESC)
                or is_page_metadata(ln)
                or is_heading_or_subheading(ln)
                or is_party_metadata(ln)
                or is_date_metadata(ln)
                or "UIN:" in ln
                or "Registration Number" in ln
                or "Page 1 of" in ln
            ):
                front_matter_line_count += 1
        if front_matter_line_count >= 2 and front_matter_line_count / len(lines) >= 0.5:
            return {
                "content_type": CONTENT_TYPE_DOCUMENT_FRONT_MATTER,
                "is_substantive": False,
                "classification": "NOT_APPLICABLE",
                "risk_score": 0.0,
                "human_review": False,
                "clause_eligibility_score": 0.0,
                "filter_reason": "Document front matter / cover page metadata"
            }

    # 21. Operative Definitions (Phase 2 & Test F)
    # Operative definitions must remain available for downstream analysis & context
    if is_definition_clause(raw):
        return {
            "content_type": CONTENT_TYPE_DEFINITION,
            "is_substantive": True,
            "classification": "NOT_APPLICABLE",  # Definitions carry 0 base risk
            "risk_score": 0.0,
            "human_review": False,
            "clause_eligibility_score": 0.85,
            "filter_reason": None,
        }

    # 22. Evaluate Operative Legal Language
    eligibility_score = calculate_clause_eligibility_score(raw)
    is_operative = has_operative_legal_language(raw)

    # Check for short substantive terms like 'The policy is non-assignable.' or 'Grace period: 30 days.'
    is_short_substantive = False
    lower_raw = raw.lower()
    if (
        "non-assignable" in lower_raw
        or "not assignable" in lower_raw
        or lower_raw.startswith("grace period")
        or lower_raw.startswith("free look period")
        or "cancellation:" in lower_raw
        or "governing law:" in lower_raw
    ):
        is_short_substantive = True

    if is_operative or is_short_substantive or eligibility_score >= 0.3:
        return {
            "content_type": CONTENT_TYPE_SUBSTANTIVE_CLAUSE,
            "is_substantive": True,
            "classification": "PENDING_DISTILBERT",
            "risk_score": None,  # Will be computed by risk engine
            "human_review": False,
            "clause_eligibility_score": max(0.6, eligibility_score),
            "filter_reason": None
        }

    # 23. Uncertain Genuine Contractual Text (Preserved for Human Review)
    # Requires genuine sentence structure or legal terms; structural metadata already filtered above
    words = raw.split()
    if len(words) >= 5 and any(w.isalpha() for w in words):
        has_letters = sum(1 for c in raw if c.isalpha())
        if has_letters >= 15:
            return {
                "content_type": CONTENT_TYPE_SUBSTANTIVE_CLAUSE,
                "is_substantive": True,
                "classification": CONTENT_TYPE_UNCERTAIN,
                "risk_score": 0.0,
                "human_review": True,
                "clause_eligibility_score": 0.35,
                "filter_reason": "Uncertain structure preserved for human verification"
            }

    # 24. Fallback for remaining brief non-operative fragments (stray words, symbols)
    return {
        "content_type": CONTENT_TYPE_OTHER_NON_SUBSTANTIVE,
        "is_substantive": False,
        "classification": "NOT_APPLICABLE",
        "risk_score": 0.0,
        "human_review": False,
        "clause_eligibility_score": 0.1,
        "filter_reason": "Non-operative document fragment"
    }
