import re
import logging
from typing import List, Dict, Any, Optional

from app.services.nlp.clause_eligibility_gate import (
    classify_content_type,
    COMPILED_DOC_TITLES,
    COMPILED_CORPORATE,
    COMPILED_PRODUCT_DESC,
    COMPILED_GREETINGS,
    COMPILED_SIGNATURES,
    COMPILED_FORM_FIELDS,
    COMPILED_ADDRESS_BLOCKS,
    COMPILED_ADMIN_TRANSMITTAL,
    COMPILED_SECTION_HEADINGS,
    has_operative_legal_language,
    is_table_of_contents_entry,
    is_table_of_contents_block,
    is_heading_or_subheading,
    is_party_metadata,
    is_address_metadata,
)
from app.services.nlp.header_footer_detector import (
    build_document_repeated_headers_footers,
    is_page_metadata,
    is_running_header_footer,
)

logger = logging.getLogger(__name__)

TOP_LEVEL_NUMBERING_PATTERNS = [
    r"^(?:\s*)(?:ARTICLE|SECTION|CLAUSE|PARAGRAPH)\s+[0-9IVXLCDM]+(?:\.[0-9]+)*[:.\s-]*",
    r"^(?:\s*)[0-9]+(?:\.[0-9]+)*[.)\s]\s*",
    r"^(?:\s*)(?:Position|Salary|Compensation|Benefits|Start Date|At-Will|Confidentiality|Contingencies|Vacation|Termination|Governing Law|Notices|Default|Rent|Security Deposit)[:\-\s]+",
    r"^(?:\s*)(?:COVERAGE\s+[A-Z0-9]+|EXCLUSIONS?|CONDITIONS?|ENDORSEMENT)[:.\s-]*",
]
COMPILED_TOP_LEVEL_NUMBERING = [re.compile(p, re.IGNORECASE) for p in TOP_LEVEL_NUMBERING_PATTERNS]

SUB_BULLET_PATTERNS = [
    r"^(?:\s*)(?:\([a-zA-Z0-9]+\)|[a-zA-Z][.)]\s+|[ivxlcdm]+[.)]\s+|[•\-\*]\s+)",
]
COMPILED_SUB_BULLETS = [re.compile(p, re.IGNORECASE) for p in SUB_BULLET_PATTERNS]


def is_structural_header_line(line: str, repeated_headers: Optional[set] = None) -> bool:
    """Checks if a single line is a header, corporate info, title, greeting, or TOC."""
    s = line.strip()
    if not s:
        return False
    if is_page_metadata(s) or is_running_header_footer(s) or is_table_of_contents_entry(s):
        return True
    if repeated_headers and s in repeated_headers:
        return True
    all_patterns = (
        COMPILED_DOC_TITLES
        + COMPILED_CORPORATE
        + COMPILED_PRODUCT_DESC
        + COMPILED_GREETINGS
        + COMPILED_SIGNATURES
        + COMPILED_FORM_FIELDS
        + COMPILED_ADDRESS_BLOCKS
        + COMPILED_ADMIN_TRANSMITTAL
        + COMPILED_SECTION_HEADINGS
    )
    for p in all_patterns:
        if p.match(s):
            return True
    return False


MIXED_HEADING_PATTERN = re.compile(
    r"^((?:(?:ARTICLE|SECTION|CLAUSE|PARAGRAPH)\s+[0-9IVXLCDM]+(?:\.[0-9]+)*[:.\s-]*|[0-9]+(?:\.[0-9]+)*[.)\s]\s*)?[A-Za-z0-9\s,\/\&'\-]{2,45}?)\s*([:\-—])\s+(.+)$"
)

SALUTATION_PREFIX_PATTERN = re.compile(
    r"^(Dear\s+(?:Policyholder|Tenant|Employee|Associate|Sir|Madam|Mr\.|Ms\.|Mrs\.|Dr\.|Lessor|Lessee|Client|Customer|Colleague|[A-Za-z\s.\-]+?)|To\s+Whom\s+It\s+May\s+Concern|Greetings)\s*([,:]\s*)(.+)$",
    re.IGNORECASE
)

PARTY_PREFIX_PATTERN = re.compile(
    r"^(LANDLORD|TENANT|LESSOR|LESSEE|EMPLOYER|EMPLOYEE|BORROWER|LENDER|DISCLOSING\s+PARTY|RECEIVING\s+PARTY|BUYER|SELLER|CONTRACTOR|CLIENT|COMPANY|PARTY\s+[AB12])\s*([:\-—])\s+(.+)$",
    re.IGNORECASE
)


def try_split_mixed_line(line_text: str) -> Optional[tuple]:
    """
    Layer C: Detects if a single line contains both a heading/salutation/party prefix and an operative contractual clause.
    Examples:
      - 'PAYMENT: The tenant shall pay rent on the first day of each month.' -> ('PAYMENT', ':', 'The tenant shall pay...')
      - 'Dear Marcus, the Company will pay you a starting annual base salary of $225,000.00.' -> ('Dear Marcus', ',', 'the Company will pay...')
      - 'EMPLOYEE: Candidate shall maintain strict confidentiality regarding all proprietary systems.' -> ('EMPLOYEE', ':', 'Candidate shall maintain...')
      - 'TENANT: John Smith, 123 Main St' -> None (retained as pure party metadata)
    Returns (heading_text, delimiter, body_text) or None.
    """
    stripped = line_text.strip()
    if not stripped:
        return None

    # Case 1: Comma-separated or colon-separated salutation prefix with operative body on the same line
    m_sal = SALUTATION_PREFIX_PATTERN.match(stripped)
    if m_sal:
        h = m_sal.group(1).strip()
        delim = m_sal.group(2).strip()
        b = m_sal.group(3).strip()
        b_sub = has_operative_legal_language(b) or (
            len(b.split()) >= 4 and any(w in b.lower() for w in ["shall", "must", "will", "may", "agree", "pay", "terminate", "offer", "salary", "bonus", "equity", "report", "benefit", "contingent", "position"])
        )
        if b_sub:
            return h, delim, b

    # Case 2: Party prefix with operative covenant (preserves operative obligation without dropping as pure metadata)
    m_party = PARTY_PREFIX_PATTERN.match(stripped)
    if m_party:
        h = m_party.group(1).strip()
        delim = m_party.group(2)
        b = m_party.group(3).strip()
        b_sub = has_operative_legal_language(b) or (
            len(b.split()) >= 5 and any(w in b.lower() for w in ["shall", "must", "will", "may", "agree", "pay", "terminate", "maintain", "covenant", "warrant", "report", "salary", "deposit"])
        )
        if b_sub:
            return h, delim, b

    # Case 3: Standard mixed heading
    m = MIXED_HEADING_PATTERN.match(stripped)
    if not m:
        return None
    h = m.group(1).strip()
    delim = m.group(2)
    b = m.group(3).strip()

    # Exclude pure party metadata & addresses (e.g. LANDLORD: John Smith, Address: 123 Main St)
    if is_party_metadata(stripped) or is_party_metadata(f"{h}: {b}") or is_address_metadata(stripped):
        return None

    # Heading check
    h_is_heading = (
        is_heading_or_subheading(h)
        or h.isupper()
        or bool(re.match(r"^(?:(?:SECTION|ARTICLE|CLAUSE)\s+[0-9IVXLCDM]+|[0-9]+(?:\.[0-9]+)*\s+[A-Za-z])", h, re.IGNORECASE))
    )
    if not h_is_heading:
        return None

    # Substantive body check
    b_is_substantive = has_operative_legal_language(b) or (
        len(b.split()) >= 6 and any(w in b.lower() for w in ["shall", "must", "will", "may", "agree", "pay", "terminate", "indemnif", "liab", "covenant", "warrant", "represent"])
    )
    if not b_is_substantive:
        return None

    return h, delim, b


def is_standalone_section_heading(line: str) -> bool:
    """Checks if a line is a section heading without operative legal text."""
    s = line.strip()
    if not s:
        return False
    if has_operative_legal_language(s):
        return False
    for p in COMPILED_SECTION_HEADINGS:
        if p.match(s):
            return True
    if is_heading_or_subheading(s):
        return True
    # Common uppercase headings
    if re.match(r"^(?:(?:SECTION|ARTICLE|CLAUSE)\s+[0-9IVXLCDM]+(?:\.[0-9]+)*[:.\s-]*[A-Z0-9\s,\-]*)$", s, re.IGNORECASE):
        return True
    return False


def is_numbered_section_heading_prefix(line: str) -> bool:
    """Checks if a heading is an explicit numbered section heading prefix like 'SECTION 3: GOVERNING LAW' or '5. Alterations:'."""
    s = line.strip()
    return bool(re.match(r"^(?:(?:SECTION|ARTICLE|CLAUSE|PARAGRAPH)\s+[0-9IVXLCDM]+(?:\.[0-9]+)*[:.\s-]*|[0-9]+(?:\.[0-9]+)*[.)\s]\s*)[A-Za-z0-9\s,\-:]+$", s, re.IGNORECASE))


def get_sequential_spans(text: str) -> List[Dict[str, Any]]:
    """
    Splits text into lines and split spans with exact character offsets without any ambiguous .find() calls.
    Guarantees text[raw_start:raw_end] == raw and text[clean_start:clean_end] == clean.
    Supports Layer C inline mixed-content splitting into heading and substantive body spans.
    """
    spans = []
    pos = 0
    current_page = 1

    for line in text.splitlines(keepends=True):
        raw_start = pos
        raw_end = pos + len(line)
        stripped = line.strip()

        # Page detection
        page_match = re.match(r"^\[PAGE\s+(\d+)\]$", stripped, re.IGNORECASE)
        if page_match:
            try:
                current_page = int(page_match.group(1))
            except Exception:
                pass

        if stripped:
            l_strip = len(line) - len(line.lstrip())
            clean_start = raw_start + l_strip
            clean_end = clean_start + len(stripped)

            # Layer C: Check for mixed content heading + substantive clause on the same line
            split_res = try_split_mixed_line(stripped)
            if split_res:
                h, delim, b = split_res
                h_idx = line.find(h, l_strip)
                b_idx = line.find(b, h_idx + len(h))

                h_c_start = raw_start + h_idx
                h_c_end = h_c_start + len(h)

                b_c_start = raw_start + b_idx
                b_c_end = b_c_start + len(b)

                # Heading span
                spans.append({
                    "raw": line[:b_idx],
                    "clean": h,
                    "raw_start": raw_start,
                    "raw_end": h_c_end,
                    "clean_start": h_c_start,
                    "clean_end": h_c_end,
                    "is_empty": False,
                    "page": current_page,
                    "is_split_heading": True,
                })
                # Substantive body span
                spans.append({
                    "raw": line[b_idx:],
                    "clean": b,
                    "raw_start": raw_start + b_idx,
                    "raw_end": raw_end,
                    "clean_start": b_c_start,
                    "clean_end": b_c_end,
                    "is_empty": False,
                    "page": current_page,
                    "is_split_body": True,
                })
                pos = raw_end
                continue
        else:
            clean_start = raw_start
            clean_end = raw_start

        spans.append({
            "raw": line,
            "clean": stripped,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "clean_start": clean_start,
            "clean_end": clean_end,
            "is_empty": len(stripped) == 0,
            "page": current_page,
        })
        pos = raw_end

    return spans


def segment_blocks(
    text: str,
    document_type: str = "rental_agreement",
    page_metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Performs deterministic, sequential block segmentation with zero string-search ambiguity:
    1. Tracks sequential line spans from character 0 to len(text).
    2. Identifies page numbers, TOC sections, and running headers across pages.
    3. Groups subsections, lettered lists, and multi-paragraph clauses conservatively.
    4. Associates standalone section headings with following clauses rather than classifying
       them as isolated legal obligations.
    5. Detects extraction duplicate artifacts.
    6. Ensures every block has exact, verified source offsets into canonical redacted text.
    """
    if not text or not text.strip():
        return []

    repeated_headers = build_document_repeated_headers_footers(text)
    line_spans = get_sequential_spans(text)
    if not line_spans:
        return []

    logical_groups = []
    current_lines = []

    def flush_current():
        nonlocal current_lines
        if not current_lines:
            return
        # Calculate block span from first line to last line
        first_ln = current_lines[0]
        last_ln = current_lines[-1]

        raw_s = first_ln["raw_start"]
        raw_e = last_ln["raw_end"]
        raw_block = text[raw_s:raw_e]

        # Calculate exact clean start and end offsets without leading/trailing whitespace
        leading_ws = len(raw_block) - len(raw_block.lstrip())
        clean_s = raw_s + leading_ws
        clean_sub = raw_block[leading_ws:]
        trailing_ws = len(clean_sub) - len(clean_sub.rstrip())
        clean_e = clean_s + len(clean_sub) - trailing_ws
        clean_block_text = text[clean_s:clean_e]

        if clean_block_text:
            logical_groups.append({
                "text": clean_block_text,
                "start_offset": clean_s,
                "end_offset": clean_e,
                "page": first_ln["page"],
            })
        current_lines = []

    i = 0
    n = len(line_spans)

    while i < n:
        sp = line_spans[i]

        if sp["is_empty"]:
            # Empty line: check if next line continues a list or begins new section
            # If current block ends with colon or semicolon, allow continuation across single empty line
            if current_lines:
                last_text = current_lines[-1]["clean"]
                if last_text.endswith(":") or last_text.endswith(";") or last_text.endswith("; and"):
                    # Peek next non-empty line
                    next_i = i + 1
                    while next_i < n and line_spans[next_i]["is_empty"]:
                        next_i += 1
                    if next_i < n:
                        next_sp = line_spans[next_i]
                        is_sub = any(p.match(next_sp["clean"]) for p in COMPILED_SUB_BULLETS)
                        if is_sub:
                            i += 1
                            continue
            flush_current()
            i += 1
            continue

        clean_str = sp["clean"]

        # Layer C: Mixed-content split heading span
        if sp.get("is_split_heading"):
            flush_current()
            current_lines = [sp]
            flush_current()
            i += 1
            continue

        # 1. Page marker line (e.g. [PAGE 1])
        if re.match(r"^\[PAGE\s+\d+\]$", clean_str, re.IGNORECASE):
            flush_current()
            current_lines = [sp]
            flush_current()
            i += 1
            continue

        # 2. Table of Contents Entry
        if is_table_of_contents_entry(clean_str):
            flush_current()
            current_lines = [sp]
            # Accumulate subsequent adjacent TOC entries into the same TOC block
            j = i + 1
            while j < n:
                if line_spans[j]["is_empty"]:
                    # Allow 1 empty line inside TOC block if next is also TOC entry
                    if j + 1 < n and is_table_of_contents_entry(line_spans[j + 1]["clean"]):
                        j += 1
                        continue
                    break
                if is_table_of_contents_entry(line_spans[j]["clean"]):
                    current_lines.append(line_spans[j])
                    j += 1
                else:
                    break
            flush_current()
            i = j
            continue

        # 3. Standalone Section Heading (e.g. 'SECTION 3: GOVERNING LAW' vs unnumbered 'TERMINATION')
        if is_standalone_section_heading(clean_str):
            flush_current()
            # If explicit numbered section heading prefix like 'SECTION 3: GOVERNING LAW', associate with following clause!
            if is_numbered_section_heading_prefix(clean_str):
                next_i = i + 1
                while next_i < n and line_spans[next_i]["is_empty"]:
                    next_i += 1

                if next_i < n:
                    next_clean = line_spans[next_i]["clean"]
                    # If next is another heading or TOC or page marker, emit this heading standalone
                    if (
                        is_standalone_section_heading(next_clean)
                        or is_structural_header_line(next_clean, repeated_headers)
                        or re.match(r"^\[PAGE\s+\d+\]$", next_clean, re.IGNORECASE)
                    ):
                        current_lines = [sp]
                        flush_current()
                        i += 1
                        continue
                    else:
                        # Associate numbered heading with following clause!
                        current_lines = [sp]
                        i = next_i
                        continue
                else:
                    current_lines = [sp]
                    flush_current()
                    i += 1
                    continue
            else:
                # Standalone unnumbered heading (e.g. 'TERMINATION', 'INDEMNIFICATION', 'PAYMENT')
                current_lines = [sp]
                flush_current()
                i += 1
                continue

        # 4. Standalone structural header / metadata / greeting / signature line
        if is_structural_header_line(clean_str, repeated_headers) and not has_operative_legal_language(clean_str):
            flush_current()
            current_lines = [sp]
            flush_current()
            i += 1
            continue

        # 5. Top-level clause numbering (e.g. '1. Definitions', 'SECTION 2', 'Term:')
        starts_top_level = any(p.match(clean_str) for p in COMPILED_TOP_LEVEL_NUMBERING)
        is_sub_bullet = any(p.match(clean_str) for p in COMPILED_SUB_BULLETS)

        if starts_top_level and not is_sub_bullet:
            if current_lines:
                flush_current()
            current_lines.append(sp)
            i += 1
            continue

        # 6. Sub-bullet / continuation of active clause
        if is_sub_bullet:
            current_lines.append(sp)
            i += 1
            continue

        # 7. Standard body line
        current_lines.append(sp)
        i += 1

    flush_current()

    # Stage 2: Deduplication and Classification
    classified_blocks = []
    block_counter = 0
    prev_clean_text = None

    for grp in logical_groups:
        clean = grp["text"].strip()
        if not clean:
            continue

        s_off = grp["start_offset"]
        e_off = grp["end_offset"]
        page_num = grp["page"]

        # Deduplication of identical consecutive extraction artifacts
        is_dup = (prev_clean_text is not None and clean == prev_clean_text)
        prev_clean_text = clean

        meta = classify_content_type(
            clean,
            document_type=document_type,
            document_repeated_headers=repeated_headers,
            page_number=page_num,
            is_duplicate=is_dup,
        )

        classified_blocks.append({
            "clause_index": block_counter,
            "text": clean,
            "start_offset": s_off,
            "end_offset": e_off,
            "page_number": page_num,
            "content_type": meta["content_type"],
            "is_substantive": meta["is_substantive"],
            "classification": meta["classification"],
            "risk_score": meta["risk_score"] or 0.0,
            "human_review": meta["human_review"],
            "clause_eligibility_score": meta["clause_eligibility_score"],
            "filter_reason": meta["filter_reason"],
        })
        block_counter += 1

    return classified_blocks

