import re
from typing import Dict, Any, List, Set

PAGE_NUMBER_PATTERNS = [
    r"^(?:\[PAGE\s+\d+\])$",
    r"^(?:Page\s+\d+(?:\s+of\s+\d+)?)$",
    r"^(?:Page\s+[ivxlcdm]+(?:\s+of\s+[ivxlcdm]+)?)$",
    r"^(?:\d+\s+of\s+\d+)$",
    r"^(?:[ivxlcdm]+\s+of\s+[ivxlcdm]+)$",
    r"^(?:\d+\s*/\s*\d+)$",
    r"^(?:[ivxlcdm]+\s*/\s*[ivxlcdm]+)$",
    r"^(?:-\s*\d+\s*-)$",
    r"^(?:-\s*[ivxlcdm]+\s*-)$",
    r"^(?:[—–]\s*\d+\s*[—–])$",
    r"^(?:[—–]\s*[ivxlcdm]+\s*[—–])$",
    r"^(?:[•*]\s*\d+\s*[•*])$",
    r"^(?:[•*]\s*[ivxlcdm]+\s*[•*])$",
    r"^(?:Page\s+No\.?\s*[:\-\s]*(?:\d+|[ivxlcdm]+))$",
    r"^(?:\d{1,3})$",  # Standalone single/double/triple digit on its own line
    r"^(?:[ivxlcdm]{1,5})$",  # Standalone short roman numeral
]

COMPILED_PAGE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PAGE_NUMBER_PATTERNS]

RUNNING_HEADER_FOOTER_PATTERNS = [
    r"^.*(?:Page\s+\d+\s+of\s+\d+).*$",
    r"^.*(?:Page\s+[ivxlcdm]+\s+of\s+[ivxlcdm]+).*$",
    r"^.*\(UIN:\s*[A-Z0-9]+\).*Page\s+\d+.*$",
    r"^.*Policy\s+Document.*Page\s+\d+.*$",
    r"^.*Confidential.*(?:Page\s+\d+|Page\s+[ivxlcdm]+).*$",
    r"^.*(?:Confidential\s+and\s+Proprietary|Privileged\s+and\s+Confidential|Attorney-Client\s+Privilege).*$",
    r"^.*(?:©|\(c\)|Copyright)\s*\d{4}.*All\s+rights?\s+reserved.*$",
    r"^(?:Exhibit\s+[A-Z0-9]+|Schedule\s+[A-Z0-9]+|Annexure\s+[A-Z0-9]+)$",
]

COMPILED_RUNNING_PATTERNS = [re.compile(p, re.IGNORECASE) for p in RUNNING_HEADER_FOOTER_PATTERNS]

def is_page_metadata(text: str) -> bool:
    if not text:
        return False
    s = text.strip()
    for cp in COMPILED_PAGE_PATTERNS:
        if cp.match(s):
            return True
    return False

def is_running_header_footer(text: str) -> bool:
    if not text:
        return False
    s = text.strip()
    for cp in COMPILED_RUNNING_PATTERNS:
        if cp.match(s):
            return True
    return False

def build_document_repeated_headers_footers(full_text: str) -> Set[str]:
    """
    Scans document pages to find verbatim or near-verbatim lines repeated on 2 or more distinct pages.
    Guarantees that lines with substantive legal language are never flagged as running headers.
    """
    if not full_text:
        return set()

    # Split into pages if [PAGE N] tags are present
    page_splits = [p.strip() for p in re.split(r"\[PAGE\s+\d+\]", full_text) if p.strip()]
    if len(page_splits) <= 1:
        # Fallback: split by 3+ newlines or form feeds
        page_splits = [p.strip() for p in re.split(r"\n{3,}|\f", full_text) if p.strip()]

    if len(page_splits) < 2:
        # Cannot have cross-page repeated headers in a single-page document
        return set()

    from app.services.nlp.clause_eligibility_gate import has_operative_legal_language

    page_top_lines_sets = []
    page_bot_lines_sets = []

    for page_chunk in page_splits:
        lines = [line.strip() for line in page_chunk.splitlines() if line.strip()]
        if not lines:
            continue
        page_top_lines_sets.append(set(lines[:3]))
        page_bot_lines_sets.append(set(lines[-3:]))

    repeated = set()
    from collections import Counter
    top_page_counts = Counter()
    for s in page_top_lines_sets:
        for line in s:
            top_page_counts[line] += 1

    bot_page_counts = Counter()
    for s in page_bot_lines_sets:
        for line in s:
            bot_page_counts[line] += 1

    for line, count in top_page_counts.items():
        if count >= 2 and 5 < len(line) < 150:
            if not has_operative_legal_language(line):
                repeated.add(line)

    for line, count in bot_page_counts.items():
        if count >= 2 and 5 < len(line) < 150:
            if not has_operative_legal_language(line):
                repeated.add(line)

    return repeated
