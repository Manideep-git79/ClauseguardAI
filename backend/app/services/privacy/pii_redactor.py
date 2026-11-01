import re
import logging
from presidio_analyzer import AnalyzerEngine

logger = logging.getLogger(__name__)

ENTITIES_TO_REDACT = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
]

ENTITY_LABEL_PREFIXES = {
    "PERSON": "Person",
    "EMAIL_ADDRESS": "Email",
    "PHONE_NUMBER": "Phone",
    "LOCATION": "Location",
    "US_SSN": "SSN",
    "CREDIT_CARD": "Credit Card",
    "IBAN_CODE": "Bank Account",
    "IP_ADDRESS": "IP Address",
}

_analyzer = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
        }
        provider = NlpEngineProvider(nlp_configuration=config)
        nlp_engine = provider.create_engine()
        _analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    return _analyzer

class DocumentEntityResolver:
    """
    Maintains document-scoped, stable, sequential pseudonymous mappings.
    Assigns neutral sequential labels: [Person 1], [Email 1], [Phone 1], etc.
    Ensures repeated mentions of the same identified entity reuse the same label.
    Prevents false merging of distinct people sharing only a first name.
    Never assigns contextual roles (Tenant, Landlord, Party A, Party B, etc.).
    """
    def __init__(self):
        self.counters = {}
        # Mapping from normalized entity key to assigned label
        self.registry = {}
        # Distinct registered persons: list of dicts with {"tokens": tuple, "full_normalized": str, "label": str}
        self.registered_persons = []

    def get_or_assign_label(self, entity_type: str, raw_value: str) -> str:
        prefix = ENTITY_LABEL_PREFIXES.get(entity_type, entity_type.replace('_', ' ').title())

        if entity_type == "PERSON":
            return self._resolve_person(raw_value)
        elif entity_type == "EMAIL_ADDRESS":
            norm = raw_value.strip().lower()
            key = f"EMAIL::{norm}"
            return self._get_or_create(key, prefix)
        elif entity_type in ("PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "IBAN_CODE"):
            digits = re.sub(r'\D', '', raw_value)
            key = f"{entity_type}::{digits if len(digits) >= 6 else raw_value.strip().lower()}"
            return self._get_or_create(key, prefix)
        elif entity_type == "LOCATION":
            norm = re.sub(r'\s+', ' ', raw_value.strip().lower())
            key = f"LOCATION::{norm}"
            return self._get_or_create(key, prefix)
        else:
            norm = raw_value.strip().lower()
            key = f"{entity_type}::{norm}"
            return self._get_or_create(key, prefix)

    def _get_or_create(self, key: str, prefix: str) -> str:
        if key in self.registry:
            return self.registry[key]
        idx = self.counters.get(prefix, 0) + 1
        self.counters[prefix] = idx
        label = f"[{prefix} {idx}]"
        self.registry[key] = label
        return label

    def _resolve_person(self, raw_name: str) -> str:
        # Strip common courtesy titles and extra punctuation
        cleaned = re.sub(r'^[^\w]+|[^\w.]+$', '', raw_name.strip())
        courtesy_titles = {"mr", "mr.", "ms", "ms.", "mrs", "mrs.", "dr", "dr.", "prof", "prof."}
        parts = cleaned.split()
        if parts and parts[0].lower() in courtesy_titles:
            parts = parts[1:]

        normalized_full = " ".join(parts).lower()
        if not normalized_full:
            normalized_full = cleaned.lower()

        tokens = tuple(p.lower() for p in parts if len(p) > 1)

        # Check existing registered persons
        for p in self.registered_persons:
            # 1. Exact full-name match
            if p["full_normalized"] == normalized_full:
                return p["label"]

            # 2. Strict matching for multi-token full names (e.g. "John Q. Smith" vs "John Smith")
            if len(tokens) >= 2 and len(p["tokens"]) >= 2:
                # If first name and last name match exactly:
                if tokens[0] == p["tokens"][0] and tokens[-1] == p["tokens"][-1]:
                    return p["label"]

            # NOTE: We deliberately do NOT match solely on first name (e.g. "John" vs "John Smith")
            # to strictly obey: "Do not merge different people solely because they share a first name or a common name."

        # If not matched, register as a new distinct person
        prefix = "Person"
        idx = self.counters.get(prefix, 0) + 1
        self.counters[prefix] = idx
        label = f"[{prefix} {idx}]"

        self.registered_persons.append({
            "full_normalized": normalized_full,
            "tokens": tokens,
            "label": label,
        })
        return label

def redact(raw_text: str) -> tuple[str, list[dict]]:
    """
    Redact personally identifiable information (PII) before any model or frontend sees it.
    Uses Presidio for detection and assigns sequential neutral pseudonyms ([Person 1], [Email 1], etc.).
    Calculates exact start and end offsets relative to the produced redacted_text.
    Returns: (redacted_text, pii_findings_list)
    """
    if not raw_text or not raw_text.strip():
        return raw_text, []

    analyzer = get_analyzer()

    # Process large texts in chunks (10,000 chars split cleanly on newlines/whitespace)
    # to maintain strict memory bounds (<90MB vs >500MB unchunked spaCy parsing)
    CHUNK_SIZE = 10000
    if len(raw_text) > CHUNK_SIZE:
        raw_results = []
        pos = 0
        while pos < len(raw_text):
            end_pos = min(pos + CHUNK_SIZE, len(raw_text))
            if end_pos < len(raw_text):
                nl = raw_text.rfind("\n", pos, end_pos)
                if nl > pos + 5000:
                    end_pos = nl + 1
                else:
                    sp = raw_text.rfind(" ", pos, end_pos)
                    if sp > pos + 5000:
                        end_pos = sp + 1
            chunk_text = raw_text[pos:end_pos]
            chunk_results = analyzer.analyze(
                text=chunk_text,
                entities=ENTITIES_TO_REDACT,
                language="en"
            )
            for r in chunk_results:
                r.start += pos
                r.end += pos
                raw_results.append(r)
            pos = end_pos
        results = raw_results
    else:
        results = analyzer.analyze(
            text=raw_text,
            entities=ENTITIES_TO_REDACT,
            language="en"
        )

    if not results:
        return raw_text, []

    # Sort primarily by start ascending, then longest span descending, then highest score
    sorted_raw_results = sorted(
        results,
        key=lambda r: (r.start, -(r.end - r.start), -r.score)
    )

    # Disjoint interval selection to resolve overlaps strictly
    disjoint_results = []
    last_end = 0
    for r in sorted_raw_results:
        if r.start >= last_end and r.end <= len(raw_text):
            disjoint_results.append(r)
            last_end = r.end

    resolver = DocumentEntityResolver()

    # Pre-resolve pseudonymous labels for each finding
    resolved_findings = []
    for r in disjoint_results:
        raw_val = raw_text[r.start:r.end]
        label = resolver.get_or_assign_label(r.entity_type, raw_val)
        resolved_findings.append({
            "raw_start": r.start,
            "raw_end": r.end,
            "entity_type": r.entity_type,
            "confidence": float(r.score),
            "label": label,
        })

    # Assemble redacted_text and calculate exact offsets against redacted_text
    redacted_chunks = []
    current_raw_pos = 0
    current_redacted_pos = 0
    final_findings = []

    for item in resolved_findings:
        raw_start = item["raw_start"]
        raw_end = item["raw_end"]

        # Text prefix before this finding
        prefix = raw_text[current_raw_pos:raw_start]
        redacted_chunks.append(prefix)
        current_redacted_pos += len(prefix)

        label = item["label"]
        redacted_start = current_redacted_pos
        redacted_end = current_redacted_pos + len(label)

        redacted_chunks.append(label)
        current_redacted_pos += len(label)
        current_raw_pos = raw_end

        final_findings.append({
            "entity_type": item["entity_type"],
            "start": redacted_start,
            "end": redacted_end,
            "confidence": item["confidence"],
            "replacement_label": label,
        })

    # Remaining text after the last finding
    suffix = raw_text[current_raw_pos:]
    redacted_chunks.append(suffix)

    redacted_text = "".join(redacted_chunks)

    # Validation: ensure every finding slice in redacted_text exactly matches its label
    for f in final_findings:
        actual_slice = redacted_text[f["start"]:f["end"]]
        if actual_slice != f["replacement_label"]:
            logger.error("Offset misalignment detected: expected '%s', found '%s'", f["replacement_label"], actual_slice)

    logger.info("PII redaction complete: %d findings replaced with neutral pseudonyms.", len(final_findings))
    return redacted_text, final_findings
