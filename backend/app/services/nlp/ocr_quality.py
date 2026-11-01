import re
import math
from typing import Tuple, Dict, Any

COMMON_LEGAL_WORDS = {
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "this", "be", "by", "on", "with",
    "as", "policy", "shall", "insured", "insurer", "company", "corporation", "agreement",
    "contract", "clause", "section", "article", "terms", "conditions", "premium", "coverage",
    "claim", "benefit", "loss", "damage", "liability", "tenant", "landlord", "rent", "lease",
    "premises", "employee", "employer", "salary", "employment", "notice", "period", "days",
    "date", "party", "parties", "herein", "thereof", "hereby", "subject", "provided", "upon",
    "within", "under", "any", "all", "such", "not", "or", "an", "at", "from", "may", "will",
    "must", "no", "if", "other", "have", "has", "been", "paid", "payment", "due", "amount"
}

VOWELS = set("aeiouAEIOU")

KNOWN_ACRONYMS = {"llc", "ltd", "inc", "corp", "ein", "uin", "ssn", "gst", "cin", "pan", "pdf", "doc", "usa", "uk", "gmt", "utc", "iso", "act", "law", "no", "id", "ref"}

def is_consonant_cluster_gibberish(word: str) -> bool:
    clean = re.sub(r"[^a-zA-Z]", "", word).lower()
    if len(clean) >= 3 and not any(c in VOWELS for c in clean):
        if clean not in KNOWN_ACRONYMS:
            return True
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", clean):
        return True
    return False

def assess_ocr_quality(text: str) -> Dict[str, Any]:
    """
    Evaluates OCR text quality and detects corrupt or unreadable noise.
    Returns:
    {
        "is_ocr_noise": bool,
        "quality_score": float (0.0 to 1.0, 1.0 being cleanest),
        "malformed_ratio": float,
        "reason": str | None
    }
    """
    if not text or not text.strip():
        return {
            "is_ocr_noise": False,
            "quality_score": 1.0,
            "malformed_ratio": 0.0,
            "reason": None
        }

    raw = text.strip()
    words = raw.split()
    total_words = len(words)
    total_chars = len(raw)

    if total_words == 0 or total_chars == 0:
        return {
            "is_ocr_noise": False,
            "quality_score": 1.0,
            "malformed_ratio": 0.0,
            "reason": None
        }

    # 1. Non-alphanumeric symbol ratio (excluding common legal punctuation like . , ; : ' " - ( ) / $ % §)
    standard_chars = re.sub(r"[a-zA-Z0-9\s.,;:()'\-\"$%&/§#@\[\]]", "", raw)
    symbol_anomaly_ratio = len(standard_chars) / max(1, total_chars)

    # 2. Count malformed / gibberish tokens
    malformed_count = 0
    consonant_gibberish_count = 0
    short_fragment_count = 0
    legal_word_count = 0

    for w in words:
        clean = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", w).lower()
        if not clean:
            malformed_count += 1
            continue

        if clean in COMMON_LEGAL_WORDS:
            legal_word_count += 1
            continue

        # Check for single char fragments (like 'f', 't', 'm' floating alone)
        if len(clean) == 1 and clean.isalpha() and clean not in ("a", "i"):
            short_fragment_count += 1
            malformed_count += 1
            continue

        # Check for mixed case noise within a word like 'fTM' or 'lPS'
        if re.search(r"[a-z][A-Z]{2,}", w) or re.search(r"[A-Z]{2,}[a-z][A-Z]", w):
            malformed_count += 1

        # Check unpronounceable consonant clusters
        if is_consonant_cluster_gibberish(clean):
            consonant_gibberish_count += 1
            malformed_count += 1
            continue

        # Check if word has abnormal character repetitions like 'aaaaaaa' or 'xxxxx'
        if re.search(r"(.)\1{4,}", clean):
            malformed_count += 1
            continue

    malformed_ratio = malformed_count / total_words

    # Safety override: If the block contains clear operative legal structure, it's not noise
    lower_raw = raw.lower()
    has_operative_phrase = any(
        p in lower_raw
        for p in [
            "shall pay", "shall terminate", "shall be", "is non-assignable", "agree to",
            "in accordance with", "terms and conditions", "grace period", "subject to",
            "free look period", "policyholder shall", "tenant shall", "employee shall"
        ]
    )

    if has_operative_phrase and legal_word_count >= 3:
        return {
            "is_ocr_noise": False,
            "quality_score": max(0.6, 1.0 - malformed_ratio),
            "malformed_ratio": round(malformed_ratio, 3),
            "reason": None
        }

    # Evaluate OCR noise criteria
    is_noise = False
    reason = None

    if symbol_anomaly_ratio > 0.25:
        is_noise = True
        reason = f"High character anomaly ratio ({symbol_anomaly_ratio:.1%})"
    elif malformed_ratio >= 0.45 and legal_word_count < 2:
        is_noise = True
        reason = f"High malformed token ratio ({malformed_ratio:.1%}) with lack of lexical coherence"
    elif consonant_gibberish_count >= 2 and total_words <= 12 and legal_word_count < 2:
        is_noise = True
        reason = f"Unpronounceable token sequence ({consonant_gibberish_count} gibberish tokens)"
    elif short_fragment_count >= 3 and (short_fragment_count / total_words) > 0.35:
        is_noise = True
        reason = f"Severe token fragmentation ({short_fragment_count} isolated single characters)"

    quality_score = max(0.0, min(1.0, 1.0 - (malformed_ratio * 0.7 + symbol_anomaly_ratio * 0.3)))

    return {
        "is_ocr_noise": is_noise,
        "quality_score": round(quality_score, 3),
        "malformed_ratio": round(malformed_ratio, 3),
        "reason": reason
    }
