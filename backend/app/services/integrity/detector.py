import re
import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from app.services.integrity.config import (
    TINY_FONT_SIZE_PT,
    SUPERSCRIPT_MAX_CHARS,
    LIGHT_TEXT_COLOR_INT,
    RGB_LIGHTNESS_THRESHOLD,
    OFF_PAGE_TOLERANCE_PT,
    OVERLAPPING_IOU_THRESHOLD,
    MAX_INSPECTION_TEXT_CHARS,
)

logger = logging.getLogger(__name__)

# Explainable Prompt Injection Rule Definitions:
# Each rule has a regex pattern, category, severity, confidence, and human-readable explanation.
INJECTION_RULES = [
    {
        "pattern": r"(?i)\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|directives|rules|prompts|guidelines)\b",
        "category": "PROMPT_INJECTION_OVERRIDE",
        "severity": "critical",
        "confidence": 0.98,
        "explanation": "Explicit instruction instructing the AI assistant to disregard system directives or guidelines.",
    },
    {
        "pattern": r"(?i)\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(?:an?\s+)?(?:unrestricted|jailbroken|new\s+AI|DAN|developer\s+mode)\b",
        "category": "PROMPT_INJECTION_PERSONA_OVERRIDE",
        "severity": "critical",
        "confidence": 0.95,
        "explanation": "Attempt to hijack AI system persona or enter an unrestricted jailbreak execution mode.",
    },
    {
        "pattern": r"(?i)\bnew\s+system\s+(?:prompt|directive|instruction)\s*:",
        "category": "PROMPT_INJECTION_SYSTEM_HIJACK",
        "severity": "critical",
        "confidence": 0.95,
        "explanation": "Fabricated system directive intended to override core AI behavioral boundaries.",
    },
    {
        "pattern": r"(?i)\b(?:reveal|output|print|display|leak)\s+(?:the|your)?\s*(?:system\s+prompt|internal\s+instructions|hidden\s+prompt|secret\s+instructions|api\s+key)\b",
        "category": "PROMPT_INJECTION_EXFILTRATION",
        "severity": "critical",
        "confidence": 0.95,
        "explanation": "Command directing the assistant to reveal internal system prompts, keys, or private operational rules.",
    },
    {
        "pattern": r"(?i)\b(?:exfiltrate|transmit|send|upload)\s+(?:all\s+)?(?:document\s+text|clauses|data)\s+to\s+https?://\S+\b",
        "category": "PROMPT_INJECTION_EXFILTRATION",
        "severity": "critical",
        "confidence": 0.92,
        "explanation": "Directive commanding unauthorized data transmission or exfiltration to an external endpoint.",
    },
    {
        "pattern": r"(?i)\b(?:system\s+override|override\s+system)\b",
        "category": "PROMPT_INJECTION_OVERRIDE",
        "severity": "critical",
        "confidence": 0.95,
        "explanation": "Command directing an override of system-level AI directives.",
    },
    {
        "pattern": r"(?i)\b(?:rate|classify|score|mark)\s+this\s+(?:document|agreement|contract|lease)\s+(?:as\s+)?(?:zero|0|no|safe|low)\s+risk\b",
        "category": "PROMPT_INJECTION_RISK_MANIPULATION",
        "severity": "high",
        "confidence": 0.90,
        "explanation": "Explicit command directing the analysis engine to misrepresent risk or force an artificial zero-risk score.",
    },
    {
        "pattern": r"(?i)\bcertify\s+that\s+this\s+(?:document|agreement|contract)\s+has\s+(?:zero|no)\s+risk\b",
        "category": "PROMPT_INJECTION_RISK_MANIPULATION",
        "severity": "high",
        "confidence": 0.88,
        "explanation": "Directive commanding the system to falsely certify a document as having zero legal risk.",
    },
    {
        "pattern": r"(?i)\b(?:ignore|suppress|hide)\s+(?:all\s+)?(?:unfavorable|high\s*risk|hazardous)\s+clauses\b",
        "category": "PROMPT_INJECTION_RISK_MANIPULATION",
        "severity": "high",
        "confidence": 0.92,
        "explanation": "Directive commanding the suppression or omission of unfavorable clauses during contract review.",
    },
    {
        "pattern": r"(?i)\bdo\s+not\s+flag\s+any\s+(?:risks|liabilities|penalties|issues)\b",
        "category": "PROMPT_INJECTION_RISK_MANIPULATION",
        "severity": "high",
        "confidence": 0.88,
        "explanation": "Instruction commanding the review pipeline to omit flagging valid liabilities or legal penalties.",
    },
    {
        "pattern": r"(?i)\b(?:bypass|disable|circumvent)\s+(?:all\s+)?(?:safety|security|content)\s+(?:filters|controls|guardrails|checks)\b",
        "category": "PROMPT_INJECTION_SAFETY_BYPASS",
        "severity": "critical",
        "confidence": 0.94,
        "explanation": "Directive commanding the bypass of safety filters or security guardrails.",
    },
]

# Legitimate legal phrasing exclusions:
# Prevents false positives when ordinary contract clauses mention "ignore", "instructions", or "representations".
LEGITIMATE_LEGAL_EXCLUSIONS = [
    r"(?i)\b(?:prior|oral|informal)\s+(?:representations|agreements|negotiations|discussions)\b",
    r"(?i)\b(?:instructions\s+of\s+the|supervisor|employer|physician|court|landlord)\b",
    r"(?i)\b(?:notices?\s+sent|written\s+notice)\b",
]

def is_legitimate_legal_context(matched_sentence: str) -> bool:
    """
    Evaluates whether an imperative or suspicious keyword appears within an ordinary
    commercial or legal contract clause (e.g. superseding prior oral representations).
    """
    for exclusion_pattern in LEGITIMATE_LEGAL_EXCLUSIONS:
        if re.search(exclusion_pattern, matched_sentence):
            return True
    return False

def detect_prompt_injection(redacted_text: str) -> List[Dict[str, Any]]:
    """
    Conservative, transparent scanner for prompt injection content in already-redacted text.
    Preserves privacy by never operating on or emitting raw unredacted PII.
    """
    if not redacted_text or not redacted_text.strip():
        return []

    text_to_scan = redacted_text[:MAX_INSPECTION_TEXT_CHARS]
    findings = []
    seen_categories = set()

    # Split into approximate sentences to maintain context
    sentences = re.split(r"(?<=[.!?\n])\s+", text_to_scan)

    for sentence in sentences:
        sentence_clean = sentence.strip()
        if not sentence_clean:
            continue

        for rule in INJECTION_RULES:
            match = re.search(rule["pattern"], sentence_clean)
            if match:
                # Check for legitimate legal context before flagging
                if is_legitimate_legal_context(sentence_clean):
                    logger.debug(f"[Integrity] Filtered legitimate legal context: {sentence_clean[:60]}...")
                    continue

                snippet = sentence_clean[:140] + ("..." if len(sentence_clean) > 140 else "")
                cat = rule["category"]

                # Deduplicate repeated identical categories from the same document
                dedup_key = (cat, match.group(0).lower())
                if dedup_key in seen_categories:
                    continue
                seen_categories.add(dedup_key)

                findings.append({
                    "id": str(uuid.uuid4()),
                    "category": cat,
                    "severity": rule["severity"],
                    "confidence": rule["confidence"],
                    "detection_method": "heuristic_rule_scan",
                    "page_number": None,
                    "bbox": None,
                    "explanation": rule["explanation"],
                    "evidence_snippet": snippet,
                    "review_status": "Flagged for Review",
                })

    return findings

def calculate_bbox_iou(box1: List[float], box2: List[float]) -> float:
    """Calculates Intersection-over-Union (IoU) of two bounding boxes [x0, y0, x1, y1]."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area <= 0:
        return 0.0
    return intersection_area / union_area

def detect_visual_hidden_text_pdf(file_path: str, page_metadata: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Inspects PDF layout, color, geometry, and rendering properties for obscured or hidden content.
    Identifies:
    1. Text positioned off-canvas (outside normal visible page boundaries).
    2. White text rendered on default white background (color matching).
    3. Extremely tiny font sizes (<= TINY_FONT_SIZE_PT), distinct from superscripts.
    4. Invisible text rendering mode (mode 3 Tr).
    5. Overlapping / duplicated text layers.
    """
    findings = []
    if not os.path.isfile(file_path):
        return findings

    try:
        import fitz
        doc = fitz.open(file_path)
    except Exception as e:
        logger.debug(f"[Integrity] Failed to open PDF for visual inspection: {e}")
        return findings

    try:
        for idx, page in enumerate(doc):
            page_num = idx + 1
            pw = page.rect.width
            ph = page.rect.height

            # --- Check 1: Off-canvas text detection using expanded textpage ---
            try:
                expanded_rect = fitz.Rect(-500, -500, pw + 500, ph + 500)
                tp = page.get_textpage(clip=expanded_rect)
                exp_dict = tp.extractDICT()

                for b in exp_dict.get("blocks", []):
                    if b.get("type", 0) != 0:
                        continue
                    b_box = b.get("bbox", (0, 0, 0, 0))
                    is_off_page = (
                        b_box[2] < -OFF_PAGE_TOLERANCE_PT or
                        b_box[3] < -OFF_PAGE_TOLERANCE_PT or
                        b_box[0] > pw + OFF_PAGE_TOLERANCE_PT or
                        b_box[1] > ph + OFF_PAGE_TOLERANCE_PT
                    )
                    if is_off_page:
                        # Extract text
                        block_text = ""
                        for line in b.get("lines", []):
                            for sp in line.get("spans", []):
                                block_text += sp.get("text", "") + " "
                        block_text = block_text.strip()
                        if block_text:
                            snippet = block_text[:120] + ("..." if len(block_text) > 120 else "")
                            findings.append({
                                "id": str(uuid.uuid4()),
                                "category": "OFF_PAGE_COORDINATES",
                                "severity": "high",
                                "confidence": 0.95,
                                "detection_method": "layout_viewport_inspection",
                                "page_number": page_num,
                                "bbox": [round(c, 1) for c in b_box],
                                "explanation": f"Text is positioned outside visible page boundaries ({round(pw, 1)}x{round(ph, 1)} pt) and clipped by viewport.",
                                "evidence_snippet": snippet,
                                "review_status": "Flagged for Review",
                            })
            except Exception as off_err:
                logger.debug(f"[Integrity] Page {page_num} off-page check error: {off_err}")

            # --- Check 2: Color matching (white text on white background) & Tiny Font ---
            try:
                page_dict = page.get_text("dict")
                page_spans = []

                for b in page_dict.get("blocks", []):
                    if b.get("type", 0) != 0:
                        continue
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            stext = span.get("text", "").strip()
                            if not stext:
                                continue
                            ssize = round(span.get("size", 0.0), 1)
                            scolor = span.get("color", 0)
                            sflags = span.get("flags", 0)
                            sbbox = [round(c, 1) for c in span.get("bbox", (0, 0, 0, 0))]

                            page_spans.append({
                                "text": stext,
                                "size": ssize,
                                "color": scolor,
                                "flags": sflags,
                                "bbox": sbbox,
                            })

                            # Check for white text (color == 16777215 or 0xFFFFFF)
                            is_white_color = (scolor == LIGHT_TEXT_COLOR_INT)
                            if is_white_color:
                                snippet = stext[:120] + ("..." if len(stext) > 120 else "")
                                findings.append({
                                    "id": str(uuid.uuid4()),
                                    "category": "HIDDEN_TEXT_COLOR_MATCH",
                                    "severity": "high",
                                    "confidence": 0.92,
                                    "detection_method": "span_color_inspection",
                                    "page_number": page_num,
                                    "bbox": sbbox,
                                    "explanation": "Text foreground color matches default page background (#FFFFFF), rendering it invisible.",
                                    "evidence_snippet": snippet,
                                    "review_status": "Flagged for Review",
                                })

                            # Check for unusually tiny text (size <= TINY_FONT_SIZE_PT)
                            # Exclude 1-2 char superscript symbols (flags & 1 or len <= 2)
                            is_superscript = bool(sflags & 1) or (len(stext) <= SUPERSCRIPT_MAX_CHARS and stext.isdigit())
                            if ssize <= TINY_FONT_SIZE_PT and not is_superscript and len(stext) > SUPERSCRIPT_MAX_CHARS:
                                snippet = stext[:120] + ("..." if len(stext) > 120 else "")
                                findings.append({
                                    "id": str(uuid.uuid4()),
                                    "category": "HIDDEN_TEXT_TINY_FONT",
                                    "severity": "medium",
                                    "confidence": 0.88,
                                    "detection_method": "typography_scale_inspection",
                                    "page_number": page_num,
                                    "bbox": sbbox,
                                    "explanation": f"Text rendered in unusually small font size ({ssize}pt), below normal reading legibility.",
                                    "evidence_snippet": snippet,
                                    "review_status": "Flagged for Review",
                                })

                # --- Check 3: Overlapping text layers on same page ---
                if len(page_spans) >= 2:
                    for i in range(len(page_spans)):
                        for j in range(i + 1, min(i + 15, len(page_spans))):
                            s1 = page_spans[i]
                            s2 = page_spans[j]
                            if s1["text"] == s2["text"]:
                                continue
                            iou = calculate_bbox_iou(s1["bbox"], s2["bbox"])
                            if iou >= OVERLAPPING_IOU_THRESHOLD:
                                snippet = f"Layer 1: '{s1['text'][:40]}' | Layer 2: '{s2['text'][:40]}'"
                                findings.append({
                                    "id": str(uuid.uuid4()),
                                    "category": "OVERLAPPING_TEXT",
                                    "severity": "medium",
                                    "confidence": 0.82,
                                    "detection_method": "spatial_overlap_inspection",
                                    "page_number": page_num,
                                    "bbox": s1["bbox"],
                                    "explanation": f"Multiple distinct text spans occupy overlapping coordinates (IoU {round(iou, 2)}).",
                                    "evidence_snippet": snippet,
                                    "review_status": "Flagged for Review",
                                })
            except Exception as span_err:
                logger.debug(f"[Integrity] Page {page_num} span inspection error: {span_err}")

            # --- Check 4: Invisible text rendering mode (type == 3 Tr) ---
            try:
                trace = page.get_texttrace()
                if trace:
                    for item in trace:
                        if item.get("type") == 3:  # Mode 3: Neither fill nor stroke
                            chars_tuple = item.get("chars", ())
                            if chars_tuple:
                                invisible_str = "".join([chr(c[0]) for c in chars_tuple if 32 <= c[0] <= 126]).strip()
                                # Check if invisible text contains substantive letters (not just spaces/dashes)
                                if len(invisible_str) > 8:
                                    snippet = invisible_str[:120] + ("..." if len(invisible_str) > 120 else "")
                                    bbox = [round(c, 1) for c in item.get("bbox", (0, 0, 0, 0))]
                                    findings.append({
                                        "id": str(uuid.uuid4()),
                                        "category": "INVISIBLE_TEXT_RENDER_MODE",
                                        "severity": "high",
                                        "confidence": 0.94,
                                        "detection_method": "pdf_render_mode_inspection",
                                        "page_number": page_num,
                                        "bbox": bbox,
                                        "explanation": "Text is rendered in PDF rendering mode 3 (neither fill nor stroke), rendering glyphs invisible.",
                                        "evidence_snippet": snippet,
                                        "review_status": "Flagged for Review",
                                    })
            except Exception as trace_err:
                logger.debug(f"[Integrity] Page {page_num} texttrace inspection error: {trace_err}")

    finally:
        doc.close()

    return findings

def detect_visual_hidden_text_docx(file_path: str) -> List[Dict[str, Any]]:
    """
    Inspects DOCX package XML properties for runs explicitly marked with hidden formatting.
    """
    findings = []
    if not os.path.isfile(file_path):
        return findings

    try:
        import docx
        doc = docx.Document(file_path)
    except Exception as e:
        logger.debug(f"[Integrity] Failed to open DOCX for inspection: {e}")
        return findings

    try:
        for p_idx, p in enumerate(doc.paragraphs):
            for r in p.runs:
                is_hidden = False
                if r.font and r.font.hidden:
                    is_hidden = True
                elif r._r is not None:
                    vanish = r._r.xpath(".//w:vanish")
                    if vanish:
                        is_hidden = True

                if is_hidden and r.text and r.text.strip():
                    stext = r.text.strip()
                    snippet = stext[:120] + ("..." if len(stext) > 120 else "")
                    findings.append({
                        "id": str(uuid.uuid4()),
                        "category": "HIDDEN_DOCX_ELEMENT",
                        "severity": "high",
                        "confidence": 0.95,
                        "detection_method": "docx_xml_inspection",
                        "page_number": max(1, (p_idx // 20) + 1),
                        "bbox": None,
                        "explanation": "DOCX run element is explicitly marked with hidden/vanish formatting attribute.",
                        "evidence_snippet": snippet,
                        "review_status": "Flagged for Review",
                    })
    except Exception as ex:
        logger.debug(f"[Integrity] DOCX hidden scan failed: {ex}")

    return findings

def analyze_document_integrity(
    file_path: str,
    ext: str,
    raw_text: str,
    redacted_text: str,
    page_metadata: Optional[List[Dict[str, Any]]] = None,
    document_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Master integrity analysis entry point:
    1. Runs visual & structural layout inspection on physical file (PDF/DOCX).
    2. Runs prompt injection detection on already-redacted text.
    3. Guarantees safety: catches all exceptions, logs cleanly without PII, returns findings list.
    """
    findings = []
    clean_ext = ext.lower().lstrip(".")

    try:
        # Step 1: Physical layout and hidden text inspection
        if clean_ext == "pdf":
            visual_findings = detect_visual_hidden_text_pdf(file_path, page_metadata=page_metadata)
            findings.extend(visual_findings)
        elif clean_ext == "docx":
            docx_findings = detect_visual_hidden_text_docx(file_path)
            findings.extend(docx_findings)
        elif clean_ext == "txt":
            # Plain text files have no coordinate layout or font properties
            logger.debug("[Integrity] Plain text file format has no visual layout layer to inspect.")

        # Step 2: Prompt injection scan on redacted text
        injection_findings = detect_prompt_injection(redacted_text)
        findings.extend(injection_findings)

        logger.info(
            f"[Integrity] Document integrity analysis completed: {len(findings)} signal(s) detected "
            f"(Visual/Structural: {len(findings) - len(injection_findings)}, Injection: {len(injection_findings)})"
        )
    except Exception as e:
        logger.error(f"[Integrity] Unexpected error during integrity analysis: {e}")

    return findings
