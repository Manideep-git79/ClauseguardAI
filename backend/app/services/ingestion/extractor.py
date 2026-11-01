import io
import os
import re
import gc
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image
import fitz
import docx
from app.services.ingestion.ocr import ocr_image_detailed

logger = logging.getLogger(__name__)

PRESERVED_HYPHEN_TERMS = {
    "non-compete",
    "non-solicitation",
    "non-disclosure",
    "court-appointed",
    "well-known",
    "pro-rata",
    "attorney-in-fact",
    "cross-default",
    "force-majeure",
    "tax-exempt",
    "third-party",
    "arms-length",
    "hold-harmless",
    "good-faith",
    "inter-alia",
    "time-sensitive",
    "end-user",
    "right-of-way",
    "co-signer",
    "pre-existing",
    "sub-lease",
}

class UnsupportedFormatError(Exception):
    pass

class EmptyDocumentError(Exception):
    pass

class EncryptedPdfError(Exception):
    pass

class CorruptedFileError(Exception):
    pass

def clean_ocr_text(text: str) -> str:
    """
    Cleans OCR/native extraction artifacts while preserving strict legal meaning.
    - Normalizes line-break hyphenations (e.g. 'termina-\\n tion' -> 'termination')
      while preserving legitimate hyphenated terms (e.g. 'non-compete').
    - Normalizes excessive blank lines and trailing whitespace.
    - Never changes dates, numbers, clause numbers, names, or legal words.
    """
    if not text:
        return ""

    def hyphen_replacer(match):
        part1 = match.group(1)
        part2 = match.group(2)
        compound = f"{part1.lower()}-{part2.lower()}"
        if compound in PRESERVED_HYPHEN_TERMS:
            return f"{part1}-{part2}"
        return f"{part1}{part2}"

    cleaned = re.sub(r"\b([a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,})\b", hyphen_replacer, text)

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())

    return cleaned.strip()

def _normalize_token_text(text: str) -> str:
    """Lowercases, strips punctuation, and collapses whitespace for fuzzy deduplication."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()

def merge_native_and_ocr_text(native_text: str, ocr_text: str) -> str:
    """
    Intelligently combines native extracted text and OCR extracted text without duplication.
    Preserves document reading order, page numbers, and paragraph structure.
    """
    clean_nat = clean_ocr_text(native_text)
    clean_ocr = clean_ocr_text(ocr_text)

    if not clean_nat:
        return clean_ocr
    if not clean_ocr:
        return clean_nat

    norm_nat = _normalize_token_text(clean_nat)
    norm_ocr = _normalize_token_text(clean_ocr)

    # If one text is completely subsumed by the other, return the more comprehensive one
    if norm_nat and norm_nat in norm_ocr:
        return clean_ocr
    if norm_ocr and norm_ocr in norm_nat:
        return clean_nat

    # Line-by-line deduplication
    nat_lines = [l.strip() for l in clean_nat.splitlines() if l.strip()]
    ocr_lines = [l.strip() for l in clean_ocr.splitlines() if l.strip()]

    norm_nat_lines = {_normalize_token_text(l) for l in nat_lines if _normalize_token_text(l)}

    unique_ocr_lines = []
    for line in ocr_lines:
        n_l = _normalize_token_text(line)
        if not n_l:
            continue
        if n_l in norm_nat_lines:
            continue
        is_dup = False
        for nl in norm_nat_lines:
            if n_l in nl or (len(n_l) > 15 and nl in n_l):
                is_dup = True
                break
        if not is_dup:
            unique_ocr_lines.append(line)

    if not unique_ocr_lines:
        return clean_nat

    footer_pattern = re.compile(r"^(page\s+\d+(\s+of\s+\d+)?|\d+|confidential)$", re.IGNORECASE)
    is_footer = len(nat_lines) == 1 and bool(footer_pattern.match(nat_lines[0]))

    if is_footer:
        merged = "\n\n".join(["\n".join(unique_ocr_lines), clean_nat])
    else:
        merged = "\n\n".join([clean_nat, "\n".join(unique_ocr_lines)])

    return clean_ocr_text(merged)

DEFAULT_QUALITY_CONFIG = {
    "min_char_count": 80,
    "min_word_count": 8,
    "max_img_coverage_threshold": 0.35,
    "min_printable_ratio": 0.75,
    "min_valid_char_ratio": 0.45,
    "max_replacement_chars": 2,
    "max_question_mark_ratio": 0.35,
}

def assess_page_extraction_quality(
    page: fitz.Page,
    extracted_text: str,
    config: Dict[str, Any] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Intelligently evaluates whether a document page requires OCR fallback based on multiple signals:
    - Character count and word count (non-whitespace)
    - Printable character ratio and encoding corruption indicators
    - Binary noise / invalid character ratio
    - Page image coverage ratio
    - Detection of header/footer-only text when a substantial scanned image exists

    Returns:
        (needs_ocr: bool, reason: str, metrics: Dict[str, Any])
    """
    cfg = dict(DEFAULT_QUALITY_CONFIG)
    if config:
        cfg.update(config)

    stripped = (extracted_text or "").strip()
    char_count = len(stripped)
    words = stripped.split()
    word_count = len(words)

    printable_count = sum(1 for c in stripped if c.isprintable() or c in "\n\r\t")
    printable_ratio = round(printable_count / max(1, char_count), 3)

    max_img_coverage = 0.0
    total_img_coverage = 0.0
    image_count = 0

    if page is not None:
        try:
            images = page.get_images(full=True)
            image_count = len(images) if images else 0
            page_area = max(1.0, page.rect.width * page.rect.height)
            if images:
                for img_info in images:
                    xref = img_info[0]
                    try:
                        rects = page.get_image_rects(xref)
                        for r in rects:
                            cov = (r.width * r.height) / page_area
                            total_img_coverage += cov
                            if cov > max_img_coverage:
                                max_img_coverage = cov
                    except Exception:
                        pass
                max_img_coverage = round(min(1.0, max_img_coverage), 3)
                total_img_coverage = round(min(1.0, total_img_coverage), 3)
        except Exception:
            pass

    header_footer_pattern = re.compile(
        r"^(page\s+\d+(\s+of\s+\d+)?|\d+|confidential(\s+agreement)?|draft|exhibit\s+[a-z0-9]+)$",
        re.IGNORECASE,
    )
    is_header_footer_only = bool(
        char_count < 100
        and (
            header_footer_pattern.match(stripped)
            or (len(stripped.splitlines()) <= 2 and word_count <= 8)
        )
    )

    metrics = {
        "char_count": char_count,
        "word_count": word_count,
        "printable_ratio": printable_ratio,
        "max_img_coverage": max_img_coverage,
        "total_img_coverage": total_img_coverage,
        "image_count": image_count,
        "is_header_footer_only": is_header_footer_only,
    }

    # Signal 1: Pure scanned page with zero native text
    if char_count == 0:
        if image_count > 0:
            return True, "scanned_page_zero_text", metrics
        return False, "blank_page_no_text", metrics

    # Signal 2: Scanned image dominates page (>35% area) with minimal or header/footer-only native text
    if max_img_coverage > cfg["max_img_coverage_threshold"] and (char_count < 120 or is_header_footer_only):
        return True, "scanned_image_dominates_page", metrics

    # Signal 3: Encoding corruption/broken font glyphs (replacement character \ufffd or excessive '?')
    replacement_count = stripped.count("\ufffd")
    question_count = stripped.count("?")
    if replacement_count > cfg["max_replacement_chars"] or (char_count > 20 and question_count > char_count * cfg["max_question_mark_ratio"]):
        return True, "encoding_artifacts_detected", metrics

    # Signal 4: Unprintable binary noise
    if printable_ratio < cfg["min_printable_ratio"]:
        return True, "unprintable_binary_noise", metrics

    # Signal 5: Corrupted glyph noise (not valid legal alphanumeric/punctuation)
    if char_count > 150:
        valid_chars = sum(1 for c in stripped if c.isalnum() or c.isspace() or c in ".,:;$-/%()|[]'\"–—#&*+=@_~")
        valid_ratio = valid_chars / max(1, char_count)
        if valid_ratio < cfg["min_valid_char_ratio"]:
            return True, "suspicious_corrupted_noise", metrics

    return False, "sufficient_native_text", metrics

def should_use_ocr(page: fitz.Page, extracted_text: str) -> Tuple[bool, str]:
    """Backwards compatibility wrapper for extraction quality check."""
    needs_ocr, reason, _ = assess_page_extraction_quality(page, extracted_text)
    return needs_ocr, reason

def extract_pdf_with_metadata(file_path: str) -> Tuple[str, int, List[Dict[str, Any]]]:
    """
    Processes each PDF page individually with intelligent extraction and layout preservation:
    - Verifies document encryption and page count.
    - Preserves column reading order, bounding boxes, text spans, and font attributes.
    - Inspects and extracts text from ALL embedded images inside the page via OCR.
    - If a page is scanned or lacks text, runs high-resolution page OCR.
    - Retains page boundaries [PAGE X] and prevents duplicate text.
    - Never silently drops pages if OCR fails on one page.
    """
    if not os.path.isfile(file_path):
        raise CorruptedFileError("PDF file does not exist on disk.")

    if os.path.getsize(file_path) == 0:
        raise EmptyDocumentError("PDF file is empty (0 bytes).")

    try:
        doc = fitz.open(file_path)
    except Exception as open_err:
        raise CorruptedFileError(f"Failed to open PDF document: {open_err}")

    if doc.is_encrypted or doc.needs_pass:
        doc.close()
        raise EncryptedPdfError("Document is password-protected or encrypted. Please provide an unencrypted document.")

    page_count = len(doc)
    if page_count == 0:
        doc.close()
        raise EmptyDocumentError("PDF document contains zero pages.")

    page_results = []
    page_metadata = []
    embedded_ocr_budget = 3  # Maximum embedded images to OCR across the entire document

    logger.info(f"[PDF] Processing document: {os.path.basename(file_path)} | Total Pages: {page_count}")

    for idx, page in enumerate(doc):
        page_num = idx + 1
        page_w = round(page.rect.width, 1)
        page_h = round(page.rect.height, 1)

        # Extract structured text with column-aware reading order and font spans
        page_blocks_layout = []
        text_block_strings = []

        try:
            page_dict = page.get_text("dict", sort=True)
            raw_blocks = page_dict.get("blocks", [])

            for b_idx, b in enumerate(raw_blocks):
                if b.get("type", 0) != 0:
                    continue

                lines_layout = []
                line_texts = []
                block_font_sizes = []

                for line in b.get("lines", []):
                    span_layouts = []
                    span_texts = []
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        if not span_text.strip():
                            continue
                        font_size = round(span.get("size", 0.0), 1)
                        block_font_sizes.append(font_size)
                        is_bold = bool(span.get("flags", 0) & 2 or "bold" in span.get("font", "").lower())
                        span_layouts.append({
                            "text": span_text,
                            "font": span.get("font", ""),
                            "size": font_size,
                            "color": span.get("color", 0),
                            "alpha": span.get("alpha", 255),
                            "flags": span.get("flags", 0),
                            "is_bold": is_bold,
                            "bbox": [round(c, 1) for c in span.get("bbox", (0, 0, 0, 0))],
                        })
                        span_texts.append(span_text)

                    if span_texts:
                        line_str = " ".join(span_texts)
                        line_texts.append(line_str)
                        lines_layout.append({
                            "bbox": [round(c, 1) for c in line.get("bbox", (0, 0, 0, 0))],
                            "spans": span_layouts,
                            "text": line_str,
                        })

                if line_texts:
                    block_text = "\n".join(line_texts)
                    text_block_strings.append(block_text)
                    avg_size = round(float(sum(block_font_sizes) / len(block_font_sizes)), 1) if block_font_sizes else 0.0
                    page_blocks_layout.append({
                        "block_index": b_idx,
                        "bbox": [round(c, 1) for c in b.get("bbox", (0, 0, 0, 0))],
                        "lines": lines_layout,
                        "avg_font_size": avg_size,
                        "text": block_text,
                    })
        except Exception as dict_err:
            logger.debug(f"[PAGE {page_num}] get_text('dict') failed ({dict_err}), falling back to blocks")

        if text_block_strings:
            native_text = "\n\n".join(text_block_strings)
        else:
            native_text = page.get_text("text", sort=True) or ""

        clean_native = clean_ocr_text(native_text)
        needs_page_ocr, reason = should_use_ocr(page, clean_native)

        images = page.get_images(full=True)
        embedded_image_texts = []
        embedded_images_found_count = len(images) if images else 0

        if not needs_page_ocr:
            # FAST-PATH FOR NATIVE DIGITAL PAGES:
            # When the page already has sufficient native digital text, retain it.
            # However, if the page contains embedded images (e.g. an addendum, exhibit, or clause block),
            # inspect and extract candidate textual images without duplicating native text.
            if images and embedded_ocr_budget > 0:
                seen_xrefs = set()
                candidate_images = []
                for img_info in images:
                    xref = img_info[0]
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    try:
                        base_image = doc.extract_image(xref)
                        w = base_image.get("width", 0)
                        h = base_image.get("height", 0)
                        area = w * h
                        if w >= 150 and h >= 40 and area >= 8000:
                            candidate_images.append((area, xref, base_image))
                    except Exception:
                        pass

                if candidate_images:
                    candidate_images.sort(key=lambda x: x[0], reverse=True)
                    for area, xref, base_image in candidate_images[:2]:
                        if embedded_ocr_budget <= 0:
                            break
                        try:
                            img_bytes = base_image.get("image")
                            if not img_bytes:
                                continue
                            with Image.open(io.BytesIO(img_bytes)) as pil_img:
                                ocr_res = ocr_image_detailed(pil_img)
                                raw_extracted = ocr_res.get("text", "").strip()
                                cleaned_extracted = clean_ocr_text(raw_extracted)
                                if cleaned_extracted and len(cleaned_extracted.split()) >= 1:
                                    if _normalize_token_text(cleaned_extracted) not in _normalize_token_text(clean_native):
                                        embedded_image_texts.append(cleaned_extracted)
                                        embedded_ocr_budget -= 1
                                        logger.info(
                                            f"[PAGE {page_num}] Extracted text from embedded image xref {xref} "
                                            f"({len(cleaned_extracted)} chars)"
                                        )
                        except Exception as img_err:
                            logger.warning(f"[PAGE {page_num}] Failed extracting image xref {xref}: {img_err}")

            if embedded_image_texts:
                final_page_text = merge_native_and_ocr_text(clean_native, "\n\n".join(embedded_image_texts))
                extraction_source = "hybrid"
                extraction_method = "hybrid"
                ocr_used = True
                ocr_conf = 92.0
            else:
                final_page_text = clean_native
                extraction_source = "native_text"
                extraction_method = "native"
                ocr_used = False
                ocr_conf = 100.0

            page_results.append(f"[PAGE {page_num}]\n{final_page_text}")
            page_metadata.append({
                "page_number": page_num,
                "width": page_w,
                "height": page_h,
                "extraction_source": extraction_source,
                "extraction_method": extraction_method,
                "text_length": len(final_page_text),
                "ocr_used": ocr_used,
                "ocr_confidence": ocr_conf,
                "blocks": page_blocks_layout,
                "embedded_images_found": embedded_images_found_count,
                "embedded_images_extracted": len(embedded_image_texts),
                "reason": reason,
            })
            logger.info(f"[PAGE {page_num}] Native/Hybrid extraction complete ({len(final_page_text)} chars, source: {extraction_source})")
        else:
            # SCANNED OR INSUFFICIENT-TEXT PAGE:
            # Check for direct embedded scans first, or fallback to rasterizing the page.
            candidate_images = []
            if images:
                seen_xrefs = set()
                for img_info in images:
                    xref = img_info[0]
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    try:
                        base_image = doc.extract_image(xref)
                        w = base_image.get("width", 0)
                        h = base_image.get("height", 0)
                        area = w * h
                        if w >= 100 and h >= 40 and area >= 5000:
                            candidate_images.append((area, xref, base_image))
                    except Exception:
                        pass

            direct_ocr_method = "embedded_image_ocr"
            direct_ocr_confidence = 90.0

            if candidate_images:
                candidate_images.sort(key=lambda x: x[0], reverse=True)
                for area, xref, base_image in candidate_images[:2]:
                    try:
                        img_bytes = base_image.get("image")
                        if not img_bytes:
                            continue
                        with Image.open(io.BytesIO(img_bytes)) as pil_img:
                            ocr_res = ocr_image_detailed(pil_img)
                            raw_extracted = ocr_res.get("text", "").strip()
                            cleaned_extracted = clean_ocr_text(raw_extracted)
                            if cleaned_extracted and len(cleaned_extracted.split()) >= 1:
                                if _normalize_token_text(cleaned_extracted) not in _normalize_token_text(clean_native):
                                    embedded_image_texts.append(cleaned_extracted)
                                    direct_ocr_method = ocr_res.get("method", "embedded_image_ocr")
                                    direct_ocr_confidence = ocr_res.get("ocr_confidence", 90.0)
                    except Exception as img_err:
                        logger.warning(f"[PAGE {page_num}] Failed extracting image xref {xref}: {img_err}")

            embedded_text_block = "\n\n".join(embedded_image_texts).strip()

            if embedded_text_block and len(embedded_text_block) > 60:
                final_page_text = merge_native_and_ocr_text(clean_native, embedded_text_block) if clean_native else embedded_text_block
                is_hybrid = bool(clean_native and _normalize_token_text(clean_native) not in _normalize_token_text(embedded_text_block))
                extraction_source = "hybrid" if is_hybrid else "ocr"
                extraction_method = "embedded_image_ocr" if not is_hybrid else "hybrid"

                page_results.append(f"[PAGE {page_num}]\n{final_page_text}")
                page_metadata.append({
                    "page_number": page_num,
                    "width": page_w,
                    "height": page_h,
                    "extraction_source": extraction_source,
                    "extraction_method": extraction_method,
                    "text_length": len(final_page_text),
                    "ocr_used": True,
                    "ocr_confidence": direct_ocr_confidence,
                    "blocks": page_blocks_layout,
                    "embedded_images_found": embedded_images_found_count,
                    "embedded_images_extracted": len(embedded_image_texts),
                    "reason": reason,
                })
                logger.info(f"[PAGE {page_num}] Direct embedded scan extraction complete ({len(final_page_text)} chars)")
            else:
                logger.info(f"[PAGE {page_num}] Native text insufficient ({reason}) → Rasterizing page for OCR...")
                try:
                    zoom = 150.0 / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img_bytes = pix.tobytes("png")
                    del pix

                    with Image.open(io.BytesIO(img_bytes)) as img:
                        ocr_res = ocr_image_detailed(img)
                    del img_bytes

                    raster_ocr_text = clean_ocr_text(ocr_res.get("text", ""))

                    ocr_blocks_layout = []
                    if ocr_res.get("boxes"):
                        for b_i, obox in enumerate(ocr_res["boxes"]):
                            ocr_blocks_layout.append({
                                "block_index": b_i,
                                "bbox": obox.get("bbox", [0, 0, 0, 0]),
                                "text": obox.get("text", ""),
                                "confidence": obox.get("confidence", 0.0),
                            })

                    # Intelligent merge native text and raster OCR text to prevent duplicates
                    if clean_native and raster_ocr_text:
                        final_page_text = merge_native_and_ocr_text(clean_native, raster_ocr_text)
                        norm_nat = _normalize_token_text(clean_native)
                        norm_ocr = _normalize_token_text(raster_ocr_text)
                        if norm_nat and norm_nat not in norm_ocr and norm_ocr not in norm_nat:
                            extraction_source = "hybrid"
                            extraction_method = "hybrid"
                        else:
                            extraction_source = "ocr"
                            extraction_method = ocr_res.get("method", "rapidocr")
                    elif raster_ocr_text:
                        final_page_text = raster_ocr_text
                        extraction_source = "ocr"
                        extraction_method = ocr_res.get("method", "rapidocr")
                    else:
                        final_page_text = clean_native
                        extraction_source = "native_text"
                        extraction_method = "native"

                    page_results.append(f"[PAGE {page_num}]\n{final_page_text}")
                    page_metadata.append({
                        "page_number": page_num,
                        "width": page_w,
                        "height": page_h,
                        "extraction_source": extraction_source,
                        "extraction_method": extraction_method,
                        "text_length": len(final_page_text),
                        "ocr_used": True,
                        "ocr_confidence": ocr_res.get("ocr_confidence", 0.0),
                        "word_count": ocr_res.get("word_count", 0),
                        "blocks": page_blocks_layout if page_blocks_layout else ocr_blocks_layout,
                        "embedded_images_found": embedded_images_found_count,
                        "embedded_images_extracted": len(embedded_image_texts),
                        "reason": reason,
                    })
                    logger.info(
                        f"[PAGE {page_num}] OCR complete via {ocr_res.get('method')} "
                        f"(Confidence: {ocr_res.get('ocr_confidence')}%, Chars: {len(final_page_text)})"
                    )
                except Exception as e:
                    logger.error(f"[PAGE {page_num}] OCR rendering failed: {e}")
                    fallback_text = clean_native or embedded_text_block
                    page_results.append(f"[PAGE {page_num}]\n{fallback_text}")
                    page_metadata.append({
                        "page_number": page_num,
                        "width": page_w,
                        "height": page_h,
                        "extraction_source": "ocr_failed",
                        "extraction_method": "ocr_failed",
                        "text_length": len(fallback_text),
                        "ocr_used": True,
                        "ocr_confidence": 0.0,
                        "warning": f"Page {page_num} OCR processing failed: {str(e)}",
                        "error": str(e),
                        "blocks": page_blocks_layout,
                        "embedded_images_found": embedded_images_found_count,
                        "embedded_images_extracted": len(embedded_image_texts),
                        "reason": reason,
                    })

        # Reclaim memory periodically between pages
        if idx % 5 == 0:
            gc.collect()

    doc.close()
    full_text = "\n\n".join(page_results)
    return full_text, page_count, page_metadata

def extract_pdf(file_path: str) -> Tuple[str, int]:
    """
    Standard interface for PDF extraction: returns (full_text, page_count).
    """
    full_text, page_count, _ = extract_pdf_with_metadata(file_path)
    return full_text, page_count

def read_plain_text(file_path: str) -> Tuple[str, int]:
    """
    Reads plain text files with encoding verification, null byte checking, and cleaning.
    """
    if not os.path.isfile(file_path):
        raise CorruptedFileError("File does not exist on disk.")
    if os.path.getsize(file_path) == 0:
        raise EmptyDocumentError("Text file is empty (0 bytes).")

    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    if b"\x00" in raw_bytes:
        raise CorruptedFileError("Binary data detected in text file: contains null bytes.")

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw_bytes.decode("latin-1")
        except Exception as e:
            raise CorruptedFileError(f"Unreadable text encoding: {e}")

    cleaned = clean_ocr_text(text)
    if not cleaned.strip():
        raise EmptyDocumentError("Text file contains no readable content.")

    page_count = max(1, (len(cleaned) // 3000) + 1)
    return cleaned, page_count

def extract_docx_with_metadata(file_path: str) -> Tuple[str, int, List[Dict[str, Any]]]:
    """
    Extracts text from DOCX files preserving paragraph and table structure in document order.
    Also inspects embedded images in the DOCX package and executes OCR on images that contain
    textual legal content not already available in the document body.
    """
    if not os.path.isfile(file_path):
        raise CorruptedFileError("File does not exist on disk.")
    if os.path.getsize(file_path) == 0:
        raise EmptyDocumentError("DOCX file is empty (0 bytes).")

    try:
        doc = docx.Document(file_path)
    except Exception as e:
        raise CorruptedFileError(f"Failed to open DOCX package: {e}")

    content = []
    for child in doc.element.body:
        if child.tag.endswith("p"):
            p = docx.text.paragraph.Paragraph(child, doc)
            if p.text.strip():
                content.append(p.text.strip())
        elif child.tag.endswith("tbl"):
            tbl = docx.table.Table(child, doc)
            for row in tbl.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    content.append(" | ".join(row_cells))

    native_text = clean_ocr_text("\n\n".join(content))
    norm_native = _normalize_token_text(native_text)

    # Inspect embedded images in the DOCX container
    embedded_image_texts = []
    image_parts_found = 0

    try:
        if hasattr(doc, "part") and hasattr(doc.part, "related_parts"):
            for rel_id, rel in doc.part.related_parts.items():
                content_type = getattr(rel, "content_type", "")
                if "image" in content_type:
                    image_parts_found += 1
                    try:
                        img_bytes = rel.blob
                        with Image.open(io.BytesIO(img_bytes)) as pil_img:
                            w, h = pil_img.size
                            if w >= 100 and h >= 40 and (w * h) >= 6000:
                                ocr_res = ocr_image_detailed(pil_img)
                                img_text = clean_ocr_text(ocr_res.get("text", ""))
                                if img_text and len(img_text.split()) >= 1:
                                    norm_img = _normalize_token_text(img_text)
                                    if norm_img and norm_img not in norm_native:
                                        embedded_image_texts.append(img_text)
                                        logger.info(
                                            f"[DOCX] Extracted text from embedded image {rel_id}: {len(img_text)} chars"
                                        )
                    except Exception as img_err:
                        logger.debug(f"[DOCX] Embedded image {rel_id} inspection failed: {img_err}")
    except Exception as rel_err:
        logger.debug(f"[DOCX] Failed inspecting related parts: {rel_err}")

    if embedded_image_texts:
        final_text = merge_native_and_ocr_text(native_text, "\n\n".join(embedded_image_texts))
        ocr_used = True
        extraction_source = "hybrid" if native_text else "ocr"
        extraction_method = "docx+embedded_image_ocr"
    else:
        final_text = native_text
        ocr_used = False
        extraction_source = "native_text"
        extraction_method = "docx_native"

    if not final_text.strip():
        raise EmptyDocumentError("DOCX document contains no readable text.")

    page_count = max(1, (len(final_text) // 3000) + 1)
    metadata = [{
        "page_number": 1,
        "extraction_source": extraction_source,
        "extraction_method": extraction_method,
        "ocr_used": ocr_used,
        "text_length": len(final_text),
        "embedded_images_found": image_parts_found,
        "embedded_images_extracted": len(embedded_image_texts),
    }]
    return final_text, page_count, metadata

def extract_docx(file_path: str) -> Tuple[str, int]:
    """
    Extracts text from DOCX files preserving paragraph and table structure in document order.
    """
    text, page_count, _ = extract_docx_with_metadata(file_path)
    return text, page_count

def extract_image_with_metadata(file_path: str) -> Tuple[str, int, List[Dict[str, Any]]]:
    """
    Extracts text from standalone image uploads (PNG, JPG, JPEG) using RapidOCR.
    Reuses existing preprocessing, inverted-color retries, and bounding box extraction.
    """
    if not os.path.isfile(file_path):
        raise CorruptedFileError("Image file does not exist on disk.")
    if os.path.getsize(file_path) == 0:
        raise EmptyDocumentError("Image file is empty (0 bytes).")

    try:
        with Image.open(file_path) as pil_img:
            pil_img.load()
            w, h = pil_img.size
            ocr_res = ocr_image_detailed(pil_img)
    except Exception as e:
        raise CorruptedFileError(f"Failed to process image with OCR: {e}")

    extracted_text = clean_ocr_text(ocr_res.get("text", ""))
    if not extracted_text:
        raise EmptyDocumentError("Image contains no readable text.")

    formatted_text = f"[PAGE 1]\n{extracted_text}"
    metadata = [{
        "page_number": 1,
        "width": w,
        "height": h,
        "extraction_source": "ocr",
        "extraction_method": ocr_res.get("method", "rapidocr"),
        "text_length": len(extracted_text),
        "ocr_used": True,
        "ocr_confidence": ocr_res.get("ocr_confidence", 0.0),
        "word_count": ocr_res.get("word_count", len(extracted_text.split())),
        "blocks": ocr_res.get("boxes", []),
    }]
    return formatted_text, 1, metadata

def extract_text_and_metadata(file_path: str, ext: str) -> Tuple[str, int, list]:
    """
    Main extraction dispatcher returning extracted text, page count, and layout metadata.
    Returns: (extracted_text, page_count, page_metadata)
    """
    ext = ext.lower().lstrip(".")
    if ext == "txt":
        t, p = read_plain_text(file_path)
        return t, p, [{
            "page_number": 1,
            "extraction_source": "native_text",
            "extraction_method": "plain_text",
            "ocr_used": False,
            "text_length": len(t),
        }]
    elif ext == "docx":
        return extract_docx_with_metadata(file_path)
    elif ext == "pdf":
        return extract_pdf_with_metadata(file_path)
    elif ext in ("png", "jpg", "jpeg"):
        return extract_image_with_metadata(file_path)
    else:
        raise UnsupportedFormatError(f"Unsupported file format: .{ext}")

def extract_text(file_path: str, ext: str) -> Tuple[str, int]:
    """
    Main extraction dispatcher for backwards compatibility.
    Returns: (extracted_text, page_count)
    """
    text, pages, _ = extract_text_and_metadata(file_path, ext)
    return text, pages

