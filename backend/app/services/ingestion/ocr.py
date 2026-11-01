import io
import os
import logging
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image, ImageOps, ImageFilter
try:
    import cv2
except Exception as _cv_err:
    cv2 = None
import pytesseract
from flask import current_app

logger = logging.getLogger(__name__)

_rapidocr_engine = None

def get_tesseract_lang() -> str:
    """Configurable Tesseract OCR language (default 'eng')."""
    return os.environ.get("TESSERACT_LANG", "eng")

def get_tesseract_config() -> str:
    """Configurable Tesseract page segmentation mode (default '--oem 3 --psm 3')."""
    return os.environ.get("TESSERACT_CONFIG", "--oem 3 --psm 3")

def configure_tesseract_cmd():
    """Configures pytesseract command from TESSERACT_CMD env or Flask config if present."""
    cmd = os.environ.get("TESSERACT_CMD")
    if not cmd and current_app:
        cmd = current_app.config.get("TESSERACT_CMD")
    if cmd and os.path.exists(cmd):
        pytesseract.pytesseract.tesseract_cmd = cmd

_tesseract_available_cached = None

def is_tesseract_available() -> bool:
    """Verifies whether Tesseract executable is installed and runnable (cached)."""
    global _tesseract_available_cached
    if _tesseract_available_cached is not None:
        return _tesseract_available_cached
    try:
        configure_tesseract_cmd()
        pytesseract.get_tesseract_version()
        _tesseract_available_cached = True
    except Exception:
        _tesseract_available_cached = False
    return _tesseract_available_cached

def get_rapidocr():
    """Initializes or returns singleton RapidOCR ONNX fallback engine."""
    global _rapidocr_engine
    if _rapidocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _rapidocr_engine = RapidOCR()
            logger.info("RapidOCR fallback engine initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize RapidOCR fallback: {e}")
    return _rapidocr_engine

def preprocess_for_tesseract(image: Image.Image) -> Image.Image:
    """
    Fast grayscale, Gaussian blur, and Otsu binary thresholding for Tesseract.
    Handles transparency, CMYK, and 1-bit modes cleanly.
    """
    try:
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            alpha = image.convert("RGBA").split()[-1]
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=alpha)
            image = bg
        elif image.mode == "CMYK":
            image = image.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        cv_img = np.array(image)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return Image.fromarray(thresh)
    except Exception as e:
        logger.debug(f"Tesseract preprocessing failed, falling back to Pillow: {e}")
        try:
            return ImageOps.grayscale(image.convert("RGB")).filter(ImageFilter.SHARPEN)
        except Exception:
            return image

def preprocess_for_rapidocr(image: Image.Image) -> np.ndarray:
    """
    Prepares images for RapidOCR deep learning model.
    Handles transparency by compositing onto pure white background.
    Preserves natural RGB and performs high-quality upscaling on small images
    for optimal character stroke detection.
    """
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        alpha = image.convert("RGBA").split()[-1]
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=alpha)
        image = bg
    elif image.mode == "CMYK":
        image = image.convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    # Bound image dimensions to prevent ONNX memory spikes and long stalls on cloud containers
    max_dim = max(w, h)
    if max_dim > 1200:
        scale = 1200.0 / max_dim
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    elif w < 300 and h < 80 and w > 0 and h > 0:
        scale = min(2.5, max(1.5, 300.0 / w, 80.0 / h))
        if scale > 1.2:
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.array(image)

def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Backwards compatibility alias for image preprocessing."""
    return preprocess_for_tesseract(image)

def ocr_image_detailed(image: Image.Image) -> Dict[str, Any]:
    """
    Performs OCR on an image with full audit metadata and bounding box extraction.
    Tries Tesseract if available, or seamlessly leverages RapidOCR deep-learning engine.
    Includes inverted-color retry for white-on-dark text and transparency handling.
    Never crashes the calling process.
    """
    if image is None or not isinstance(image, Image.Image):
        return {
            "text": "",
            "method": "ocr_none",
            "ocr_confidence": 0.0,
            "word_count": 0,
            "boxes": [],
            "error": "Invalid image: input is None or not a PIL.Image instance.",
        }

    if image.width == 0 or image.height == 0:
        return {
            "text": "",
            "method": "ocr_none",
            "ocr_confidence": 0.0,
            "word_count": 0,
            "boxes": [],
            "error": "Image has 0 width or height.",
        }

    boxes = []

    if is_tesseract_available():
        try:
            processed = preprocess_for_tesseract(image)
            lang = get_tesseract_lang()
            config = get_tesseract_config()
            configure_tesseract_cmd()

            data = pytesseract.image_to_data(processed, lang=lang, config=config, output_type=pytesseract.Output.DICT)
            text = pytesseract.image_to_string(processed, lang=lang, config=config).strip()

            confs = [float(c) for c in data.get("conf", []) if str(c).strip() not in ("-1", "")]
            avg_conf = round(float(np.mean(confs)), 2) if confs else 0.0
            word_count = len([w for w in data.get("text", []) if w.strip()])

            for i, w in enumerate(data.get("text", [])):
                w_str = str(w).strip()
                if w_str:
                    l = data["left"][i]
                    t = data["top"][i]
                    wb = data["width"][i]
                    hb = data["height"][i]
                    c = float(data["conf"][i]) if str(data["conf"][i]).strip() not in ("-1", "") else 0.0
                    boxes.append({
                        "text": w_str,
                        "bbox": [l, t, l + wb, t + hb],
                        "confidence": round(c, 1),
                    })

            if text and len(text.split()) >= 1:
                logger.info(f"[OCR] Tesseract extraction succeeded (Confidence: {avg_conf}%, Words: {word_count})")
                return {
                    "text": text,
                    "method": "tesseract",
                    "ocr_confidence": avg_conf,
                    "word_count": word_count,
                    "boxes": boxes,
                    "error": None,
                }
        except Exception as e:
            logger.info(f"[OCR] Tesseract failed ({e}). Proceeding to RapidOCR...")

    rapid = get_rapidocr()
    if rapid:
        try:
            cv_img = preprocess_for_rapidocr(image)
            result, _ = rapid(cv_img)

            if not result and np.mean(cv_img) < 85:
                try:
                    rgb_img = image.convert("RGB")
                    inv_img = ImageOps.invert(rgb_img)
                    cv_img_inv = preprocess_for_rapidocr(inv_img)
                    result, _ = rapid(cv_img_inv)
                except Exception:
                    pass

            if result:
                lines = []
                confs = []
                rapid_boxes = []
                for item in result:
                    lines.append(item[1])
                    conf_val = float(item[2]) * 100.0 if len(item) > 2 and isinstance(item[2], (int, float)) else 85.0
                    confs.append(conf_val)
                    if len(item) > 0 and isinstance(item[0], (list, np.ndarray)):
                        xs = [pt[0] for pt in item[0]]
                        ys = [pt[1] for pt in item[0]]
                        rapid_boxes.append({
                            "text": item[1],
                            "bbox": [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)],
                            "confidence": round(conf_val, 1),
                        })

                combined_text = "\n".join(lines).strip()
                avg_conf = round(float(np.mean(confs)), 2) if confs else 85.0
                word_count = len(combined_text.split())

                if combined_text:
                    logger.info(f"[OCR] RapidOCR extraction succeeded (Confidence: {avg_conf}%, Words: {word_count})")
                    return {
                        "text": combined_text,
                        "method": "rapidocr",
                        "ocr_confidence": avg_conf,
                        "word_count": word_count,
                        "boxes": rapid_boxes,
                        "error": None,
                    }
        except Exception as ex:
            logger.error(f"[OCR] RapidOCR extraction failed: {ex}")

    return {
        "text": "",
        "method": "ocr_none",
        "ocr_confidence": 0.0,
        "word_count": 0,
        "boxes": [],
        "error": None,
    }

def ocr_image(image: Image.Image) -> str:
    """Standard interface: returns extracted text string."""
    res = ocr_image_detailed(image)
    return res.get("text", "")

