import unittest
import sys
import tempfile
import os
import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import fitz
from app.services.ingestion.extractor import (
    extract_pdf_with_metadata,
    extract_pdf,
    clean_ocr_text,
    should_use_ocr,
)

class TestRobustPdfOcr(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.gettempdir()

    def test_hyphen_normalization(self):
        text = "This agreement allows early termina-\n  tion under section 4."
        cleaned = clean_ocr_text(text)
        self.assertIn("termination", cleaned)

        legal_compound = "The employee agrees to a reasonable non-compete covenant."
        cleaned_compound = clean_ocr_text(legal_compound)
        self.assertIn("non-compete", cleaned_compound)

    def test_native_text_pdf(self):
        pdf_path = os.path.join(self.temp_dir, "test_native.pdf")
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text((50, 50), "ARTICLE 1: TERM AND TERMINATION\nThis lease shall continue for twelve full calendar months with standard notice covenants.")
        doc.save(pdf_path)
        doc.close()

        full_text, page_count, metadata = extract_pdf_with_metadata(pdf_path)
        self.assertEqual(page_count, 1)
        self.assertIn("[PAGE 1]", full_text)
        self.assertIn("ARTICLE 1: TERM AND TERMINATION", full_text)
        self.assertEqual(metadata[0]["extraction_method"], "native")
        self.assertFalse(metadata[0]["ocr_used"])
        os.remove(pdf_path)

    def test_scanned_image_pdf(self):
        pdf_path = os.path.join(self.temp_dir, "test_scanned.pdf")
        img = Image.new("RGB", (600, 200), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 40), "CONFIDENTIAL SETTLEMENT CLAUSE", fill="black")
        draw.text((20, 80), "The parties agree to mutual non-disclosure.", fill="black")
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        doc = fitz.open()
        page = doc.new_page(width=600, height=200)
        page.insert_image(page.rect, stream=img_bytes.read())
        doc.save(pdf_path)
        doc.close()

        full_text, page_count, metadata = extract_pdf_with_metadata(pdf_path)
        self.assertEqual(page_count, 1)
        self.assertIn("[PAGE 1]", full_text)
        self.assertTrue(metadata[0]["ocr_used"])
        self.assertIn(metadata[0]["extraction_method"], ["tesseract", "rapidocr", "embedded_image_ocr"])
        self.assertTrue(len(full_text) > 10)
        os.remove(pdf_path)

    def test_mixed_pdf(self):
        pdf_path = os.path.join(self.temp_dir, "test_mixed.pdf")
        doc = fitz.open()

        p1 = doc.new_page()
        p1.insert_text((50, 50), "PAGE 1 CONTENT: Standard employment offer letter with statutory at-will provisions and compensation schedule.")

        img = Image.new("RGB", (600, 200), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 50), "EXHIBIT A: RESTRICTIVE COVENANTS", fill="black")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        p2 = doc.new_page(width=600, height=200)
        p2.insert_image(p2.rect, stream=img_bytes.read())

        doc.save(pdf_path)
        doc.close()

        full_text, page_count, metadata = extract_pdf_with_metadata(pdf_path)
        self.assertEqual(page_count, 2)
        self.assertEqual(metadata[0]["extraction_method"], "native")
        self.assertFalse(metadata[0]["ocr_used"])
        self.assertTrue(metadata[1]["ocr_used"])
        self.assertIn(metadata[1]["extraction_method"], ["tesseract", "rapidocr", "embedded_image_ocr"])
        self.assertIn("[PAGE 1]", full_text)
        self.assertIn("[PAGE 2]", full_text)
        os.remove(pdf_path)

    def test_embedded_image_inside_text_page(self):
        """
        Verifies that when a PDF page has digital text AND an embedded image containing text,
        both the native digital text and the embedded image text are extracted cleanly.
        """
        pdf_path = os.path.join(self.temp_dir, "test_embedded_inside_text.pdf")
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        page.insert_text((50, 50), "COMMERCIAL LEASE AGREEMENT SECTION 1: PREMISES\nLandlord leases to Tenant the commercial property.")

        img = Image.new("RGB", (600, 150), color="white")
        d = ImageDraw.Draw(img)
        d.text((20, 20), "ADDENDUM CLAUSE: RENT ESCALATION", fill="black")
        d.text((20, 60), "Base rent shall escalate by five percent annually on lease anniversary.", fill="black")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        rect = fitz.Rect(50, 150, 550, 280)
        page.insert_image(rect, stream=img_bytes.read())
        page.insert_text((50, 320), "SECTION 2: DEFAULT REMEDIES\nFailure to cure within ten days constitutes default.")

        doc.save(pdf_path)
        doc.close()

        full_text, page_count, metadata = extract_pdf_with_metadata(pdf_path)
        self.assertEqual(page_count, 1)
        self.assertIn("[PAGE 1]", full_text)
        self.assertIn("COMMERCIAL LEASE AGREEMENT", full_text)
        self.assertIn("SECTION 2: DEFAULT REMEDIES", full_text)
        self.assertTrue(metadata[0]["ocr_used"])
        self.assertGreaterEqual(metadata[0]["embedded_images_found"], 1)
        self.assertGreaterEqual(metadata[0]["embedded_images_extracted"], 1)
        full_lower = full_text.lower()
        self.assertTrue(
            "escalat" in full_lower or "rent" in full_lower or "addendum" in full_lower,
            f"Expected image text in extraction, got: {full_text}"
        )
        os.remove(pdf_path)

    def test_invalid_and_zero_dim_image_ocr(self):
        """Verifies that ocr_image_detailed handles None, non-image types, and 0x0 images gracefully."""
        from app.services.ingestion.ocr import ocr_image_detailed

        # None input
        res_none = ocr_image_detailed(None)
        self.assertEqual(res_none["method"], "ocr_none")
        self.assertEqual(res_none["text"], "")
        self.assertEqual(res_none["boxes"], [])
        self.assertIsNotNone(res_none.get("error"))

        # String input instead of PIL Image
        res_str = ocr_image_detailed("not an image")
        self.assertEqual(res_str["method"], "ocr_none")
        self.assertEqual(res_str["text"], "")

        # 0x0 image
        zero_img = Image.new("RGB", (0, 0))
        res_zero = ocr_image_detailed(zero_img)
        self.assertEqual(res_zero["method"], "ocr_none")
        self.assertEqual(res_zero["text"], "")
        self.assertIn("0 width or height", res_zero.get("error", ""))

    def test_cmyk_and_transparency_color_modes(self):
        """Verifies preprocessing handles CMYK, 1-bit monochrome, and RGBA transparency cleanly."""
        from app.services.ingestion.ocr import preprocess_for_tesseract, preprocess_for_rapidocr

        # CMYK image
        cmyk_img = Image.new("CMYK", (100, 50), color=(10, 20, 30, 40))
        tess_cmyk = preprocess_for_tesseract(cmyk_img)
        self.assertIsNotNone(tess_cmyk)
        rapid_cmyk = preprocess_for_rapidocr(cmyk_img)
        self.assertIsNotNone(rapid_cmyk)

        # 1-bit monochrome image
        mono_img = Image.new("1", (100, 50), color=1)
        tess_mono = preprocess_for_tesseract(mono_img)
        self.assertIsNotNone(tess_mono)
        rapid_mono = preprocess_for_rapidocr(mono_img)
        self.assertIsNotNone(rapid_mono)

        # RGBA image with alpha transparency
        rgba_img = Image.new("RGBA", (100, 50), color=(255, 128, 0, 128))
        tess_rgba = preprocess_for_tesseract(rgba_img)
        self.assertIsNotNone(tess_rgba)
        rapid_rgba = preprocess_for_rapidocr(rgba_img)
        self.assertIsNotNone(rapid_rgba)

    def test_ocr_boxes_structure(self):
        """Verifies that OCR returns structured bounding boxes when text is recognized."""
        from app.services.ingestion.ocr import ocr_image_detailed

        img = Image.new("RGB", (400, 100), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 30), "INDEMNITY CLAUSE", fill="black")

        res = ocr_image_detailed(img)
        self.assertIn("boxes", res)
        self.assertIsInstance(res["boxes"], list)
        if res["boxes"]:
            box = res["boxes"][0]
            self.assertIn("text", box)
            self.assertIn("bbox", box)
            self.assertEqual(len(box["bbox"]), 4)
            self.assertIn("confidence", box)

    def test_ocr_failure_does_not_drop_other_pages(self):
        """Verifies that an OCR failure or exception on one page does not drop successfully extracted pages."""
        from unittest.mock import patch

        pdf_path = os.path.join(self.temp_dir, "test_ocr_resilience.pdf")
        doc = fitz.open()

        # Page 1: Native text
        p1 = doc.new_page()
        p1.insert_text((50, 50), "ARTICLE 1: VALID NATIVE TEXT CONTENT THAT MUST BE PRESERVED.")

        # Page 2: Scanned image page that triggers OCR
        img = Image.new("RGB", (400, 200), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        p2 = doc.new_page(width=400, height=200)
        p2.insert_image(p2.rect, stream=img_bytes.read())

        doc.save(pdf_path)
        doc.close()

        # Patch ocr_image_detailed to raise an unexpected runtime error when called
        with patch("app.services.ingestion.extractor.ocr_image_detailed", side_effect=RuntimeError("Simulated OCR engine failure")):
            full_text, page_count, metadata = extract_pdf_with_metadata(pdf_path)

        self.assertEqual(page_count, 2)
        # Page 1 must still be present!
        self.assertIn("[PAGE 1]", full_text)
        self.assertIn("ARTICLE 1: VALID NATIVE TEXT CONTENT THAT MUST BE PRESERVED", full_text)
        # Page 2 must still have metadata recorded, not crashed
        self.assertEqual(len(metadata), 2)
        self.assertEqual(metadata[0]["extraction_method"], "native")
        self.assertIn("Simulated OCR engine failure", str(metadata[1].get("error", "")))
        os.remove(pdf_path)

if __name__ == "__main__":
    unittest.main()

