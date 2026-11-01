import unittest
import sys
import tempfile
import os
import io
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image, ImageDraw

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import fitz
import docx

from app.services.ingestion.extractor import (
    extract_pdf_with_metadata,
    extract_pdf,
    extract_docx_with_metadata,
    extract_docx,
    extract_image_with_metadata,
    extract_text_and_metadata,
    assess_page_extraction_quality,
    should_use_ocr,
    merge_native_and_ocr_text,
    clean_ocr_text,
)
from app.utils.validators import (
    validate_file_content,
    allowed_file,
    ALLOWED_EXTENSIONS,
    CorruptedFileError,
    EmptyFileError,
    FileLimitExceededError,
)

class TestConditionalOcrFallback(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.gettempdir()
        self.created_files = []

    def tearDown(self):
        for f in self.created_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def _track_file(self, path):
        self.created_files.append(path)
        return path

    # -------------------------------------------------------------------------
    # TEST A: Text-based PDF uses native extraction without unnecessary OCR
    # -------------------------------------------------------------------------
    def test_a_text_based_pdf_uses_native_extraction_without_ocr(self):
        pdf_path = self._track_file(os.path.join(self.temp_dir, "test_req_a_native.pdf"))
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        legal_text = (
            "ARTICLE 1: TERM AND TERMINATION\n"
            "This Agreement shall commence on the Effective Date and continue for a period of two (2) years.\n\n"
            "ARTICLE 2: COMPENSATION AND PAYMENT TERMS\n"
            "Client shall pay all undisputed invoices within thirty (30) calendar days of receipt.\n\n"
            "ARTICLE 3: GOVERNING LAW\n"
            "This Agreement shall be governed by and construed in accordance with the laws of Delaware."
        )
        page.insert_text((50, 50), legal_text)
        doc.save(pdf_path)
        doc.close()

        with patch("app.services.ingestion.extractor.ocr_image_detailed") as mock_ocr:
            full_text, page_count, metadata = extract_pdf_with_metadata(pdf_path)

            mock_ocr.assert_not_called()
            self.assertEqual(page_count, 1)
            self.assertIn("[PAGE 1]", full_text)
            self.assertIn("ARTICLE 1: TERM AND TERMINATION", full_text)
            self.assertIn("ARTICLE 2: COMPENSATION", full_text)
            self.assertEqual(metadata[0]["extraction_source"], "native_text")
            self.assertEqual(metadata[0]["extraction_method"], "native")
            self.assertFalse(metadata[0]["ocr_used"])
            self.assertEqual(metadata[0]["ocr_confidence"], 100.0)

    # -------------------------------------------------------------------------
    # TEST B: Fully scanned PDF uses OCR for its image-based pages
    # -------------------------------------------------------------------------
    def test_b_fully_scanned_pdf_uses_ocr(self):
        pdf_path = self._track_file(os.path.join(self.temp_dir, "test_req_b_scanned.pdf"))
        img = Image.new("RGB", (700, 250), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((30, 40), "SECTION 8: LIMITATION OF LIABILITY", fill="black")
        draw.text((30, 90), "In no event shall either party be liable for punitive damages.", fill="black")
        draw.text((30, 140), "Total cumulative liability shall not exceed the fees paid in twelve months.", fill="black")

        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        doc = fitz.open()
        page = doc.new_page(width=700, height=250)
        page.insert_image(page.rect, stream=img_bytes.read())
        doc.save(pdf_path)
        doc.close()

        full_text, page_count, metadata = extract_pdf_with_metadata(pdf_path)
        self.assertEqual(page_count, 1)
        self.assertIn("[PAGE 1]", full_text)
        self.assertTrue(metadata[0]["ocr_used"])
        self.assertIn(metadata[0]["extraction_source"], ["ocr", "hybrid"])
        self.assertIn(metadata[0]["extraction_method"], ["rapidocr", "tesseract", "embedded_image_ocr", "hybrid"])
        self.assertTrue(len(full_text) > 20)
        text_lower = full_text.lower()
        self.assertTrue(
            "liab" in text_lower or "party" in text_lower or "damage" in text_lower or "punitive" in text_lower,
            f"Expected OCR recognized text, got: {full_text}"
        )

    # -------------------------------------------------------------------------
    # TEST C: Mixed PDF uses native on text pages and OCR on scanned pages
    # -------------------------------------------------------------------------
    def test_c_mixed_pdf_per_page_decisions(self):
        pdf_path = self._track_file(os.path.join(self.temp_dir, "test_req_c_mixed.pdf"))
        doc = fitz.open()

        # Page 1: Native Digital
        p1 = doc.new_page(width=612, height=792)
        p1.insert_text(
            (50, 50),
            "PAGE 1 CONTRACT PROVISIONS:\n"
            "Tenant shall maintain comprehensive general liability insurance throughout the term."
        )

        # Page 2: Scanned Image
        img = Image.new("RGB", (650, 220), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((25, 40), "SCHEDULE B: ESCROW AND INDEMNIFICATION", fill="black")
        draw.text((25, 80), "The escrow agent shall release funds upon mutual written authorization.", fill="black")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        p2 = doc.new_page(width=650, height=220)
        p2.insert_image(p2.rect, stream=img_bytes.read())

        doc.save(pdf_path)
        doc.close()

        full_text, page_count, metadata = extract_pdf_with_metadata(pdf_path)
        self.assertEqual(page_count, 2)

        # Verify Page 1 was purely native
        self.assertEqual(metadata[0]["extraction_source"], "native_text")
        self.assertEqual(metadata[0]["extraction_method"], "native")
        self.assertFalse(metadata[0]["ocr_used"])

        # Verify Page 2 used OCR
        self.assertTrue(metadata[1]["ocr_used"])
        self.assertIn(metadata[1]["extraction_source"], ["ocr", "hybrid"])

        self.assertIn("[PAGE 1]", full_text)
        self.assertIn("[PAGE 2]", full_text)
        self.assertIn("liability insurance", full_text.lower())

    # -------------------------------------------------------------------------
    # TEST D: Native page number / footer + image contractual body triggers OCR
    # -------------------------------------------------------------------------
    def test_d_native_page_number_with_scanned_body_triggers_ocr(self):
        pdf_path = self._track_file(os.path.join(self.temp_dir, "test_req_d_header_scanned.pdf"))
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        # Insert only a native digital page number / footer at the bottom
        page.insert_text((50, 750), "Page 1 of 5")

        # Insert a large scanned image containing the operative legal text
        img = Image.new("RGB", (550, 600), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((30, 40), "CONFIDENTIAL NON-SOLICITATION COVENANT", fill="black")
        draw.text((30, 90), "Employee shall not solicit customers for twelve calendar months.", fill="black")
        draw.text((30, 140), "Remedies for breach include immediate injunctive relief without bond.", fill="black")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        rect = fitz.Rect(30, 50, 580, 700)
        page.insert_image(rect, stream=img_bytes.read())

        doc.save(pdf_path)
        doc.close()

        full_text, page_count, metadata = extract_pdf_with_metadata(pdf_path)
        self.assertEqual(page_count, 1)

        # Quality assessment must detect that native text is only a footer while image dominates
        self.assertTrue(metadata[0]["ocr_used"], "OCR should have triggered despite native 'Page 1 of 5'")
        self.assertIn(metadata[0]["extraction_source"], ["hybrid", "ocr"])

        # Both the body text and the footer should be retained without error
        text_lower = full_text.lower()
        self.assertTrue(
            "solicit" in text_lower or "covenant" in text_lower or "relief" in text_lower,
            f"Expected scanned body text, got: {full_text}"
        )
        self.assertIn("Page 1 of 5", full_text)

    # -------------------------------------------------------------------------
    # TEST E: Mixed page with native text + scanned content does not duplicate
    # -------------------------------------------------------------------------
    def test_e_mixed_page_no_text_loss_and_no_duplication(self):
        pdf_path = self._track_file(os.path.join(self.temp_dir, "test_req_e_hybrid.pdf"))
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        # Native top header and paragraph
        page.insert_text((50, 50), "COMMERCIAL LEASE AGREEMENT\nSECTION 1: PREMISES AND USE\nLandlord leases to Tenant.")

        # Embedded scanned image with addendum
        img = Image.new("RGB", (500, 150), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 30), "ADDENDUM A: UTILITY AND HVAC CHARGES", fill="black")
        draw.text((20, 70), "Tenant agrees to pay electricity and HVAC expenses monthly.", fill="black")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        rect = fitz.Rect(50, 150, 550, 300)
        page.insert_image(rect, stream=img_bytes.read())

        # Native bottom paragraph
        page.insert_text((50, 350), "SECTION 2: ASSIGNMENT AND SUBLETTING\nNo assignment without Landlord consent.")

        doc.save(pdf_path)
        doc.close()

        full_text, page_count, metadata = extract_pdf_with_metadata(pdf_path)
        self.assertEqual(page_count, 1)
        self.assertTrue(metadata[0]["ocr_used"])
        self.assertEqual(metadata[0]["extraction_source"], "hybrid")

        # Verify no loss of native or image content
        self.assertIn("COMMERCIAL LEASE AGREEMENT", full_text)
        self.assertIn("SECTION 1: PREMISES", full_text)
        self.assertIn("SECTION 2: ASSIGNMENT", full_text)
        text_lower = full_text.lower()
        self.assertTrue("hvac" in text_lower or "utility" in text_lower or "addendum" in text_lower)

        # Verify NO duplicate occurrences of native section headings
        self.assertEqual(full_text.count("COMMERCIAL LEASE AGREEMENT"), 1)
        self.assertEqual(full_text.count("SECTION 1: PREMISES"), 1)
        self.assertEqual(full_text.count("SECTION 2: ASSIGNMENT"), 1)

    # -------------------------------------------------------------------------
    # TEST F: DOCX with ordinary paragraphs + embedded scanned image
    # -------------------------------------------------------------------------
    def test_f_docx_with_embedded_scanned_image(self):
        docx_path = self._track_file(os.path.join(self.temp_dir, "test_req_f_embedded.docx"))
        doc = docx.Document()
        doc.add_paragraph("EMPLOYMENT AGREEMENT: Base Salary and Bonus Structure.")
        doc.add_paragraph("The employee shall be paid a base salary of $120,000 per annum.")

        # Create image with restrictive covenant
        img = Image.new("RGB", (600, 120), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 30), "SCHEDULE 1: INTELLECTUAL PROPERTY ASSIGNMENT", fill="black")
        draw.text((20, 70), "All inventions created during employment belong solely to Company.", fill="black")

        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        doc.add_picture(img_bytes)
        doc.add_paragraph("IN WITNESS WHEREOF, the parties execute this Agreement.")
        doc.save(docx_path)

        full_text, page_count, metadata = extract_docx_with_metadata(docx_path)
        self.assertTrue(metadata[0]["ocr_used"])
        self.assertEqual(metadata[0]["extraction_source"], "hybrid")
        self.assertEqual(metadata[0]["extraction_method"], "docx+embedded_image_ocr")
        self.assertIn("EMPLOYMENT AGREEMENT", full_text)
        self.assertIn("base salary", full_text)
        self.assertIn("IN WITNESS WHEREOF", full_text)

        text_lower = full_text.lower()
        self.assertTrue(
            "inve" in text_lower or "creat" in text_lower or "sole" in text_lower or "property" in text_lower,
            f"Expected OCR extracted image content in DOCX, got: {full_text}"
        )

    # -------------------------------------------------------------------------
    # TEST G: DOCX with no embedded textual images does not trigger unnecessary OCR
    # -------------------------------------------------------------------------
    def test_g_docx_no_images_does_not_trigger_ocr(self):
        docx_path = self._track_file(os.path.join(self.temp_dir, "test_req_g_text_only.docx"))
        doc = docx.Document()
        doc.add_paragraph("CONSULTING SERVICES AGREEMENT")
        doc.add_paragraph("Consultant shall perform services with professional skill and diligence.")
        doc.add_paragraph("Fees shall be billed at $200 per hour up to a maximum cap of $50,000.")
        doc.save(docx_path)

        with patch("app.services.ingestion.extractor.ocr_image_detailed") as mock_ocr:
            full_text, page_count, metadata = extract_docx_with_metadata(docx_path)

            mock_ocr.assert_not_called()
            self.assertFalse(metadata[0]["ocr_used"])
            self.assertEqual(metadata[0]["extraction_source"], "native_text")
            self.assertEqual(metadata[0]["extraction_method"], "docx_native")
            self.assertIn("CONSULTING SERVICES AGREEMENT", full_text)

    # -------------------------------------------------------------------------
    # TEST H: Standalone image upload (PNG, JPG) is processed using OCR engine
    # -------------------------------------------------------------------------
    def test_h_standalone_image_upload_processed_via_ocr(self):
        png_path = self._track_file(os.path.join(self.temp_dir, "test_req_h_image.png"))
        img = Image.new("RGB", (700, 200), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((25, 40), "MUTUAL NON-DISCLOSURE AGREEMENT", fill="black")
        draw.text((25, 90), "Receiving Party shall preserve confidentiality of Proprietary Info.", fill="black")
        img.save(png_path)

        # Verify validator accepts PNG
        self.assertTrue(allowed_file("test_req_h_image.png"))
        validate_file_content(png_path, "png")

        # Test extraction
        full_text, page_count, metadata = extract_text_and_metadata(png_path, "png")
        self.assertEqual(page_count, 1)
        self.assertIn("[PAGE 1]", full_text)
        self.assertTrue(metadata[0]["ocr_used"])
        self.assertEqual(metadata[0]["extraction_source"], "ocr")
        self.assertIn(metadata[0]["extraction_method"], ["rapidocr", "tesseract"])
        text_lower = full_text.lower()
        self.assertTrue(
            "disclosure" in text_lower or "confidential" in text_lower or "proprietary" in text_lower or "mutual" in text_lower,
            f"Expected OCR recognized text from image upload, got: {full_text}"
        )

    # -------------------------------------------------------------------------
    # TEST I: Failed OCR page is flagged with warning and does not silently misrepresent
    # -------------------------------------------------------------------------
    def test_i_failed_ocr_page_flagged_with_warning(self):
        pdf_path = self._track_file(os.path.join(self.temp_dir, "test_req_i_failed.pdf"))
        img = Image.new("RGB", (500, 200), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        doc = fitz.open()
        p1 = doc.new_page(width=500, height=200)
        p1.insert_image(p1.rect, stream=img_bytes.read())
        doc.save(pdf_path)
        doc.close()

        with patch("app.services.ingestion.extractor.ocr_image_detailed", side_effect=RuntimeError("Simulated neural OCR failure")):
            full_text, page_count, metadata = extract_pdf_with_metadata(pdf_path)

            self.assertEqual(page_count, 1)
            self.assertEqual(metadata[0]["extraction_source"], "ocr_failed")
            self.assertEqual(metadata[0]["extraction_method"], "ocr_failed")
            self.assertTrue(metadata[0]["ocr_used"])
            self.assertEqual(metadata[0]["ocr_confidence"], 0.0)
            self.assertIn("warning", metadata[0])
            self.assertIn("Simulated neural OCR failure", metadata[0]["warning"])

    # -------------------------------------------------------------------------
    # TEST J: Downstream pipeline gating and regression tests continuity
    # -------------------------------------------------------------------------
    def test_j_downstream_substantive_eligibility_and_segmentation(self):
        from app.services.nlp.block_segmenter import segment_blocks
        from app.services.nlp.clause_eligibility_gate import classify_content_type

        # Simulate text containing OCR output with mixed headers, page numbers, and substantive clauses
        ocr_mixed_text = (
            "[PAGE 1]\n"
            "Page 1 of 3\n\n"
            "STANDARD RESIDENTIAL LEASE AGREEMENT\n\n"
            "1. TERM OF LEASE\n"
            "The Lease Term shall commence on October 1, 2026 and continue month-to-month thereafter until terminated.\n\n"
            "2. SECURITY DEPOSIT AND DEFAULT PENALTIES\n"
            "Tenant shall forfeit entire security deposit and pay double rent immediately upon any default."
        )

        blocks = segment_blocks(ocr_mixed_text, document_type="rental_agreement")
        self.assertGreater(len(blocks), 0)

        # Filter out non-substantive blocks
        substantive = [b for b in blocks if b.get("is_substantive")]
        non_substantive = [b for b in blocks if not b.get("is_substantive")]

        # Headings or page number must be non-substantive
        self.assertTrue(len(non_substantive) >= 1)
        non_sub_texts = " ".join(b["text"].lower() for b in non_substantive)
        self.assertTrue("page 1 of 3" in non_sub_texts or "standard residential" in non_sub_texts or "term of lease" in non_sub_texts)

        # Operative clauses must be substantive
        self.assertTrue(len(substantive) >= 1)
        sub_texts = " ".join(b["text"].lower() for b in substantive)
        self.assertTrue("security deposit" in sub_texts or "commence on october" in sub_texts)

    # -------------------------------------------------------------------------
    # TEST EXTRA: Quality Assessment Decision Function Verification
    # -------------------------------------------------------------------------
    def test_quality_decision_function_signals(self):
        # Signal 1: Pure scanned page zero text
        needs_ocr, reason, m = assess_page_extraction_quality(None, "")
        self.assertFalse(needs_ocr)
        self.assertEqual(reason, "blank_page_no_text")

        # Signal 2: Corrupted text with excessive replacement characters
        corrupt_text = "Standard lease with \ufffd\ufffd\ufffd corrupt unicode font glyphs"
        needs_ocr, reason, m = assess_page_extraction_quality(None, corrupt_text)
        self.assertTrue(needs_ocr)
        self.assertEqual(reason, "encoding_artifacts_detected")

        # Signal 3: Unprintable binary noise
        unprintable_noise = "Valid\x00\x01\x02\x03\x04\x05\x06\x07Text"
        needs_ocr, reason, m = assess_page_extraction_quality(None, unprintable_noise)
        self.assertTrue(needs_ocr)
        self.assertEqual(reason, "unprintable_binary_noise")

        # Signal 4: Clean substantive native text
        clean_native = (
            "ARTICLE 1: DEFINITIONS\n"
            "As used in this Agreement, the following terms shall have the meanings specified below."
        )
        needs_ocr, reason, m = assess_page_extraction_quality(None, clean_native)
        self.assertFalse(needs_ocr)
        self.assertEqual(reason, "sufficient_native_text")

    # -------------------------------------------------------------------------
    # TEST EXTRA: Merge native and OCR text deduplication
    # -------------------------------------------------------------------------
    def test_merge_native_and_ocr_deduplication(self):
        native = "PAGE 1\nARTICLE 1: TERM\nLease is for one year."
        ocr = "PAGE 1\nARTICLE 1: TERM\nLease is for one year.\nARTICLE 2: RENT\nRent is $1,500."

        merged = merge_native_and_ocr_text(native, ocr)
        # Should cleanly merge without duplicate ARTICLE 1
        self.assertEqual(merged.count("ARTICLE 1: TERM"), 1)
        self.assertIn("ARTICLE 2: RENT", merged)

if __name__ == "__main__":
    unittest.main()
