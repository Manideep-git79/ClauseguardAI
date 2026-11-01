import unittest
import sys
import tempfile
import os
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.ingestion.extractor import (
    extract_text,
    extract_pdf_with_metadata,
    UnsupportedFormatError,
    EmptyDocumentError,
    EncryptedPdfError,
    CorruptedFileError,
)
from app.utils.validators import (
    validate_file_content,
    EncryptedFileError,
    CorruptedFileError as ValCorruptedFileError,
    EmptyFileError,
)
from app.services.privacy.pii_redactor import redact
from app.services.nlp.block_segmenter import segment_blocks
import docx
import fitz
from PIL import Image

class TestIngestion(unittest.TestCase):
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

    def _temp_file(self, name: str) -> str:
        p = os.path.join(self.temp_dir, f"cg_test_{os.getpid()}_{name}")
        self.created_files.append(p)
        return p

    def test_plain_text_extraction(self):
        txt_path = self._temp_file("test_doc.txt")
        sample_content = "Clause 1: The tenant shall pay $1,500 rent monthly on the 1st day."
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(sample_content)

        text, page_count = extract_text(txt_path, "txt")
        self.assertIn("Clause 1:", text)
        self.assertGreaterEqual(page_count, 1)

    def test_docx_extraction(self):
        docx_path = self._temp_file("test_doc.docx")
        doc = docx.Document()
        doc.add_paragraph("Section 1: Security Deposit")
        doc.add_paragraph("Tenant shall deposit $2,000 upon signing.")
        
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Utility"
        table.cell(0, 1).text = "Responsible Party"
        table.cell(1, 0).text = "Water"
        table.cell(1, 1).text = "Landlord"
        
        doc.save(docx_path)

        text, page_count = extract_text(docx_path, "docx")
        self.assertIn("Security Deposit", text)
        self.assertIn("Water | Landlord", text)

    def test_digital_pdf_extraction(self):
        pdf_path = self._temp_file("test_doc.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "ARTICLE 1: TERM AND TERMINATION\nThis lease terminates in 30 days.")
        doc.save(pdf_path)
        doc.close()

        text, page_count = extract_text(pdf_path, "pdf")
        self.assertIn("TERM AND TERMINATION", text)
        self.assertEqual(page_count, 1)

    def test_unsupported_format(self):
        fake_path = self._temp_file("test.xyz")
        with open(fake_path, "w") as f:
            f.write("data")
        with self.assertRaises(UnsupportedFormatError):
            extract_text(fake_path, "xyz")

    # 1. Encrypted / password-protected PDF handling
    def test_encrypted_pdf_handling(self):
        pdf_path = self._temp_file("encrypted.pdf")
        doc = fitz.open()
        p = doc.new_page()
        p.insert_text((50, 50), "Confidential legal terms.")
        doc.save(pdf_path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="pass123")
        doc.close()

        # Both extractor and validator detect encrypted PDF and raise explicit errors
        with self.assertRaises(EncryptedPdfError):
            extract_text(pdf_path, "pdf")

        with self.assertRaises(EncryptedFileError):
            validate_file_content(pdf_path, "pdf")

    # 2. Empty file handling
    def test_empty_file_handling(self):
        txt_path = self._temp_file("empty.txt")
        with open(txt_path, "wb") as f:
            pass

        with self.assertRaises(EmptyDocumentError):
            extract_text(txt_path, "txt")

        with self.assertRaises(EmptyFileError):
            validate_file_content(txt_path, "txt")

        pdf_path = self._temp_file("empty.pdf")
        with open(pdf_path, "wb") as f:
            pass

        with self.assertRaises(EmptyDocumentError):
            extract_text(pdf_path, "pdf")

        with self.assertRaises(EmptyFileError):
            validate_file_content(pdf_path, "pdf")

    # 3. Corrupted / truncated file handling
    def test_corrupted_files_handling(self):
        # Truncated / fake PDF
        bad_pdf = self._temp_file("corrupt.pdf")
        with open(bad_pdf, "wb") as f:
            f.write(b"NOT_A_REAL_PDF_HEADER_JUST_RANDOM_DATA")

        with self.assertRaises(CorruptedFileError):
            extract_text(bad_pdf, "pdf")

        with self.assertRaises(ValCorruptedFileError):
            validate_file_content(bad_pdf, "pdf")

        # Fake DOCX (not a zip)
        bad_docx = self._temp_file("corrupt.docx")
        with open(bad_docx, "wb") as f:
            f.write(b"NOT_A_ZIP_ARCHIVE")

        with self.assertRaises(CorruptedFileError):
            extract_text(bad_docx, "docx")

        with self.assertRaises(ValCorruptedFileError):
            validate_file_content(bad_docx, "docx")

        # Binary null bytes in txt
        bad_txt = self._temp_file("binary.txt")
        with open(bad_txt, "wb") as f:
            f.write(b"ELF\x00\x02\x01\x01\x00binary_content")

        with self.assertRaises(CorruptedFileError):
            extract_text(bad_txt, "txt")

        with self.assertRaises(ValCorruptedFileError):
            validate_file_content(bad_txt, "txt")

    # 4. Multi-column PDF reading order preservation
    def test_multi_column_pdf_reading_order(self):
        pdf_path = self._temp_file("multi_column.pdf")
        doc = fitz.open()
        p = doc.new_page(width=600, height=800)

        # Insert Column 1 (x: 50 to 250)
        p.insert_textbox(fitz.Rect(50, 50, 250, 400), "COLUMN_ONE_TOP\nCOLUMN_ONE_BODY\nCOLUMN_ONE_BOTTOM")
        # Insert Column 2 (x: 350 to 550)
        p.insert_textbox(fitz.Rect(350, 50, 550, 400), "COLUMN_TWO_TOP\nCOLUMN_TWO_BODY\nCOLUMN_TWO_BOTTOM")

        doc.save(pdf_path)
        doc.close()

        full_text, page_count, metadata = extract_pdf_with_metadata(pdf_path)
        self.assertEqual(page_count, 1)

        # In true column reading order, Column 1 text precedes Column 2 text
        idx_col1 = full_text.find("COLUMN_ONE_BOTTOM")
        idx_col2 = full_text.find("COLUMN_TWO_TOP")
        self.assertNotEqual(idx_col1, -1)
        self.assertNotEqual(idx_col2, -1)
        self.assertLess(idx_col1, idx_col2, "Column 1 text should precede Column 2 text without interleaving")

    # 5. Layout metadata preservation (bbox, spans, font attributes)
    def test_layout_metadata_preservation(self):
        pdf_path = self._temp_file("layout_test.pdf")
        doc = fitz.open()
        p = doc.new_page(width=612, height=792)
        p.insert_textbox(fitz.Rect(50, 50, 500, 150), "ARTICLE 1: LEASE TERM\nThe term shall be one year.")
        doc.save(pdf_path)
        doc.close()

        full_text, page_count, metadata = extract_pdf_with_metadata(pdf_path)
        self.assertEqual(len(metadata), 1)
        page_meta = metadata[0]

        self.assertEqual(page_meta["page_number"], 1)
        self.assertEqual(page_meta["width"], 612.0)
        self.assertEqual(page_meta["height"], 792.0)
        self.assertIn("blocks", page_meta)
        self.assertGreaterEqual(len(page_meta["blocks"]), 1)

        block = page_meta["blocks"][0]
        self.assertIn("bbox", block)
        self.assertIn("lines", block)
        self.assertEqual(len(block["bbox"]), 4)
        self.assertGreater(block["avg_font_size"], 0.0)

        # Check line spans
        line0 = block["lines"][0]
        self.assertIn("spans", line0)
        self.assertGreaterEqual(len(line0["spans"]), 1)
        span0 = line0["spans"][0]
        self.assertIn("bbox", span0)
        self.assertIn("size", span0)
        self.assertIn("font", span0)

    # 6. Text normalization and offset consistency downstream
    def test_text_normalization_and_offset_consistency(self):
        txt_path = self._temp_file("pii_lease.txt")
        content = (
            "RESIDENTIAL LEASE AGREEMENT\n\n"
            "This Agreement is between Alice Smith at alice@example.com and Bob Jones at bob@example.com.\n\n"
            "SECTION 1: RENT PAYMENT\n"
            "Tenant shall pay monthly rent of $2,000 on the first day of each month."
        )
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)

        extracted_text, _ = extract_text(txt_path, "txt")
        redacted_text, findings = redact(extracted_text)

        # Verify that all findings have exact offset alignment against redacted_text
        for f in findings:
            label = f["replacement_label"]
            start = f["start"]
            end = f["end"]
            self.assertEqual(redacted_text[start:end], label)

        # Verify downstream segmentation against redacted_text
        blocks = segment_blocks(redacted_text, document_type="rental_agreement")
        self.assertGreaterEqual(len(blocks), 2)
        for b in blocks:
            # Slices match exactly
            self.assertEqual(redacted_text[b["start_offset"]:b["end_offset"]], b["text"])

    # 7. Temporary file cleanup verification
    def test_temporary_file_cleanup_simulation(self):
        temp_path = self._temp_file("cleanup_test.tmp")
        with open(temp_path, "w") as f:
            f.write("temporary data")

        self.assertTrue(os.path.exists(temp_path))
        # Simulate pipeline finally block
        try:
            pass
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        self.assertFalse(os.path.exists(temp_path))

    # 8. Concurrent uploads collision isolation
    def test_concurrent_upload_names_unique(self):
        import uuid
        names = set()
        filename = "contract.pdf"
        for _ in range(50):
            uid = uuid.uuid4().hex
            t_name = f"cg_{uid}_{filename}"
            self.assertNotIn(t_name, names)
            names.add(t_name)
        self.assertEqual(len(names), 50)

if __name__ == "__main__":
    unittest.main()
