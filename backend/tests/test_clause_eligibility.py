import unittest
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.nlp.clause_eligibility_gate import classify_content_type
from app.services.nlp.ocr_quality import assess_ocr_quality
from app.services.nlp.header_footer_detector import is_page_metadata, is_running_header_footer

class TestClauseEligibility(unittest.TestCase):
    def test_a_page_metadata(self):
        # TEST A: "Page 1 of 21" -> PAGE_METADATA, NOT_APPLICABLE, risk = 0, human_review = false
        res = classify_content_type("Page 1 of 21", document_type="insurance_policy")
        self.assertEqual(res["content_type"], "PAGE_METADATA")
        self.assertFalse(res["is_substantive"])
        self.assertEqual(res["classification"], "NOT_APPLICABLE")
        self.assertEqual(res["risk_score"], 0)
        self.assertFalse(res["human_review"])

    def test_b_document_title(self):
        # TEST B: "LIC'S NEW JEEVAN SHANTI" -> DOCUMENT_TITLE, NOT_APPLICABLE, risk = 0
        res = classify_content_type("LIC'S NEW JEEVAN SHANTI", document_type="insurance_policy")
        self.assertEqual(res["content_type"], "DOCUMENT_TITLE")
        self.assertFalse(res["is_substantive"])
        self.assertEqual(res["classification"], "NOT_APPLICABLE")
        self.assertEqual(res["risk_score"], 0)

    def test_c_product_description(self):
        # TEST C: "(A Non-Linked, Non-Participating, Individual, Single Premium, Deferred Annuity Plan)"
        # -> PRODUCT_DESCRIPTION, NOT_APPLICABLE, risk = 0
        text = "(A Non-Linked, Non-Participating, Individual, Single Premium, Deferred Annuity Plan)"
        res = classify_content_type(text, document_type="insurance_policy")
        self.assertEqual(res["content_type"], "PRODUCT_DESCRIPTION")
        self.assertFalse(res["is_substantive"])
        self.assertEqual(res["classification"], "NOT_APPLICABLE")
        self.assertEqual(res["risk_score"], 0)

    def test_d_corporate_information(self):
        # TEST D: "LIFE INSURANCE CORPORATION OF INDIA" -> CORPORATE_INFORMATION, NOT_APPLICABLE, risk = 0
        res = classify_content_type("LIFE INSURANCE CORPORATION OF INDIA", document_type="insurance_policy")
        self.assertEqual(res["content_type"], "CORPORATE_INFORMATION")
        self.assertFalse(res["is_substantive"])
        self.assertEqual(res["classification"], "NOT_APPLICABLE")
        self.assertEqual(res["risk_score"], 0)

    def test_e_registration_number(self):
        # TEST E: "Registration Number: 512" -> CORPORATE_INFORMATION, NOT_APPLICABLE, risk = 0
        res = classify_content_type("Registration Number: 512", document_type="insurance_policy")
        self.assertEqual(res["content_type"], "CORPORATE_INFORMATION")
        self.assertFalse(res["is_substantive"])
        self.assertEqual(res["classification"], "NOT_APPLICABLE")
        self.assertEqual(res["risk_score"], 0)

    def test_f_greeting(self):
        # TEST F: "Dear Policyholder," -> GREETING, NOT_APPLICABLE, risk = 0
        res = classify_content_type("Dear Policyholder,", document_type="insurance_policy")
        self.assertEqual(res["content_type"], "GREETING")
        self.assertFalse(res["is_substantive"])
        self.assertEqual(res["classification"], "NOT_APPLICABLE")
        self.assertEqual(res["risk_score"], 0)

    def test_g_signature_block(self):
        # TEST G: "Thanking you. Yours faithfully," -> SIGNATURE_BLOCK, NOT_APPLICABLE, risk = 0
        res = classify_content_type("Thanking you. Yours faithfully,", document_type="insurance_policy")
        self.assertEqual(res["content_type"], "SIGNATURE_BLOCK")
        self.assertFalse(res["is_substantive"])
        self.assertEqual(res["classification"], "NOT_APPLICABLE")
        self.assertEqual(res["risk_score"], 0)

    def test_h_ocr_noise(self):
        # TEST H: "LIC LPS THT fTM JEEINSURANCE..." -> OCR_NOISE, NOT_APPLICABLE, risk = 0
        res = classify_content_type("LIC LPS THT fTM JEEINSURANCE...", document_type="insurance_policy")
        self.assertEqual(res["content_type"], "OCR_NOISE")
        self.assertFalse(res["is_substantive"])
        self.assertEqual(res["classification"], "NOT_APPLICABLE")
        self.assertEqual(res["risk_score"], 0)

    def test_i_substantive_clause(self):
        # TEST I: "The policyholder shall pay the premium on the due date." -> SUBSTANTIVE_CLAUSE
        res = classify_content_type("The policyholder shall pay the premium on the due date.", document_type="insurance_policy")
        self.assertEqual(res["content_type"], "SUBSTANTIVE_CLAUSE")
        self.assertTrue(res["is_substantive"])

    def test_j_short_substantive_clause(self):
        # TEST J: "The policy is non-assignable." -> SUBSTANTIVE_CLAUSE (must NOT be filtered for being short)
        res = classify_content_type("The policy is non-assignable.", document_type="insurance_policy")
        self.assertEqual(res["content_type"], "SUBSTANTIVE_CLAUSE")
        self.assertTrue(res["is_substantive"])

    def test_k_conditional_substantive_clause(self):
        # TEST K: "The policy shall terminate if the premium remains unpaid." -> SUBSTANTIVE_CLAUSE
        res = classify_content_type("The policy shall terminate if the premium remains unpaid.", document_type="insurance_policy")
        self.assertEqual(res["content_type"], "SUBSTANTIVE_CLAUSE")
        self.assertTrue(res["is_substantive"])

    def test_multiline_front_matter(self):
        front_matter = (
            "Sample Policy Document - LIC's New Jeevan Shanti\n"
            "(UIN: 512N338V01)\n"
            "Page 1 of 21\n"
            "LIFE INSURANCE CORPORATION OF INDIA\n"
            "(Established by the Life Insurance Corporation Act, 1956)\n"
            "Registration Number: 512\n"
            "LIC'S NEW JEEVAN SHANTI\n"
            "(UIN: 512N338V01)\n"
            "(A Non-Linked, Non-Participating, Individual, Single Premium, Deferred Annuity Plan)"
        )
        res = classify_content_type(front_matter, document_type="insurance_policy")
        self.assertEqual(res["content_type"], "DOCUMENT_FRONT_MATTER")
        self.assertFalse(res["is_substantive"])
        self.assertEqual(res["classification"], "NOT_APPLICABLE")
        self.assertEqual(res["risk_score"], 0)

    def test_l_mixed_page(self):
        # TEST L: Page contains front matter followed by "The policyholder shall..."
        # Expected: first blocks excluded, actual legal paragraph retained and substantive
        from app.services.nlp.block_segmenter import segment_blocks
        page_text = (
            "LIC'S NEW JEEVAN SHANTI\n"
            "(UIN: 512N338V01)\n"
            "Registration Number: 512\n"
            "(A Non-Linked, Non-Participating, Individual, Single Premium, Deferred Annuity Plan)\n\n"
            "The policyholder shall pay the premium on or before the due date."
        )
        blocks = segment_blocks(page_text, document_type="insurance_policy")
        self.assertGreaterEqual(len(blocks), 2)
        # Verify first blocks are non-substantive
        for b in blocks[:-1]:
            self.assertFalse(b["is_substantive"])
            self.assertEqual(b["classification"], "NOT_APPLICABLE")
            self.assertEqual(b["risk_score"], 0)
        # Verify last block is substantive clause
        last_block = blocks[-1]
        self.assertTrue(last_block["is_substantive"])
        self.assertEqual(last_block["content_type"], "SUBSTANTIVE_CLAUSE")
        self.assertIn("The policyholder shall pay the premium", last_block["text"])

    def test_m_address_block(self):
        # TEST M: "Address and e-mail id of Branch Office" -> ADDRESS_BLOCK
        res = classify_content_type("(Address and e-mail id of Branch Office):", document_type="insurance_policy")
        self.assertEqual(res["content_type"], "ADDRESS_BLOCK")
        self.assertFalse(res["is_substantive"])
        self.assertEqual(res["classification"], "NOT_APPLICABLE")
        self.assertEqual(res["risk_score"], 0)

    def test_n_admin_transmittal(self):
        # TEST N: "We have pleasure in forwarding herewith..." -> DOCUMENT_FRONT_MATTER
        text = "We have pleasure in forwarding herewith the above policy document comprising of Part A to Part G"
        res = classify_content_type(text, document_type="insurance_policy")
        self.assertEqual(res["content_type"], "DOCUMENT_FRONT_MATTER")
        self.assertFalse(res["is_substantive"])
        self.assertEqual(res["classification"], "NOT_APPLICABLE")
        self.assertEqual(res["risk_score"], 0)

if __name__ == "__main__":
    unittest.main()
