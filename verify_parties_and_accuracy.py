import sys
import os
import uuid
import json
from pathlib import Path

backend_dir = Path("backend")
sys.path.insert(0, str(backend_dir))

from app import create_app
from app.extensions import db
from app.models.document import Document
from app.models.clause import Clause
from app.services.pipeline import process_document_pipeline
from app.services.rag.chat_engine import answer_document_question, deterministic_document_qa

def run_e2e_verification():
    app = create_app()
    with app.app_context():
        print("=== 1. Testing Contract A Ordinary Agreement End-to-End ===")
        contract_path = Path("sample_documents/synthetic_contracts/contract_a_ordinary_agreement.txt")
        assert contract_path.exists(), "Contract A file not found!"

        doc_id = uuid.uuid4()
        temp_file = Path(f"sample_documents/synthetic_contracts/temp_{doc_id}.txt")
        temp_file.write_text(contract_path.read_text(encoding="utf-8"), encoding="utf-8")

        doc = Document(
            id=doc_id,
            filename="contract_a_ordinary_agreement.txt",
            original_format="txt",
            document_type="rental_agreement",
            status="uploaded"
        )
        db.session.add(doc)
        db.session.commit()

        # Run pipeline
        print("Executing full document pipeline...")
        processed_doc = process_document_pipeline(str(doc_id), str(temp_file), "txt")
        print(f"Pipeline finished! Status: {processed_doc.status}, Risk: {processed_doc.overall_risk_score} ({processed_doc.risk_band})")

        # Verify parties_metadata
        p_meta = processed_doc.parties_metadata or {}
        print("\n--- Extracted Parties & Signatories ---")
        print("Agreement Title:", p_meta.get("agreement_title"))
        print("Effective Date: ", p_meta.get("effective_date"))
        print("Governing Law:  ", p_meta.get("governing_law"))
        print("Parties:        ", json.dumps(p_meta.get("parties", []), indent=2))
        print("Signatories:    ", json.dumps(p_meta.get("signatories", []), indent=2))

        parties = p_meta.get("parties", [])
        signatories = p_meta.get("signatories", [])

        assert len(parties) >= 2, f"Expected at least 2 parties, got {len(parties)}"
        party_names = [p["name"] for p in parties]
        print(f"Party Names Identified: {party_names}")
        assert any("Acme Enterprise Solutions" in n for n in party_names), "Acme Enterprise Solutions Inc. not found in parties!"
        assert any("Apex Global Consulting" in n for n in party_names), "Apex Global Consulting LLC not found in parties!"

        assert len(signatories) >= 2, f"Expected at least 2 signatories, got {len(signatories)}"
        sig_names = [s["name"] for s in signatories]
        print(f"Signatory Names Identified: {sig_names}")
        assert any("John Smith" in n for n in sig_names), "John Smith not found in signatories!"
        assert any("Jane Doe" in n for n in sig_names), "Jane Doe not found in signatories!"

        print("\n>>> CONTRACT A PARTY & NAME IDENTIFICATION TEST: PASSED! <<<\n")

        # 2. Test QA / Chat on Party Identification
        print("=== 2. Testing Chat Engine QA on Party Identification ===")
        questions = [
            "Who are the parties to this agreement?",
            "What is the name of the customer?",
            "Who signed this agreement and what are their titles?",
        ]

        for q in questions:
            print(f"\n--- User Question: \"{q}\" ---")
            qa_res = answer_document_question(str(doc_id), q)
            ans = qa_res.get("answer", "")
            print(f"Deciva Answer:\n{ans}")
            assert not "does not contain information" in ans.lower(), f"Failed to answer question grounded in document: {ans}"
            if "customer" in q.lower():
                assert "Acme Enterprise Solutions" in ans, f"Failed to identify Acme as customer in answer: {ans}"
            if "sign" in q.lower():
                assert "John Smith" in ans and "Jane Doe" in ans, f"Failed to identify John Smith and Jane Doe in answer: {ans}"

        print("\n>>> CHAT QA PARTY & NAME EXTRACTION TEST: PASSED! <<<\n")

        # 3. Test California Lease Party Identification
        print("=== 3. Testing Real California Lease Party Identification ===")
        lease_path = Path("sample_documents/real_california_residential_lease.txt")
        if lease_path.exists():
            l_doc_id = uuid.uuid4()
            l_temp = Path(f"sample_documents/temp_{l_doc_id}.txt")
            l_temp.write_text(lease_path.read_text(encoding="utf-8"), encoding="utf-8")

            l_doc = Document(
                id=l_doc_id,
                filename="real_california_residential_lease.txt",
                original_format="txt",
                document_type="rental_agreement",
                status="uploaded"
            )
            db.session.add(l_doc)
            db.session.commit()

            processed_lease = process_document_pipeline(str(l_doc_id), str(l_temp), "txt")
            l_meta = processed_lease.parties_metadata or {}
            print("Lease Title:   ", l_meta.get("agreement_title"))
            print("Lease Parties: ", json.dumps(l_meta.get("parties", []), indent=2))
            l_parties = [p["name"] for p in l_meta.get("parties", [])]
            assert any("Robert M. Sterling" in n or "Sterling" in n for n in l_parties), f"Landlord Sterling not identified: {l_parties}"
            assert any("Emily Chen" in n or "Chen" in n for n in l_parties), f"Tenant Emily Chen not identified: {l_parties}"
            print(">>> CALIFORNIA LEASE PARTIES IDENTIFICATION: PASSED! <<<")

        print("\n=======================================================")
        print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY WITH 100% PASS!")
        print("=======================================================")

if __name__ == "__main__":
    run_e2e_verification()
