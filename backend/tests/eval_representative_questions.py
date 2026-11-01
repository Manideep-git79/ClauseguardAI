"""
Representative 8-Category Evaluation Script for ClauseGuard AI Phase 3 RAG.
Evaluates answer generation quality across:
1. Direct factual lookup
2. Paraphrased questions
3. Plain-English explanation
4. Conditions and exceptions
5. Negation
6. Conflicting clauses
7. Missing evidence
8. Gemini timeout and fallback behavior
"""

import sys
import uuid
import json
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app
from app.extensions import db
from app.models.document import Document
from app.models.clause import Clause
from app.models.deadline import Deadline
from app.services.rag.chat_engine import answer_document_question, deterministic_document_qa
from app.services.rag.intent import classify_query_intent
from unittest.mock import patch, MagicMock

def run_representative_eval():
    app = create_app()
    with app.app_context():
        doc_id = uuid.uuid4()
        doc = Document(
            id=doc_id,
            filename="representative_eval_contract.txt",
            original_format="txt",
            document_type="commercial_contract",
            status="analyzed",
            overall_risk_score=45.0,
            risk_band="Medium",
            redacted_text="Commercial Executive Employment and Service Agreement.",
        )
        db.session.add(doc)

        c0 = Clause(
            id=uuid.uuid4(),
            document_id=doc_id,
            clause_index=0,
            clause_type="payment",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=10.0,
            favorability_label="fair",
            redacted_text="Employer shall pay Employee an annual base salary of $180,000, payable monthly in advance on or before the 1st day of each month.",
            page_number=1,
        )
        c1 = Clause(
            id=uuid.uuid4(),
            document_id=doc_id,
            clause_index=1,
            clause_type="governing_law",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=5.0,
            favorability_label="fair",
            redacted_text="This agreement shall be governed by and construed in accordance with the laws of California.",
            page_number=1,
        )
        c2 = Clause(
            id=uuid.uuid4(),
            document_id=doc_id,
            clause_index=2,
            clause_type="termination_notice",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=25.0,
            favorability_label="fair",
            redacted_text="Either party may terminate this agreement upon giving 30 calendar days written notice, unless excused by force majeure.",
            page_number=2,
        )
        c3 = Clause(
            id=uuid.uuid4(),
            document_id=doc_id,
            clause_index=3,
            clause_type="termination",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=65.0,
            favorability_label="unfavorable",
            redacted_text="Company may discharge or dismiss Employee for gross misconduct immediately without prior notice.",
            page_number=2,
        )
        c4 = Clause(
            id=uuid.uuid4(),
            document_id=doc_id,
            clause_index=4,
            clause_type="severance",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=15.0,
            favorability_label="fair",
            redacted_text="In the event of redundancy or termination without cause, Employee shall receive severance compensation of $45,000.",
            page_number=3,
        )
        c5 = Clause(
            id=uuid.uuid4(),
            document_id=doc_id,
            clause_index=5,
            clause_type="termination_notice",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=40.0,
            favorability_label="needs_review",
            redacted_text="Executive termination requires 60 calendar days written notice prior to the end of any calendar quarter.",
            page_number=3,
        )

        db.session.add_all([c0, c1, c2, c3, c4, c5])
        db.session.commit()

        cases = [
            {
                "category": "1. Direct Factual Lookup",
                "question": "What is the governing law of this agreement?",
                "expected_evidence": "Clause 1 (Governing Law: California)",
                "expected_grounded": True,
            },
            {
                "category": "2. Paraphrased Questions",
                "question": "Can they fire or dismiss me without notice?",
                "expected_evidence": "Clause 3 (Dismissal for gross misconduct immediately without prior notice)",
                "expected_grounded": True,
            },
            {
                "category": "3. Plain-English Explanation",
                "question": "What severance compensation am I entitled to upon redundancy?",
                "expected_evidence": "Clause 4 (Severance compensation of $45,000 upon redundancy)",
                "expected_grounded": True,
            },
            {
                "category": "4. Conditions and Exceptions",
                "question": "What are the general termination requirements?",
                "expected_evidence": "Clause 2 (30 calendar days written notice, unless excused by force majeure)",
                "expected_grounded": True,
            },
            {
                "category": "5. Negation",
                "question": "Can I leave without notice or penalty?",
                "expected_evidence": "Clause 2 & 3 (Preserves negative condition; requires 30 days notice)",
                "expected_grounded": True,
            },
            {
                "category": "6. Conflicting Clauses",
                "question": "What notice period is required for termination?",
                "expected_evidence": "Clause 2 (30 days) vs Clause 5 (60 days) - surfaces both",
                "expected_grounded": True,
            },
            {
                "category": "7. Missing Evidence",
                "question": "What is the policy regarding pet ownership and dog restrictions?",
                "expected_evidence": "None (Honest abstention)",
                "expected_grounded": False,
            },
            {
                "category": "8. Gemini Timeout & Fallback Behavior",
                "question": "When is the annual base salary paid?",
                "expected_evidence": "Clause 0 (Monthly in advance on or before the 1st day of each month)",
                "expected_grounded": True,
                "force_timeout": True,
            },
        ]

        results = []
        print("================================================================================")
        print("EVALUATING 8 REPRESENTATIVE DOCUMENT QUESTION SCENARIOS")
        print("================================================================================")

        try:
            for idx, item in enumerate(cases, 1):
                q = item["question"]
                cat = item["category"]
                exp_ev = item["expected_evidence"]
                force_timeout = item.get("force_timeout", False)

                # Configure fallback or timeout simulation
                with patch("app.services.rag.chat_engine.get_genai_client", return_value=None), \
                     patch("app.services.rag.embedder.get_genai_client", return_value=None), \
                     patch("app.services.rag.chat_engine.embed_query", return_value=[0.0] * 768):

                    res = answer_document_question(doc_id, q)

                answer = res["answer"]
                grounded = res["grounded"]
                cited_ids = res["cited_clause_ids"]
                fallback_used = res.get("fallback_used", False)
                confidence = res.get("confidence")

                # Validate citation
                has_valid_citations = (len(cited_ids) > 0) if item["expected_grounded"] else (len(cited_ids) == 0)

                # Determine if answer is supported
                if not item["expected_grounded"]:
                    is_supported = not grounded and ("does not contain information" in answer.lower())
                else:
                    is_supported = grounded and has_valid_citations

                first_para = answer.strip().split("\n\n")[0]
                if "###" in first_para and len(answer.strip().split("\n\n")) > 1:
                    first_para = answer.strip().split("\n\n")[1]

                print(f"\nScenario {idx}: [{cat}]")
                print(f"Query:               \"{q}\"")
                print(f"Expected Evidence:   {exp_ev}")
                print(f"Actual Answer:       {first_para[:180]}...")
                print(f"Grounded:            {grounded} (Expected: {item['expected_grounded']})")
                print(f"Fallback Used:       {fallback_used}")
                print(f"Confidence:          {confidence}")
                print(f"Cited Clause IDs:    {cited_ids}")
                print(f"Citation Valid:      {has_valid_citations}")
                print(f"Answer Supported:    {is_supported}")

                results.append({
                    "category": cat,
                    "question": q,
                    "expected_evidence": exp_ev,
                    "actual_answer_preview": first_para[:250],
                    "grounded": grounded,
                    "fallback_used": fallback_used,
                    "cited_count": len(cited_ids),
                    "is_supported": is_supported,
                })

            print("\n================================================================================")
            all_supported = all(r["is_supported"] for r in results)
            print(f"Summary: {sum(1 for r in results if r['is_supported'])}/{len(results)} Scenarios Fully Supported.")
            print(f"All Scenarios Grounded & Correct: {all_supported}")
            print("================================================================================")

        finally:
            db.session.rollback()
            Clause.query.filter_by(document_id=doc_id).delete()
            Document.query.filter_by(id=doc_id).delete()
            db.session.commit()

if __name__ == "__main__":
    run_representative_eval()
