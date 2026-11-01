"""
Benchmark and retrieval-quality measurement script for ClauseGuard AI RAG.
Measures:
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- Answerability Accuracy
- False Rejection Rate
- Unsupported-Answer Rate
- Citation Correctness
- End-to-End Latency
"""

import sys
import time
import uuid
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app
from app.extensions import db
from app.models.document import Document
from app.models.clause import Clause
from app.models.missing_clause import MissingClause
from app.services.rag.retriever import retrieve_relevant_clauses
from app.services.rag.chat_engine import answer_document_question
from unittest.mock import patch

def run_rag_benchmark():
    app = create_app()
    with app.app_context():
        # Setup mock test environment doubles
        with patch("app.services.rag.chat_engine.get_genai_client", return_value=None), \
             patch("app.services.rag.embedder.get_genai_client", return_value=None), \
             patch("app.services.rag.chat_engine.embed_query", return_value=[0.0] * 768):

            doc_id = uuid.uuid4()
            doc = Document(
                id=doc_id,
                filename="benchmark_contract.txt",
                original_format="txt",
                document_type="commercial_contract",
                status="analyzed",
                overall_risk_score=62.0,
                risk_band="High",
                redacted_text="Commercial Agreement with high-risk indemnity and termination provisions.",
            )
            db.session.add(doc)

            c0 = Clause(
                id=uuid.uuid4(),
                document_id=doc_id,
                clause_index=0,
                clause_type="preamble",
                is_substantive=True,
                risk_score=5.0,
                favorability_label="neutral",
                redacted_text="This Consulting Agreement is entered between Alpha Tech and Beta Corp.",
                page_number=1,
            )
            c1 = Clause(
                id=uuid.uuid4(),
                document_id=doc_id,
                clause_index=1,
                clause_type="rent_payment",
                is_substantive=True,
                risk_score=15.0,
                favorability_label="fair",
                redacted_text="Client shall pay professional fees of $8,000 monthly in advance on the 1st day of each month.",
                page_number=1,
            )
            c2 = Clause(
                id=uuid.uuid4(),
                document_id=doc_id,
                clause_index=2,
                clause_type="indemnity",
                is_substantive=True,
                risk_score=85.0,
                favorability_label="unfavorable",
                redacted_text="Consultant shall indemnify, defend, and hold harmless Client against any losses, claims, damages, liabilities, and attorney fees without limitation.",
                page_number=2,
            )
            c3 = Clause(
                id=uuid.uuid4(),
                document_id=doc_id,
                clause_index=3,
                clause_type="termination_notice",
                is_substantive=True,
                risk_score=60.0,
                favorability_label="unfavorable",
                redacted_text="Client may terminate or discontinue the appointment immediately without prior notice upon any convenience.",
                page_number=3,
            )
            c4 = Clause(
                id=uuid.uuid4(),
                document_id=doc_id,
                clause_index=4,
                clause_type="liquidated_damages",
                is_substantive=True,
                risk_score=75.0,
                favorability_label="unfavorable",
                redacted_text="In the event Consultant withdraws early, Consultant shall forfeit accrued compensation and pay an early termination penalty fee of $20,000.",
                page_number=4,
            )
            c5 = Clause(
                id=uuid.uuid4(),
                document_id=doc_id,
                clause_index=5,
                clause_type="governing_law",
                is_substantive=True,
                risk_score=5.0,
                favorability_label="fair",
                redacted_text="This agreement is governed by the laws of California.",
                page_number=5,
            )
            db.session.add_all([c0, c1, c2, c3, c4, c5])
            db.session.commit()

            # Benchmark evaluation queries with ground truth
            test_cases = [
                # Answerable queries: (query, expected_clause_indices, expected_answerable)
                ("any risks", [2, 3, 4], True),
                ("what are the risks", [2, 3, 4], True),
                ("any unfavourable clauses", [2, 3, 4], True),
                ("unfavorable terms", [2, 3, 4], True),
                ("can they fire me", [3], True),
                ("what happens if I leave early", [4], True),
                ("do I need to pay if I leave", [4], True),
                ("what do I have to pay", [1], True),
                ("what is the governing law", [5], True),
                ("explain clause 2", [2], True),
                ("what is the fee", [1], True),
                ("is there anything dangerous", [2, 3, 4], True),
                # Unanswerable queries:
                ("what is the pet policy for dogs and cats", [], False),
                ("who pays for international air travel", [], False),
                ("what is the dress code for casual Fridays", [], False),
            ]

            K = 3
            recall_scores = []
            precision_scores = []
            mrr_scores = []
            answerability_correct = 0
            false_rejections = 0
            unsupported_correct = 0
            total_answerable = 0
            total_unanswerable = 0
            citation_correct_count = 0
            total_citations = 0
            latencies = []

            # Pre-compute clause ID mappings before the loop to prevent ORM instance expiration on commit
            clause_id_by_index = {c.clause_index: str(c.id) for c in [c0, c1, c2, c3, c4, c5]}

            try:
                for query, expected_indices, is_answerable in test_cases:
                    t0 = time.perf_counter()
                    res = answer_document_question(doc_id, query)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    latencies.append(elapsed_ms)

                    is_grounded = res.get("grounded", False)
                    cited_ids = res.get("cited_clause_ids", [])
                    retrieved_clauses = res.get("retrieved_clauses", [])

                    # Answerability evaluation
                    if is_answerable:
                        total_answerable += 1
                        if is_grounded:
                            answerability_correct += 1
                        else:
                            false_rejections += 1
                    else:
                        total_unanswerable += 1
                        if not is_grounded:
                            answerability_correct += 1
                            unsupported_correct += 1

                    # Retrieval metrics for answerable queries
                    if is_answerable and expected_indices:
                        retrieved_indices = [c["clause_index"] for c in retrieved_clauses[:K]]
                        hits = set(retrieved_indices).intersection(set(expected_indices))

                        # Recall@K
                        recall = len(hits) / len(expected_indices)
                        recall_scores.append(recall)

                        # Precision@K
                        precision = len(hits) / min(K, len(retrieved_indices)) if retrieved_indices else 0.0
                        precision_scores.append(precision)

                        # MRR
                        reciprocal_rank = 0.0
                        for rank, c_idx in enumerate(retrieved_indices, 1):
                            if c_idx in expected_indices:
                                reciprocal_rank = 1.0 / rank
                                break
                        mrr_scores.append(reciprocal_rank)

                        # Citation correctness
                        if cited_ids:
                            total_citations += len(cited_ids)
                            expected_cids = {clause_id_by_index[idx] for idx in expected_indices if idx in clause_id_by_index}
                            for cid in cited_ids:
                                if cid in expected_cids:
                                    citation_correct_count += 1

                # Aggregate results
                avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
                avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
                avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
                acc_answerability = answerability_correct / len(test_cases)
                false_rejection_rate = false_rejections / total_answerable if total_answerable else 0.0
                unsupported_answer_rate = unsupported_correct / total_unanswerable if total_unanswerable else 0.0
                citation_accuracy = citation_correct_count / total_citations if total_citations else 1.0
                avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

                print("======================================================")
                print("CLAUSEGUARD AI: RAG RETRIEVAL & QA BENCHMARK RESULTS")
                print("======================================================")
                print(f"Total Test Queries Evaluated:    {len(test_cases)}")
                print(f"Recall@{K}:                       {avg_recall * 100:.2f}%")
                print(f"Precision@{K}:                    {avg_precision * 100:.2f}%")
                print(f"Mean Reciprocal Rank (MRR):      {avg_mrr:.4f}")
                print(f"Answerability Accuracy:          {acc_answerability * 100:.2f}%")
                print(f"False Rejection Rate:            {false_rejection_rate * 100:.2f}%")
                print(f"Unsupported-Answer Correctness:  {unsupported_answer_rate * 100:.2f}%")
                print(f"Citation Correctness:            {citation_accuracy * 100:.2f}%")
                print(f"Average End-to-End Latency:      {avg_latency:.2f} ms")
                print("======================================================")

                return {
                    "recall_at_k": avg_recall,
                    "precision_at_k": avg_precision,
                    "mrr": avg_mrr,
                    "answerability_accuracy": acc_answerability,
                    "false_rejection_rate": false_rejection_rate,
                    "unsupported_answer_rate": unsupported_answer_rate,
                    "citation_correctness": citation_accuracy,
                    "avg_latency_ms": avg_latency,
                }
            finally:
                db.session.rollback()
                Clause.query.filter_by(document_id=doc_id).delete()
                Document.query.filter_by(id=doc_id).delete()
                db.session.commit()

if __name__ == "__main__":
    run_rag_benchmark()
