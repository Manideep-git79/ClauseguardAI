import os
import sys
import uuid
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.utils.gemini_chat import (
    GEMINI_CANDIDATE_MODELS,
    STALE_CHAT_MODELS,
    build_candidate_models,
    generate_content_with_cascade,
    format_conversation_history,
    ask_gemini_or_fallback,
)
from app.services.rag.chat_engine import (
    _is_conversational_query,
    _extract_cited_indices,
    SYSTEM_INSTRUCTION,
)


class TestRAGMeaningFirstAndCascade(unittest.TestCase):
    """Database-independent regression tests for model cascade, meaning-first prompt logic, and citation extraction."""

    def test_build_candidate_models_deprioritizes_stale_models(self):
        # Legacy stalling model in Railway/env must NOT sit ahead of fast working models
        models = build_candidate_models("gemini-3.8-flash")
        self.assertNotEqual(models[0], "gemini-3.8-flash")
        self.assertEqual(models[:len(GEMINI_CANDIDATE_MODELS)], GEMINI_CANDIDATE_MODELS)
        self.assertEqual(models[-1], "gemini-3.8-flash")

    def test_build_candidate_models_prioritizes_valid_custom_model(self):
        models = build_candidate_models("gemini-3.5-pro")
        self.assertEqual(models[0], "gemini-3.5-pro")
        for m in GEMINI_CANDIDATE_MODELS:
            self.assertIn(m, models)

    def test_generate_content_with_cascade_fails_over_on_timeout(self):
        mock_client = MagicMock()

        def side_effect(model, contents):
            if model == GEMINI_CANDIDATE_MODELS[0]:
                raise TimeoutError("Read timed out")
            resp = MagicMock()
            resp.text = "Direct grounded answer [Clause 1, Page 1]."
            return resp

        mock_client.models.generate_content.side_effect = side_effect
        text, used_model = generate_content_with_cascade(mock_client, "Test prompt")
        self.assertEqual(text, "Direct grounded answer [Clause 1, Page 1].")
        self.assertEqual(used_model, GEMINI_CANDIDATE_MODELS[1])

    def test_generate_content_with_cascade_skips_empty_responses(self):
        mock_client = MagicMock()
        calls = []

        def side_effect(model, contents):
            calls.append(model)
            resp = MagicMock()
            resp.text = "" if len(calls) == 1 else "Valid answer"
            return resp

        mock_client.models.generate_content.side_effect = side_effect
        text, used_model = generate_content_with_cascade(mock_client, "Test prompt")
        self.assertEqual(text, "Valid answer")
        self.assertEqual(len(calls), 2)

    def test_conversational_query_detection(self):
        self.assertTrue(_is_conversational_query("hi"))
        self.assertTrue(_is_conversational_query("Hello!"))
        self.assertTrue(_is_conversational_query("Thank you"))
        self.assertTrue(_is_conversational_query("who are you?"))
        self.assertFalse(_is_conversational_query("What is the notice period for termination?"))
        self.assertFalse(_is_conversational_query("What does indemnification mean in this lease?"))

    def test_extract_cited_indices_ignores_page_numbers_and_handles_multi_clauses(self):
        self.assertEqual(_extract_cited_indices("Yes [Clause 2, Page 5]."), [2])
        self.assertEqual(_extract_cited_indices("See [Clause 1, Page 2] and [Clause 4, Page 9]."), [1, 4])
        self.assertEqual(_extract_cited_indices("Compare [Clause 0, 3] for details."), [0, 3])
        self.assertEqual(_extract_cited_indices("As stated in Clause 7."), [7])
        self.assertEqual(_extract_cited_indices("No citations here."), [])

    def test_format_conversation_history_bounds_turns(self):
        turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn {i}"} for i in range(10)]
        formatted = format_conversation_history(turns)
        self.assertIn("<CONVERSATION_HISTORY>", formatted)
        self.assertNotIn("Turn 0", formatted)
        self.assertIn("Turn 9", formatted)

    def test_system_instruction_supports_meaning_first_and_dual_mode(self):
        self.assertIn("UNDERSTAND THE USER'S INTENT & MEANING FIRST", SYSTEM_INSTRUCTION)
        self.assertIn("DOCUMENT-SPECIFIC QUESTION", SYSTEM_INSTRUCTION)
        self.assertIn("NORMAL / GENERAL QUESTION", SYSTEM_INSTRUCTION)
        self.assertIn("DIRECT PLAIN-ENGLISH ANSWER FIRST", SYSTEM_INSTRUCTION)
        self.assertIn("UNTRUSTED DOCUMENT EVIDENCE", SYSTEM_INSTRUCTION)

    def test_general_mode_ask_gemini_or_fallback_with_mocked_cascade(self):
        with patch("app.utils.gemini_chat.os.environ.get", side_effect=lambda k, d=None: "fake-key" if k == "GEMINI_API_KEY" else d), \
             patch("app.utils.gemini_chat.get_genai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.text = "Indemnification is a contractual promise to compensate for specified losses."
            mock_client.models.generate_content.return_value = mock_resp
            mock_get_client.return_value = mock_client

            res = ask_gemini_or_fallback(
                "What does indemnification mean?",
                context_text="",
                history=[{"role": "user", "content": "hi"}],
            )
            self.assertFalse(res["fallbackUsed"])
            self.assertIn("Indemnification", res["answer"])
            prompt_sent = mock_client.models.generate_content.call_args.kwargs["contents"]
            self.assertIn("<CONVERSATION_HISTORY>", prompt_sent)
            self.assertIn("Understand Meaning First", prompt_sent)

    def test_answer_document_question_end_to_end_mocked_db_grounded_and_normal(self):
        from flask import Flask
        from app.services.rag import chat_engine

        dummy_app = Flask(__name__)
        doc_id = uuid.uuid4()
        fake_doc = SimpleNamespace(
            id=doc_id,
            filename="offer_letter.pdf",
            document_type="job_offer_letter",
            parties_metadata=None,
            overall_risk_score=18.0,
            risk_band="Low",
            missing_clauses=[],
            redacted_text="Offer of Employment. Base salary is $130,000. Outside moonlighting requires consent.",
        )
        fake_clauses = [
            {
                "clause_id": str(uuid.uuid4()),
                "clause_index": 0,
                "clause_type": "compensation",
                "favorability_label": "fair",
                "risk_score": 5.0,
                "page_number": 1,
                "redacted_text": "Employee will receive an annual base salary of $130,000 paid bi-weekly.",
                "similarity": 0.91,
            },
            {
                "clause_id": str(uuid.uuid4()),
                "clause_index": 1,
                "clause_type": "exclusivity",
                "favorability_label": "fair",
                "risk_score": 15.0,
                "page_number": 2,
                "redacted_text": "Employee shall not engage in outside moonlighting or competing work without prior written approval.",
                "similarity": 0.88,
            },
        ]

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp

        fake_session_obj = SimpleNamespace(id=uuid.uuid4())
        mock_db = MagicMock()
        mock_db.session.get.side_effect = lambda model, ident: (
            fake_doc if getattr(model, "__name__", "") == "Document" else fake_session_obj
        )

        def fake_add(obj):
            if not getattr(obj, "id", None):
                obj.id = uuid.uuid4()

        mock_db.session.add.side_effect = fake_add

        with dummy_app.app_context(), \
             patch.object(chat_engine, "db", mock_db), \
             patch.object(chat_engine, "ChatSession", side_effect=lambda **kw: SimpleNamespace(id=uuid.uuid4(), **kw)), \
             patch.object(chat_engine, "ChatMessage", side_effect=lambda **kw: SimpleNamespace(id=uuid.uuid4(), **kw)), \
             patch.object(chat_engine, "Deadline") as mock_dl_cls, \
             patch("app.services.rag.chat_engine.embed_query", return_value=[0.1] * 768), \
             patch("app.services.rag.chat_engine.retrieve_relevant_clauses", return_value=[]), \
             patch("app.services.rag.chat_engine._load_all_document_clauses", return_value=fake_clauses), \
             patch("app.services.rag.chat_engine.get_genai_client", return_value=mock_client):

            mock_dl_cls.query.filter_by.return_value.all.return_value = []

            # Case 1: Document-specific semantic paraphrase where retrieve_relevant_clauses returned []
            # Full clause inventory still allows Gemini to answer and cite [Clause 1, Page 2]!
            mock_resp.text = "No, you cannot do freelance side work without prior written approval under [Clause 1, Page 2]."
            res_doc = chat_engine.answer_document_question(doc_id, "Can I do freelance side gigs on weekends?")
            self.assertTrue(res_doc["grounded"])
            self.assertFalse(res_doc["fallback_used"])
            self.assertIn(fake_clauses[1]["clause_id"], res_doc["cited_clause_ids"])
            self.assertEqual(res_doc["sources"][0]["page_number"], 2)

            # Case 2: Normal / general conceptual question while document is loaded
            mock_resp.text = "At-will employment means either party may terminate the relationship at any time for any lawful reason."
            res_norm = chat_engine.answer_document_question(doc_id, "What does at-will employment mean?")
            self.assertFalse(res_norm["grounded"])
            self.assertFalse(res_norm["fallback_used"])
            self.assertEqual(res_norm["cited_clause_ids"], [])
            self.assertEqual(res_norm["confidence"], 0.88)

            # Case 3: Pure conversational greeting while document is loaded
            mock_resp.text = "Hello! I have offer_letter.pdf ready. Ask me anything about your pay, terms, or general contract rules!"
            res_conv = chat_engine.answer_document_question(doc_id, "hi")
            self.assertFalse(res_conv["grounded"])
            self.assertFalse(res_conv["fallback_used"])
            self.assertIn("Hello", res_conv["answer"])

            # Case 4: Asking about a different document owned by the same user triggers <OTHER_DOCUMENT_REDIRECT>
            other_doc_id = str(uuid.uuid4())
            with patch(
                "app.services.rag.chat_engine._load_user_doc_metadata",
                return_value=[
                    {"id": str(doc_id), "filename": "offer_letter.pdf", "document_type": "job_offer_letter"},
                    {"id": other_doc_id, "filename": "Rental_Agreement.pdf", "document_type": "rental_agreement"},
                ],
            ):
                mock_resp.text = "This chat is scoped exclusively to offer_letter.pdf. Please switch to the Account-Wide AI Chatbot for Rental_Agreement.pdf."
                res_redir = chat_engine.answer_document_question(
                    doc_id,
                    "What is the late fee in Rental_Agreement.pdf?",
                    user_id=uuid.uuid4(),
                )
                prompt_used = mock_client.models.generate_content.call_args.kwargs["contents"]
                self.assertIn("OTHER-DOCUMENT BOUNDARY", prompt_used)
                self.assertIn("Rental_Agreement.pdf", prompt_used)
                self.assertIn("Rental_Agreement.pdf", res_redir["answer"])

    def test_answer_document_question_rejects_mismatched_session(self):
        from flask import Flask
        from app.services.rag import chat_engine

        dummy_app = Flask(__name__)
        doc_a = uuid.uuid4()
        doc_b = uuid.uuid4()
        fake_session = SimpleNamespace(id=uuid.uuid4(), document_id=doc_b)
        fake_doc = SimpleNamespace(
            id=doc_a,
            filename="doc_a.pdf",
            document_type="rental_agreement",
            parties_metadata=None,
            overall_risk_score=10.0,
            risk_band="Low",
            missing_clauses=[],
            redacted_text="Clause text",
        )
        mock_db = MagicMock()
        mock_db.session.get.side_effect = lambda model, ident: (
            fake_doc if getattr(model, "__name__", "") == "Document" else fake_session
        )

        with dummy_app.app_context(), patch.object(chat_engine, "db", mock_db):
            with self.assertRaises(ValueError):
                chat_engine.answer_document_question(doc_a, "What is the deposit?", session_id=fake_session.id)


class TestEmbedderAndPipelineAlignment(unittest.TestCase):
    """Verify overlapping chunk embeddings, vector normalization pooling, and blank-clause alignment safety."""

    def test_split_text_with_overlap_long_clause_covers_tail(self):
        from app.services.rag.embedder import _split_text_with_overlap

        short_slices = _split_text_with_overlap("Short clause text.", max_chars=1600)
        self.assertEqual(short_slices, ["Short clause text."])

        body = ("Section detail clause content. " * 90) + "UNIQUE_TAIL_PENALTY_TERMS_END."
        slices = _split_text_with_overlap(body, max_chars=1600, overlap=200, max_slices=6)
        self.assertGreater(len(slices), 1)
        self.assertLessEqual(len(slices), 6)
        self.assertIn("UNIQUE_TAIL_PENALTY_TERMS_END.", slices[-1])

    def test_average_and_normalize_vectors_returns_unit_vector(self):
        import math
        from app.services.rag.embedder import _average_and_normalize_vectors

        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        pooled = _average_and_normalize_vectors([v1, v2])
        self.assertIsNotNone(pooled)
        norm = math.sqrt(sum(x * x for x in pooled))
        self.assertAlmostEqual(norm, 1.0, places=6)
        self.assertAlmostEqual(pooled[0], pooled[1], places=6)

    def test_batch_embed_texts_pools_oversized_item_and_preserves_none_for_empty(self):
        from app.services.rag import embedder

        mock_client = MagicMock()

        def fake_embed_content(model, contents, config):
            # Return unique normalized 768-dim vectors per slice
            embs = []
            for idx, _ in enumerate(contents):
                vec = [0.0] * 768
                vec[idx % 10] = 1.0
                embs.append(SimpleNamespace(values=vec))
            return SimpleNamespace(embeddings=embs)

        mock_client.models.embed_content.side_effect = fake_embed_content

        long_clause = ("Termination and indemnification obligations. " * 70) + "TAIL_CLAUSE_END"
        with patch.object(embedder, "get_genai_client", return_value=mock_client):
            results = embedder.batch_embed_texts(["Normal short clause", "", long_clause], batch_size=20)

        self.assertEqual(len(results), 3)
        self.assertIsNotNone(results[0])
        self.assertEqual(len(results[0]), 768)
        self.assertIsNone(results[1])
        self.assertIsNotNone(results[2])
        self.assertEqual(len(results[2]), 768)

    def test_pipeline_stage7_skips_blank_clauses_without_misalignment(self):
        # Verify the Stage 7 source pattern aligns zip(embeddable_clauses, batch_embs) only on non-empty redacted_text
        c1 = SimpleNamespace(redacted_text="First substantive clause", embedding=None)
        c_blank = SimpleNamespace(redacted_text="   ", embedding=None)
        c2 = SimpleNamespace(redacted_text="Second substantive clause", embedding=None)
        clause_objs = [c1, c_blank, c2]
        embeddable = [c for c in clause_objs if c.redacted_text and c.redacted_text.strip()]
        fake_embs = [[0.1] * 768, [0.2] * 768]
        for c, emb in zip(embeddable, fake_embs):
            c.embedding = emb
        self.assertEqual(c1.embedding, fake_embs[0])
        self.assertIsNone(c_blank.embedding)
        self.assertEqual(c2.embedding, fake_embs[1])


class TestQueryUnderstanding(unittest.TestCase):
    """Verify typo repair, slang/abbreviation expansion, cross-doc redirect detection, and conversational rewriting."""

    def test_repair_typos_and_expand_slang_and_abbreviations(self):
        from app.services.rag.query_understanding import (
            repair_typos_and_normalize,
            expand_slang_and_abbreviations,
        )

        fixed = repair_typos_and_normalize("What is the termiantion notise penality?")
        self.assertEqual(fixed, "What is the termination notice penalty?")

        exp1 = " ".join(expand_slang_and_abbreviations("What is the money held by landlord?"))
        self.assertIn("security deposit", exp1.lower())

        exp2 = " ".join(expand_slang_and_abbreviations("What is my CTC and PTO?"))
        self.assertIn("annual salary", exp2.lower())
        self.assertIn("paid time off", exp2.lower())

        exp3 = " ".join(expand_slang_and_abbreviations("Do I get extra time to pay rent?"))
        self.assertIn("grace period", exp3.lower())

    def test_detect_other_document_reference_metadata_only(self):
        from app.services.rag.query_understanding import detect_other_document_reference

        active_id = str(uuid.uuid4())
        other_id = str(uuid.uuid4())
        docs_meta = [
            {"id": active_id, "filename": "Employment_Agreement.pdf", "document_type": "job_offer_letter"},
            {"id": other_id, "filename": "Rental Agreement.pdf", "document_type": "rental_agreement"},
        ]

        hit_other = detect_other_document_reference("What is my security deposit in Rental Agreement.pdf?", active_id, docs_meta)
        self.assertIsNotNone(hit_other)
        self.assertEqual(hit_other["filename"], "Rental Agreement.pdf")

        hit_same = detect_other_document_reference("What is my notice period in Employment Agreement?", active_id, docs_meta)
        self.assertIsNone(hit_same)

    def test_understand_and_rewrite_query_resolves_followup_and_skips_clear_single_turn(self):
        from app.services.rag.query_understanding import understand_and_rewrite_query

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = SimpleNamespace(
            text='{"standalone_query": "Does the 30 days written termination notice period apply to both parties?", '
                 '"semantic_queries": ["mutual termination notice period both parties"], '
                 '"category": "DOCUMENT_SPECIFIC", "target_doc_hints": [], '
                 '"clarification_needed": false, "clarification_question": null}'
        )

        # Follow-up anaphora with history triggers fast LLM rewrite
        qu_followup = understand_and_rewrite_query(
            "Does that apply to both parties?",
            history=[
                {"role": "user", "content": "What is the termination period?"},
                {"role": "assistant", "content": "The agreement requires 30 days written notice."},
            ],
            mode="document",
            client=mock_client,
        )
        self.assertTrue(qu_followup.used_llm_rewrite)
        self.assertIn("30 days written termination notice", qu_followup.standalone_query.lower())
        self.assertTrue(mock_client.models.generate_content.called)

        # Clear single-turn query skips LLM rewrite overhead
        mock_client.reset_mock()
        qu_clear = understand_and_rewrite_query(
            "Can I terminate without cause?",
            history=None,
            mode="document",
            client=mock_client,
        )
        self.assertFalse(qu_clear.used_llm_rewrite)
        self.assertFalse(mock_client.models.generate_content.called)


class TestAccountWideRetrieverAndChatEngine(unittest.TestCase):
    """Verify strictly account-scoped retrieval, cross-document comparison, citation verification, and chat routes."""

    def test_retrieve_relevant_clauses_for_user_guards_and_balances_docs(self):
        from app.services.rag import retriever

        self.assertEqual(retriever.retrieve_relevant_clauses_for_user(None, query_texts=["notice"]), [])

        user_id = uuid.uuid4()
        doc_a = uuid.uuid4()
        doc_b = uuid.uuid4()
        docs = [
            SimpleNamespace(id=doc_a, filename="Rental_Agreement.pdf", document_type="rental_agreement", overall_risk_score=40.0, risk_band="Medium"),
            SimpleNamespace(id=doc_b, filename="Employment_Agreement.pdf", document_type="job_offer_letter", overall_risk_score=25.0, risk_band="Low"),
        ]
        clauses = [
            SimpleNamespace(id=uuid.uuid4(), document_id=doc_a, clause_index=i, clause_type="termination", favorability_label="fair", risk_score=30.0, page_number=1, is_substantive=True, content_type="SUBSTANTIVE_CLAUSE", redacted_text=f"Rental notice period clause {i} requiring 30 days written notice.")
            for i in range(4)
        ] + [
            SimpleNamespace(id=uuid.uuid4(), document_id=doc_b, clause_index=i, clause_type="termination", favorability_label="fair", risk_score=20.0, page_number=2, is_substantive=True, content_type="SUBSTANTIVE_CLAUSE", redacted_text=f"Employment notice period clause {i} requiring 60 days written notice.")
            for i in range(4)
        ]

        q_docs = MagicMock()
        q_docs.filter.return_value = q_docs
        q_docs.order_by.return_value = q_docs
        q_docs.all.return_value = docs

        q_clauses = MagicMock()
        q_clauses.filter.return_value = q_clauses
        q_clauses.order_by.return_value = q_clauses
        q_clauses.all.return_value = clauses

        mock_db = MagicMock()
        mock_db.session.query.side_effect = lambda model: q_docs if getattr(model, "__name__", "") == "Document" else q_clauses

        with patch.object(retriever, "db", mock_db):
            hits = retriever.retrieve_relevant_clauses_for_user(
                user_id=user_id,
                query_vectors=None,
                query_texts=["Compare notice period across agreements"],
                top_k=4,
                per_doc_cap=2,
            )

        self.assertEqual(len(hits), 4)
        hit_filenames = [h.get("filename") for h in hits]
        self.assertEqual(hit_filenames.count("Rental_Agreement.pdf"), 2)
        self.assertEqual(hit_filenames.count("Employment_Agreement.pdf"), 2)

    def test_format_history_and_extract_account_citations(self):
        from app.services.rag.chat_engine import (
            _format_account_history_with_docs,
            _extract_account_citations,
        )

        hist_msgs = [
            SimpleNamespace(role="USER", content="Compare notice periods.", sources=[]),
            SimpleNamespace(
                role="ASSISTANT",
                content="Rental requires 30 days and Employment requires 60 days.",
                sources=[
                    {"document_name": "Rental_Agreement.pdf"},
                    {"document_name": "Employment_Agreement.pdf"},
                ],
            ),
        ]
        formatted = _format_account_history_with_docs(hist_msgs)
        self.assertIn("[Documents Referenced: Rental_Agreement.pdf, Employment_Agreement.pdf]", formatted)

        c1 = {"clause_id": "id-1", "document_id": "d-1", "filename": "Rental_Agreement.pdf", "clause_index": 1, "page_number": 2, "redacted_text": "30 days notice"}
        c2 = {"clause_id": "id-2", "document_id": "d-2", "filename": "Employment_Agreement.pdf", "clause_index": 0, "page_number": 1, "redacted_text": "60 days notice"}
        lookup = {
            ("rental_agreement.pdf", 1): c1,
            ("employment_agreement.pdf", 0): c2,
        }
        answer = (
            "Your documents differ: Rental requires 30 days [Rental_Agreement.pdf | Clause 1, Page 2] "
            "while Employment requires 60 days [Employment_Agreement.pdf | Clause 0, Page 1]. "
            "Ignore fabricated [Other_User_Secret.pdf | Clause 9, Page 1]."
        )
        cited = _extract_account_citations(answer, lookup, ["Rental_Agreement.pdf", "Employment_Agreement.pdf"])
        self.assertEqual([c["clause_id"] for c in cited], ["id-1", "id-2"])

    def test_answer_account_question_cross_doc_comparison_and_deterministic_fallback(self):
        from flask import Flask
        from app.services.rag import chat_engine

        dummy_app = Flask(__name__)
        user_id = uuid.uuid4()
        doc_a = str(uuid.uuid4())
        doc_b = str(uuid.uuid4())
        docs_meta = [
            {"id": doc_a, "filename": "Rental_Agreement.pdf", "document_type": "rental_agreement", "overall_risk_score": 40.0, "risk_band": "Medium"},
            {"id": doc_b, "filename": "Employment_Agreement.pdf", "document_type": "job_offer_letter", "overall_risk_score": 25.0, "risk_band": "Low"},
        ]
        retrieved = [
            {"clause_id": "c-rent-1", "document_id": doc_a, "filename": "Rental_Agreement.pdf", "clause_index": 1, "clause_type": "termination", "favorability_label": "fair", "risk_score": 30.0, "page_number": 2, "redacted_text": "Tenant must provide 30 days written notice before vacating.", "similarity": 0.92},
            {"clause_id": "c-emp-0", "document_id": doc_b, "filename": "Employment_Agreement.pdf", "clause_index": 0, "clause_type": "termination", "favorability_label": "fair", "risk_score": 20.0, "page_number": 1, "redacted_text": "Employee must provide 60 days written notice prior to resignation.", "similarity": 0.90},
        ]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = SimpleNamespace(
            text="Your agreements establish different notice periods:\n- Rental Agreement: 30 days written notice [Rental_Agreement.pdf | Clause 1, Page 2]\n- Employment Agreement: 60 days written notice [Employment_Agreement.pdf | Clause 0, Page 1]"
        )

        mock_db = MagicMock()
        mock_msg_cls = MagicMock()
        mock_msg_cls.side_effect = lambda **kw: SimpleNamespace(id=uuid.uuid4(), **kw)
        mock_msg_cls.user_id = MagicMock()
        mock_msg_cls.document_id = MagicMock()
        mock_msg_cls.created_at = MagicMock()
        mock_msg_cls.query.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_dl_cls = MagicMock()
        mock_dl_cls.query.filter.return_value.all.return_value = []

        with dummy_app.app_context(), \
             patch.object(chat_engine, "db", mock_db), \
             patch.object(chat_engine, "ChatMessage", mock_msg_cls), \
             patch.object(chat_engine, "Deadline", mock_dl_cls), \
             patch("app.services.rag.chat_engine._load_user_doc_metadata", return_value=docs_meta), \
             patch("app.services.rag.chat_engine.batch_embed_texts", return_value=[[0.1] * 768]), \
             patch("app.services.rag.chat_engine.retrieve_relevant_clauses_for_user", return_value=retrieved), \
             patch("app.services.rag.chat_engine.get_genai_client", return_value=mock_client):

            res = chat_engine.answer_account_question(user_id, "Compare notice periods across all my contracts")

        self.assertEqual(res["scope"], "account")
        self.assertTrue(res["grounded"])
        self.assertFalse(res["fallback_used"])
        self.assertEqual(len(res["sources"]), 2)
        self.assertEqual({s["document_name"] for s in res["sources"]}, {"Rental_Agreement.pdf", "Employment_Agreement.pdf"})

        # Offline deterministic fallback check
        with dummy_app.app_context(), \
             patch.object(chat_engine, "db", mock_db), \
             patch.object(chat_engine, "ChatMessage", mock_msg_cls), \
             patch.object(chat_engine, "Deadline", mock_dl_cls), \
             patch("app.services.rag.chat_engine._load_user_doc_metadata", return_value=docs_meta), \
             patch("app.services.rag.chat_engine.retrieve_relevant_clauses_for_user", return_value=retrieved), \
             patch("app.services.rag.chat_engine.get_genai_client", return_value=None):

            res_fb = chat_engine.answer_account_question(user_id, "What are my notice periods?")

        self.assertTrue(res_fb["fallback_used"])
        self.assertIn("Rental_Agreement.pdf", res_fb["answer"])
        self.assertIn("Employment_Agreement.pdf", res_fb["answer"])

    def test_chat_api_routes_scope_history_and_session_validation(self):
        from flask import Flask, g
        from app.api import chat as chat_api

        app = Flask(__name__)
        fake_user = SimpleNamespace(id=uuid.uuid4(), email="user@example.com", role="user")

        # 1. POST /chat with omitted documentId routes to answer_account_question
        with app.test_request_context("/api/chat", method="POST", json={"question": "Compare my contracts"}), \
             patch.object(chat_api, "answer_account_question", return_value={"id": "m1", "answer": "Comparison", "grounded": True, "confidence": 0.9, "sources": [], "scope": "account"}) as mock_acc, \
             patch("app.services.audit_service.log_audit_event"):
            g.user = fake_user
            resp, status = chat_api.post_chat_message.__wrapped__()
            self.assertEqual(status, 200)
            mock_acc.assert_called_once_with(user_id=fake_user.id, question="Compare my contracts")

        # 2. POST /chat with scope="general" uses ask_gemini_or_fallback
        mock_msg_cls = MagicMock()
        mock_msg_cls.side_effect = lambda **kw: SimpleNamespace(**{"id": uuid.uuid4(), **kw})
        with app.test_request_context("/api/chat", method="POST", json={"question": "What is force majeure?", "scope": "general"}), \
             patch.object(chat_api, "answer_account_question") as mock_acc_unused, \
             patch.object(chat_api, "ask_gemini_or_fallback", return_value={"answer": "Force majeure excuses unforeseeable events.", "grounded": False, "confidence": 0.85, "sources": []}) as mock_gen, \
             patch.object(chat_api, "ChatMessage", mock_msg_cls), \
             patch.object(chat_api, "db", MagicMock()), \
             patch("app.services.audit_service.log_audit_event"):
            g.user = fake_user
            resp, status = chat_api.post_chat_message.__wrapped__()
            self.assertEqual(status, 200)
            mock_acc_unused.assert_not_called()
            mock_gen.assert_called_once()

        # 3. GET /chat/history returns 400 on invalid documentId & isolates standalone thread when omitted
        with app.test_request_context("/api/chat/history?documentId=bad-uuid", method="GET"):
            g.user = fake_user
            resp, status = chat_api.get_conversation_history.__wrapped__()
            self.assertEqual(status, 400)

        mock_hist_q = MagicMock()
        mock_hist_q.filter.return_value = mock_hist_q
        mock_hist_q.order_by.return_value.limit.return_value.all.return_value = []
        mock_hist_cls = MagicMock()
        mock_hist_cls.query.filter_by.return_value = mock_hist_q
        mock_hist_cls.document_id = MagicMock()
        with app.test_request_context("/api/chat/history", method="GET"), \
             patch.object(chat_api, "ChatMessage", mock_hist_cls):
            g.user = fake_user
            resp, status = chat_api.get_conversation_history.__wrapped__()
            self.assertEqual(status, 200)
            mock_hist_cls.document_id.is_.assert_called_once_with(None)

        # 4. POST /documents/<doc_id>/chat returns 403 on mismatched session ValueError
        doc_id = uuid.uuid4()
        mock_doc_cls = MagicMock()
        mock_doc_cls.query.get_or_404.return_value = SimpleNamespace(id=doc_id, user_id=fake_user.id)
        with app.test_request_context(f"/api/documents/{doc_id}/chat", method="POST", json={"question": "Notice?", "session_id": str(uuid.uuid4())}), \
             patch.object(chat_api, "Document", mock_doc_cls), \
             patch.object(chat_api, "answer_document_question", side_effect=ValueError("Chat session does not belong to the specified document.")):
            g.user = fake_user
            resp, status = chat_api.chat_with_document_legacy.__wrapped__(doc_id)
            self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
