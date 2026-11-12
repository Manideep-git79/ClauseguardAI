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


if __name__ == "__main__":
    unittest.main()
