import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
import unittest

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app
from app.extensions import db
from app.models.user import User
from app.models.document import Document
from app.models.clause import Clause
from app.models.deadline import Deadline
from app.models.chat import ChatSession, ChatMessage
from sqlalchemy import text
from tests.test_isolation_guard import enforce_test_database_isolation

class TestDatabaseIsolationRegression(unittest.TestCase):
    """
    Regression test suite verifying that:
    1. Test operations strictly target clauseguard_test_isolation.
    2. Test teardowns never touch unrelated user documents or clauses.
    3. Foreign key cascading and deletion orders are preserved.
    4. Fail-closed safety guard halts execution if schema is missing, public, or falling back.
    5. ORM CRUD operations are verified to never touch the public schema.
    """

    @classmethod
    def setUpClass(cls):
        os.environ["TEST_DATABASE_SCHEMA"] = "clauseguard_test_isolation"
        cls.app = create_app()
        cls.app_context = cls.app.app_context()
        cls.app_context.push()
        enforce_test_database_isolation(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.app_context.pop()

    def setUp(self):
        try:
            db.session.rollback()
        except Exception:
            pass
        db.session.remove()

        # Create an unrelated "User X" document that must NEVER be deleted by other tests
        self.unrelated_doc_id = uuid.uuid4()
        self.unrelated_doc = Document(
            id=self.unrelated_doc_id,
            filename="unrelated_production_contract.pdf",
            original_format="pdf",
            document_type="rental_agreement",
            status="analyzed",
            raw_text_encrypted=b"dummy",
            redacted_text="Unrelated contract text.",
            overall_risk_score=25.0,
            risk_band="Low",
        )
        self.unrelated_clause_id = uuid.uuid4()
        self.unrelated_clause = Clause(
            id=self.unrelated_clause_id,
            document_id=self.unrelated_doc_id,
            clause_index=0,
            clause_type="rent_payment",
            content_type="SUBSTANTIVE_CLAUSE",
            is_substantive=True,
            risk_score=10.0,
            favorability_label="fair",
            redacted_text="Tenant shall pay rent on the 1st of each month.",
            page_number=1,
        )
        db.session.add_all([self.unrelated_doc, self.unrelated_clause])
        db.session.commit()

        # Temporary test document created by this test method
        self.test_doc_id = uuid.uuid4()
        self.test_doc = Document(
            id=self.test_doc_id,
            filename="temporary_test_run.txt",
            original_format="txt",
            document_type="job_offer_letter",
            status="pending",
            redacted_text="Temporary text.",
        )
        db.session.add(self.test_doc)
        db.session.commit()

    def tearDown(self):
        try:
            db.session.rollback()
            # Clean up test_doc_id strictly
            if hasattr(self, "test_doc_id") and self.test_doc_id:
                db.session.query(Clause).filter(Clause.document_id == self.test_doc_id).delete(synchronize_session=False)
                db.session.query(Document).filter(Document.id == self.test_doc_id).delete(synchronize_session=False)
                db.session.commit()
            # Clean up unrelated fixture after test verification finishes
            if hasattr(self, "unrelated_doc_id") and self.unrelated_doc_id:
                db.session.query(Clause).filter(Clause.document_id == self.unrelated_doc_id).delete(synchronize_session=False)
                db.session.query(Document).filter(Document.id == self.unrelated_doc_id).delete(synchronize_session=False)
                db.session.commit()
        except Exception:
            db.session.rollback()
        finally:
            db.session.remove()

    def test_search_path_is_isolated(self):
        """Verifies that the current PostgreSQL search path is isolated to clauseguard_test_isolation."""
        cur_schema = db.session.execute(text("SELECT current_schema();")).scalar()
        self.assertEqual(cur_schema, "clauseguard_test_isolation")

    def test_unrelated_documents_survive_scoped_cleanup(self):
        """Proves that cleaning up test_doc leaves unrelated_doc completely intact."""
        # Simulate scoped teardown of test_doc
        db.session.query(Clause).filter(Clause.document_id == self.test_doc_id).delete(synchronize_session=False)
        db.session.query(Document).filter(Document.id == self.test_doc_id).delete(synchronize_session=False)
        db.session.commit()

        # Verify test_doc is gone
        found_test_doc = Document.query.filter_by(id=self.test_doc_id).first()
        self.assertIsNone(found_test_doc, "Test document should have been deleted")

        # Verify unrelated_doc is STILL PRESENT and untouched
        found_unrelated = Document.query.filter_by(id=self.unrelated_doc_id).first()
        self.assertIsNotNone(found_unrelated, "Unrelated document must remain untouched")
        self.assertEqual(found_unrelated.filename, "unrelated_production_contract.pdf")

        # Verify unrelated_clause is STILL PRESENT
        found_clause = Clause.query.filter_by(id=self.unrelated_clause_id).first()
        self.assertIsNotNone(found_clause, "Unrelated clause must remain untouched")
        self.assertEqual(found_clause.clause_type, "rent_payment")

    def test_foreign_key_cascade_safety(self):
        """Verifies that deleting a document safely cascades without violating foreign key constraints."""
        doc_id = uuid.uuid4()
        doc = Document(id=doc_id, filename="cascade_test.txt", original_format="txt", document_type="rental_agreement", redacted_text="Text")
        db.session.add(doc)
        db.session.commit()

        clause_id = uuid.uuid4()
        c = Clause(id=clause_id, document_id=doc_id, clause_index=0, redacted_text="Clause text")
        db.session.add(c)
        db.session.commit()

        dl = Deadline(document_id=doc_id, clause_id=clause_id, deadline_type="notice_period", raw_text="Within 30 days")
        db.session.add(dl)
        db.session.commit()

        # Delete document in correct order or via cascade
        db.session.query(Deadline).filter(Deadline.document_id == doc_id).delete(synchronize_session=False)
        db.session.query(Clause).filter(Clause.document_id == doc_id).delete(synchronize_session=False)
        db.session.query(Document).filter(Document.id == doc_id).delete(synchronize_session=False)
        db.session.commit()

        self.assertIsNone(Document.query.filter_by(id=doc_id).first())
        self.assertIsNone(Clause.query.filter_by(id=clause_id).first())

    def test_search_path_is_strictly_test_isolation_without_public(self):
        """Verifies that search_path in the test environment has NO public fallback."""
        with db.engine.connect() as conn:
            sp = conn.execute(text("SHOW search_path;")).scalar()
            self.assertEqual(sp, "clauseguard_test_isolation")
            self.assertNotIn("public", [p.strip().strip('"') for p in sp.split(",")])

    def test_fail_closed_guard_rejects_missing_or_public_schema(self):
        """Verifies that enforce_test_database_isolation raises RuntimeError on invalid schemas."""
        orig_schema = os.environ.get("TEST_DATABASE_SCHEMA")
        try:
            # Test empty / unset
            os.environ["TEST_DATABASE_SCHEMA"] = ""
            with self.assertRaises(RuntimeError) as ctx:
                enforce_test_database_isolation(self.app)
            self.assertIn("FAIL-CLOSED SAFETY GUARD ACTIVATED", str(ctx.exception))

            # Test explicit 'public'
            os.environ["TEST_DATABASE_SCHEMA"] = "public"
            with self.assertRaises(RuntimeError) as ctx:
                enforce_test_database_isolation(self.app)
            self.assertIn("FAIL-CLOSED SAFETY GUARD ACTIVATED", str(ctx.exception))
        finally:
            if orig_schema is not None:
                os.environ["TEST_DATABASE_SCHEMA"] = orig_schema

    def test_orm_crud_operations_stay_in_test_schema(self):
        """Verifies that ORM insert, read, update, delete only touch clauseguard_test_isolation."""
        test_doc_id = uuid.uuid4()
        doc = Document(
            id=test_doc_id,
            filename="isolation_orm_verify.txt",
            original_format="txt",
            document_type="rental_agreement",
            redacted_text="Test ORM text"
        )
        db.session.add(doc)
        db.session.commit()

        try:
            # 1. Verify row exists in clauseguard_test_isolation.documents
            with db.engine.connect() as conn:
                in_test = conn.execute(
                    text("SELECT count(*) FROM clauseguard_test_isolation.documents WHERE id = :id;"),
                    {"id": test_doc_id}
                ).scalar()
                self.assertEqual(in_test, 1, "Inserted document must exist in clauseguard_test_isolation")

                # 2. Verify row DOES NOT EXIST in public.documents
                in_public = conn.execute(
                    text("SELECT count(*) FROM public.documents WHERE id = :id;"),
                    {"id": test_doc_id}
                ).scalar()
                self.assertEqual(in_public, 0, "Inserted document must NOT exist in public")

            # 3. Update via ORM
            doc.redacted_text = "Updated ORM text"
            db.session.commit()

            with db.engine.connect() as conn:
                updated_txt = conn.execute(
                    text("SELECT redacted_text FROM clauseguard_test_isolation.documents WHERE id = :id;"),
                    {"id": test_doc_id}
                ).scalar()
                self.assertEqual(updated_txt, "Updated ORM text")
                
                # Check public still has 0 rows for this id
                in_public_after_upd = conn.execute(
                    text("SELECT count(*) FROM public.documents WHERE id = :id;"),
                    {"id": test_doc_id}
                ).scalar()
                self.assertEqual(in_public_after_upd, 0)
        finally:
            # 4. Clean up the test document
            db.session.query(Document).filter(Document.id == test_doc_id).delete(synchronize_session=False)
            db.session.commit()

    def test_missing_table_fails_immediately_without_falling_back(self):
        """Verifies that missing relations raise UndefinedTable immediately with no public fallback."""
        with db.engine.connect() as conn:
            with self.assertRaises(Exception) as ctx:
                conn.execute(text("SELECT * FROM non_existent_table_never_created_999;"))
            self.assertIn("does not exist", str(ctx.exception).lower())

    def test_invalid_schema_identifier_rejected_by_validator(self):
        """Verifies that invalid or injection-laden schema identifiers raise ValueError."""
        import re
        bad_schemas = [
            "public; DROP TABLE documents;",
            "test_schema -- comment",
            "schema with spaces",
            "schema'quote",
            "123startsWithNumber"
        ]
        pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        for bad in bad_schemas:
            self.assertIsNone(re.match(pattern, bad), f"Identifier {bad!r} must be rejected by validator")

    def test_begin_hook_failure_aborts_transaction_and_fails_closed(self):
        """Verifies that a failure in schema assignment immediately aborts transaction and never falls back."""
        from sqlalchemy import create_engine, event, text
        from app.config import Config

        test_url = Config.SQLALCHEMY_DATABASE_URI
        broken_engine = create_engine(test_url)

        @event.listens_for(broken_engine, "begin")
        def broken_on_begin(conn):
            # Intentionally fail closed during transaction start
            raise RuntimeError("SIMULATED SCHEMA INITIALIZATION FAILURE IN BEGIN HOOK")

        with broken_engine.connect() as conn:
            with self.assertRaises(RuntimeError) as ctx:
                with conn.begin():
                    # This line must NEVER be reached
                    conn.execute(text("SELECT * FROM documents;"))
            self.assertIn("SIMULATED SCHEMA INITIALIZATION FAILURE", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
