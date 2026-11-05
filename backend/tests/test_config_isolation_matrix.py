import os
import sys
import unittest
import importlib
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import app.config
from tests.test_isolation_guard import enforce_test_database_isolation

class TestConfigIsolationMatrix(unittest.TestCase):
    """
    Automated regression suite verifying every configuration combination
    and guaranteeing that no test runner can ever connect to the production
    database string or production compute endpoint.
    """

    def setUp(self):
        self.orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)

    def test_comb1_testing_enabled_without_test_db_url_raises(self):
        """Combination 1: TESTING=true with NO TEST_DATABASE_URL raises RuntimeError."""
        os.environ["TESTING"] = "true"
        os.environ.pop("TEST_DATABASE_URL", None)
        os.environ.pop("TEST_DATABASE_SCHEMA", None)

        with self.assertRaises(RuntimeError) as ctx:
            importlib.reload(app.config)
        self.assertIn("TEST_DATABASE_URL is strictly required whenever TESTING is enabled", str(ctx.exception))

    def test_comb2b_schema_only_without_testing_flag_fails_closed(self):
        """Setting TEST_DATABASE_SCHEMA alone without TESTING flag must also fail closed."""
        os.environ.pop("TESTING", None)
        os.environ["TEST_DATABASE_SCHEMA"] = "clauseguard_test_isolation"
        os.environ.pop("TEST_DATABASE_URL", None)

        with self.assertRaises(RuntimeError) as ctx:
            importlib.reload(app.config)
        self.assertIn("TEST_DATABASE_URL is strictly required whenever TESTING is enabled", str(ctx.exception))

    def test_comb2_testing_with_schema_only_without_test_db_url_raises(self):
        """Combination 2: Never allow TEST_DATABASE_SCHEMA alone to select DATABASE_URL."""
        os.environ["TESTING"] = "true"
        os.environ["TEST_DATABASE_SCHEMA"] = "clauseguard_test_isolation"
        os.environ.pop("TEST_DATABASE_URL", None)

        with self.assertRaises(RuntimeError) as ctx:
            importlib.reload(app.config)
        self.assertIn("TEST_DATABASE_URL is strictly required whenever TESTING is enabled", str(ctx.exception))

    def test_comb3_test_db_url_equals_prod_without_schema_raises(self):
        """Combination 3: TEST_DATABASE_URL == DATABASE_URL without schema raises RuntimeError."""
        os.environ["TESTING"] = "true"
        os.environ["DATABASE_URL"] = "postgresql://user:pass@ep-test-host.neon.tech/neondb"
        os.environ["TEST_DATABASE_URL"] = "postgresql://user:pass@ep-test-host.neon.tech/neondb"
        os.environ.pop("TEST_DATABASE_SCHEMA", None)

        with self.assertRaises(RuntimeError) as ctx:
            importlib.reload(app.config)
        self.assertIn("CRITICAL SAFETY VIOLATION", str(ctx.exception))

    def test_comb4_test_db_url_equals_prod_WITH_schema_raises(self):
        """Combination 4: Reject production branch connection even when TEST_DATABASE_SCHEMA is supplied."""
        os.environ["TESTING"] = "true"
        os.environ["DATABASE_URL"] = "postgresql://user:pass@ep-prod-branch.neon.tech/neondb"
        os.environ["TEST_DATABASE_URL"] = "postgresql://user:pass@ep-prod-branch.neon.tech/neondb"
        os.environ["TEST_DATABASE_SCHEMA"] = "clauseguard_test_isolation"

        with self.assertRaises(RuntimeError) as ctx:
            importlib.reload(app.config)
        self.assertIn("CRITICAL SAFETY VIOLATION", str(ctx.exception))
        self.assertIn("Tests are strictly forbidden from connecting to the production database string", str(ctx.exception))

    def test_comb5_test_db_url_targets_prod_neon_endpoint_raises(self):
        """Combination 5: Reject TEST_DATABASE_URL sharing production Neon endpoint prefix."""
        os.environ["TESTING"] = "true"
        os.environ["DATABASE_URL"] = "postgresql://prod_user:prod_pass@ep-jolly-hall-b5q4upq4-pooler.c-7.us-east-2.aws.neon.tech/neondb"
        # Test URL uses different user/pass/port but shares the same production compute endpoint!
        os.environ["TEST_DATABASE_URL"] = "postgresql://other_user:other_pass@ep-jolly-hall-b5q4upq4.c-7.us-east-2.aws.neon.tech/neondb"

        with self.assertRaises(RuntimeError) as ctx:
            importlib.reload(app.config)
        self.assertIn("targets production compute endpoint", str(ctx.exception))

    def test_comb6_normal_production_traffic_unaffected(self):
        """Combination 6: When TESTING is false, normal production configuration is unchanged."""
        os.environ.pop("TESTING", None)
        os.environ.pop("TEST_DATABASE_URL", None)
        os.environ.pop("TEST_DATABASE_SCHEMA", None)
        os.environ["FLASK_ENV"] = "production"
        os.environ["DATABASE_URL"] = "postgresql://prod_user:prod_pass@ep-jolly-hall-b5q4upq4-pooler.c-7.us-east-2.aws.neon.tech/neondb"

        importlib.reload(app.config)
        self.assertEqual(
            app.config.Config.SQLALCHEMY_DATABASE_URI,
            "postgresql+psycopg2://prod_user:prod_pass@ep-jolly-hall-b5q4upq4-pooler.c-7.us-east-2.aws.neon.tech/neondb"
        )

    def test_comb7_valid_isolated_test_database_url_succeeds(self):
        """Combination 7: Distinct test branch URL succeeds and selects TEST_DATABASE_URL."""
        os.environ["TESTING"] = "true"
        os.environ["DATABASE_URL"] = "postgresql://prod_user:prod_pass@ep-jolly-hall-b5q4upq4-pooler.c-7.us-east-2.aws.neon.tech/neondb"
        os.environ["TEST_DATABASE_URL"] = "postgresql://test_runner:test_pass@ep-isolated-test-branch-1234.c-7.us-east-2.aws.neon.tech/neondb"

        importlib.reload(app.config)
        self.assertIn("ep-isolated-test-branch-1234", app.config.Config.SQLALCHEMY_DATABASE_URI)

    def test_comb8_test_runner_guard_rejects_production_connection(self):
        """Combination 8: enforce_test_database_isolation halts runner if connected to production."""
        from unittest.mock import MagicMock
        mock_app = MagicMock()
        mock_app.config = {
            "SQLALCHEMY_DATABASE_URI": "postgresql://neondb_owner:secret@ep-jolly-hall-b5q4upq4-pooler.c-7.us-east-2.aws.neon.tech/neondb"
        }
        os.environ["DATABASE_URL"] = "postgresql://neondb_owner:secret@ep-jolly-hall-b5q4upq4-pooler.c-7.us-east-2.aws.neon.tech/neondb"

        with self.assertRaises(RuntimeError) as ctx:
            enforce_test_database_isolation(mock_app)
        self.assertIn("FAIL-CLOSED SAFETY GUARD ACTIVATED", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()

