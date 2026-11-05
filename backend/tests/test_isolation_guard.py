import os
from urllib.parse import urlparse
from sqlalchemy import text
from app.extensions import db

def enforce_test_database_isolation(app):
    """
    Fail-closed test safety protection:
    Refuses test execution unless:
    1. The active database connection URI does NOT point to the production database or endpoint.
    2. TEST_DATABASE_SCHEMA (if used) is explicitly set and not 'public'.
    3. The active database connection verifies that current_schema() matches TEST_DATABASE_SCHEMA.
    4. The active search_path does NOT include 'public'.
    """
    db_uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
    prod_uri = os.environ.get("DATABASE_URL", "")

    # Reject running tests directly on production database URL
    if db_uri and prod_uri:
        if db_uri.strip().lower() == prod_uri.strip().lower():
            raise RuntimeError(
                "FAIL-CLOSED SAFETY GUARD ACTIVATED: Test suite is connected to the production DATABASE_URL! "
                "Tests must run on a dedicated TEST_DATABASE_URL pointing to an isolated test branch."
            )

        parsed_db = urlparse(db_uri)
        parsed_prod = urlparse(prod_uri)
        prod_host = (parsed_prod.hostname or "").lower()
        test_host = (parsed_db.hostname or "").lower()

        # Reject production Neon endpoint
        if prod_host and ("neon.tech" in prod_host or "aws.neon.tech" in prod_host):
            prod_prefix = prod_host.split(".")[0].replace("-pooler", "")
            if prod_prefix and prod_prefix in test_host:
                raise RuntimeError(
                    f"FAIL-CLOSED SAFETY GUARD ACTIVATED: Test runner connected to production Neon endpoint '{prod_host}'! "
                    "Execution halted to protect production data."
                )

    schema = os.environ.get("TEST_DATABASE_SCHEMA")
    if schema:
        if schema.strip().lower() in ("", "public", "pg_catalog", "information_schema"):
            raise RuntimeError(
                f"FAIL-CLOSED SAFETY GUARD ACTIVATED: Destructive tests cannot run against production schema! "
                f"TEST_DATABASE_SCHEMA must be explicitly set to a dedicated test schema (current value: {repr(schema)})."
            )

        with app.app_context():
            with db.engine.connect() as conn:
                # 1. Positive Identity Verification for Neon connections:
                # When connecting to a Neon host, positive verification of project/branch identity is required.
                # Silent bypass or exception suppression is strictly forbidden.
                is_neon = "neon.tech" in test_host or "aws.neon.tech" in test_host
                if is_neon:
                    try:
                        live_project = conn.execute(text("SHOW neon.project_id;")).scalar()
                        live_branch = conn.execute(text("SHOW neon.branch_id;")).scalar()
                    except Exception as err:
                        raise RuntimeError(
                            f"FAIL-CLOSED SAFETY GUARD ACTIVATED: Failed to query Neon project/branch settings ({err}). "
                            "Positive database identity verification is strictly required for Neon test connections."
                        )

                    if not live_project or not live_branch:
                        raise RuntimeError(
                            "FAIL-CLOSED SAFETY GUARD ACTIVATED: neon.project_id or neon.branch_id returned empty! "
                            "Cannot verify test database identity. Halting execution."
                        )

                    # Reject production Neon project
                    prod_project = os.environ.get("PROD_NEON_PROJECT_ID", "super-heart-35339659")
                    if live_project.strip() == prod_project:
                        raise RuntimeError(
                            f"FAIL-CLOSED SAFETY GUARD ACTIVATED: Active database session is running on PRODUCTION Neon project '{live_project}'! "
                            "Tests must run on a completely separate test project. Execution halted to protect production data."
                        )

                    # Reject production Neon branch
                    prod_branch = os.environ.get("PROD_NEON_BRANCH_ID", "br-dry-art-b5bwc8dy")
                    if live_branch.strip() == prod_branch:
                        raise RuntimeError(
                            f"FAIL-CLOSED SAFETY GUARD ACTIVATED: Active database session is running on PRODUCTION Neon branch '{live_branch}'! "
                            "Tests must run on a dedicated test branch or separate project. Execution halted to protect production data."
                        )

                    # If an expected test project ID is configured, enforce positive match
                    expected_test_project = os.environ.get("EXPECTED_TEST_NEON_PROJECT_ID")
                    if expected_test_project and live_project.strip() != expected_test_project.strip():
                        raise RuntimeError(
                            f"FAIL-CLOSED SAFETY GUARD ACTIVATED: Connected Neon project '{live_project}' does not match "
                            f"EXPECTED_TEST_NEON_PROJECT_ID '{expected_test_project}'! Execution halted."
                        )

                active_schema = conn.execute(text("SELECT current_schema();")).scalar()
                search_path = conn.execute(text("SHOW search_path;")).scalar()

                if active_schema != schema:
                    raise RuntimeError(
                        f"FAIL-CLOSED SAFETY GUARD ACTIVATED: Active database schema is '{active_schema}', "
                        f"expected isolated schema '{schema}'! Halting test execution to protect production data."
                    )

                search_path_parts = [p.strip().strip('"') for p in search_path.split(",")]
                if "public" in search_path_parts:
                    raise RuntimeError(
                        f"FAIL-CLOSED SAFETY GUARD ACTIVATED: search_path contains 'public' fallback: '{search_path}'. "
                        f"Isolation requires strictly isolated schema without public fallback."
                    )
