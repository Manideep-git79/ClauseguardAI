import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

backend_dir = Path(__file__).resolve().parent.parent
dotenv_path = backend_dir / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

raw_url = os.environ.get("DATABASE_URL")
if not raw_url:
    print("ERROR: DATABASE_URL is not set.")
    sys.exit(1)

if raw_url.startswith("postgres://"):
    raw_url = raw_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif raw_url.startswith("postgresql://"):
    raw_url = raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)

target_schema = os.environ.get("TEST_DATABASE_SCHEMA", "clauseguard_test_isolation")

engine = create_engine(raw_url)

def sync_schema():
    print(f"[*] Synchronizing isolated test schema: '{target_schema}' from 'public'...")
    with engine.connect() as conn:
        with conn.begin():
            # 1. Create schema
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {target_schema};"))
            
            # 2. Query all tables in public
            tables = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)).fetchall()
            table_names = [t[0] for t in tables]
            print(f"[*] Found {len(table_names)} tables in public: {', '.join(table_names)}")

            # 3. Clone tables if not existing
            for tbl in table_names:
                exists = conn.execute(text(f"""
                    SELECT count(*) FROM information_schema.tables 
                    WHERE table_schema = '{target_schema}' AND table_name = '{tbl}';
                """)).scalar()
                if not exists:
                    print(f"  -> Creating table {target_schema}.{tbl} (LIKE public.{tbl} INCLUDING ALL)")
                    conn.execute(text(f"CREATE TABLE {target_schema}.{tbl} (LIKE public.{tbl} INCLUDING ALL);"))
                else:
                    # Sync any missing columns
                    cols_pub = conn.execute(text(f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_schema = 'public' AND table_name = '{tbl}';
                    """)).fetchall()
                    cols_test = conn.execute(text(f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_schema = '{target_schema}' AND table_name = '{tbl}';
                    """)).fetchall()
                    test_col_names = set(c[0] for c in cols_test)
                    for col_name, data_type in cols_pub:
                        if col_name not in test_col_names:
                            print(f"  -> Adding missing column {col_name} ({data_type}) to {target_schema}.{tbl}")
                            conn.execute(text(f"ALTER TABLE {target_schema}.{tbl} ADD COLUMN {col_name} {data_type};"))

            # 4. Verify foreign key constraints
            print("[*] Verifying foreign keys on isolated test schema...")
            fks = [
                ("documents", "fk_documents_user", "FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL"),
                ("clauses", "fk_clauses_document", "FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE"),
                ("deadlines", "fk_deadlines_document", "FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE"),
                ("deadlines", "fk_deadlines_clause", "FOREIGN KEY (clause_id) REFERENCES clauses(id) ON DELETE SET NULL"),
                ("chat_sessions", "fk_chat_sessions_document", "FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE"),
                ("chat_messages", "fk_chat_messages_session", "FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE"),
                ("chat_messages", "fk_chat_messages_document", "FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE"),
                ("chat_messages", "fk_chat_messages_user", "FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL"),
                ("missing_clauses", "fk_missing_clauses_document", "FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE"),
                ("audit_logs", "fk_audit_logs_user", "FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL"),
                ("pii_findings", "fk_pii_findings_document", "FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE"),
                ("sessions", "fk_sessions_user", "FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE"),
            ]
            for tbl, fk_name, fk_def in fks:
                has_fk = conn.execute(text(f"""
                    SELECT count(*) FROM information_schema.table_constraints
                    WHERE table_schema = '{target_schema}' AND table_name = '{tbl}' AND constraint_name = '{fk_name}';
                """)).scalar()
                if not has_fk:
                    print(f"  -> Adding FK {fk_name} on {target_schema}.{tbl}")
                    conn.execute(text(f"ALTER TABLE {target_schema}.{tbl} ADD CONSTRAINT {fk_name} {fk_def};"))

    print(f"[+] Schema '{target_schema}' successfully synchronized with zero public data leakage.")

if __name__ == "__main__":
    sync_schema()

