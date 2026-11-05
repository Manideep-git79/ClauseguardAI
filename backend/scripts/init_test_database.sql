-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Table: users
CREATE TABLE users (
	id UUID NOT NULL, 
	name VARCHAR(255) NOT NULL, 
	email VARCHAR(255) NOT NULL, 
	password_hash VARCHAR(255) NOT NULL, 
	role VARCHAR(50) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
);

-- Table: sessions
CREATE TABLE sessions (
	id UUID NOT NULL, 
	user_id UUID NOT NULL, 
	device_fingerprint VARCHAR(255), 
	ip VARCHAR(64), 
	revoked BOOLEAN NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	last_seen TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- Table: documents
CREATE TABLE documents (
	id UUID NOT NULL, 
	filename VARCHAR(255) NOT NULL, 
	original_format VARCHAR(10) NOT NULL, 
	document_type VARCHAR(32) NOT NULL, 
	status VARCHAR(20) NOT NULL, 
	processing_stage VARCHAR(64), 
	error_message TEXT, 
	page_count INTEGER, 
	raw_text_encrypted BYTEA, 
	redacted_text TEXT, 
	overall_risk_score FLOAT, 
	risk_band VARCHAR(32), 
	model_version VARCHAR(64), 
	parties_metadata JSON, 
	uploaded_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	analyzed_at TIMESTAMP WITH TIME ZONE, 
	user_id UUID, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- Table: clauses
CREATE TABLE clauses (
	id UUID NOT NULL, 
	document_id UUID NOT NULL, 
	clause_index INTEGER NOT NULL, 
	clause_type VARCHAR(64), 
	clause_type_confidence FLOAT, 
	favorability_label VARCHAR(16), 
	favorability_confidence FLOAT, 
	risk_score FLOAT, 
	redacted_text TEXT NOT NULL, 
	raw_text_encrypted BYTEA, 
	start_offset INTEGER, 
	end_offset INTEGER, 
	page_number INTEGER, 
	embedding VECTOR(768), 
	content_type VARCHAR(64), 
	is_substantive BOOLEAN, 
	filter_reason TEXT, 
	human_review BOOLEAN, 
	clause_eligibility_score FLOAT, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE
);

-- Table: deadlines
CREATE TABLE deadlines (
	id UUID NOT NULL, 
	document_id UUID NOT NULL, 
	clause_id UUID, 
	deadline_type VARCHAR(64), 
	raw_text TEXT NOT NULL, 
	parsed_date DATE, 
	relative_days INTEGER, 
	confidence VARCHAR(16) NOT NULL, 
	obligation_title VARCHAR(256), 
	responsible_party VARCHAR(128), 
	required_action TEXT, 
	trigger_event VARCHAR(256), 
	trigger_date DATE, 
	duration_count INTEGER, 
	duration_unit VARCHAR(32), 
	day_count_convention VARCHAR(32), 
	recurrence VARCHAR(64), 
	conditions TEXT, 
	page_number INTEGER, 
	start_offset INTEGER, 
	end_offset INTEGER, 
	status VARCHAR(32), 
	review_reason TEXT, 
	calculation_note TEXT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE, 
	FOREIGN KEY(clause_id) REFERENCES clauses (id) ON DELETE SET NULL
);

-- Table: integrity_findings
CREATE TABLE integrity_findings (
	id UUID NOT NULL, 
	document_id UUID NOT NULL, 
	category VARCHAR(64) NOT NULL, 
	severity VARCHAR(16) NOT NULL, 
	confidence FLOAT NOT NULL, 
	detection_method VARCHAR(64) NOT NULL, 
	page_number INTEGER, 
	bbox JSON, 
	explanation TEXT NOT NULL, 
	evidence_snippet TEXT, 
	review_status VARCHAR(32) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE
);

-- Table: missing_clauses
CREATE TABLE missing_clauses (
	id UUID NOT NULL, 
	document_id UUID NOT NULL, 
	clause_type VARCHAR(64) NOT NULL, 
	severity VARCHAR(16) NOT NULL, 
	checklist_note TEXT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE
);

-- Table: pii_findings
CREATE TABLE pii_findings (
	id UUID NOT NULL, 
	document_id UUID NOT NULL, 
	entity_type VARCHAR(32) NOT NULL, 
	start_offset INTEGER, 
	end_offset INTEGER, 
	confidence FLOAT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE
);

-- Table: chat_sessions
CREATE TABLE chat_sessions (
	id UUID NOT NULL, 
	document_id UUID NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE
);

-- Table: chat_messages
CREATE TABLE chat_messages (
	id UUID NOT NULL, 
	session_id UUID, 
	document_id UUID, 
	user_id UUID, 
	role VARCHAR(20) NOT NULL, 
	content TEXT NOT NULL, 
	confidence FLOAT, 
	grounded BOOLEAN, 
	sources JSONB, 
	retrieved_clause_ids UUID[], 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE, 
	FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE SET NULL, 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- Table: audit_logs
CREATE TABLE audit_logs (
	id UUID NOT NULL, 
	user_id UUID, 
	user_email VARCHAR(255), 
	action VARCHAR(100) NOT NULL, 
	resource_type VARCHAR(50), 
	resource_id VARCHAR(100), 
	details JSONB, 
	ip_address VARCHAR(64), 
	created_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE SET NULL
);