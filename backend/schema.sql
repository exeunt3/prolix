CREATE TABLE IF NOT EXISTS traces (
  trace_id UUID PRIMARY KEY,
  created_at TIMESTAMP NOT NULL,
  object_label VARCHAR(128) NOT NULL,
  vector_domain VARCHAR(64) NOT NULL,
  concept_path JSONB NOT NULL,
  paragraph_text TEXT NOT NULL,
  ending_type VARCHAR(16) NOT NULL,
  safety_flag BOOLEAN NOT NULL DEFAULT FALSE,
  dark_flag BOOLEAN NOT NULL DEFAULT FALSE
);
