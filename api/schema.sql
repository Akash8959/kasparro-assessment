-- Schema for Kasparro FSE Assessment runs and predictions

CREATE TABLE IF NOT EXISTS runs (
    run_id UUID PRIMARY KEY,
    split VARCHAR(32) NOT NULL,
    agent_version VARCHAR(16) NOT NULL,
    prompt_version VARCHAR(16) NOT NULL,
    dataset_version VARCHAR(64) NOT NULL,
    evaluator_version VARCHAR(64) NOT NULL,
    fact_version VARCHAR(64) NOT NULL,
    provider VARCHAR(64) NOT NULL,
    model VARCHAR(64) NOT NULL,
    settings JSONB NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    git_revision VARCHAR(64),
    file_hashes JSONB NOT NULL,
    metrics JSONB,
    status VARCHAR(32) NOT NULL
);

CREATE TABLE IF NOT EXISTS answer_results (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    answer_id VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL,
    latency_ms INTEGER NOT NULL,
    error JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_run_answer UNIQUE (run_id, answer_id)
);

CREATE TABLE IF NOT EXISTS claim_results (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    answer_id VARCHAR(32) NOT NULL,
    claim_id VARCHAR(64) NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    quote TEXT NOT NULL,
    label VARCHAR(32) NOT NULL,
    fact_ids JSONB NOT NULL,
    reason TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_run_answer_span UNIQUE (run_id, answer_id, start_char, end_char)
);

CREATE INDEX IF NOT EXISTS idx_answer_results_run_id ON answer_results(run_id);
CREATE INDEX IF NOT EXISTS idx_claim_results_run_id ON claim_results(run_id);
