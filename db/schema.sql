-- ============================================================
-- HerdSync Database Schema
-- Run against the RDS Postgres instance
-- ============================================================

-- Providers
CREATE TABLE IF NOT EXISTS providers (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    company         VARCHAR(200),
    phone           VARCHAR(50),
    email           VARCHAR(200),
    address         TEXT,
    active_since    DATE,
    status          VARCHAR(20) DEFAULT 'active',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Animals (unified serial_id sequence across all species)
CREATE TABLE IF NOT EXISTS animals (
    serial_id   SERIAL PRIMARY KEY,
    species     VARCHAR(20) NOT NULL CHECK (species IN ('chicken', 'goat', 'lamb')),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_animals_species ON animals(species);

-- Chickens
CREATE TABLE IF NOT EXISTS chickens (
    serial_id       INTEGER PRIMARY KEY REFERENCES animals(serial_id),
    live_weight     NUMERIC(10,2),
    hang_weight     NUMERIC(10,2),
    hang_portion    NUMERIC(10,4),
    whole_hw        NUMERIC(10,2),
    prov_id         INTEGER REFERENCES providers(id),
    kill_date       DATE,
    process_date    DATE,
    purchase_date   DATE,
    delivery_date   DATE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Goats
CREATE TABLE IF NOT EXISTS goats (
    serial_id       INTEGER PRIMARY KEY REFERENCES animals(serial_id),
    hook_id         VARCHAR(50),
    description     TEXT,
    live_weight     NUMERIC(10,2),
    hang_weight     NUMERIC(10,2),
    hang_portion    NUMERIC(10,4),
    whole_hw        NUMERIC(10,2),
    grade           VARCHAR(20),
    prov_id         INTEGER REFERENCES providers(id),
    kill_date       DATE,
    process_date    DATE,
    purchase_date   DATE,
    delivery_date   DATE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Lambs
CREATE TABLE IF NOT EXISTS lambs (
    serial_id       INTEGER PRIMARY KEY REFERENCES animals(serial_id),
    description     TEXT,
    live_weight     NUMERIC(10,2),
    hang_weight     NUMERIC(10,2),
    hang_portion    NUMERIC(10,4),
    whole_hw        NUMERIC(10,2),
    grade           VARCHAR(20),
    prov_id         INTEGER REFERENCES providers(id),
    kill_date       DATE,
    process_date    DATE,
    purchase_date   DATE,
    delivery_date   DATE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Grading results (goats and lambs)
CREATE TABLE IF NOT EXISTS grade_results (
    id                  SERIAL PRIMARY KEY,
    serial_id           INTEGER NOT NULL REFERENCES animals(serial_id),
    grade               VARCHAR(20),
    live_weight         NUMERIC(10,2),
    all_views_ok        BOOLEAN,
    measurements        JSONB,
    grade_details       JSONB,
    manual_override_history JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Raw capture S3 keys
    side_raw_s3_key     TEXT,
    top_raw_s3_key      TEXT,
    front_raw_s3_key    TEXT,

    -- Debug overlay S3 keys
    side_debug_s3_key   TEXT,
    top_debug_s3_key    TEXT,
    front_debug_s3_key  TEXT,

    -- Timing
    capture_sec         NUMERIC(8,2),
    ec2_sec             NUMERIC(8,2),
    total_sec           NUMERIC(8,2),

    -- Warnings
    warnings            TEXT[],

    graded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_grade_results_serial ON grade_results(serial_id);
CREATE INDEX IF NOT EXISTS idx_grade_results_date ON grade_results(graded_at);

ALTER TABLE grade_results
    ADD COLUMN IF NOT EXISTS grade_details JSONB;

ALTER TABLE grade_results
    ADD COLUMN IF NOT EXISTS manual_override_history JSONB NOT NULL DEFAULT '[]'::jsonb;

-- updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN
    CREATE TRIGGER trg_providers_updated BEFORE UPDATE ON providers FOR EACH ROW EXECUTE FUNCTION update_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
    CREATE TRIGGER trg_chickens_updated BEFORE UPDATE ON chickens FOR EACH ROW EXECUTE FUNCTION update_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
    CREATE TRIGGER trg_goats_updated BEFORE UPDATE ON goats FOR EACH ROW EXECUTE FUNCTION update_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
    CREATE TRIGGER trg_lambs_updated BEFORE UPDATE ON lambs FOR EACH ROW EXECUTE FUNCTION update_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================
-- Audit Logs
-- Tracks all mutations (create, update, delete) across the system.
-- Written by db-proxy after successful upstream responses.
-- ============================================================

CREATE TABLE IF NOT EXISTS audit_logs (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ DEFAULT NOW(),
    username        TEXT NOT NULL,
    role            TEXT,
    action          VARCHAR(20) NOT NULL,
    resource_type   VARCHAR(30) NOT NULL,
    resource_id     TEXT,
    detail          JSONB,
    ip_address      TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_username ON audit_logs(username);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type);
