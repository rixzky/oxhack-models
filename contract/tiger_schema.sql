-- Tiger Data (TimescaleDB / PostgreSQL) schema. OPTIONAL storage.
-- Mirrors the parquet tables in CONTRACT.md one to one (same column names).
-- Part 1 owns loading. Part 3 owns reading (through its Storage interface).
-- If DATABASE_URL is not set, everyone uses the parquet files instead.

CREATE TABLE IF NOT EXISTS matches (
  match_id TEXT PRIMARY KEY, league TEXT NOT NULL, season TEXT NOT NULL,
  kickoff_utc TIMESTAMPTZ NOT NULL, kickoff_precision TEXT NOT NULL,
  home_team TEXT NOT NULL, away_team TEXT NOT NULL,
  matchweek_key TEXT NOT NULL, feature_cutoff_utc TIMESTAMPTZ NOT NULL
);

-- Outcomes are kept in their own table, separate from prediction inputs.
CREATE TABLE IF NOT EXISTS outcomes (
  match_id TEXT PRIMARY KEY REFERENCES matches(match_id),
  home_goals INT NOT NULL, away_goals INT NOT NULL, result TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quotes (
  quote_id TEXT NOT NULL, match_id TEXT NOT NULL REFERENCES matches(match_id),
  bookmaker TEXT NOT NULL, snapshot TEXT NOT NULL,
  observed_at_utc TIMESTAMPTZ,              -- source observation time (may be NULL)
  timestamp_precision TEXT NOT NULL,
  ingested_at_utc TIMESTAMPTZ NOT NULL,     -- when WE loaded it; never used as observation time
  odds_home DOUBLE PRECISION, odds_draw DOUBLE PRECISION, odds_away DOUBLE PRECISION,
  source TEXT NOT NULL, is_valid BOOLEAN NOT NULL, invalid_reason TEXT,
  -- hypertable time column: observation time, else ingestion time
  ts TIMESTAMPTZ GENERATED ALWAYS AS (COALESCE(observed_at_utc, ingested_at_utc)) STORED,
  PRIMARY KEY (quote_id, ts)
);
-- Hypertable if TimescaleDB is available (Tiger Data). Safe to skip on plain Postgres.
-- SELECT create_hypertable('quotes', 'ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS predictions (
  match_id TEXT NOT NULL, model_id TEXT NOT NULL, model_version TEXT NOT NULL,
  split TEXT NOT NULL, decision_time_utc TIMESTAMPTZ NOT NULL,
  feature_cutoff_utc TIMESTAMPTZ NOT NULL, training_cutoff_utc TIMESTAMPTZ NOT NULL,
  p_home DOUBLE PRECISION NOT NULL, p_draw DOUBLE PRECISION NOT NULL, p_away DOUBLE PRECISION NOT NULL,
  xg_home DOUBLE PRECISION, xg_away DOUBLE PRECISION,
  PRIMARY KEY (match_id, model_id, model_version)
);

CREATE TABLE IF NOT EXISTS decisions (
  decision_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, match_id TEXT NOT NULL,
  model_id TEXT NOT NULL, split TEXT NOT NULL, decision_time_utc TIMESTAMPTZ NOT NULL,
  batch_key TEXT NOT NULL, quote_id TEXT, action TEXT NOT NULL, selection TEXT,
  p_model DOUBLE PRECISION, p_stressed DOUBLE PRECISION,
  odds_offered DOUBLE PRECISION, odds_adjusted DOUBLE PRECISION,
  ev_raw DOUBLE PRECISION, ev_stressed DOUBLE PRECISION, stake DOUBLE PRECISION NOT NULL,
  reason_code TEXT NOT NULL, reason_codes_all TEXT NOT NULL, explanation TEXT NOT NULL,
  settled_pnl DOUBLE PRECISION, closing_fair_prob DOUBLE PRECISION, clv DOUBLE PRECISION,
  config_hash TEXT NOT NULL, is_exploratory BOOLEAN NOT NULL
);

-- THE replay query: latest eligible PRE_CLOSING quote at or before a replay cutoff.
-- :cutoff defaults to the match kickoff_utc; the UI may choose an earlier cutoff.
-- Quotes with NULL observation time are NOT returned here; the decision engine then
-- falls back to the reference quote and flags it STALE_QUOTE (unverified time).
-- :match_id, :bookmaker, :cutoff are bind parameters.
-- SELECT * FROM quotes
--  WHERE match_id = :match_id AND bookmaker = :bookmaker AND snapshot = 'PRE_CLOSING'
--    AND is_valid AND observed_at_utc IS NOT NULL AND observed_at_utc <= :cutoff
--  ORDER BY observed_at_utc DESC LIMIT 1;
