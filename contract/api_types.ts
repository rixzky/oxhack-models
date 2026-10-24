// API contract between the Part 3 backend (FastAPI) and the Part 3 frontend (React). Contract v1.2.0.
// The backend only READS artifact files from Parts 1 and 2 and calls contract/decisions.py for sliders.
// SINGLE SOURCE OF TRUTH for response shapes. Example payloads: contract/golden/api/*.json
// Conventions: outcome arrays are always [H, D, A]; times are ISO 8601 UTC strings ending in "Z";
// odds are decimal; probabilities are 0..1; money is in "units" (paper bankroll).
// Every numeric field may be null when it cannot be computed. The UI must show "n/a", never 0.

export type Outcome = "H" | "D" | "A";
export type DatasetMode = "REAL" | "SYNTHETIC";
export type Precision = "EXACT" | "DATE" | "WINDOW" | "UNKNOWN";
export type Snapshot = "PRE_CLOSING" | "CLOSING";
export type Split = "warmup" | "train" | "calibration" | "validation" | "holdout";
export type ReasonCode =
  | "ACCEPTED" | "EV_BELOW_THRESHOLD" | "FAILS_STRESS" | "INVALID_PRICE" | "MISSING_QUOTE"
  | "MISSING_PREDICTION" | "STALE_QUOTE" | "LEAGUE_NOT_ELIGIBLE" | "EXPOSURE_LIMIT_MATCH"
  | "EXPOSURE_LIMIT_TOTAL" | "INSUFFICIENT_BANKROLL";
export type Triple = [number, number, number];

// GET /api/meta
export interface Meta {
  contract_version: string;
  dataset_mode: DatasetMode;          // UI shows a prominent banner when SYNTHETIC
  storage_backend: "tiger" | "local"; // what is ACTUALLY active, never what is hoped for
  gemini_enabled: boolean;            // true only if configured AND a test call succeeded
  leagues: string[];
  seasons: string[];
  models: string[];
  frozen_config_hash: string | null;
  holdout_locked: boolean;            // true once holdout_locked.json exists
}

// GET /api/efficiency-map
export interface EfficiencyRow {
  league: string; season: string; snapshot: Snapshot; bookmaker: string; n_matches: number;
  log_loss: number; brier: number; mean_overround: number;
  calib_slope: number | null; fav_longshot_gap: number | null; draw_bias: number | null;
}
export interface EfficiencyMap {
  dataset_mode: DatasetMode;
  sign_conventions: Record<string, string>; // show these as legend text
  rows: EfficiencyRow[];
}

// GET /api/fixtures?league=E0&season=2025-26
export interface Fixture {
  match_id: string; league: string; season: string; kickoff_utc: string;
  home_team: string; away_team: string; split: Split; has_closing: boolean;
}

export interface QuoteView {
  quote_id: string; bookmaker: string; snapshot: Snapshot;
  observed_at_utc: string | null; timestamp_precision: Precision; source: string; odds: Triple;
  // Bookmaker behind each of the three prices. All "B365" for the reference quote; may differ for best price.
  bookmakers?: [string, string, string];
}
export type PriceSource = "REFERENCE" | "BEST_AVAILABLE";
export type PolicyId = "naive" | "controlled" | "controlled_best_price";

export interface PolicyConfig {
  min_ev: number; prob_stress_delta: number; price_haircut: number;
  staking_mode: "FLAT" | "QUARTER_KELLY"; flat_stake_frac: number; kelly_multiplier: number;
  max_stake_frac_per_match: number; max_open_exposure_frac: number; initial_bankroll: number;
  max_quote_age_hours: number; allow_precisions: Precision[]; league_gate: boolean;
  price_source: PriceSource; best_price_max_prob_gap: number;
}

// GET /api/replay/{match_id}?model_id=&policy_id=controlled|controlled_best_price&prob_stress_delta=&price_haircut=&min_ev=
// Overriding any policy parameter makes is_exploratory = true. Closing odds and the result
// are NEVER in this payload; they come only from /reveal.
export interface Replay {
  dataset_mode: DatasetMode;
  is_exploratory: boolean;
  config_hash: string;
  policy_id: PolicyId;
  price_source: PriceSource;
  match: { match_id: string; league: string; season: string; kickoff_utc: string;
           home_team: string; away_team: string; split: Split };
  information_cutoff_utc: string;
  offered_quote: QuoteView | null;
  market_probs: Triple | null;
  model: { model_id: string; model_version: string; probs: Triple } | null;
  per_outcome: {
    outcome: Outcome; p_model: number; p_stressed: number; q_market: number; // q_market is always Bet365 margin-free
    odds_offered: number; odds_adjusted: number; bookmaker: string | null; ev_raw: number; ev_stressed: number;
  }[];
  decision: {
    action: "BET" | "NO_BET"; selection: Outcome | null; stake: number; bookmaker?: string | null;
    reason_code: ReasonCode; reason_codes_all: ReasonCode[]; explanation: string;
  };
  // Optional ML explanation from Part 2 (LightGBM SHAP or MarketNet terms). Empty array if none.
  top_features: { outcome: Outcome; rank: number; feature: string;
                  feature_value: number | null; contribution: number }[];
  config_used: PolicyConfig;
}

// GET /api/replay/{match_id}/reveal
export interface Reveal {
  match_id: string;
  closing_quote: QuoteView | null;
  closing_fair_probs: Triple | null;
  result: { home_goals: number; away_goals: number; result: Outcome };
  clv: number | null;          // only if a bet was placed and a closing quote exists
  settled_pnl: number | null;  // only if a bet was placed
}

// GET /api/evidence?model_id=&split=validation|holdout  (built from forecast_metrics.json, policy_metrics.json, feature_importance.json)
export interface PolicySummary {
  policy_id: PolicyId; label: string;
  n_decisions: number; n_bets: number; n_abstain: number;
  turnover: number; profit: number; roi: number | null; max_drawdown: number;
  mean_clv: number | null; sharpe_per_matchweek: number | null; deflated_sharpe: number | null;
  reason_counts: Partial<Record<ReasonCode, number>>;
  bootstrap_ci_95: { roi: [number | null, number | null]; [metric: string]: [number | null, number | null] };
}
export interface CalibrationBin { bin_lo: number; bin_hi: number; n: number; mean_pred: number; obs_freq: number }
export interface ForecastRow {
  model_id: string; split: Split; n: number; log_loss: number; brier_sum3: number;
  calibration_bins: CalibrationBin[];
}
export interface FeatureImportance {
  model_id: string; model_version: string; method: string; split: Split;
  features: { feature: string; importance: number }[];
}
export interface Evidence {
  dataset_mode: DatasetMode; split: Split; model_id: string; is_locked_holdout: boolean;
  forecast: ForecastRow[];
  policies: PolicySummary[];
  equity_curve: { batch_key: string; time_utc: string; policy_id: string; bankroll: number; drawdown: number }[];
  feature_importance: FeatureImportance[];
  limitations: string[];
}

// GET /api/audit  (passes through artifacts/data_audit.json from Part 1)
export interface Audit {
  dataset_mode: DatasetMode;
  raw_counts: Record<string, number>;
  accepted_counts: Record<string, number>;
  invalid_quotes_by_reason: Record<string, number>;
  timestamp_precision_counts: Record<string, number>;
  duplicates_removed: number;
  missing_value_counts: Record<string, number>;
  flagged_outliers: { quote_id: string; reason: string }[];
  exclusions: { match_id: string; reason: string }[];
  note: string;
}

// GET /api/scenarios  (constructed demo inputs, NEVER shown as historical findings)
export interface Scenarios {
  label: string;
  scenarios: { id: string; title: string; p_model: Triple; odds: Triple;
               timestamp_precision: Precision;
               decision: Replay["decision"]; per_outcome: Replay["per_outcome"] }[];
}

// POST /api/explain/{match_id}  (Gemini: explains an existing decision record only)
export interface Explain { available: boolean; text: string | null; reason?: string }
