"""Table schemas, enums, ID helpers and validators. SINGLE SOURCE OF TRUTH.

Every parquet file a part WRITES must pass `validate(name, df)` before it is
shared. Every part READING a file should call it too. Validation needs only
pandas + numpy so it runs in all three repos.

Change process: only the contract owner edits this file, bumps
CONTRACT_VERSION, and posts the new copy to all three repos.
"""
from __future__ import annotations

import re
from datetime import timedelta

import numpy as np
import pandas as pd

from .maths import PROB_SUM_TOL

CONTRACT_VERSION = "1.2.0"

# ---------------------------------------------------------------- enums
LEAGUES = ["E0", "E1", "SP1", "D1", "I1", "F1", "N1", "P1"]  # football-data codes
FIRST_SEASON = "2005-06"
HOLDOUT_SEASON = "2025-26"
SNAPSHOTS = ["PRE_CLOSING", "CLOSING"]          # never "OPENING"
PRECISIONS = ["EXACT", "DATE", "WINDOW", "UNKNOWN"]
# Individual bookmakers (football-data column prefixes; Part 1 ingests those present):
# B365 Bet365, PS Pinnacle, BW Bet&Win, IW Interwetten, WH William Hill, VC VC Bet.
# MAX / AVG are football-data aggregate columns: efficiency map only, never executable.
BOOKMAKERS = ["B365", "PS", "BW", "IW", "WH", "VC", "MAX", "AVG"]
AGGREGATE_BOOKMAKERS = ["MAX", "AVG"]
REFERENCE_BOOKMAKER = "B365"                     # probability benchmark, all seasons (contract 1.2.0)
# Bookmakers never used for execution in a given season (provider warning: Pinnacle stale from 2025-07-23).
EXECUTION_EXCLUDE = {"2025-26": ["PS"]}
RESULTS = ["H", "D", "A"]
SPLITS = ["warmup", "train", "calibration", "validation", "holdout"]
MODEL_IDS = ["market", "elo", "dixon_coles", "marketnet", "lgbm"]
DATASET_MODES = ["REAL", "SYNTHETIC"]
ACTIONS = ["BET", "NO_BET"]
STAKING_MODES = ["FLAT", "QUARTER_KELLY"]
POLICY_IDS = ["naive", "controlled", "controlled_best_price"]
REASON_CODES = [
    "ACCEPTED",
    "EV_BELOW_THRESHOLD",
    "FAILS_STRESS",
    "INVALID_PRICE",
    "MISSING_QUOTE",
    "MISSING_PREDICTION",
    "STALE_QUOTE",
    "LEAGUE_NOT_ELIGIBLE",
    "EXPOSURE_LIMIT_MATCH",
    "EXPOSURE_LIMIT_TOTAL",
    "INSUFFICIENT_BANKROLL",
]

# ---------------------------------------------------------------- ID helpers
def slug(name: str) -> str:
    """Team slug: lowercase, non-alphanumerics -> '-', trimmed. 'Man United' -> 'man-united'."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# Season exceptions: the source file's season wins. 2019-20 includes fixtures replayed
# in Jun-Aug 2020 after the COVID suspension (Q1, contract 1.2.0).
SEASON_DATE_EXCEPTIONS = {"2019-20": ("2019-07-01", "2020-08-31")}


def season_date_bounds(season: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive date bounds a match in `season` must fall within (UTC dates)."""
    if season in SEASON_DATE_EXCEPTIONS:
        a, b = SEASON_DATE_EXCEPTIONS[season]
    else:
        y = int(season[:4])
        a, b = f"{y}-07-01", f"{y + 1}-06-30"
    return pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)


def season_str(start_year: int) -> str:
    """2023 -> '2023-24'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def make_match_id(league: str, season: str, kickoff_utc: pd.Timestamp, home: str, away: str) -> str:
    """E0_2023-24_20230812_arsenal_nottm-forest  (home/away are CANONICAL names)."""
    return f"{league}_{season}_{kickoff_utc.strftime('%Y%m%d')}_{slug(home)}_{slug(away)}"


def feature_cutoff(kickoff_utc: pd.Timestamp) -> pd.Timestamp:
    """Conservative information cutoff for PRE_CLOSING decisions.

    = kickoff date floored to 00:00 UTC, minus 2 days. Features may only use
    matches whose kickoff_utc < this value. See CONTRACT.md section 5.
    """
    return kickoff_utc.floor("D") - timedelta(days=2)


def matchweek_key(league: str, kickoff_utc: pd.Timestamp) -> str:
    """Bootstrap / batching unit: league + ISO year-week, e.g. 'E0_2023-W32'."""
    iso = kickoff_utc.isocalendar()
    return f"{league}_{iso.year}-W{iso.week:02d}"


# ---------------------------------------------------------------- schemas
# dtype tokens: str, float, int, bool, ts (tz-aware UTC timestamp), cat:<ENUM>
# A trailing '?' means nullable. Extra columns are allowed only if prefixed 'x_'.
SCHEMAS: dict[str, dict[str, str]] = {
    "matches": {
        "match_id": "str", "league": "cat:LEAGUES", "season": "str",
        "kickoff_utc": "ts", "kickoff_precision": "cat:PRECISIONS",
        "home_team": "str", "away_team": "str",
        "matchweek_key": "str", "feature_cutoff_utc": "ts",
    },
    "outcomes": {
        "match_id": "str", "home_goals": "int", "away_goals": "int", "result": "cat:RESULTS",
    },
    "quotes": {
        "quote_id": "str", "match_id": "str", "bookmaker": "cat:BOOKMAKERS",
        "snapshot": "cat:SNAPSHOTS", "observed_at_utc": "ts?",
        "timestamp_precision": "cat:PRECISIONS", "ingested_at_utc": "ts",
        "odds_home": "float?", "odds_draw": "float?", "odds_away": "float?",
        "source": "str", "is_valid": "bool", "invalid_reason": "str?",
    },
    "market_probs": {
        "match_id": "str", "bookmaker": "cat:BOOKMAKERS", "snapshot": "cat:SNAPSHOTS",
        "is_reference": "bool",
        "q_home": "float", "q_draw": "float", "q_away": "float", "overround": "float",
    },
    "efficiency_map": {
        "league": "cat:LEAGUES", "season": "str", "snapshot": "cat:SNAPSHOTS",
        "bookmaker": "cat:BOOKMAKERS", "n_matches": "int",
        "log_loss": "float", "brier": "float", "mean_overround": "float",
        "calib_slope": "float?",          # logistic recalibration slope of q (1 = calibrated)
        "fav_longshot_gap": "float?",     # mean(y - q) for longshots (q<0.25) minus for favourites (q>0.6)
        "draw_bias": "float?",            # mean(y_draw - q_draw)
    },
    "predictions": {
        "match_id": "str", "model_id": "cat:MODEL_IDS", "model_version": "str",
        "split": "cat:SPLITS", "decision_time_utc": "ts",
        "feature_cutoff_utc": "ts", "training_cutoff_utc": "ts",
        "p_home": "float", "p_draw": "float", "p_away": "float",
        "xg_home": "float?", "xg_away": "float?",
    },
    "decisions": {
        "decision_id": "str", "run_id": "str", "policy_id": "cat:POLICY_IDS", "match_id": "str", "model_id": "cat:MODEL_IDS",
        "split": "cat:SPLITS", "decision_time_utc": "ts", "batch_key": "str",
        "quote_id": "str?", "action": "cat:ACTIONS", "selection": "cat:RESULTS?",
        "bookmaker": "str?",              # bookmaker whose price was executed (null if NO_BET)
        "p_model": "float?", "p_stressed": "float?",
        "odds_offered": "float?", "odds_adjusted": "float?",
        "ev_raw": "float?", "ev_stressed": "float?",
        "stake": "float", "reason_code": "cat:REASON_CODES",
        "reason_codes_all": "str",        # '|'-joined, every check that failed (or 'ACCEPTED')
        "explanation": "str",
        "settled_pnl": "float?", "closing_fair_prob": "float?", "clv": "float?",
        "config_hash": "str", "is_exploratory": "bool",
    },
    # Optional (Part 2): per-match top feature contributions, for the replay screen.
    # LightGBM: SHAP values on the probability of `outcome`. MarketNet: W_k * x_k terms.
    "prediction_explanations": {
        "match_id": "str", "model_id": "cat:MODEL_IDS", "model_version": "str",
        "outcome": "cat:RESULTS", "rank": "int", "feature": "str",
        "feature_value": "float?", "contribution": "float",
    },
}

ENUMS = {
    "LEAGUES": LEAGUES, "PRECISIONS": PRECISIONS, "RESULTS": RESULTS, "BOOKMAKERS": BOOKMAKERS,
    "SNAPSHOTS": SNAPSHOTS, "SPLITS": SPLITS, "MODEL_IDS": MODEL_IDS, "ACTIONS": ACTIONS,
    "REASON_CODES": REASON_CODES, "POLICY_IDS": POLICY_IDS,
}

PRIMARY_KEYS = {
    "matches": ["match_id"],
    "outcomes": ["match_id"],
    "quotes": ["quote_id"],
    "market_probs": ["match_id", "bookmaker", "snapshot"],
    "efficiency_map": ["league", "season", "snapshot", "bookmaker"],
    "predictions": ["match_id", "model_id", "model_version"],
    "decisions": ["decision_id"],
    "prediction_explanations": ["match_id", "model_id", "model_version", "outcome", "rank"],
}

PROB_COLS = {
    "market_probs": ("q_home", "q_draw", "q_away"),
    "predictions": ("p_home", "p_draw", "p_away"),
}


class ContractError(ValueError):
    pass


def _check_col(df: pd.DataFrame, col: str, spec: str, table: str) -> None:
    nullable = spec.endswith("?")
    spec = spec.rstrip("?")
    s = df[col]
    if not nullable and s.isna().any():
        raise ContractError(f"{table}.{col} has nulls but is not nullable")
    nn = s.dropna()
    if spec == "ts":
        if not isinstance(s.dtype, pd.DatetimeTZDtype) or str(s.dt.tz) != "UTC":
            raise ContractError(f"{table}.{col} must be tz-aware UTC datetime, got {s.dtype}")
    elif spec == "float":
        if not pd.api.types.is_float_dtype(s):
            raise ContractError(f"{table}.{col} must be float, got {s.dtype}")
        if not np.isfinite(nn.to_numpy()).all():
            raise ContractError(f"{table}.{col} has non-finite values")
    elif spec == "int":
        if not pd.api.types.is_integer_dtype(s):
            raise ContractError(f"{table}.{col} must be integer, got {s.dtype}")
    elif spec == "bool":
        if not pd.api.types.is_bool_dtype(s):
            raise ContractError(f"{table}.{col} must be bool, got {s.dtype}")
    elif spec == "str":
        if not nn.map(lambda v: isinstance(v, str)).all():
            raise ContractError(f"{table}.{col} must be str")
    elif spec.startswith("cat:"):
        allowed = ENUMS[spec[4:]]
        bad = set(nn.unique()) - set(allowed)
        if bad:
            raise ContractError(f"{table}.{col} has values {bad} not in {spec[4:]}")


def validate(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Raise ContractError if df does not match SCHEMAS[table]. Returns df."""
    schema = SCHEMAS[table]
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ContractError(f"{table}: missing columns {missing}")
    extra = [c for c in df.columns if c not in schema and not c.startswith("x_")]
    if extra:
        raise ContractError(f"{table}: unexpected columns {extra} (prefix extras with 'x_')")
    for col, spec in schema.items():
        _check_col(df, col, spec, table)
    if df.duplicated(PRIMARY_KEYS[table]).any():
        raise ContractError(f"{table}: duplicate primary key {PRIMARY_KEYS[table]}")
    if table in PROB_COLS:
        p = df[list(PROB_COLS[table])].to_numpy(dtype=float)
        if (p < 0).any() or (np.abs(p.sum(axis=1) - 1) > PROB_SUM_TOL).any():
            raise ContractError(f"{table}: probabilities negative or not summing to 1")
    if table == "quotes":
        v = df[df["is_valid"]]
        o = v[["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=float)
        if np.isnan(o).any() or (o <= 1.0).any():
            raise ContractError("quotes: a row with is_valid=True has missing odds or odds <= 1")
    if table == "predictions":
        if (df["feature_cutoff_utc"] > df["decision_time_utc"]).any():
            raise ContractError("predictions: feature_cutoff_utc after decision_time_utc")
        if (df["training_cutoff_utc"] > df["feature_cutoff_utc"]).any():
            raise ContractError("predictions: training_cutoff_utc after feature_cutoff_utc (leakage)")
    if table == "market_probs":
        ref = df[df["is_reference"]]
        if ref.duplicated(["match_id", "snapshot"]).any():
            raise ContractError("market_probs: more than one reference quote per match and snapshot")
    return df


def read_table(path, table: str) -> pd.DataFrame:
    """Read a parquet file and validate it."""
    return validate(table, pd.read_parquet(path))


def write_table(df: pd.DataFrame, path, table: str) -> None:
    """Validate then write parquet (pyarrow)."""
    validate(table, df)
    df.to_parquet(path, index=False)
