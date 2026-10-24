"""Entry point: python -m modelling.run --models market,elo [--golden] [--include-holdout]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from contract.schemas import read_table, write_table

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "./artifacts"))
GOLDEN_TABLES = Path("contract/golden/tables")
GOLDEN_JSON = Path("contract/golden/artifacts_json")

ALL_MODELS = ["market", "elo", "dixon_coles", "marketnet", "lgbm"]


def _load_inputs(golden: bool):
    src = GOLDEN_TABLES if golden else ARTIFACTS_DIR
    matches = read_table(src / "matches.parquet", "matches")
    market_probs = read_table(src / "market_probs.parquet", "market_probs")
    outcomes = read_table(src / "outcomes.parquet", "outcomes")

    splits_path = (GOLDEN_JSON / "splits.json") if golden else (ARTIFACTS_DIR / "splits.json")
    splits_by_season: dict[str, str] = json.loads(splits_path.read_text())["by_season"]

    ref_pre = market_probs[
        market_probs["is_reference"] & (market_probs["snapshot"] == "PRE_CLOSING")
    ].copy()

    return matches, ref_pre, outcomes, splits_by_season


def _fix_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all *_utc columns are tz-aware UTC (validator requirement)."""
    for col in ["decision_time_utc", "feature_cutoff_utc", "training_cutoff_utc"]:
        if col in df.columns:
            s = pd.to_datetime(df[col], utc=True)
            df[col] = s.dt.tz_convert("UTC")
    return df


def _fix_nullable_floats(df: pd.DataFrame) -> pd.DataFrame:
    """xg_home / xg_away are float? — keep as float64 with NaN."""
    for col in ["xg_home", "xg_away"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", action="store_true",
                        help="read inputs from contract/golden/ instead of artifacts/")
    parser.add_argument("--models", default="market,elo",
                        help="comma-separated model names")
    parser.add_argument("--include-holdout", action="store_true")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    unknown = [m for m in model_names if m not in ALL_MODELS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Choose from {ALL_MODELS}")

    matches, ref_pre, outcomes, splits_by_season = _load_inputs(args.golden)

    all_preds: list[pd.DataFrame] = []

    for name in model_names:
        print(f"  running model: {name}")
        if name == "market":
            from modelling import market
            preds = market.predict_all(matches, ref_pre, splits_by_season, args.include_holdout)
        elif name == "elo":
            from modelling import elo
            preds = elo.predict_all(matches, outcomes, splits_by_season, args.include_holdout)
        elif name == "dixon_coles":
            from modelling import dixon_coles
            preds = dixon_coles.predict_all(matches, outcomes, splits_by_season, args.include_holdout)
        elif name == "marketnet":
            from modelling import marketnet
            preds = marketnet.predict_all(matches, ref_pre, outcomes, splits_by_season, args.include_holdout)
        elif name == "lgbm":
            from modelling import lgbm
            preds = lgbm.predict_all(matches, ref_pre, outcomes, splits_by_season, args.include_holdout)
        else:
            raise ValueError(name)

        preds = _fix_timestamps(preds)
        preds = _fix_nullable_floats(preds)
        all_preds.append(preds)

    predictions = pd.concat(all_preds, ignore_index=True)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "predictions.parquet"
    write_table(predictions, out_path, "predictions")
    print(f"Wrote {len(predictions)} rows -> {out_path}")


if __name__ == "__main__":
    main()
