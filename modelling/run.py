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
ML_MODELS = {"marketnet", "lgbm"}


def _read_parquet(path, table: str) -> pd.DataFrame:
    """Read parquet using pyarrow (golden) or fastparquet (real artifacts)."""
    try:
        return read_table(path, table)
    except Exception:
        from contract.schemas import validate
        df = pd.read_parquet(path, engine="fastparquet")
        for col in ["kickoff_utc", "feature_cutoff_utc"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
        return validate(table, df)


def _load_inputs(golden: bool):
    src = GOLDEN_TABLES if golden else ARTIFACTS_DIR
    matches = _read_parquet(src / "matches.parquet", "matches")
    market_probs = _read_parquet(src / "market_probs.parquet", "market_probs")
    outcomes = _read_parquet(src / "outcomes.parquet", "outcomes")

    splits_path = (GOLDEN_JSON / "splits.json") if golden else (ARTIFACTS_DIR / "splits.json")
    splits_by_season: dict[str, str] = json.loads(splits_path.read_text())["by_season"]

    ref_pre = market_probs[
        market_probs["is_reference"] & (market_probs["snapshot"] == "PRE_CLOSING")
    ].copy()

    return matches, ref_pre, outcomes, splits_by_season


def _fix_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["decision_time_utc", "feature_cutoff_utc", "training_cutoff_utc"]:
        if col in df.columns:
            s = pd.to_datetime(df[col], utc=True)
            df[col] = s.dt.tz_convert("UTC")
    return df


def _fix_nullable_floats(df: pd.DataFrame) -> pd.DataFrame:
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

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    trials_path = ARTIFACTS_DIR / "trials_log.jsonl"

    all_preds: list[pd.DataFrame] = []
    all_explanations: list[pd.DataFrame] = []
    all_importances: list[dict] = []
    feat_table = None   # built once and shared by ML models

    for name in model_names:
        print(f"  running model: {name}")

        if name in ML_MODELS and feat_table is None:
            print("    building feature table...")
            from modelling import features as feat_mod
            matches_with_split = matches.copy()
            matches_with_split["split"] = matches_with_split["season"].map(splits_by_season)
            dc_preds = next(
                (p for p in all_preds if p["model_id"].iloc[0] == "dixon_coles"),
                None,
            )
            # also check existing artifacts (only if match IDs overlap — guards against stale files)
            if dc_preds is None:
                ap = ARTIFACTS_DIR / "predictions.parquet"
                if ap.exists():
                    try:
                        existing = _read_parquet(ap, "predictions")
                        dc_sub = existing[existing["model_id"] == "dixon_coles"]
                        current_ids = set(matches["match_id"])
                        if len(dc_sub) > 0 and len(dc_sub[dc_sub["match_id"].isin(current_ids)]) > 0:
                            dc_preds = dc_sub[dc_sub["match_id"].isin(current_ids)]
                    except Exception:
                        pass
            feat_table = feat_mod.build(matches_with_split, ref_pre, outcomes, dc_preds)
            feat_table.to_parquet(ARTIFACTS_DIR / "x_features.parquet", index=False)
            print(f"    feature table: {len(feat_table)} rows x {len(feat_mod.FEATURE_COLS)} features")

        if name == "market":
            from modelling import market
            preds = market.predict_all(matches, ref_pre, splits_by_season, args.include_holdout)
            expl = pd.DataFrame()
            imp = {}
        elif name == "elo":
            from modelling import elo
            preds = elo.predict_all(matches, outcomes, splits_by_season, args.include_holdout)
            expl = pd.DataFrame()
            imp = {}
        elif name == "dixon_coles":
            from modelling import dixon_coles
            preds = dixon_coles.predict_all(matches, outcomes, splits_by_season, args.include_holdout)
            expl = pd.DataFrame()
            imp = {}
        elif name == "marketnet":
            from modelling import marketnet
            preds, expl, imp = marketnet.predict_all(
                matches, ref_pre, outcomes, splits_by_season,
                args.include_holdout, feat_table=feat_table, trials_path=trials_path,
            )
        elif name == "lgbm":
            from modelling import lgbm
            preds, expl, imp = lgbm.predict_all(
                matches, ref_pre, outcomes, splits_by_season,
                args.include_holdout, feat_table=feat_table, trials_path=trials_path,
            )
        else:
            raise ValueError(name)

        preds = _fix_timestamps(preds)
        preds = _fix_nullable_floats(preds)
        all_preds.append(preds)
        if len(expl) > 0:
            all_explanations.append(expl)
        if imp:
            all_importances.append(imp)

    predictions = pd.concat(all_preds, ignore_index=True)
    out_path = ARTIFACTS_DIR / "predictions.parquet"
    write_table(predictions, out_path, "predictions")
    print(f"Wrote {len(predictions)} rows -> {out_path}")

    if all_explanations:
        explanations = pd.concat(all_explanations, ignore_index=True)
        write_table(explanations, ARTIFACTS_DIR / "prediction_explanations.parquet",
                    "prediction_explanations")
        print(f"Wrote {len(explanations)} explanation rows")

    if all_importances:
        manifest_path = (GOLDEN_JSON / "manifest.json") if args.golden else (ARTIFACTS_DIR / "manifest.json")
        dataset_mode = "SYNTHETIC"
        if manifest_path.exists():
            dataset_mode = json.loads(manifest_path.read_text()).get("dataset_mode", "SYNTHETIC")
        fi_out = {"dataset_mode": dataset_mode, "rows": all_importances}
        (ARTIFACTS_DIR / "feature_importance.json").write_text(json.dumps(fi_out, indent=2))
        print(f"Wrote feature_importance.json ({len(all_importances)} models)")


if __name__ == "__main__":
    main()
