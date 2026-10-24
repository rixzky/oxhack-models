"""Entry point: python -m evaluation.run [--golden] [--split validation]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from contract.schemas import read_table, validate
from evaluation import forecast, policy

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "./artifacts"))
GOLDEN_TABLES = Path("contract/golden/tables")
GOLDEN_JSON = Path("contract/golden/artifacts_json")


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read parquet tolerating PyArrow/fastparquet version differences."""
    try:
        return pd.read_parquet(path)
    except OSError:
        return pd.read_parquet(path, engine="fastparquet")


def _count_trials(artifacts_dir: Path) -> int:
    p = artifacts_dir / "trials_log.jsonl"
    if not p.exists():
        return 1
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    return max(len(lines), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--split", default=None,
                        help="restrict forecast evaluation to one split (e.g. validation)")
    args = parser.parse_args()

    src = GOLDEN_TABLES if args.golden else ARTIFACTS_DIR
    json_src = GOLDEN_JSON if args.golden else ARTIFACTS_DIR

    predictions = validate("predictions", _read_parquet(src / "predictions.parquet"))
    outcomes = validate("outcomes", _read_parquet(src / "outcomes.parquet"))

    # matches for league breakdown (best-effort; not present in all golden sets)
    matches = None
    matches_path = src / "matches.parquet"
    if matches_path.exists():
        try:
            matches = _read_parquet(matches_path)
        except Exception:
            pass

    if args.split:
        predictions = predictions[predictions["split"] == args.split]

    manifest_path = json_src / "manifest.json"
    dataset_mode = "SYNTHETIC" if args.golden else "REAL"
    if manifest_path.exists():
        dataset_mode = json.loads(manifest_path.read_text()).get("dataset_mode", dataset_mode)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- forecast metrics ---
    fm = forecast.compute(predictions, outcomes, dataset_mode=dataset_mode, matches=matches)
    fm_out = ARTIFACTS_DIR / "forecast_metrics.json"
    fm_out.write_text(json.dumps(fm, indent=2))
    print(f"Wrote {len(fm['rows'])} rows -> {fm_out}")

    # --- policy metrics ---
    dec_path = (GOLDEN_TABLES / "decisions.parquet") if args.golden else (ARTIFACTS_DIR / "decisions.parquet")
    if not dec_path.exists():
        print("No decisions.parquet found — skipping policy metrics.")
        return

    decisions = _read_parquet(dec_path)
    n_trials = _count_trials(ARTIFACTS_DIR)
    initial_bankroll = 1000.0

    pm = policy.compute(decisions, n_trials=n_trials, initial_bankroll=initial_bankroll,
                        dataset_mode=dataset_mode)
    pm_out = ARTIFACTS_DIR / "policy_metrics.json"
    pm_out.write_text(json.dumps(pm, indent=2))
    print(f"Wrote {len(pm['rows'])} rows -> {pm_out}")


if __name__ == "__main__":
    main()
