"""Entry point: python -m evaluation.run [--golden] [--split validation]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from contract.schemas import read_table
from evaluation import forecast

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "./artifacts"))
GOLDEN_TABLES = Path("contract/golden/tables")
GOLDEN_JSON = Path("contract/golden/artifacts_json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--split", default=None,
                        help="restrict evaluation to one split (e.g. validation)")
    args = parser.parse_args()

    src = GOLDEN_TABLES if args.golden else ARTIFACTS_DIR

    predictions = read_table(src / "predictions.parquet", "predictions")
    outcomes = read_table(src / "outcomes.parquet", "outcomes")

    if args.split:
        predictions = predictions[predictions["split"] == args.split]

    manifest_path = (GOLDEN_JSON / "manifest.json") if args.golden else (ARTIFACTS_DIR / "manifest.json")
    dataset_mode = "SYNTHETIC"
    if manifest_path.exists():
        dataset_mode = json.loads(manifest_path.read_text()).get("dataset_mode", "SYNTHETIC")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = forecast.compute(predictions, outcomes, dataset_mode=dataset_mode)
    out = ARTIFACTS_DIR / "forecast_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {len(metrics['rows'])} rows -> {out}")


if __name__ == "__main__":
    main()
