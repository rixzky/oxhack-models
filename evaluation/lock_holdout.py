"""Lock the holdout result — write once, refuse to overwrite.

Usage:
    python -m evaluation.lock_holdout [--artifacts-dir ./artifacts]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

from contract.schemas import CONTRACT_VERSION

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "./artifacts"))


def lock(
    forecast_metrics: dict,
    policy_metrics: dict,
    frozen_config: dict | None,
    output_path: Path,
) -> None:
    """Write holdout_locked.json once. Raises FileExistsError if already present."""
    if output_path.exists():
        raise FileExistsError(
            f"holdout_locked.json already exists at {output_path}. Cannot overwrite."
        )

    holdout_forecast = [r for r in forecast_metrics.get("rows", []) if r["split"] == "holdout"]
    holdout_policy = [r for r in policy_metrics.get("rows", []) if r["split"] == "holdout"]

    result = {
        "dataset_mode": forecast_metrics.get("dataset_mode", "REAL"),
        "contract_version": CONTRACT_VERSION,
        "config_hash": frozen_config.get("config_hash") if frozen_config else None,
        "locked_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "model_id": frozen_config.get("model_id") if frozen_config else None,
        "model_version": frozen_config.get("model_version") if frozen_config else None,
        "split": "holdout",
        "is_locked_holdout": True,
        "forecast": holdout_forecast,
        "policies": holdout_policy[0]["policies"] if holdout_policy else [],
        "equity_curve": holdout_policy[0]["equity_curve"] if holdout_policy else [],
        "note": "Written by evaluation.lock_holdout. Cannot be overwritten.",
    }
    output_path.write_text(json.dumps(result, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    args = parser.parse_args()

    art = Path(args.artifacts_dir)
    fm = json.loads((art / "forecast_metrics.json").read_text())
    pm = json.loads((art / "policy_metrics.json").read_text())

    frozen_path = art / "frozen_config.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else None

    out = art / "holdout_locked.json"
    lock(fm, pm, frozen, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
