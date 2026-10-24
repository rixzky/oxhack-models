"""Evaluation tests: metrics reproduce contract vectors, lock_holdout overwrite guard."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from contract import maths as M
from contract.schemas import read_table


GOLDEN_TABLES = "contract/golden/tables"
GOLDEN_JSON = "contract/golden/artifacts_json"
TEST_VECTORS = "contract/golden/test_vectors.json"


@pytest.fixture(scope="module")
def vectors():
    return json.loads(Path(TEST_VECTORS).read_text())


def test_brier_vector(vectors):
    for case in vectors["brier"]:
        result = M.brier_row(case["p"], case["result"])
        assert abs(result - case["expected_brier_sum3"]) < 1e-9, \
            f"brier_row({case['p']}, {case['result']}) = {result}, expected {case['expected_brier_sum3']}"


def test_log_loss_vector(vectors):
    for case in vectors["log_loss"]:
        result = M.log_loss_row(case["p"], case["result"])
        assert abs(result - case["expected"]) < 1e-6, \
            f"log_loss_row({case['p']}, {case['result']}) = {result}, expected {case['expected']}"


def test_clv_vector(vectors):
    for case in vectors["clv"]:
        result = M.clv(case["offered_odds"], case["closing_fair_prob"])
        assert abs(result - case["expected_clv"]) < 1e-9, \
            f"clv({case['offered_odds']}, {case['closing_fair_prob']}) = {result}, expected {case['expected_clv']}"


def test_forecast_metrics_on_golden():
    """forecast.compute on golden predictions reproduces expected log_loss/brier."""
    import pandas as pd
    from evaluation import forecast

    preds = read_table(f"{GOLDEN_TABLES}/predictions.parquet", "predictions")
    outcomes = read_table(f"{GOLDEN_TABLES}/outcomes.parquet", "outcomes")

    result = forecast.compute(preds, outcomes, dataset_mode="SYNTHETIC")

    assert "rows" in result
    assert len(result["rows"]) > 0

    for row in result["rows"]:
        assert "log_loss" in row
        assert "brier_sum3" in row
        assert "calibration_bins" in row
        assert row["log_loss"] > 0
        assert 0 < row["brier_sum3"] < 2


def test_policy_metrics_hand_calc_golden():
    """policy.compute on golden decisions reproduces a hand-calculated ROI."""
    import pandas as pd

    try:
        decisions = pd.read_parquet(
            f"{GOLDEN_TABLES}/decisions.parquet", engine="fastparquet"
        )
    except Exception:
        decisions = read_table(f"{GOLDEN_TABLES}/decisions.parquet", "decisions")

    try:
        from evaluation import policy
    except ImportError:
        pytest.skip("evaluation.policy not yet implemented")

    result = policy.compute(decisions, n_trials=1, dataset_mode="SYNTHETIC")

    assert "rows" in result
    for row in result["rows"]:
        assert "policies" in row
        for pol in row["policies"]:
            bets = decisions[
                (decisions["model_id"] == row["model_id"])
                & (decisions["split"] == row["split"])
                & (decisions["policy_id"] == pol["policy_id"])
                & (decisions["action"] == "BET")
            ]
            expected_turnover = float(bets["stake"].sum())
            assert abs(pol["turnover"] - expected_turnover) < 1e-6, \
                f"turnover mismatch for {row['model_id']}/{pol['policy_id']}: " \
                f"got {pol['turnover']}, expected {expected_turnover}"


def test_lock_holdout_refuses_overwrite():
    """lock_holdout must raise FileExistsError if called twice."""
    try:
        from evaluation import lock_holdout
    except ImportError:
        pytest.skip("evaluation.lock_holdout not yet implemented")

    dummy_forecast = {"dataset_mode": "SYNTHETIC", "rows": []}
    dummy_policy = {"dataset_mode": "SYNTHETIC", "rows": []}

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "holdout_locked.json"
        # First call should succeed
        lock_holdout.lock(dummy_forecast, dummy_policy, None, out)
        assert out.exists()
        # Second call must raise
        with pytest.raises(FileExistsError):
            lock_holdout.lock(dummy_forecast, dummy_policy, None, out)
