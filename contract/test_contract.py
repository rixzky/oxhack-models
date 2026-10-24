"""Contract tests. Every repo runs these:  python -m pytest contract -q

They prove three things:
  1. the reference maths matches the shared numeric vectors,
  2. the golden tables pass the schema validators,
  3. this repo's copy of the contract is the agreed version.
Part-specific tests live in each part's own tests/ folder.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from contract import maths as M
from contract import schemas as S

G = Path(__file__).parent / "golden"
V = json.loads((G / "test_vectors.json").read_text())
TOL = V["tolerance"]


def test_contract_version_matches_vectors():
    assert V["contract_version"] == S.CONTRACT_VERSION


@pytest.mark.parametrize("case", V["normalise"])
def test_normalise(case):
    q, ov = M.implied_probs(case["odds"])
    assert q == pytest.approx(case["expected_q"], abs=TOL)
    assert ov == pytest.approx(case["expected_overround"], abs=TOL)


@pytest.mark.parametrize("case", V["ev"])
def test_ev(case):
    assert M.expected_profit(case["p"], case["odds"]) == pytest.approx(case["expected_ev"], abs=TOL)


def test_regression_edge_is_not_ev():
    # margin-free market prob 0.50 vs model 0.52 looks like +0.02 "edge" but loses money at 1.90
    assert M.expected_profit(0.52, 1.90) < 0


@pytest.mark.parametrize("case", V["adverse_price"])
def test_adverse(case):
    assert M.adverse_odds(case["odds"], case["price_haircut"]) == pytest.approx(case["expected_adjusted_odds"], abs=TOL)


@pytest.mark.parametrize("case", V["stress"])
def test_stress(case):
    assert M.stressed_prob(case["p"], case["delta"]) == pytest.approx(case["expected_p_stressed"], abs=TOL)


@pytest.mark.parametrize("case", V["kelly"])
def test_kelly(case):
    assert M.full_kelly(case["p"], case["odds"]) == pytest.approx(case["expected_full_kelly"], abs=TOL)


def test_clv_brier_logloss():
    c = V["clv"][0]
    assert M.clv(c["offered_odds"], c["closing_fair_prob"]) == pytest.approx(c["expected_clv"], abs=TOL)
    b = V["brier"][0]
    assert M.brier_row(b["p"], b["result"]) == pytest.approx(b["expected_brier_sum3"], abs=TOL)
    l = V["log_loss"][0]
    assert M.log_loss_row(l["p"], l["result"]) == pytest.approx(l["expected"], abs=TOL)


@pytest.mark.parametrize("table", list(S.SCHEMAS))
def test_golden_tables_validate(table):
    path = G / "tables" / f"{table}.parquet"
    if not path.exists():
        pytest.skip(f"no golden file for {table} (produced by Part 3 at runtime)")
    S.validate(table, pd.read_parquet(path))


def test_match_id_format():
    ko = pd.Timestamp("2023-08-12T14:00:00Z")
    assert S.make_match_id("E0", "2023-24", ko, "Arsenal", "Nott'm Forest") == "E0_2023-24_20230812_arsenal_nott-m-forest"
    assert S.feature_cutoff(ko) == pd.Timestamp("2023-08-10T00:00:00Z")
    assert S.matchweek_key("E0", ko) == "E0_2023-W32"


def test_validator_rejects_leakage():
    p = pd.read_parquet(G / "tables" / "predictions.parquet").head(3).copy()
    p["training_cutoff_utc"] = p["decision_time_utc"] + pd.Timedelta(days=1)
    with pytest.raises(S.ContractError):
        S.validate("predictions", p)


# ---------------------------------------------------------------- shared decision function
from contract.decisions import best_price_quote, decide_match  # noqa: E402


def _quote(q):
    if q is None:
        return None
    obs = q["observed_at_utc"]
    return {**q, "observed_at_utc": None if obs is None else pd.Timestamp(obs)}


@pytest.mark.parametrize("case", V["decision"])
def test_decide_match_vectors(case):
    d = decide_match(case["probs"], _quote(case["quote"]), pd.Timestamp(case["kickoff_utc"]), case["policy"])
    e = case["expected"]
    for k in ("action", "selection", "reason_code", "reason_codes_all", "best_outcome", "bookmaker"):
        assert d[k] == e[k], k
    for k in ("ev_raw", "ev_stressed", "proposed_stake"):
        if e[k] is None:
            assert d[k] is None
        else:
            assert d[k] == pytest.approx(e[k], abs=TOL)


def test_decide_match_regression_rejects_short_favourite():
    q = {"quote_id": "x", "odds": [1.90, 3.60, 4.80], "is_valid": True, "timestamp_precision": "WINDOW",
         "observed_at_utc": pd.Timestamp("2025-01-03T14:00:00Z")}
    d = decide_match([0.52, 0.28, 0.20], q, pd.Timestamp("2025-01-04T15:00:00Z"))
    assert d["action"] == "NO_BET"


@pytest.mark.parametrize("case", V["best_price"])
def test_best_price_vectors(case):
    qs = [{**q, "observed_at_utc": pd.Timestamp(q["observed_at_utc"])} for q in case["quotes"]]
    b = best_price_quote(qs, case["season"])
    if case["expected"] is None:
        assert b is None
    else:
        for k in ("odds", "bookmakers", "ref_odds", "timestamp_precision"):
            assert b[k] == case["expected"][k], k


def test_quote_at_or_after_kickoff_rejected():
    ko = pd.Timestamp("2025-01-04T15:00:00Z")
    q = {"quote_id": "x", "odds": [3.10, 3.50, 2.80], "is_valid": True, "timestamp_precision": "WINDOW",
         "observed_at_utc": ko}
    assert decide_match([0.30, 0.28, 0.42], q, ko)["reason_code"] == "STALE_QUOTE"


def test_market_benchmark_uses_reference_odds_not_best_price():
    ko = pd.Timestamp("2025-01-04T15:00:00Z")
    q = {"quote_id": "x", "odds": [3.20, 3.50, 2.90], "ref_odds": [3.10, 3.50, 2.80], "is_valid": True,
         "timestamp_precision": "WINDOW", "observed_at_utc": ko - pd.Timedelta(hours=20)}
    d = decide_match([0.30, 0.28, 0.42], q, ko)
    from contract.maths import implied_probs
    assert [r["q_market"] for r in d["per_outcome"]] == pytest.approx(implied_probs([3.10, 3.50, 2.80])[0])
