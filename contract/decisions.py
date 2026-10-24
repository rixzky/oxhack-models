"""Single-match decision function. SINGLE SOURCE OF TRUTH for CONTRACT section 6.1-6.2.

Used by:
  * Part 1 (decisions/): the backtest calls this per match, then applies batching,
    bankroll and exposure caps (CONTRACT 6.3) on top.
  * Part 3 (app/): the replay endpoint calls this when the user moves the stress /
    haircut / min-EV sliders. The UI never computes decisions itself.

Pure function, no I/O. Tested against contract/golden/test_vectors.json.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd

from . import maths as M

DEFAULT_POLICY: dict[str, Any] = {
    "min_ev": 0.02, "prob_stress_delta": 0.02, "price_haircut": 0.0,
    "staking_mode": "FLAT", "flat_stake_frac": 0.01, "kelly_multiplier": 0.25,
    "max_stake_frac_per_match": 0.02, "max_open_exposure_frac": 0.20,
    "initial_bankroll": 1000.0, "max_quote_age_hours": 120,
    "allow_precisions": ["EXACT", "DATE", "WINDOW"], "league_gate": False,
}

# The "naive" comparison policy (CONTRACT 7): EV > 0 at raw offered odds, no checks.
NAIVE_POLICY: dict[str, Any] = {
    **DEFAULT_POLICY, "min_ev": 1e-12, "prob_stress_delta": 0.0, "price_haircut": 0.0,
    "allow_precisions": ["EXACT", "DATE", "WINDOW", "UNKNOWN"], "max_quote_age_hours": 24 * 365 * 100,
    "league_gate": False, "max_stake_frac_per_match": 1.0, "max_open_exposure_frac": 1.0,
}

NAMES = {"H": "Home", "D": "Draw", "A": "Away"}


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def decide_match(
    probs: Optional[Sequence[float]],
    quote: Optional[dict],
    kickoff_utc: pd.Timestamp,
    policy: Optional[dict] = None,
    league_eligible: bool = True,
    available_bankroll: Optional[float] = None,
) -> dict:
    """Decide one match. Does NOT apply batch/total-exposure caps (Part 1 does that).

    probs: model probabilities [H, D, A] or None if no prediction.
    quote: reference PRE_CLOSING quote as a dict with keys
           quote_id, odds ([H, D, A]), is_valid, timestamp_precision, observed_at_utc
           (pd.Timestamp or None); or None if no quote.
    Returns a dict with action, selection, reason_code, reason_codes_all, per_outcome,
    the chosen outcome's numbers, proposed_stake and a deterministic explanation.
    """
    pol = {**DEFAULT_POLICY, **(policy or {})}
    bank = pol["initial_bankroll"] if available_bankroll is None else available_bankroll
    fails: list[str] = []

    if probs is None:
        fails.append("MISSING_PREDICTION")
    else:
        M.check_probs(probs)
    if quote is None:
        fails.append("MISSING_QUOTE")
    elif not quote.get("is_valid", False):
        fails.append("INVALID_PRICE")
    else:
        prec = quote.get("timestamp_precision")
        obs = quote.get("observed_at_utc")
        if prec not in pol["allow_precisions"]:
            fails.append("STALE_QUOTE")
        elif obs is None or pd.isna(obs):
            if "UNKNOWN" not in pol["allow_precisions"]:
                fails.append("STALE_QUOTE")
        elif (kickoff_utc - obs) > pd.Timedelta(hours=pol["max_quote_age_hours"]):
            fails.append("STALE_QUOTE")
    if pol["league_gate"] and not league_eligible:
        fails.append("LEAGUE_NOT_ELIGIBLE")

    per_outcome: list[dict] = []
    best = None
    can_price = probs is not None and quote is not None and quote.get("is_valid", False)
    if can_price:
        odds = list(quote["odds"])
        q_mkt = M.implied_probs(odds)[0]
        for k, lab in enumerate(M.OUTCOMES):
            oa = M.adverse_odds(odds[k], pol["price_haircut"])
            ps = M.stressed_prob(probs[k], pol["prob_stress_delta"])
            row = {"outcome": lab, "p_model": probs[k], "p_stressed": ps, "q_market": q_mkt[k],
                   "odds_offered": odds[k], "odds_adjusted": oa,
                   "ev_raw": M.expected_profit(probs[k], oa), "ev_stressed": M.expected_profit(ps, oa)}
            per_outcome.append(row)
        best = max(per_outcome, key=lambda r: r["ev_raw"])  # one position per match max
        if best["ev_raw"] < pol["min_ev"]:
            fails.append("EV_BELOW_THRESHOLD")
        elif best["ev_stressed"] < pol["min_ev"]:
            fails.append("FAILS_STRESS")

    stake = 0.0
    if not fails and best is not None:
        if pol["staking_mode"] == "FLAT":
            stake = pol["flat_stake_frac"] * pol["initial_bankroll"]
        else:
            stake = bank * pol["kelly_multiplier"] * M.full_kelly(best["p_stressed"], best["odds_adjusted"])
        cap = pol["max_stake_frac_per_match"] * bank
        if cap <= 0:
            fails.append("EXPOSURE_LIMIT_MATCH")
            stake = 0.0
        else:
            stake = min(stake, cap)
        if stake > bank:
            fails.append("INSUFFICIENT_BANKROLL")
            stake = 0.0

    action = "NO_BET" if fails else "BET"
    code = fails[0] if fails else "ACCEPTED"
    out = {
        "action": action,
        "selection": best["outcome"] if action == "BET" else None,
        "best_outcome": best["outcome"] if best else None,
        "reason_code": code,
        "reason_codes_all": fails or ["ACCEPTED"],
        "per_outcome": per_outcome,
        "quote_id": quote.get("quote_id") if quote else None,
        "p_model": best["p_model"] if best else None,
        "p_stressed": best["p_stressed"] if best else None,
        "odds_offered": best["odds_offered"] if best else None,
        "odds_adjusted": best["odds_adjusted"] if best else None,
        "ev_raw": best["ev_raw"] if best else None,
        "ev_stressed": best["ev_stressed"] if best else None,
        "proposed_stake": round(stake, 6),
        "policy": pol,
    }
    out["explanation"] = explain(out)
    return out


def explain(d: dict) -> str:
    """Deterministic plain-English explanation built only from the decision record."""
    pol = d["policy"]
    thr = _pct(pol["min_ev"])
    code = d["reason_code"]
    b = d["best_outcome"]
    if code == "ACCEPTED":
        return (f"Bet {NAMES[b]} at {d['odds_offered']:.2f}: expected profit {_pct(d['ev_raw'])} per unit, "
                f"still {_pct(d['ev_stressed'])} after assuming our probability is "
                f"{pol['prob_stress_delta'] * 100:.1f}pp too high, above the {thr} threshold. "
                f"Stake {d['proposed_stake']:.2f} units.")
    if code == "EV_BELOW_THRESHOLD":
        return (f"No bet: the best option, {NAMES[b]} at {d['odds_offered']:.2f}, has expected profit "
                f"{_pct(d['ev_raw'])} per unit at the offered price, below the {thr} threshold.")
    if code == "FAILS_STRESS":
        return (f"No bet: {NAMES[b]} looks like {_pct(d['ev_raw'])} at {d['odds_offered']:.2f}, but if our "
                f"probability is {pol['prob_stress_delta'] * 100:.1f}pp too high it falls to "
                f"{_pct(d['ev_stressed'])}, below the {thr} threshold.")
    messages = {
        "MISSING_PREDICTION": "No bet: there is no model prediction for this match.",
        "MISSING_QUOTE": "No bet: there is no pre-closing quote from the reference bookmaker.",
        "INVALID_PRICE": "No bet: the quoted odds failed validation (missing or not above 1.00).",
        "STALE_QUOTE": "No bet: the quote's observation time is unknown or too old to trust.",
        "LEAGUE_NOT_ELIGIBLE": "No bet: past seasons show this league's market is too efficient to target.",
        "EXPOSURE_LIMIT_MATCH": "No bet: the per-match exposure limit allows no stake.",
        "EXPOSURE_LIMIT_TOTAL": "No bet: the total open exposure limit is already used.",
        "INSUFFICIENT_BANKROLL": "No bet: not enough settled bankroll is available.",
    }
    return messages.get(code, f"No bet: {code}.")
