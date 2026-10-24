"""Single-match decision function and best-price builder. SINGLE SOURCE OF TRUTH for
CONTRACT sections 6.1, 6.2 and 6.4.

Used by:
  * Part 1 (decisions/): the backtest builds the quote (reference or best price) and calls
    `decide_match` per match, then applies batching, bankroll and exposure caps (6.3).
  * Part 3 (app/): the replay endpoint calls the same functions when the user moves the
    stress / haircut / min-EV sliders or switches price source. The UI never computes
    decisions itself.

Pure functions, no I/O. Tested against contract/golden/test_vectors.json.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from . import maths as M
from .schemas import AGGREGATE_BOOKMAKERS, EXECUTION_EXCLUDE, PRECISIONS, REFERENCE_BOOKMAKER

DEFAULT_POLICY: dict[str, Any] = {
    "min_ev": 0.02, "prob_stress_delta": 0.02, "price_haircut": 0.0,
    "staking_mode": "FLAT", "flat_stake_frac": 0.01, "kelly_multiplier": 0.25,
    "max_stake_frac_per_match": 0.02, "max_open_exposure_frac": 0.20,
    "initial_bankroll": 1000.0, "max_quote_age_hours": 120,
    "allow_precisions": ["EXACT", "DATE", "WINDOW"], "league_gate": False,
    "price_source": "REFERENCE",       # REFERENCE (Bet365 only) or BEST_AVAILABLE (line shopping)
    "best_price_max_prob_gap": 0.10,   # outlier guard for BEST_AVAILABLE (see best_price_quote)
}

# The "naive" comparison policy (CONTRACT 7): EV > 0 at raw Bet365 odds, no checks.
NAIVE_POLICY: dict[str, Any] = {
    **DEFAULT_POLICY, "min_ev": 1e-12, "prob_stress_delta": 0.0, "price_haircut": 0.0,
    "allow_precisions": ["EXACT", "DATE", "WINDOW", "UNKNOWN"], "max_quote_age_hours": 24 * 365 * 100,
    "league_gate": False, "max_stake_frac_per_match": 1.0, "max_open_exposure_frac": 1.0,
}

# Same checks as DEFAULT_POLICY, but executes at the best valid price across bookmakers.
BEST_PRICE_POLICY: dict[str, Any] = {**DEFAULT_POLICY, "price_source": "BEST_AVAILABLE"}

POLICIES = {"naive": NAIVE_POLICY, "controlled": DEFAULT_POLICY, "controlled_best_price": BEST_PRICE_POLICY}

NAMES = {"H": "Home", "D": "Draw", "A": "Away"}


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def best_price_quote(
    quotes: Iterable[dict],
    season: str,
    max_prob_gap: float = DEFAULT_POLICY["best_price_max_prob_gap"],
) -> Optional[dict]:
    """Build the BEST_AVAILABLE executable quote for one match from its PRE_CLOSING quotes.

    quotes: dicts for ONE match, ONE snapshot (PRE_CLOSING), each with keys
            quote_id, bookmaker, odds [H, D, A], is_valid, timestamp_precision,
            observed_at_utc, and optional is_outlier (bool, from Part 1's audit flags).
    Rules (CONTRACT 6.1):
      * candidates: valid, not flagged outliers, individual bookmakers only (no MAX/AVG),
        not excluded for the season (EXECUTION_EXCLUDE);
      * the Bet365 reference quote must be valid (it is the probability benchmark);
      * per outcome, drop a bookmaker's price if its own margin-free probability differs
        from the Bet365 margin-free probability by more than `max_prob_gap` (stale guard);
      * per outcome take the highest remaining odds; ties go to Bet365, then alphabetical;
      * timing is the most conservative of the chosen components: worst precision and
        latest observed time.
    Returns None if there is no valid Bet365 reference quote.
    """
    qs = [q for q in quotes if q.get("is_valid", False)]
    ref = next((q for q in qs if q["bookmaker"] == REFERENCE_BOOKMAKER), None)
    if ref is None:
        return None
    q_ref = M.implied_probs(ref["odds"])[0]
    excluded = set(AGGREGATE_BOOKMAKERS) | set(EXECUTION_EXCLUDE.get(season, []))
    cands = [q for q in qs if q["bookmaker"] not in excluded and not q.get("is_outlier", False)]
    if ref not in cands:
        cands.append(ref)
    chosen = []
    for k in range(3):
        opts = []
        for q in cands:
            qk = M.implied_probs(q["odds"])[0][k]
            if q is ref or abs(qk - q_ref[k]) <= max_prob_gap:
                opts.append(q)
        opts.sort(key=lambda q: (-q["odds"][k], q["bookmaker"] != REFERENCE_BOOKMAKER, q["bookmaker"]))
        chosen.append(opts[0])
    worst_prec = max((c["timestamp_precision"] for c in chosen), key=PRECISIONS.index)
    obs_list = [c.get("observed_at_utc") for c in chosen]
    obs = None if any(o is None or pd.isna(o) for o in obs_list) else max(obs_list)
    return {
        "quote_id": "BEST:" + "|".join(c["quote_id"] for c in chosen),
        "odds": [chosen[k]["odds"][k] for k in range(3)],
        "bookmakers": [chosen[k]["bookmaker"] for k in range(3)],
        "ref_odds": list(ref["odds"]),
        "is_valid": True,
        "timestamp_precision": worst_prec,
        "observed_at_utc": obs,
    }


def reference_quote(quotes: Iterable[dict]) -> Optional[dict]:
    """The Bet365 PRE_CLOSING quote for one match (valid or not), or None."""
    ref = next((q for q in quotes if q["bookmaker"] == REFERENCE_BOOKMAKER), None)
    if ref is None:
        return None
    return {**ref, "bookmakers": [REFERENCE_BOOKMAKER] * 3, "ref_odds": list(ref["odds"])}


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
    quote: the executable PRE_CLOSING quote: from `reference_quote` (price_source
           REFERENCE) or `best_price_quote` (BEST_AVAILABLE), or a plain dict with keys
           quote_id, odds, is_valid, timestamp_precision, observed_at_utc, and optional
           bookmakers [3] and ref_odds [3] (Bet365 odds, used for the market benchmark).
           None if no quote.
    Returns action, selection, reason_code, reason_codes_all, per_outcome, the chosen
    outcome's numbers (incl. bookmaker), proposed_stake and a deterministic explanation.
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
        has_obs = obs is not None and not pd.isna(obs)
        if prec not in pol["allow_precisions"]:
            fails.append("STALE_QUOTE")
        elif not has_obs:
            if "UNKNOWN" not in pol["allow_precisions"]:
                fails.append("STALE_QUOTE")
        elif obs >= kickoff_utc:
            # observed_at_utc stores the LATEST plausible capture time (CONTRACT 4.1);
            # a price that may have been captured at or after kickoff is not usable.
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
        books = list(quote.get("bookmakers") or [None, None, None])
        q_mkt = M.implied_probs(list(quote.get("ref_odds") or odds))[0]
        for k, lab in enumerate(M.OUTCOMES):
            oa = M.adverse_odds(odds[k], pol["price_haircut"])
            ps = M.stressed_prob(probs[k], pol["prob_stress_delta"])
            per_outcome.append({
                "outcome": lab, "p_model": probs[k], "p_stressed": ps, "q_market": q_mkt[k],
                "odds_offered": odds[k], "odds_adjusted": oa, "bookmaker": books[k],
                "ev_raw": M.expected_profit(probs[k], oa), "ev_stressed": M.expected_profit(ps, oa)})
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
        "bookmaker": best["bookmaker"] if (best and action == "BET") else None,
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
    at = ""
    if d.get("bookmaker"):
        at = f" with {d['bookmaker']}"
    if code == "ACCEPTED":
        return (f"Bet {NAMES[b]} at {d['odds_offered']:.2f}{at}: expected profit {_pct(d['ev_raw'])} per unit, "
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
        "MISSING_QUOTE": "No bet: there is no pre-closing Bet365 quote for this match.",
        "INVALID_PRICE": "No bet: the quoted odds failed validation (missing or not above 1.00).",
        "STALE_QUOTE": ("No bet: the quote's capture time is unknown, too old, or not safely "
                        "before kickoff."),
        "LEAGUE_NOT_ELIGIBLE": "No bet: past seasons show this league's market is too efficient to target.",
        "EXPOSURE_LIMIT_MATCH": "No bet: the per-match exposure limit allows no stake.",
        "EXPOSURE_LIMIT_TOTAL": "No bet: the total open exposure limit is already used.",
        "INSUFFICIENT_BANKROLL": "No bet: not enough settled bankroll is available.",
    }
    return messages.get(code, f"No bet: {code}.")
