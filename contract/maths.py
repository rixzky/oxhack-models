"""Reference maths for the whole project. SINGLE SOURCE OF TRUTH.

Every part imports these functions instead of re-implementing them.
If you think a formula here is wrong, tell the contract owner. Do not
patch a local copy, because the three repos must stay identical.

Conventions (see CONTRACT.md section 4):
  * Outcome order is always (H, D, A) = (home win, draw, away win).
  * Odds are decimal odds (stake included), e.g. 2.10.
  * Probabilities are floats in [0, 1] that sum to 1 within PROB_SUM_TOL.
"""
from __future__ import annotations

import math
from typing import Sequence

OUTCOMES = ("H", "D", "A")
PROB_SUM_TOL = 1e-6
EPS = 1e-15  # clip for log loss


def implied_probs(odds: Sequence[float]) -> tuple[list[float], float]:
    """Margin-free market probabilities from ONE coherent 3-way quote.

    q_k = (1/O_k) / sum_j (1/O_j);  overround = sum_j (1/O_j) - 1.
    Never mix bookmakers (e.g. best price per outcome) in one call.
    """
    if len(odds) != 3:
        raise ValueError("need exactly 3 odds (H, D, A)")
    if any((o is None) or (not math.isfinite(o)) or o <= 1.0 for o in odds):
        raise ValueError(f"invalid odds {odds}")
    inv = [1.0 / o for o in odds]
    s = sum(inv)
    return [i / s for i in inv], s - 1.0


def expected_profit(p: float, odds: float) -> float:
    """Expected profit per 1 unit staked at decimal odds `odds`: p*O - 1.

    Uses the ACTUAL offered (or adverse-adjusted) odds. Removing the
    bookmaker margin builds a probability benchmark; it never improves
    the payout you can actually get.
    """
    return p * odds - 1.0


def stressed_prob(p: float, delta: float) -> float:
    """Sensitivity assumption (NOT a confidence interval): max(0, p - delta)."""
    return max(0.0, p - delta)


def adverse_odds(odds: float, price_haircut: float) -> float:
    """Adverse-price adjustment: shrink the PROFIT part of the odds.

    O_adj = 1 + (O - 1) * (1 - h),  h in [0, 1).
    h = 0.05 means we assume we only get 95% of the quoted profit
    (slippage, worse price by the time we act).
    """
    if not 0.0 <= price_haircut < 1.0:
        raise ValueError("price_haircut must be in [0, 1)")
    return 1.0 + (odds - 1.0) * (1.0 - price_haircut)


def full_kelly(p: float, odds: float) -> float:
    """Zero-fee decimal-odds Kelly fraction: max(0, (p*O - 1) / (O - 1))."""
    if odds <= 1.0:
        return 0.0
    return max(0.0, (p * odds - 1.0) / (odds - 1.0))


def log_loss_row(p: Sequence[float], result: str) -> float:
    """Natural-log loss for one match. `result` in {"H","D","A"}."""
    k = OUTCOMES.index(result)
    return -math.log(min(max(p[k], EPS), 1.0))


def brier_row(p: Sequence[float], result: str) -> float:
    """Multiclass Brier, SUM convention: sum_k (p_k - y_k)^2, range [0, 2].

    Not divided by 3. Report this convention wherever Brier is shown.
    """
    k = OUTCOMES.index(result)
    return sum((p[j] - (1.0 if j == k else 0.0)) ** 2 for j in range(3))


def clv(offered_odds: float, closing_fair_prob: float) -> float:
    """Closing line value of a bet, PROBABILITY convention.

    CLV = O_offered * q_close - 1, where q_close is the margin-free
    probability of the SAME outcome from the reference CLOSING quote
    (same bookmaker as the offered quote where possible).
    Interpretation: expected profit IF the closing line were the truth.
    It is a benchmark, not proof of profitability.
    """
    return offered_odds * closing_fair_prob - 1.0


def check_probs(p: Sequence[float]) -> None:
    """Raise if probabilities are not finite, non-negative and summing to 1."""
    if len(p) != 3:
        raise ValueError("need 3 probabilities")
    if any((not math.isfinite(x)) or x < 0.0 for x in p):
        raise ValueError(f"bad probabilities {p}")
    if abs(sum(p) - 1.0) > PROB_SUM_TOL:
        raise ValueError(f"probabilities sum to {sum(p)}, not 1")
