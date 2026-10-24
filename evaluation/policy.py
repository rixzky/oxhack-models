"""Policy metrics from Part 1's decisions ledger.

Computes per (model_id, split, policy_id): ROI, drawdown, CLV, Sharpe,
deflated Sharpe (Bailey & Lopez de Prado 2014), bootstrap CIs, line-shopping.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import norm as scipy_norm
from scipy.stats import skew as scipy_skew

from contract.schemas import CONTRACT_VERSION
from evaluation.bootstrap import roi_bootstrap_ci, roi_diff_bootstrap_ci

_POLICY_LABELS = {
    "naive": "EV > 0 at reference odds, no checks, flat stakes",
    "controlled": "Reference odds; threshold + stress + data checks + exposure caps",
    "controlled_best_price": "Best price across bookmakers; same checks and caps",
}
_POLICY_ORDER = ["naive", "controlled", "controlled_best_price"]


def _deflated_sharpe(sr_obs: float, returns: np.ndarray, n_trials: int) -> float | None:
    T = len(returns)
    if T < 4 or n_trials < 1:
        return None
    E = 0.5772  # Euler-Mascheroni constant
    # expected max SR under H0 over n_trials independent strategies
    sr_star = (1 - E) * scipy_norm.ppf(1 - 1 / n_trials) + E * scipy_norm.ppf(
        1 - 1 / (n_trials * math.e)
    )
    skew = float(scipy_skew(returns))
    kurt = float(scipy_kurtosis(returns, fisher=True))  # excess kurtosis
    var_sr = (1 + (kurt / 4) * sr_obs**2 - skew * sr_obs) / T
    if var_sr <= 0:
        return None
    return float(scipy_norm.cdf((sr_obs - sr_star) / math.sqrt(var_sr)))


def _policy_summary(
    g: pd.DataFrame,
    policy_id: str,
    initial_bankroll: float,
    n_trials: int,
) -> tuple[dict, pd.DataFrame]:
    """Returns (summary_dict, by_batch_df) for one policy group."""
    bets = g[g["action"] == "BET"]
    turnover = float(bets["stake"].sum())
    profit = float(bets["settled_pnl"].sum())
    roi = profit / turnover if turnover > 0 else None

    clv_vals = bets["clv"].dropna()
    mean_clv = float(clv_vals.mean()) if len(clv_vals) > 0 else None

    by_batch = (
        bets.groupby("batch_key")
        .agg(stake=("stake", "sum"), pnl=("settled_pnl", "sum"))
        .sort_index()
    )

    # Sharpe: batch PnL / initial_bankroll as returns
    if len(by_batch) >= 2:
        rets = (by_batch["pnl"] / initial_bankroll).values
        std = float(rets.std())
        if std > 0:
            sr = float(rets.mean() / std)
            sharpe = sr
            dsr = _deflated_sharpe(sr, rets, n_trials)
        else:
            sharpe = None
            dsr = None
    else:
        sharpe = None
        dsr = None

    ci_lo, ci_hi = roi_bootstrap_ci(by_batch)

    # Equity curve + max drawdown (over all decisions batched by batch_key)
    all_batches = (
        g.groupby("batch_key")
        .agg(
            pnl=("settled_pnl", lambda x: x[g.loc[x.index, "action"] == "BET"].sum()),
            time_utc=("decision_time_utc", "min"),
        )
        .sort_index()
    )
    bankroll = initial_bankroll
    peak = initial_bankroll
    max_dd = 0.0
    equity = []
    for bk, row in all_batches.iterrows():
        bankroll += float(row["pnl"])
        peak = max(peak, bankroll)
        dd = peak - bankroll
        max_dd = max(max_dd, dd)
        t = row["time_utc"]
        time_str = t.isoformat().replace("+00:00", "Z") if hasattr(t, "isoformat") else str(t)
        equity.append({
            "batch_key": bk,
            "time_utc": time_str,
            "policy_id": policy_id,
            "bankroll": round(bankroll, 4),
            "drawdown": round(dd, 4),
        })

    summary = {
        "policy_id": policy_id,
        "label": _POLICY_LABELS.get(policy_id, policy_id),
        "n_decisions": int(len(g)),
        "n_bets": int(len(bets)),
        "n_abstain": int((g["action"] == "NO_BET").sum()),
        "turnover": turnover,
        "profit": profit,
        "roi": roi,
        "max_drawdown": max_dd,
        "mean_clv": mean_clv,
        "sharpe_per_matchweek": sharpe,
        "deflated_sharpe": dsr,
        "reason_counts": g["reason_code"].value_counts().to_dict(),
        "bootstrap_ci_95": {"roi": [ci_lo, ci_hi]},
    }
    return summary, equity, by_batch


def compute(
    decisions: pd.DataFrame,
    n_trials: int = 1,
    initial_bankroll: float = 1000.0,
    dataset_mode: str = "REAL",
) -> dict:
    rows = []

    for (model_id, split), ms_group in decisions.groupby(["model_id", "split"]):
        policies_out = []
        equity_combined = []
        by_batch_per_policy: dict[str, pd.DataFrame] = {}

        for policy_id in _POLICY_ORDER:
            pg = ms_group[ms_group["policy_id"] == policy_id]
            if len(pg) == 0:
                continue
            summary, equity, by_batch = _policy_summary(
                pg, policy_id, initial_bankroll, n_trials
            )
            policies_out.append(summary)
            equity_combined.extend(equity)
            by_batch_per_policy[policy_id] = by_batch

        # Line-shopping: controlled_best_price vs controlled
        ctrl = by_batch_per_policy.get("controlled")
        cbp = by_batch_per_policy.get("controlled_best_price")
        ctrl_summary = next((p for p in policies_out if p["policy_id"] == "controlled"), None)
        cbp_summary = next(
            (p for p in policies_out if p["policy_id"] == "controlled_best_price"), None
        )

        if ctrl is not None and cbp is not None and ctrl_summary and cbp_summary:
            roi_c = ctrl_summary["roi"]
            roi_b = cbp_summary["roi"]
            roi_diff = (roi_b - roi_c) if (roi_c is not None and roi_b is not None) else None
            diff_lo, diff_hi = roi_diff_bootstrap_ci(ctrl, cbp)
            line_shopping = {
                "roi_diff": roi_diff,
                "ci_95": [diff_lo, diff_hi],
                "bets_diff": cbp_summary["n_bets"] - ctrl_summary["n_bets"],
                "turnover_diff": cbp_summary["turnover"] - ctrl_summary["turnover"],
            }
        else:
            line_shopping = {"roi_diff": None, "ci_95": [None, None], "bets_diff": 0, "turnover_diff": 0.0}

        rows.append({
            "model_id": str(model_id),
            "split": str(split),
            "is_locked_holdout": False,
            "policies": policies_out,
            "equity_curve": equity_combined,
            "line_shopping": line_shopping,
        })

    return {
        "dataset_mode": dataset_mode,
        "contract_version": CONTRACT_VERSION,
        "rows": rows,
    }
