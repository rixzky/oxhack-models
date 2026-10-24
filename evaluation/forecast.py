"""Forecast metrics: log loss, Brier, calibration bins per model x split.

Also computes per-league breakdown (validation only) and bootstrap CI for
log-loss difference vs the market baseline (matchweek blocks, 2000 resamples).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from contract import maths as M


def _calibration_bins(p_flat: np.ndarray, y_flat: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_flat >= lo) & (p_flat < hi)
        n = int(mask.sum())
        bins.append({
            "bin_lo": round(float(lo), 10),
            "bin_hi": round(float(hi), 10),
            "n": n,
            "mean_pred": float(p_flat[mask].mean()) if n > 0 else 0.0,
            "obs_freq": float(y_flat[mask].mean()) if n > 0 else 0.0,
        })
    return bins


def _ll_bootstrap_ci(
    ll_model: np.ndarray,
    ll_market: np.ndarray,
    batch_keys: np.ndarray,
    n_samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float | None, float | None]:
    """Bootstrap CI for mean(ll_model) - mean(ll_market) by matchweek blocks."""
    estimate = float(np.mean(ll_model) - np.mean(ll_market))
    batches = sorted(set(batch_keys))
    if len(batches) < 2:
        return estimate, None, None
    batch_arr = np.array(batches)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_samples):
        idx = rng.integers(0, len(batch_arr), size=len(batch_arr))
        chosen = batch_arr[idx]
        mask = np.isin(batch_keys, chosen)
        if mask.sum() == 0:
            continue
        diffs.append(float(np.mean(ll_model[mask]) - np.mean(ll_market[mask])))
    if len(diffs) < 2:
        return estimate, None, None
    return estimate, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _compute_split(
    split_df: pd.DataFrame,
    models: list[str],
    split: str,
    market_ll_by_match: dict | None = None,
) -> list[dict]:
    """Compute metrics for all models in one split on their common match set."""
    common_ids = None
    for mid in models:
        ids = set(split_df[split_df["model_id"] == mid]["match_id"])
        common_ids = ids if common_ids is None else common_ids & ids
    if not common_ids:
        return []
    sub = split_df[split_df["match_id"].isin(common_ids)]

    rows = []
    for model_id in models:
        mdf = sub[sub["model_id"] == model_id].copy()
        if mdf.empty:
            continue

        ll_vals = np.array([
            M.log_loss_row([r.p_home, r.p_draw, r.p_away], r.result)
            for r in mdf.itertuples()
        ])
        br_vals = np.array([
            M.brier_row([r.p_home, r.p_draw, r.p_away], r.result)
            for r in mdf.itertuples()
        ])

        p_all = np.concatenate([mdf["p_home"].values, mdf["p_draw"].values, mdf["p_away"].values])
        y_all = np.concatenate([
            (mdf["result"] == "H").astype(float).values,
            (mdf["result"] == "D").astype(float).values,
            (mdf["result"] == "A").astype(float).values,
        ])

        row: dict = {
            "model_id": str(model_id),
            "split": split,
            "n": len(mdf),
            "log_loss": float(np.mean(ll_vals)),
            "brier_sum3": float(np.mean(br_vals)),
            "calibration_bins": _calibration_bins(p_all, y_all),
        }

        # Bootstrap ll diff vs market (validation only, non-market models)
        if split == "validation" and model_id != "market" and market_ll_by_match is not None:
            mdf_ids = mdf["match_id"].values
            market_ll = np.array([market_ll_by_match.get(mid, np.nan) for mid in mdf_ids])
            valid = ~np.isnan(market_ll)
            if valid.sum() >= 2:
                # Use feature_cutoff_utc as batch key if available, else match_id prefix
                bk = mdf["feature_cutoff_utc"].astype(str).values if "feature_cutoff_utc" in mdf.columns else mdf_ids
                est, ci_lo, ci_hi = _ll_bootstrap_ci(ll_vals[valid], market_ll[valid], bk[valid])
                row["ll_diff_vs_market"] = {"estimate": est, "ci_95": [ci_lo, ci_hi]}

        rows.append(row)
    return rows


def _per_league(
    split_df: pd.DataFrame,
    models: list[str],
) -> list[dict]:
    """Per-league breakdown for validation split."""
    rows = []
    if "league" not in split_df.columns:
        return rows
    for league in sorted(split_df["league"].unique()):
        lg_df = split_df[split_df["league"] == league]
        common_ids = None
        for mid in models:
            ids = set(lg_df[lg_df["model_id"] == mid]["match_id"])
            common_ids = ids if common_ids is None else common_ids & ids
        if not common_ids:
            continue
        sub = lg_df[lg_df["match_id"].isin(common_ids)]
        for model_id in models:
            mdf = sub[sub["model_id"] == model_id]
            if mdf.empty:
                continue
            ll_vals = [M.log_loss_row([r.p_home, r.p_draw, r.p_away], r.result) for r in mdf.itertuples()]
            br_vals = [M.brier_row([r.p_home, r.p_draw, r.p_away], r.result) for r in mdf.itertuples()]
            rows.append({
                "model_id": str(model_id),
                "split": "validation",
                "league": league,
                "n": len(mdf),
                "log_loss": float(np.mean(ll_vals)),
                "brier_sum3": float(np.mean(br_vals)),
            })
    return rows


def compute(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    dataset_mode: str = "REAL",
    matches: pd.DataFrame | None = None,
) -> dict:
    """Return the forecast_metrics.json dict.

    Pass matches (with 'league' column) to get per-league validation breakdown.
    """
    merged = predictions.merge(outcomes[["match_id", "result"]], on="match_id", how="inner")
    if matches is not None and "league" in matches.columns:
        merged = merged.merge(matches[["match_id", "league"]], on="match_id", how="left")

    splits = [s for s in ["train", "calibration", "validation"] if s in merged["split"].unique()]
    models = sorted(merged["model_id"].unique())

    # Pre-compute market log-loss per match for bootstrap diff
    market_ll_by_match: dict[str, float] | None = None
    if "market" in models:
        mkt = merged[merged["model_id"] == "market"]
        market_ll_by_match = {
            r.match_id: M.log_loss_row([r.p_home, r.p_draw, r.p_away], r.result)
            for r in mkt.itertuples()
        }

    rows = []
    by_league_rows = []

    for split in splits:
        split_df = merged[merged["split"] == split]
        split_rows = _compute_split(split_df, models, split, market_ll_by_match if split == "validation" else None)
        rows.extend(split_rows)

        if split == "validation" and "league" in merged.columns:
            by_league_rows.extend(_per_league(split_df, models))

    result: dict = {
        "dataset_mode": dataset_mode,
        "brier_convention": "sum over 3 outcomes, range 0 to 2",
        "rows": rows,
    }
    if by_league_rows:
        result["by_league"] = by_league_rows

    return result
