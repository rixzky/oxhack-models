"""Forecast metrics: log loss, Brier, calibration bins per model x split."""
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


def compute(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    dataset_mode: str = "REAL",
) -> dict:
    """Return the forecast_metrics.json dict."""
    merged = predictions.merge(outcomes[["match_id", "result"]], on="match_id", how="inner")

    rows = []
    splits = [s for s in ["train", "calibration", "validation"] if s in merged["split"].unique()]
    models = merged["model_id"].unique()

    for split in splits:
        split_df = merged[merged["split"] == split]
        # inner-join match_ids across all models so comparisons are fair
        common_ids = None
        for mid in models:
            ids = set(split_df[split_df["model_id"] == mid]["match_id"])
            common_ids = ids if common_ids is None else common_ids & ids
        if not common_ids:
            continue
        split_df = split_df[split_df["match_id"].isin(common_ids)]

        for model_id in models:
            mdf = split_df[split_df["model_id"] == model_id]
            if mdf.empty:
                continue

            ll_vals = [
                M.log_loss_row([r.p_home, r.p_draw, r.p_away], r.result)
                for r in mdf.itertuples()
            ]
            br_vals = [
                M.brier_row([r.p_home, r.p_draw, r.p_away], r.result)
                for r in mdf.itertuples()
            ]

            # pool all 3 outcome probabilities for calibration bins
            p_all = np.concatenate([
                mdf["p_home"].values,
                mdf["p_draw"].values,
                mdf["p_away"].values,
            ])
            y_all = np.concatenate([
                (mdf["result"] == "H").astype(float).values,
                (mdf["result"] == "D").astype(float).values,
                (mdf["result"] == "A").astype(float).values,
            ])

            rows.append({
                "model_id": str(model_id),
                "split": split,
                "n": len(mdf),
                "log_loss": float(np.mean(ll_vals)),
                "brier_sum3": float(np.mean(br_vals)),
                "calibration_bins": _calibration_bins(p_all, y_all),
            })

    return {
        "dataset_mode": dataset_mode,
        "brier_convention": "sum over 3 outcomes, range 0 to 2",
        "rows": rows,
    }
