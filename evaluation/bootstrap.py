"""Bootstrap utilities for matchweek-block resampling."""
from __future__ import annotations

import numpy as np
import pandas as pd


def roi_bootstrap_ci(
    by_batch_df: pd.DataFrame,
    n_samples: int = 2000,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """Bootstrap 95% CI for ROI by resampling matchweek blocks.

    by_batch_df must have columns 'stake' and 'pnl', indexed by batch key.
    Returns (p2.5, p97.5) or (None, None) if fewer than 2 batches.
    """
    if len(by_batch_df) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    batches = by_batch_df[["stake", "pnl"]].values
    n = len(batches)
    rois = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        s = batches[idx, 0].sum()
        p = batches[idx, 1].sum()
        if s > 0:
            rois.append(p / s)
    if len(rois) < 2:
        return None, None
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


def roi_diff_bootstrap_ci(
    batches_controlled: pd.DataFrame,
    batches_bestprice: pd.DataFrame,
    n_samples: int = 2000,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """Bootstrap 95% CI for (best_price_roi - controlled_roi).

    Resamples the same matchweek indices from both policies so the comparison
    is paired. Both DataFrames must be indexed by batch_key with columns
    'stake' and 'pnl'. Uses the intersection of batch keys.
    """
    common = batches_controlled.index.intersection(batches_bestprice.index)
    if len(common) < 2:
        return None, None
    c = batches_controlled.loc[common, ["stake", "pnl"]].values
    b = batches_bestprice.loc[common, ["stake", "pnl"]].values
    rng = np.random.default_rng(seed)
    n = len(common)
    diffs = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        sc, pc = c[idx, 0].sum(), c[idx, 1].sum()
        sb, pb = b[idx, 0].sum(), b[idx, 1].sum()
        roi_c = pc / sc if sc > 0 else None
        roi_b = pb / sb if sb > 0 else None
        if roi_c is not None and roi_b is not None:
            diffs.append(roi_b - roi_c)
    if len(diffs) < 2:
        return None, None
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
