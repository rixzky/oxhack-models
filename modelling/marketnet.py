"""MarketNet: learned log-odds correction over the market.

p = softmax(log_q + delta),  delta = W x
W has shape (2, n_features): rows for H and A.  Draw logit is pinned to zero,
so untrained W=0 reproduces the market exactly.

L2 penalty lambda * ||W||^2 tuned on calibration split.
Walk-forward by season: for season S, fit on all warmup+train seasons before S.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from modelling.features import FEATURE_COLS

MODEL_ID = "marketnet"
MODEL_VERSION = "mn-1.1"

LAMBDA_CANDIDATES = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
HL_CANDIDATES = [90, 180, 365, 730]   # half-life in days for sample weighting
MAX_TRAIN_SEASONS = 8
LR = 5e-3
MAX_EPOCHS = 2000
PATIENCE = 50


# ---------------------------------------------------------------- model
class _Net(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros(2, n_features))

    def forward(self, log_q: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        delta_ha = X @ self.W.T          # (batch, 2)
        delta = torch.zeros(X.shape[0], 3)
        delta[:, 0] = delta_ha[:, 0]    # H
        delta[:, 2] = delta_ha[:, 1]    # A  (D stays 0)
        return F.softmax(log_q + delta, dim=1)


def _sample_weights(feat_df: pd.DataFrame, training_cutoff: pd.Timestamp, half_life: float) -> np.ndarray:
    days = (training_cutoff - feat_df["kickoff_utc"]).dt.total_seconds() / 86400
    w = np.exp(-np.log(2) * days.clip(lower=0).values / half_life).astype(np.float32)
    return w / w.mean()  # keep mean=1 so NLL scale is comparable across half-lives


def _fit(
    log_q: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,         # integer labels 0=H,1=D,2=A
    lam: float,
    weights: np.ndarray | None = None,
    seed: int = 0,
) -> _Net:
    torch.manual_seed(seed)
    n_features = X.shape[1]
    net = _Net(n_features)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    lq = torch.tensor(log_q, dtype=torch.float32)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    wt = torch.tensor(weights, dtype=torch.float32) if weights is not None else torch.ones(len(y))

    best_loss, best_state, patience = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        opt.zero_grad()
        p = net(lq, Xt)
        nll = (F.nll_loss(torch.log(p.clamp(1e-15)), yt, reduction="none") * wt).mean()
        l2 = lam * (net.W ** 2).sum()
        loss = nll + l2
        loss.backward()
        opt.step()
        v = loss.item()
        if v < best_loss - 1e-6:
            best_loss, best_state, patience = v, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_state:
        net.load_state_dict(best_state)
    return net


def _prep(feat_df: pd.DataFrame, outcomes: pd.DataFrame | None = None):
    """Return (log_q, X, y_or_None) arrays, imputing any NaN with column median."""
    X = feat_df[FEATURE_COLS].copy()
    X = X.fillna(X.median())
    log_q = np.column_stack([
        np.log(feat_df["q_home"].clip(1e-6)),
        np.log(feat_df["q_draw"].clip(1e-6)),
        np.log(feat_df["q_away"].clip(1e-6)),
    ])
    X_arr = X.values.astype(np.float32)

    # normalise: fit scaler on this data
    mu = X_arr.mean(axis=0)
    sd = X_arr.std(axis=0) + 1e-8
    X_norm = (X_arr - mu) / sd

    y = None
    if outcomes is not None:
        merged = feat_df[["match_id"]].merge(outcomes[["match_id", "result"]], on="match_id")
        label_map = {"H": 0, "D": 1, "A": 2}
        y = merged["result"].map(label_map).values.astype(np.int64)

    return log_q.astype(np.float32), X_norm, y, mu, sd


def _predict(net: _Net, log_q: np.ndarray, X_norm: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        p = net(
            torch.tensor(log_q, dtype=torch.float32),
            torch.tensor(X_norm, dtype=torch.float32),
        ).numpy()
    return p  # (n, 3)  order H D A


# ---------------------------------------------------------------- public interface
def predict_all(
    matches: pd.DataFrame,
    ref_pre: pd.DataFrame,
    outcomes: pd.DataFrame,
    splits_by_season: dict[str, str],
    include_holdout: bool = False,
    feat_table: pd.DataFrame | None = None,
    trials_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (predictions_df, explanations_df, feature_importance_dict)."""
    from modelling.features import build as build_features, FEATURE_COLS

    matches = matches.copy()
    matches["split"] = matches["season"].map(splits_by_season)

    if feat_table is None:
        feat_table = build_features(matches, ref_pre, outcomes)

    feat_table = feat_table.copy()
    feat_table["split"] = feat_table["season"].map(splits_by_season)

    seasons = sorted(feat_table["season"].unique())

    all_preds: list[dict] = []
    all_explanations: list[dict] = []

    best_lam = LAMBDA_CANDIDATES[2]   # default; updated by calibration tuning
    best_hl = HL_CANDIDATES[-1]        # default: least decay
    lam_tuned = False
    best_net = None
    best_mu = best_sd = None

    for i, season in enumerate(seasons):
        season_feat = feat_table[feat_table["season"] == season]
        season_split = season_feat["split"].iloc[0]

        if season_split == "holdout" and not include_holdout:
            continue

        train_seasons = [
            s for s in seasons[:i]
            if splits_by_season.get(s) in ("warmup", "train")
        ][-MAX_TRAIN_SEASONS:]

        # --- joint lambda + half-life tuning (once, using train → calibration) ---
        if season_split == "calibration" and not lam_tuned:
            cal_feat = season_feat
            train_feat = feat_table[feat_table["season"].isin(train_seasons)]
            if len(train_feat) >= 20:
                tc = matches[matches["season"].isin(train_seasons)]["kickoff_utc"].max()
                log_q_tr, X_tr, y_tr, mu, sd = _prep(train_feat, outcomes)
                X_cal = (feat_table[feat_table["season"].isin([season])][FEATURE_COLS]
                         .fillna(feat_table[feat_table["season"].isin([season])][FEATURE_COLS].median())
                         .values.astype(np.float32))
                X_cal = (X_cal - mu) / (sd + 1e-8)
                log_q_cal = np.column_stack([
                    np.log(cal_feat["q_home"].clip(1e-6).values),
                    np.log(cal_feat["q_draw"].clip(1e-6).values),
                    np.log(cal_feat["q_away"].clip(1e-6).values),
                ]).astype(np.float32)
                cal_outcomes = cal_feat.merge(outcomes[["match_id", "result"]], on="match_id")
                y_cal = cal_outcomes["result"].map({"H": 0, "D": 1, "A": 2}).values

                best_ll, best_lam, best_hl = np.inf, LAMBDA_CANDIDATES[2], HL_CANDIDATES[-1]
                for hl in HL_CANDIDATES:
                    w = _sample_weights(train_feat, tc, hl)
                    for lam in LAMBDA_CANDIDATES:
                        net = _fit(log_q_tr, X_tr, y_tr, lam, weights=w)
                        p_cal = _predict(net, log_q_cal, X_cal)
                        ll = -np.log(p_cal[np.arange(len(y_cal)), y_cal].clip(1e-15)).mean()
                        if ll < best_ll:
                            best_ll, best_lam, best_hl = ll, lam, hl
                if trials_path:
                    _log_trial(trials_path, best_lam, best_hl, best_ll)
            lam_tuned = True

        # --- fit on all prior training seasons ---
        if len(train_seasons) == 0:
            for _, row in season_feat.iterrows():
                all_preds.append(_stub_row(row))
            continue

        train_feat = feat_table[feat_table["season"].isin(train_seasons)]
        if len(train_feat) < 20:
            for _, row in season_feat.iterrows():
                all_preds.append(_stub_row(row))
            continue

        training_cutoff = matches[matches["season"].isin(train_seasons)]["kickoff_utc"].max()
        log_q_tr, X_tr, y_tr, mu, sd = _prep(train_feat, outcomes)
        w = _sample_weights(train_feat, training_cutoff, best_hl)
        net = _fit(log_q_tr, X_tr, y_tr, best_lam, weights=w)
        best_net, best_mu, best_sd = net, mu, sd

        # --- predict this season ---
        X_raw = (season_feat[FEATURE_COLS]
                 .fillna(season_feat[FEATURE_COLS].median())
                 .values.astype(np.float32))
        X_norm = (X_raw - mu) / (sd + 1e-8)
        log_q_pred = np.column_stack([
            np.log(season_feat["q_home"].clip(1e-6).values),
            np.log(season_feat["q_draw"].clip(1e-6).values),
            np.log(season_feat["q_away"].clip(1e-6).values),
        ]).astype(np.float32)

        probs = _predict(net, log_q_pred, X_norm)

        W = net.W.detach().numpy()   # (2, n_features)

        for j, (_, row) in enumerate(season_feat.iterrows()):
            tc = min(training_cutoff, row["feature_cutoff_utc"])  # training_cutoff set above
            all_preds.append({
                "match_id": row["match_id"],
                "model_id": MODEL_ID,
                "model_version": MODEL_VERSION,
                "split": row["split"],
                "decision_time_utc": row["feature_cutoff_utc"],
                "feature_cutoff_utc": row["feature_cutoff_utc"],
                "training_cutoff_utc": tc,
                "p_home": float(probs[j, 0]),
                "p_draw": float(probs[j, 1]),
                "p_away": float(probs[j, 2]),
                "xg_home": pd.NA,
                "xg_away": pd.NA,
            })

            # explanations (validation + holdout only): W_k * x_k per feature
            if season_split in ("validation", "holdout"):
                x = X_norm[j]
                for outcome_idx, outcome_label in [(0, "H"), (2, "A")]:
                    w_row = W[0] if outcome_idx == 0 else W[1]
                    contribs = w_row * x   # (n_features,)
                    top3_idx = np.argsort(np.abs(contribs))[::-1][:3]
                    for rank, fi in enumerate(top3_idx):
                        all_explanations.append({
                            "match_id": row["match_id"],
                            "model_id": MODEL_ID,
                            "model_version": MODEL_VERSION,
                            "outcome": outcome_label,
                            "rank": rank + 1,
                            "feature": FEATURE_COLS[fi],
                            "feature_value": float(x[fi]),
                            "contribution": float(contribs[fi]),
                        })

    # --- global feature importance (mean abs adjustment) ---
    importance: dict = {"model_id": MODEL_ID, "model_version": MODEL_VERSION,
                        "method": "mean_abs_adjustment_logit", "split": "validation",
                        "features": []}
    if best_net is not None:
        W = best_net.W.detach().numpy()
        mean_abs = (np.abs(W[0]) + np.abs(W[1])) / 2
        order = np.argsort(mean_abs)[::-1]
        importance["features"] = [
            {"feature": FEATURE_COLS[fi], "importance": float(mean_abs[fi])}
            for fi in order
        ]

    return (
        pd.DataFrame(all_preds).reset_index(drop=True),
        pd.DataFrame(all_explanations).reset_index(drop=True),
        importance,
    )


def _stub_row(row: pd.Series) -> dict:
    return {
        "match_id": row["match_id"],
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "split": row["split"],
        "decision_time_utc": row["feature_cutoff_utc"],
        "feature_cutoff_utc": row["feature_cutoff_utc"],
        "training_cutoff_utc": row["feature_cutoff_utc"],
        "p_home": row["q_home"],
        "p_draw": row["q_draw"],
        "p_away": row["q_away"],
        "xg_home": pd.NA,
        "xg_away": pd.NA,
    }


def _log_trial(path: Path, lam: float, hl: float, val_ll: float) -> None:
    import datetime, math
    ll = None if not math.isfinite(val_ll) else round(float(val_ll), 6)
    entry = {"time": datetime.datetime.utcnow().isoformat() + "Z",
             "part": 2, "model_id": MODEL_ID, "model_version": MODEL_VERSION,
             "what_changed": f"lambda={float(lam)} half_life={float(hl)}d",
             "val_log_loss": ll}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
