"""LightGBM multiclass model with SHAP explanations.

Walk-forward by season; early stopping on calibration split.
Conservative settings: shallow trees, strong regularisation.
Produces per-match top-3 SHAP contributions for the replay screen.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from modelling.features import FEATURE_COLS

MODEL_ID = "lgbm"
MODEL_VERSION = "lgbm-1.1"

HL_CANDIDATES = [90, 180, 365, 730]   # half-life in days for sample weighting
MAX_TRAIN_SEASONS = 8

LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 20,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30
LABEL_MAP = {"H": 0, "D": 1, "A": 2}
LABEL_INV = {0: "H", 1: "D", 2: "A"}


def _prep(feat_df: pd.DataFrame, outcomes: pd.DataFrame | None = None):
    X = feat_df[FEATURE_COLS].fillna(feat_df[FEATURE_COLS].median()).values.astype(np.float32)
    y = None
    if outcomes is not None:
        merged = feat_df[["match_id"]].merge(outcomes[["match_id", "result"]], on="match_id")
        y = merged["result"].map(LABEL_MAP).values.astype(np.int32)
    return X, y


def _sample_weights(feat_df: pd.DataFrame, training_cutoff: pd.Timestamp, half_life: float) -> np.ndarray:
    days = (training_cutoff - feat_df["kickoff_utc"]).dt.total_seconds() / 86400
    w = np.exp(-np.log(2) * days.clip(lower=0).values / half_life).astype(np.float32)
    return w / w.mean()


def _fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> lgb.Booster:
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS, weight=sample_weight)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=dtrain)
    callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                 lgb.log_evaluation(period=-1)]
    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    return model


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
    from modelling.features import build as build_features

    matches = matches.copy()
    matches["split"] = matches["season"].map(splits_by_season)

    if feat_table is None:
        feat_table = build_features(matches, ref_pre, outcomes)

    feat_table = feat_table.copy()
    feat_table["split"] = feat_table["season"].map(splits_by_season)

    seasons = sorted(feat_table["season"].unique())

    all_preds: list[dict] = []
    all_explanations: list[dict] = []
    best_model: lgb.Booster | None = None
    best_imp: dict = {}

    # pre-collect calibration features for early stopping + HL tuning
    cal_feat = feat_table[feat_table["split"] == "calibration"]
    X_cal, y_cal = _prep(cal_feat, outcomes) if len(cal_feat) > 0 else (None, None)

    best_hl = HL_CANDIDATES[-1]   # default: least decay; updated once on calibration
    hl_tuned = False

    for i, season in enumerate(seasons):
        season_feat = feat_table[feat_table["season"] == season]
        season_split = season_feat["split"].iloc[0]

        if season_split == "holdout" and not include_holdout:
            continue

        train_seasons = [
            s for s in seasons[:i]
            if splits_by_season.get(s) in ("warmup", "train")
        ][-MAX_TRAIN_SEASONS:]

        # --- half-life tuning (once on calibration split) ---
        if season_split == "calibration" and not hl_tuned and X_cal is not None and y_cal is not None:
            tune_train = feat_table[feat_table["season"].isin(train_seasons)]
            if len(tune_train) >= 30:
                tc = matches[matches["season"].isin(train_seasons)]["kickoff_utc"].max()
                X_tt, y_tt = _prep(tune_train, outcomes)[:2]
                best_ll_hl, best_hl = np.inf, HL_CANDIDATES[-1]
                for hl in HL_CANDIDATES:
                    sw = _sample_weights(tune_train, tc, hl)
                    m = _fit(X_tt, y_tt, X_cal, y_cal, sample_weight=sw)
                    probs = m.predict(X_cal)
                    ll = -np.log(probs[np.arange(len(y_cal)), y_cal].clip(1e-15)).mean()
                    if trials_path:
                        _log_trial(trials_path, m.num_trees(), f"hl_tune half_life={hl}d", hl, float(ll))
                    if ll < best_ll_hl:
                        best_ll_hl, best_hl = ll, hl
            hl_tuned = True

        if len(train_seasons) == 0:
            for _, row in season_feat.iterrows():
                all_preds.append(_stub_row(row, feat_table))
            continue

        train_feat = feat_table[feat_table["season"].isin(train_seasons)]
        X_train, y_train = _prep(train_feat, outcomes)

        if len(X_train) < 30 or y_train is None:
            for _, row in season_feat.iterrows():
                all_preds.append(_stub_row(row, feat_table))
            continue

        # early stopping: use calibration if available, else a holdout slice of train
        if X_cal is not None and y_cal is not None and len(X_cal) > 0:
            X_v, y_v = X_cal, y_cal
        else:
            split_at = int(len(X_train) * 0.8)
            X_v, y_v = X_train[split_at:], y_train[split_at:]
            X_train, y_train = X_train[:split_at], y_train[:split_at]

        if len(X_v) < 5:
            for _, row in season_feat.iterrows():
                all_preds.append(_stub_row(row, feat_table))
            continue

        training_cutoff = matches[matches["season"].isin(train_seasons)]["kickoff_utc"].max()
        sw = _sample_weights(train_feat, training_cutoff, best_hl)
        model = _fit(X_train, y_train, X_v, y_v, sample_weight=sw)
        best_model = model

        X_pred, _ = _prep(season_feat)
        probs = model.predict(X_pred)   # (n, 3)


        for j, (_, row) in enumerate(season_feat.iterrows()):
            tc = min(training_cutoff, row["feature_cutoff_utc"])
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

        # SHAP explanations for validation/holdout
        if season_split in ("validation", "holdout") and len(season_feat) > 0:
            explainer = shap.TreeExplainer(model)
            raw = explainer.shap_values(X_pred)
            # normalise to list of (n, n_features) arrays regardless of SHAP version
            if isinstance(raw, np.ndarray) and raw.ndim == 3:
                shap_vals = [raw[:, :, k] for k in range(raw.shape[2])]
            else:
                shap_vals = raw
            for j, (_, row) in enumerate(season_feat.iterrows()):
                for outcome_idx, outcome_label in [(0, "H"), (1, "D"), (2, "A")]:
                    sv = shap_vals[outcome_idx][j]    # (n_features,)
                    top3 = np.argsort(np.abs(sv))[::-1][:3]
                    for rank, fi in enumerate(top3):
                        all_explanations.append({
                            "match_id": row["match_id"],
                            "model_id": MODEL_ID,
                            "model_version": MODEL_VERSION,
                            "outcome": outcome_label,
                            "rank": rank + 1,
                            "feature": FEATURE_COLS[fi],
                            "feature_value": float(X_pred[j, fi]),
                            "contribution": float(sv[fi]),
                        })

    # global feature importance (mean abs SHAP on validation)
    importance: dict = {"model_id": MODEL_ID, "model_version": MODEL_VERSION,
                        "method": "mean_abs_shap", "split": "validation", "features": []}
    if best_model is not None:
        val_feat = feat_table[feat_table["split"] == "validation"]
        if len(val_feat) > 0:
            X_val_imp, _ = _prep(val_feat)
            explainer = shap.TreeExplainer(best_model)
            raw = explainer.shap_values(X_val_imp)
            if isinstance(raw, np.ndarray) and raw.ndim == 3:
                sv = [raw[:, :, k] for k in range(raw.shape[2])]
            else:
                sv = raw
            mean_abs = np.mean([np.abs(sv[k]) for k in range(3)], axis=(0, 1))
            order = np.argsort(mean_abs)[::-1]
            importance["features"] = [
                {"feature": FEATURE_COLS[fi], "importance": float(mean_abs[fi])}
                for fi in order
            ]
            best_imp = importance

    return (
        pd.DataFrame(all_preds).reset_index(drop=True),
        pd.DataFrame(all_explanations).reset_index(drop=True),
        importance,
    )


def _stub_row(row: pd.Series, feat_table: pd.DataFrame) -> dict:
    frow = feat_table[feat_table["match_id"] == row["match_id"]]
    if len(frow):
        q_h = float(frow.iloc[0]["q_home"])
        q_d = float(frow.iloc[0]["q_draw"])
        q_a = float(frow.iloc[0]["q_away"])
    else:
        q_h = q_d = q_a = 1 / 3
    return {
        "match_id": row["match_id"],
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "split": row["split"],
        "decision_time_utc": row["feature_cutoff_utc"],
        "feature_cutoff_utc": row["feature_cutoff_utc"],
        "training_cutoff_utc": row["feature_cutoff_utc"],
        "p_home": q_h, "p_draw": q_d, "p_away": q_a,
        "xg_home": pd.NA, "xg_away": pd.NA,
    }


def _log_trial(path: Path, n_trees: int, what: str, half_life: float, val_log_loss: float | None) -> None:
    import datetime, math
    ll = None if val_log_loss is None or not math.isfinite(val_log_loss) else round(val_log_loss, 6)
    entry = {"time": datetime.datetime.utcnow().isoformat() + "Z",
             "part": 2, "model_id": MODEL_ID, "model_version": MODEL_VERSION,
             "what_changed": f"{what} n_trees={n_trees} half_life={half_life}d",
             "val_log_loss": ll}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
