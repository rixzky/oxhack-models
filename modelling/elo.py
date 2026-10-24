"""Elo rating model -> H/D/A probabilities via multinomial logistic regression.

Walk-forward by season: ratings update from past results only (kickoff <
feature_cutoff_utc). The logistic regression is re-fitted each season on all
prior warmup+train seasons, so predictions for season S never see results from
season S or later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_ID = "elo"
MODEL_VERSION = "elo-1.0"

INIT_RATING = 1500.0
HOME_ADV = 100.0      # rating points added to home side for expected-score calc
K_BASE = 32.0
REGRESS_FACTOR = 0.3  # fraction of (rating - init) kept across season boundary


def _k_mult(gd: int) -> float:
    return 1.0 + 0.5 * np.log1p(abs(gd))


def _expected_score(r_a: float, r_b: float) -> float:
    """Elo expected score for side A; r_a already includes any offset."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def _build_snapshots(
    matches: pd.DataFrame, outcomes: pd.DataFrame
) -> dict[pd.Timestamp, dict[str, float]]:
    """Return {feature_cutoff_utc: {team: rating}} computed walk-forward."""
    data = (
        matches[["match_id", "home_team", "away_team", "kickoff_utc",
                 "feature_cutoff_utc", "season"]]
        .merge(outcomes[["match_id", "result", "home_goals", "away_goals"]],
               on="match_id", how="left")
        .sort_values("kickoff_utc")
        .reset_index(drop=True)
    )

    ratings: dict[str, float] = {}
    snapshots: dict[pd.Timestamp, dict[str, float]] = {}
    prev_season: str | None = None
    result_idx = 0

    for cutoff in sorted(matches["feature_cutoff_utc"].unique()):
        # absorb all results with kickoff < cutoff
        while result_idx < len(data):
            row = data.iloc[result_idx]
            if row["kickoff_utc"] >= cutoff:
                break
            result_idx += 1
            if pd.isna(row["result"]):
                continue

            home, away, season = row["home_team"], row["away_team"], row["season"]

            # regress on season boundary
            if prev_season is not None and season != prev_season:
                for t in list(ratings):
                    ratings[t] = INIT_RATING + (1.0 - REGRESS_FACTOR) * (ratings[t] - INIT_RATING)
            prev_season = season

            ratings.setdefault(home, INIT_RATING)
            ratings.setdefault(away, INIT_RATING)

            r_h = ratings[home] + HOME_ADV
            r_a = ratings[away]
            e_h = _expected_score(r_h, r_a)

            result = row["result"]
            s_h = 1.0 if result == "H" else (0.5 if result == "D" else 0.0)
            gd = abs(int(row["home_goals"]) - int(row["away_goals"]))
            k = K_BASE * _k_mult(gd)

            ratings[home] += k * (s_h - e_h)
            ratings[away] += k * ((1.0 - s_h) - (1.0 - e_h))

        snapshots[cutoff] = ratings.copy()

    return snapshots


def predict_all(
    matches: pd.DataFrame,
    outcomes: pd.DataFrame,
    splits_by_season: dict[str, str],
    include_holdout: bool = False,
) -> pd.DataFrame:
    matches = matches.copy()
    matches["split"] = matches["season"].map(splits_by_season)

    if not include_holdout:
        matches = matches[matches["split"] != "holdout"]

    snapshots = _build_snapshots(matches, outcomes)

    # rating diff per match
    feat_rows = []
    for _, m in matches.iterrows():
        snap = snapshots.get(m["feature_cutoff_utc"], {})
        r_h = snap.get(m["home_team"], INIT_RATING)
        r_a = snap.get(m["away_team"], INIT_RATING)
        feat_rows.append({
            "match_id": m["match_id"],
            "season": m["season"],
            "split": m["split"],
            "feature_cutoff_utc": m["feature_cutoff_utc"],
            "rating_diff": r_h - r_a,
        })
    feat_df = pd.DataFrame(feat_rows)

    seasons = sorted(feat_df["season"].unique())
    all_preds: list[dict] = []

    for i, season in enumerate(seasons):
        season_feat = feat_df[feat_df["season"] == season]
        season_split = season_feat["split"].iloc[0]

        if season_split == "holdout" and not include_holdout:
            continue

        # LR training: warmup+train seasons strictly before this one
        train_seasons = [
            s for s in seasons[:i]
            if splits_by_season.get(s) in ("warmup", "train")
        ]

        if len(train_seasons) == 0:
            # insufficient history: equal-probability stub
            for _, row in season_feat.iterrows():
                all_preds.append(_stub_row(row))
            continue

        train_feat = feat_df[feat_df["season"].isin(train_seasons)]
        train_with_res = train_feat.merge(
            outcomes[["match_id", "result"]], on="match_id", how="inner"
        )

        if len(train_with_res) < 30:
            for _, row in season_feat.iterrows():
                all_preds.append(_stub_row(row))
            continue

        lr = LogisticRegression(
            solver="lbfgs", max_iter=1000, C=1.0, random_state=0,
        )
        lr.fit(
            train_with_res[["rating_diff"]].values,
            train_with_res["result"].values,
        )

        training_cutoff = matches[matches["season"].isin(train_seasons)]["kickoff_utc"].max()
        classes = list(lr.classes_)
        h_i, d_i, a_i = classes.index("H"), classes.index("D"), classes.index("A")

        probs = lr.predict_proba(season_feat[["rating_diff"]].values)

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
                "p_home": float(probs[j, h_i]),
                "p_draw": float(probs[j, d_i]),
                "p_away": float(probs[j, a_i]),
                "xg_home": pd.NA,
                "xg_away": pd.NA,
            })

    return pd.DataFrame(all_preds).reset_index(drop=True)


def _stub_row(row: pd.Series) -> dict:
    return {
        "match_id": row["match_id"],
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "split": row["split"],
        "decision_time_utc": row["feature_cutoff_utc"],
        "feature_cutoff_utc": row["feature_cutoff_utc"],
        "training_cutoff_utc": row["feature_cutoff_utc"],
        "p_home": 1 / 3,
        "p_draw": 1 / 3,
        "p_away": 1 / 3,
        "xg_home": pd.NA,
        "xg_away": pd.NA,
    }
