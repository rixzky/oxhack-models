"""Market baseline: use reference PRE_CLOSING implied probabilities as-is."""
from __future__ import annotations

import pandas as pd

MODEL_ID = "market"
MODEL_VERSION = "market-1.0"


def predict_all(
    matches: pd.DataFrame,
    ref_pre: pd.DataFrame,
    splits_by_season: dict[str, str],
    include_holdout: bool = False,
) -> pd.DataFrame:
    """One prediction row per match that has a reference PRE_CLOSING quote."""
    matches = matches.copy()
    matches["split"] = matches["season"].map(splits_by_season)

    if not include_holdout:
        matches = matches[matches["split"] != "holdout"]

    # ref_pre already filtered to is_reference=True, snapshot=PRE_CLOSING
    joined = matches.merge(
        ref_pre[["match_id", "q_home", "q_draw", "q_away"]],
        on="match_id",
        how="inner",
    )

    rows = pd.DataFrame({
        "match_id": joined["match_id"],
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "split": joined["split"],
        "decision_time_utc": joined["feature_cutoff_utc"],
        "feature_cutoff_utc": joined["feature_cutoff_utc"],
        "training_cutoff_utc": joined["feature_cutoff_utc"],
        "p_home": joined["q_home"],
        "p_draw": joined["q_draw"],
        "p_away": joined["q_away"],
        "xg_home": pd.NA,
        "xg_away": pd.NA,
    })

    return rows.reset_index(drop=True)
