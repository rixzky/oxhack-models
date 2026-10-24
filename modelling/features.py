"""Feature table for MarketNet and LightGBM.

One row per match, all values known at feature_cutoff_utc:
  market q / log-q (H, D, A)
  Elo rating difference, home Elo (walk-forward)
  Dixon-Coles probs, log-ratio to market, xg
  Rest days home/away (capped at 14), rest-day difference
  League one-hot
  Favourite-longshot features: log-q of the favourite, q_draw, overround
  Lagged form: points-per-game last 5 matches for home and away (before cutoff)

No closing odds, no post-cutoff information, no result of the current match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modelling.elo import INIT_RATING, _build_snapshots

LEAGUES = ["E0", "E1", "SP1", "D1", "I1", "F1", "N1", "P1"]


# ---------------------------------------------------------------- form / rest
def _build_team_history(matches: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    """Return {team: sorted list of (kickoff_utc, points_for, points_against)}."""
    data = matches.merge(
        outcomes[["match_id", "result", "home_goals", "away_goals"]],
        on="match_id", how="left",
    ).dropna(subset=["result"])

    history: dict[str, list] = {}
    for row in data.itertuples():
        if row.result == "H":
            hp, ap = 3, 0
        elif row.result == "D":
            hp, ap = 1, 1
        else:
            hp, ap = 0, 3
        history.setdefault(row.home_team, []).append((row.kickoff_utc, hp, row.home_goals, row.away_goals))
        history.setdefault(row.away_team, []).append((row.kickoff_utc, ap, row.away_goals, row.home_goals))

    for t in history:
        history[t].sort(key=lambda x: x[0])
    return history


def _form_and_rest(team: str, cutoff: pd.Timestamp, history: dict, n: int = 5) -> tuple[float, float]:
    """Return (ppg_last_n, rest_days_capped_14) for team before cutoff."""
    entries = [(k, p) for k, p, *_ in history.get(team, []) if k < cutoff]
    rest = 14.0
    if entries:
        last_ko = entries[-1][0]
        rest = min(float((cutoff - last_ko).total_seconds() / 86400), 14.0)
    recent = [p for _, p in entries[-n:]]
    ppg = float(np.mean(recent)) if recent else 1.0   # 1.0 ≈ average prior
    return ppg, rest


# ---------------------------------------------------------------- public interface
def build(
    matches: pd.DataFrame,
    ref_pre: pd.DataFrame,
    outcomes: pd.DataFrame,
    dc_preds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build feature table. matches must have a 'split' column already assigned.

    Returns one row per match that has a reference PRE_CLOSING market quote.
    Saves to artifacts/x_features.parquet for debugging (not shared).
    """
    # --- market features ---
    market = ref_pre[["match_id", "q_home", "q_draw", "q_away", "overround"]].copy()
    base = matches.merge(market, on="match_id", how="inner")

    base["log_q_home"] = np.log(base["q_home"].clip(1e-6))
    base["log_q_draw"] = np.log(base["q_draw"].clip(1e-6))
    base["log_q_away"] = np.log(base["q_away"].clip(1e-6))

    # favourite = outcome with highest q; fav_log_q captures longshot bias signal
    q_mat = base[["q_home", "q_draw", "q_away"]].values
    fav_idx = q_mat.argmax(axis=1)
    base["fav_log_q"] = np.log(q_mat[np.arange(len(base)), fav_idx].clip(1e-6))
    base["longshot_log_q"] = np.log(q_mat[np.arange(len(base)), q_mat.argmin(axis=1)].clip(1e-6))

    # --- Elo features ---
    snapshots = _build_snapshots(matches, outcomes)
    elo_h, elo_a = [], []
    for row in base.itertuples():
        snap = snapshots.get(row.feature_cutoff_utc, {})
        elo_h.append(snap.get(row.home_team, INIT_RATING))
        elo_a.append(snap.get(row.away_team, INIT_RATING))
    base["elo_home"] = elo_h
    base["elo_away"] = elo_a
    base["elo_diff"] = base["elo_home"] - base["elo_away"]

    # --- Dixon-Coles features ---
    if dc_preds is not None:
        dc = dc_preds[dc_preds["model_id"] == "dixon_coles"][
            ["match_id", "p_home", "p_draw", "p_away", "xg_home", "xg_away"]
        ].rename(columns={
            "p_home": "dc_p_home", "p_draw": "dc_p_draw", "p_away": "dc_p_away",
        })
        base = base.merge(dc, on="match_id", how="left")
        eps = 1e-6
        base["dc_logratio_home"] = np.log(base["dc_p_home"].clip(eps) / base["q_home"].clip(eps))
        base["dc_logratio_draw"] = np.log(base["dc_p_draw"].clip(eps) / base["q_draw"].clip(eps))
        base["dc_logratio_away"] = np.log(base["dc_p_away"].clip(eps) / base["q_away"].clip(eps))
        base["xg_home"] = base["xg_home"].fillna(base["xg_home"].median())
        base["xg_away"] = base["xg_away"].fillna(base["xg_away"].median())
        base["xg_diff"] = base["xg_home"] - base["xg_away"]
    else:
        for col in ["dc_p_home", "dc_p_draw", "dc_p_away",
                    "dc_logratio_home", "dc_logratio_draw", "dc_logratio_away",
                    "xg_home", "xg_away", "xg_diff"]:
            base[col] = 0.0

    # --- rest days and form ---
    hist = _build_team_history(matches, outcomes)
    form_h, form_a, rest_h, rest_a = [], [], [], []
    for row in base.itertuples():
        fh, rh = _form_and_rest(row.home_team, row.feature_cutoff_utc, hist)
        fa, ra = _form_and_rest(row.away_team, row.feature_cutoff_utc, hist)
        form_h.append(fh); form_a.append(fa)
        rest_h.append(rh); rest_a.append(ra)

    base["form_home"] = form_h
    base["form_away"] = form_a
    base["form_diff"] = base["form_home"] - base["form_away"]
    base["rest_days_home"] = rest_h
    base["rest_days_away"] = rest_a
    base["rest_days_diff"] = base["rest_days_home"] - base["rest_days_away"]

    # --- league one-hot ---
    for lg in LEAGUES:
        base[f"league_{lg}"] = (base["league"] == lg).astype(float)

    return base.reset_index(drop=True)


FEATURE_COLS = [
    "q_home", "q_draw", "q_away",
    "log_q_home", "log_q_draw", "log_q_away",
    "overround", "fav_log_q", "longshot_log_q",
    "elo_diff", "elo_home",
    "dc_p_home", "dc_p_draw", "dc_p_away",
    "dc_logratio_home", "dc_logratio_draw", "dc_logratio_away",
    "xg_home", "xg_away", "xg_diff",
    "form_home", "form_away", "form_diff",
    "rest_days_home", "rest_days_away", "rest_days_diff",
] + [f"league_{lg}" for lg in LEAGUES]
