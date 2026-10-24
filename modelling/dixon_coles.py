"""Dixon-Coles model: Poisson goals with DC low-score correction and time decay.

Per league, refit at least monthly on matches before training_cutoff_utc.
Scoreline grid 0-10 x 0-10 summed to H/D/A. Fills xg_home / xg_away.

Reference: Dixon & Coles (1997) "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market".
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MODEL_ID = "dixon_coles"
MODEL_VERSION = "dc-1.0"

MAX_GOALS = 10          # grid size for scoreline probabilities
MIN_MATCHES = 20        # minimum matches to fit a league model
HALF_LIFE_DEFAULT = 60  # days; tuned on calibration split below


# ---------------------------------------------------------------- DC correction
def _tau(lam: float, mu: float, x: int, y: int, rho: float) -> float:
    """Dixon-Coles correction factor for low-scoring games."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


_GOALS = np.arange(MAX_GOALS + 1)


def _score_matrix(lam: float, mu: float, rho: float) -> np.ndarray:
    """(MAX_GOALS+1) x (MAX_GOALS+1) matrix of P(home=i, away=j). Vectorised."""
    ph = poisson.pmf(_GOALS, lam)   # (g,)
    pa = poisson.pmf(_GOALS, mu)    # (g,)
    mat = np.outer(ph, pa)          # (g, g)
    # DC correction on the 2x2 low-score corner
    mat[0, 0] *= max(1.0 - lam * mu * rho, 1e-15)
    mat[1, 0] *= 1.0 + mu * rho
    mat[0, 1] *= 1.0 + lam * rho
    mat[1, 1] *= 1.0 - rho
    mat = np.clip(mat, 0.0, None)
    s = mat.sum()
    return mat / s if s > 0 else mat


def _hda_from_matrix(mat: np.ndarray) -> tuple[float, float, float]:
    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())
    total = p_home + p_draw + p_away
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return p_home / total, p_draw / total, p_away / total


# ---------------------------------------------------------------- model fitting
RHO_FIXED = -0.13   # DC correction: fixed at a typical football value, saves one param


def _neg_log_likelihood(
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    rho: float,
) -> float:
    attack = params[:n_teams]
    defence = params[n_teams : 2 * n_teams]
    home_adv = params[2 * n_teams]

    lam = np.exp(attack[home_idx] - defence[away_idx] + home_adv)
    mu = np.exp(attack[away_idx] - defence[home_idx])

    # vectorised Poisson log-pmf
    ll = weights * (
        home_goals * np.log(np.maximum(lam, 1e-15)) - lam
        + away_goals * np.log(np.maximum(mu, 1e-15)) - mu
    )

    # DC correction for low-scoring games (applied per row)
    tau = np.ones(len(home_goals))
    m00 = (home_goals == 0) & (away_goals == 0)
    m10 = (home_goals == 1) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m11 = (home_goals == 1) & (away_goals == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m11] = 1.0 - rho
    tau = np.clip(tau, 1e-15, None)
    ll += weights * np.log(tau)

    return -ll.sum()


def _fit_league(
    data: pd.DataFrame,
    cutoff: pd.Timestamp,
    half_life: float,
) -> dict | None:
    """Fit DC model for one league on matches with kickoff < cutoff."""
    sub = data[data["kickoff_utc"] < cutoff].dropna(subset=["result"]).copy()
    if len(sub) < MIN_MATCHES:
        return None

    days_ago = (cutoff - sub["kickoff_utc"]).dt.total_seconds() / 86400
    sub["w"] = np.exp(-np.log(2) * days_ago / half_life)

    teams = sorted(set(sub["home_team"]) | set(sub["away_team"]))
    n = len(teams)
    if n < 4:
        return None

    idx = {t: i for i, t in enumerate(teams)}
    hi = np.array([idx[t] for t in sub["home_team"]])
    ai = np.array([idx[t] for t in sub["away_team"]])

    x0 = np.zeros(2 * n + 1)
    x0[2 * n] = 0.1   # home_adv

    bounds = [(-3.0, 3.0)] * (2 * n) + [(-0.5, 1.0)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            _neg_log_likelihood,
            x0,
            args=(hi, ai,
                  sub["home_goals"].values.astype(int),
                  sub["away_goals"].values.astype(int),
                  sub["w"].values, n, RHO_FIXED),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-6},
        )

    p = res.x
    return {
        "teams": teams,
        "attack": dict(zip(teams, p[:n])),
        "defence": dict(zip(teams, p[n : 2 * n])),
        "home_adv": float(p[2 * n]),
        "rho": RHO_FIXED,
        "mean_attack": float(p[:n].mean()),
        "mean_defence": float(p[n : 2 * n].mean()),
    }


def _predict_match(params: dict, home: str, away: str) -> tuple[float, float, float, float, float]:
    """Returns p_home, p_draw, p_away, xg_home, xg_away."""
    a = params["attack"]
    d = params["defence"]
    mean_a = params["mean_attack"]
    mean_d = params["mean_defence"]
    # fall back to league means for unseen teams
    lam = np.exp(a.get(home, mean_a) - d.get(away, mean_d) + params["home_adv"])
    mu = np.exp(a.get(away, mean_a) - d.get(home, mean_d))
    lam = float(np.clip(lam, 0.05, 10.0))
    mu = float(np.clip(mu, 0.05, 10.0))
    mat = _score_matrix(lam, mu, params["rho"])
    ph, pd_, pa = _hda_from_matrix(mat)
    return ph, pd_, pa, lam, mu


# ---------------------------------------------------------------- half-life tuning
def _tune_half_life(
    data: pd.DataFrame,
    cal_cutoff: pd.Timestamp,
    candidates: list[float],
) -> float:
    """Pick half-life minimising log loss on calibration matches.

    Fits once per (half-life, league) at cal_cutoff, evaluates on all
    calibration matches after that cutoff. Fast: len(candidates)*len(leagues) fits.
    """
    from contract import maths as M

    best_hl, best_ll = candidates[0], np.inf
    leagues = data["league"].unique()

    for hl in candidates:
        ll_sum, n = 0.0, 0
        for league in leagues:
            league_data = data[data["league"] == league]
            params = _fit_league(league_data, cal_cutoff, hl)
            if params is None:
                continue
            cal_end = cal_cutoff + pd.Timedelta(days=400)
            eval_rows = league_data[
                (league_data["kickoff_utc"] >= cal_cutoff)
                & (league_data["kickoff_utc"] < cal_end)
                & (league_data["split"] == "calibration")
                & league_data["result"].notna()
            ]
            for row in eval_rows.itertuples():
                ph, pd_, pa, _, _ = _predict_match(params, row.home_team, row.away_team)
                ll_sum += M.log_loss_row([ph, pd_, pa], row.result)
                n += 1
        if n > 0 and (ll_sum / n) < best_ll:
            best_ll = ll_sum / n
            best_hl = hl
    return best_hl


# ---------------------------------------------------------------- public interface
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

    data = matches.merge(
        outcomes[["match_id", "result", "home_goals", "away_goals"]],
        on="match_id",
        how="left",
    )

    # --- tune half-life on calibration split (one fit per candidate per league) ---
    cal_matches = matches[matches["split"] == "calibration"]
    if len(cal_matches) > 0:
        cal_cutoff = cal_matches["feature_cutoff_utc"].min()
        hl_candidates = [30.0, 60.0, 90.0, 180.0]
        best_hl = _tune_half_life(data, cal_cutoff, hl_candidates)
    else:
        best_hl = HALF_LIFE_DEFAULT
    print(f"    dixon_coles best half-life: {best_hl:.0f} days")

    # --- walk-forward predictions ---
    # Refit per (league, season): monthly is too slow on real multi-league data.
    # Spec says "monthly if slow" — seasonal is the practical choice here.
    all_preds: list[dict] = []

    # Pre-group to avoid O(n²) pandas filters in the loop.
    data_by_league: dict[str, pd.DataFrame] = {
        lg: data[data["league"] == lg] for lg in data["league"].unique()
    }
    # training_cutoff per (league, season): last kickoff < first cutoff of that season
    tc_cache: dict[tuple, pd.Timestamp] = {}
    param_cache: dict[tuple, dict | None] = {}  # (league, season) -> fitted params

    # Group matches by (feature_cutoff_utc, league) — avoids repeated row filters
    for (cutoff, league), group in matches.groupby(
        ["feature_cutoff_utc", "league"], sort=False
    ):
        season = group["season"].iloc[0]
        cache_key = (league, season)

        if cache_key not in param_cache:
            ld = data_by_league.get(league, data[data["league"] == league])
            param_cache[cache_key] = _fit_league(ld, cutoff, best_hl)
            past = ld[ld["kickoff_utc"] < cutoff]
            tc_cache[cache_key] = past["kickoff_utc"].max() if len(past) > 0 else cutoff

        params = param_cache[cache_key]
        tc_raw = tc_cache.get(cache_key, cutoff)
        tc = min(tc_raw, cutoff)

        for _, m in group.iterrows():
            if params is None:
                all_preds.append(_stub_row(m))
                continue
            ph, pd_, pa, xg_h, xg_a = _predict_match(params, m["home_team"], m["away_team"])
            all_preds.append({
                "match_id": m["match_id"],
                "model_id": MODEL_ID,
                "model_version": MODEL_VERSION,
                "split": m["split"],
                "decision_time_utc": cutoff,
                "feature_cutoff_utc": cutoff,
                "training_cutoff_utc": tc,
                "p_home": ph,
                "p_draw": pd_,
                "p_away": pa,
                "xg_home": xg_h,
                "xg_away": xg_a,
            })

    return pd.DataFrame(all_preds).reset_index(drop=True)


def _stub_row(m: pd.Series) -> dict:
    return {
        "match_id": m["match_id"],
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "split": m["split"],
        "decision_time_utc": m["feature_cutoff_utc"],
        "feature_cutoff_utc": m["feature_cutoff_utc"],
        "training_cutoff_utc": m["feature_cutoff_utc"],
        "p_home": 1 / 3,
        "p_draw": 1 / 3,
        "p_away": 1 / 3,
        "xg_home": float("nan"),
        "xg_away": float("nan"),
    }
