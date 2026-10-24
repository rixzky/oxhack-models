"""Generate the SYNTHETIC golden files used by all three parts for shape and
integration tests. These are NOT real data and must never be reported as
findings. Run from the repo root:  python -m contract.golden.generate_golden

Outputs (in contract/golden/):
  tables/*.parquet + tables/*.csv   one per contract table
  artifacts_json/*.json             manifest, splits, data_audit, forecast_metrics, frozen_config
  api/*.json                        example API responses for the UI (Part 3 frontend)
  test_vectors.json                 numeric vectors every part tests against
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from contract import maths as M
from contract import schemas as S

HERE = Path(__file__).parent
TAB = HERE / "tables"
ART = HERE / "artifacts_json"
API = HERE / "api"
for d in (TAB, ART, API):
    d.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20261024)
INGESTED = pd.Timestamp("2026-10-24T12:00:00Z")
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
SPLIT_OF = {"2021-22": "warmup", "2022-23": "train", "2023-24": "calibration",
            "2024-25": "validation", "2025-26": "holdout"}
LEAGUE_TEAMS = {
    "E0": ["Alder FC", "Birch Town", "Cedar City", "Dunmore", "Elmfield",
           "Fernley", "Granton", "Hollins", "Irwell", "Juniper Utd"],
    "D1": ["Ankerberg", "Bruckstadt", "Celle Nord", "Dornfeld",
           "Eschwald", "Falkenau", "Grunhof", "Hagenbach"],
}


# ------------------------------------------------------------ simulate
def true_probs(att_h, att_a, home_adv=0.25):
    lh, la = np.exp(home_adv + att_h - att_a) * 1.3, np.exp(att_a - att_h) * 1.1
    gh = rng.poisson(lh)
    ga = rng.poisson(la)
    # outcome probabilities by simulation of the same Poisson pair
    sims_h = rng.poisson(lh, 4000)
    sims_a = rng.poisson(la, 4000)
    p = np.array([(sims_h > sims_a).mean(), (sims_h == sims_a).mean(), (sims_h < sims_a).mean()])
    return p / p.sum(), int(gh), int(ga)


def quote_from(p, margin, noise):
    q = np.clip(p * np.exp(rng.normal(0, noise, 3)), 0.04, None)
    q = q / q.sum() * (1 + margin)
    return np.maximum(np.round(1 / q, 2), 1.02)


matches, outcomes, quotes = [], [], []
qid = 0
for league, teams in LEAGUE_TEAMS.items():
    strength = {t: rng.normal(0, 0.25) for t in teams}
    for si, season in enumerate(SEASONS):
        start = pd.Timestamp(f"{2021 + si}-08-14T14:00:00Z")  # a Saturday
        fixtures = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(fixtures)
        per_week = len(teams) // 2
        for i, (h, a) in enumerate(fixtures):
            week = i // per_week
            ko = start + timedelta(weeks=week) + (timedelta(days=1) if i % 4 == 3 else timedelta(0))
            p, gh, ga = true_probs(strength[h], strength[a])
            mid = S.make_match_id(league, season, ko, h, a)
            matches.append(dict(match_id=mid, league=league, season=season, kickoff_utc=ko,
                                kickoff_precision="EXACT", home_team=h, away_team=a,
                                matchweek_key=S.matchweek_key(league, ko),
                                feature_cutoff_utc=S.feature_cutoff(ko)))
            res = "H" if gh > ga else ("D" if gh == ga else "A")
            outcomes.append(dict(match_id=mid, home_goals=gh, away_goals=ga, result=res))
            pre_obs = ko.floor("D") - timedelta(days=1) + timedelta(hours=18)  # LATEST plausible capture (window end)
            for bk, margin, noise in (("B365", 0.05, 0.07), ("PS", 0.025, 0.06), ("BW", 0.06, 0.08), ("WH", 0.06, 0.08)):
                for snap, obs, nz in (("PRE_CLOSING", pre_obs, noise), ("CLOSING", ko, noise / 2)):
                    o = quote_from(p, margin, nz)
                    qid += 1
                    quotes.append(dict(quote_id=f"Q{qid:06d}", match_id=mid, bookmaker=bk, snapshot=snap,
                                       observed_at_utc=obs, timestamp_precision="WINDOW",
                                       ingested_at_utc=INGESTED, odds_home=float(o[0]),
                                       odds_draw=float(o[1]), odds_away=float(o[2]),
                                       source="SYNTHETIC", is_valid=True, invalid_reason=None))
        for t in teams:  # strengths drift between seasons
            strength[t] += rng.normal(0, 0.12)

matches = pd.DataFrame(matches)
outcomes = pd.DataFrame(outcomes)
quotes = pd.DataFrame(quotes)

# Inject data problems so the audit and the decision checks have something to catch.
bad = quotes.sample(6, random_state=1).index
quotes.loc[bad[:3], "odds_draw"] = np.nan
quotes.loc[bad[3:], "odds_home"] = 1.0
quotes.loc[bad, "is_valid"] = False
quotes.loc[bad[:3], "invalid_reason"] = "MISSING_ODDS"
quotes.loc[bad[3:], "invalid_reason"] = "ODDS_LE_1"
unk = quotes[(quotes.snapshot == "PRE_CLOSING") & quotes.is_valid].sample(4, random_state=2).index
quotes.loc[unk, "timestamp_precision"] = "UNKNOWN"
quotes.loc[unk, "observed_at_utc"] = pd.NaT
for c in ("observed_at_utc",):
    quotes[c] = pd.to_datetime(quotes[c], utc=True)

# ------------------------------------------------------------ market probs (reference rule)
rows = []
for (mid, snap), g in quotes[quotes.is_valid].groupby(["match_id", "snapshot"]):
    ref_bk = S.REFERENCE_BOOKMAKER
    for _, r in g.iterrows():
        q, ov = M.implied_probs([r.odds_home, r.odds_draw, r.odds_away])
        rows.append(dict(match_id=mid, bookmaker=r.bookmaker, snapshot=snap,
                         is_reference=bool(r.bookmaker == ref_bk),
                         q_home=q[0], q_draw=q[1], q_away=q[2], overround=ov))
market = pd.DataFrame(rows)

# ------------------------------------------------------------ efficiency map
mo = market[~market.bookmaker.isin(S.AGGREGATE_BOOKMAKERS)].merge(outcomes, on="match_id").merge(matches[["match_id", "league", "season"]], on="match_id")
eff = []
for (lg, se, sn, bk), g in mo.groupby(["league", "season", "snapshot", "bookmaker"]):
    P = g[["q_home", "q_draw", "q_away"]].to_numpy()
    y = g["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    Y = np.eye(3)[y]
    ll = float(np.mean([-np.log(max(P[i, y[i]], M.EPS)) for i in range(len(y))]))
    br = float(np.mean(((P - Y) ** 2).sum(axis=1)))
    flat_q, flat_y = P.ravel(), Y.ravel()
    lo, hi = flat_q < 0.25, flat_q > 0.6
    fl = float((flat_y[lo] - flat_q[lo]).mean() - (flat_y[hi] - flat_q[hi]).mean()) if lo.any() and hi.any() else None
    eff.append(dict(league=lg, season=se, snapshot=sn, bookmaker=bk, n_matches=len(g),
                    log_loss=ll, brier=br, mean_overround=float(g.overround.mean()),
                    calib_slope=None, fav_longshot_gap=fl,
                    draw_bias=float((Y[:, 1] - P[:, 1]).mean())))
eff = pd.DataFrame(eff)
eff["calib_slope"] = eff["calib_slope"].astype(float)

# ------------------------------------------------------------ predictions (golden stubs)
ref_pre = market[market.is_reference & (market.snapshot == "PRE_CLOSING")]
base = matches.merge(ref_pre, on="match_id")
preds = []
for _, r in base.iterrows():
    q = np.array([r.q_home, r.q_draw, r.q_away])
    season_start = matches.loc[matches.season == r.season, "kickoff_utc"].min()
    tc = min(season_start - timedelta(days=1), r.feature_cutoff_utc)
    for model_id, p in (("market", q), ("elo", None), ("lgbm", None)):
        if p is None:  # GOLDEN STUB, not a model: market tilted towards home + noise
            p = q * np.exp(rng.normal(0, 0.08, 3)) * np.array([1.04, 0.98, 0.98])
            p = p / p.sum()
        preds.append(dict(match_id=r.match_id, model_id=model_id,
                          model_version="golden-stub-1", split=SPLIT_OF[r.season],
                          decision_time_utc=r.feature_cutoff_utc,
                          feature_cutoff_utc=r.feature_cutoff_utc, training_cutoff_utc=tc,
                          p_home=float(p[0]), p_draw=float(p[1]), p_away=float(p[2]),
                          xg_home=np.nan, xg_away=np.nan))
preds = pd.DataFrame(preds)

# ------------------------------------------------------------ write + validate tables
for name, df in (("matches", matches), ("outcomes", outcomes), ("quotes", quotes),
                 ("market_probs", market), ("efficiency_map", eff), ("predictions", preds)):
    S.write_table(df, TAB / f"{name}.parquet", name)
    df.to_csv(TAB / f"{name}.csv", index=False)

# ------------------------------------------------------------ JSON artifacts
manifest = {
    "contract_version": S.CONTRACT_VERSION, "dataset_mode": "SYNTHETIC",
    "produced_by": "contract/golden/generate_golden.py",
    "produced_at_utc": INGESTED.isoformat().replace("+00:00", "Z"),
    "leagues": sorted(matches.league.unique().tolist()), "seasons": SEASONS,
    "sources": [{"name": "synthetic generator", "url": None, "terms_checked": None,
                 "download_date": None, "limitations": "Invented teams and prices. Shape testing only."}],
}
splits = {"contract_version": S.CONTRACT_VERSION, "by_season": SPLIT_OF,
          "holdout_season": "2025-26", "note": "Golden splits. Real splits live in artifacts/splits.json."}
audit = {
    "dataset_mode": "SYNTHETIC",
    "raw_counts": {"matches": len(matches), "quotes": len(quotes)},
    "accepted_counts": {"matches": len(matches), "quotes": int(quotes.is_valid.sum())},
    "invalid_quotes_by_reason": quotes[~quotes.is_valid].invalid_reason.value_counts().to_dict(),
    "timestamp_precision_counts": quotes.timestamp_precision.value_counts().to_dict(),
    "duplicates_removed": 0, "missing_value_counts": {"odds_draw": 3},
    "flagged_outliers": [], "exclusions": [],
    "note": "Exclusions are decided without looking at match outcomes.",
}
for name, obj in (("manifest", manifest), ("splits", splits), ("data_audit", audit)):
    (ART / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))

# forecast metrics per model/split (computed on golden predictions)
po = preds.merge(outcomes, on="match_id")
fm = []
for (mid_, sp), g in po.groupby(["model_id", "split"]):
    P = g[["p_home", "p_draw", "p_away"]].to_numpy()
    res = g.result.tolist()
    bins = []
    flat = [(P[i, k], 1.0 if res[i] == M.OUTCOMES[k] else 0.0) for i in range(len(res)) for k in range(3)]
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [(q, y) for q, y in flat if lo <= q < hi or (hi == 1 and q == 1)]
        if sel:
            bins.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2), "n": len(sel),
                         "mean_pred": float(np.mean([s[0] for s in sel])),
                         "obs_freq": float(np.mean([s[1] for s in sel]))})
    fm.append({"model_id": mid_, "split": sp, "n": len(g),
               "log_loss": float(np.mean([M.log_loss_row(P[i], res[i]) for i in range(len(res))])),
               "brier_sum3": float(np.mean([M.brier_row(P[i], res[i]) for i in range(len(res))])),
               "calibration_bins": bins})
(ART / "forecast_metrics.json").write_text(json.dumps(
    {"dataset_mode": "SYNTHETIC", "brier_convention": "sum over 3 outcomes, range 0 to 2", "rows": fm}, indent=2))



# ------------------------------------------------------------ test vectors
from contract.decisions import (decide_match, NAIVE_POLICY, DEFAULT_POLICY, BEST_PRICE_POLICY,  # noqa: E402
                                 POLICIES, best_price_quote, reference_quote)

KO = pd.Timestamp("2025-01-04T15:00:00Z")
OBS = pd.Timestamp("2025-01-03T14:00:00Z")


def qd(odds, precision="WINDOW", obs=OBS, valid=True, qid="Qtest"):
    return {"quote_id": qid, "odds": odds, "is_valid": valid,
            "timestamp_precision": precision, "observed_at_utc": obs if precision != "UNKNOWN" else None}


def summary(d):
    r = lambda x: None if x is None else round(x, 6)
    return {"action": d["action"], "selection": d["selection"], "reason_code": d["reason_code"],
            "reason_codes_all": d["reason_codes_all"], "best_outcome": d["best_outcome"],
            "ev_raw": r(d["ev_raw"]), "ev_stressed": r(d["ev_stressed"]),
            "proposed_stake": r(d["proposed_stake"]), "bookmaker": d["bookmaker"]}


base_pol = {"min_ev": 0.02, "prob_stress_delta": 0.02, "price_haircut": 0.0}
dec_cases = [
    ([0.50, 0.27, 0.23], qd([2.10, 3.40, 3.60]), base_pol),
    ([0.47, 0.27, 0.26], qd([2.10, 3.40, 3.60]), base_pol),
    ([0.30, 0.28, 0.42], qd([3.10, 3.50, 2.80]), base_pol),
    ([0.30, 0.28, 0.42], qd([3.10, 3.50, 2.80]), {**base_pol, "price_haircut": 0.10}),
    ([0.52, 0.28, 0.20], qd([1.90, 3.60, 4.80]), base_pol),
    ([0.30, 0.28, 0.42], qd([3.10, 3.50, 2.80], precision="UNKNOWN"), base_pol),
    ([0.30, 0.28, 0.42], qd([3.10, 3.50, 2.80], valid=False), base_pol),
    (None, qd([3.10, 3.50, 2.80]), base_pol),
    ([0.30, 0.28, 0.42], None, base_pol),
    ([0.30, 0.28, 0.42], qd([3.10, 3.50, 2.80], obs=KO - pd.Timedelta(hours=200)), base_pol),
    ([0.30, 0.28, 0.42], qd([3.10, 3.50, 2.80]), {**base_pol, "staking_mode": "QUARTER_KELLY"}),
    ([0.30, 0.28, 0.42], qd([3.10, 3.50, 2.80], obs=KO), base_pol),  # captured at kickoff -> STALE_QUOTE
    ([0.30, 0.28, 0.42], {**qd([3.20, 3.50, 2.90]), "bookmakers": ["BW", "B365", "WH"],
                          "ref_odds": [3.10, 3.50, 2.80]}, base_pol),  # best-price quote, Bet365 benchmark
]

# best_price_quote vectors (one match, PRE_CLOSING quotes from several bookmakers)
def bq(qid, bk, odds, valid=True, outlier=False, prec="WINDOW"):
    return {"quote_id": qid, "bookmaker": bk, "odds": odds, "is_valid": valid, "is_outlier": outlier,
            "timestamp_precision": prec, "observed_at_utc": OBS}


best_cases = [
    {"season": "2024-25", "quotes": [bq("q1", "B365", [2.10, 3.40, 3.60]), bq("q2", "PS", [2.15, 3.45, 3.70]),
                                     bq("q3", "WH", [2.05, 3.60, 3.50]), bq("q4", "MAX", [2.40, 3.80, 4.00])]},
    {"season": "2025-26", "quotes": [bq("q1", "B365", [2.10, 3.40, 3.60]), bq("q2", "PS", [2.15, 3.45, 3.70]),
                                     bq("q3", "WH", [2.05, 3.60, 3.50])]},
    {"season": "2024-25", "quotes": [bq("q1", "B365", [2.10, 3.40, 3.60]), bq("q5", "BW", [3.50, 3.40, 2.00]),
                                     bq("q6", "VC", [2.12, 3.42, 3.66], outlier=True)]},
    {"season": "2024-25", "quotes": [bq("q1", "B365", [2.10, 3.40, 3.60], valid=False), bq("q2", "PS", [2.15, 3.45, 3.70])]},
]
for c in best_cases:
    b = best_price_quote(c["quotes"], c["season"])
    c["expected"] = None if b is None else {"odds": b["odds"], "bookmakers": b["bookmakers"], "ref_odds": b["ref_odds"],
                                            "timestamp_precision": b["timestamp_precision"]}
    for q in c["quotes"]:
        q["observed_at_utc"] = OBS.isoformat().replace("+00:00", "Z")


def jsonable_quote(q):
    if q is None:
        return None
    return {**q, "observed_at_utc": None if q["observed_at_utc"] is None else q["observed_at_utc"].isoformat().replace("+00:00", "Z")}


vectors = {
    "contract_version": S.CONTRACT_VERSION,
    "tolerance": 1e-6,
    "normalise": [{"odds": o, "expected_q": [round(x, 6) for x in M.implied_probs(o)[0]],
                   "expected_overround": round(M.implied_probs(o)[1], 6)}
                  for o in ([2.10, 3.40, 3.60], [1.50, 4.20, 7.00])],
    "ev": [{"p": 0.52, "odds": 1.90, "expected_ev": -0.012,
            "note": "REGRESSION: must be rejected even though margin-free market prob is 0.50"},
           {"p": 0.55, "odds": 2.00, "expected_ev": 0.1}],
    "adverse_price": [{"odds": 2.00, "price_haircut": 0.05, "expected_adjusted_odds": 1.95},
                      {"odds": 3.40, "price_haircut": 0.10, "expected_adjusted_odds": 3.16}],
    "stress": [{"p": 0.55, "delta": 0.02, "expected_p_stressed": 0.53},
               {"p": 0.01, "delta": 0.02, "expected_p_stressed": 0.0}],
    "kelly": [{"p": 0.55, "odds": 2.00, "expected_full_kelly": 0.1, "expected_quarter_kelly": 0.025},
              {"p": 0.52, "odds": 1.90, "expected_full_kelly": 0.0, "expected_quarter_kelly": 0.0}],
    "clv": [{"offered_odds": 2.10, "closing_fair_prob": 0.50, "expected_clv": 0.05}],
    "brier": [{"p": [0.5, 0.3, 0.2], "result": "H", "expected_brier_sum3": 0.38}],
    "log_loss": [{"p": [0.5, 0.3, 0.2], "result": "D", "expected": round(float(-np.log(0.3)), 6)}],
    "decision": [{"probs": p, "quote": jsonable_quote(q), "kickoff_utc": KO.isoformat().replace("+00:00", "Z"),
                  "policy": pol, "expected": summary(decide_match(p, q, KO, pol))}
                 for p, q, pol in dec_cases],
    "best_price": best_cases,
    "bankroll": {
        "description": "Flat staking, one batch of 25 accepted bets, bankroll 1000, nothing open before. "
                       "Stake 1% of INITIAL bankroll = 10 each. Total open exposure cap 20% = 200, so bets are "
                       "taken in order (ev_stressed desc, then match_id asc) until the cap: first 20 placed, "
                       "last 5 get EXPOSURE_LIMIT_TOTAL.",
        "initial_bankroll": 1000.0, "n_accepted": 25, "flat_stake_frac": 0.01,
        "max_open_exposure_frac": 0.20, "expected_placed": 20, "expected_rejected_total_cap": 5,
    },
    "quarter_kelly_stake": {
        "description": "available_bankroll * 0.25 * full_kelly(p_stressed, odds_adjusted), then capped at 2% of available.",
        "available_bankroll": 1000.0, "p_stressed": 0.53, "odds_adjusted": 2.0,
        "expected_stake": 15.0, "cap_frac": 0.02,
    },
}
(HERE / "test_vectors.json").write_text(json.dumps(vectors, indent=2))

# ------------------------------------------------------------ golden decisions ledger (Part 1 output shape)
# GOLDEN SKETCH of CONTRACT 6.3 so Parts 2 and 3 have a ledger to read. Part 1 owns the real engine.
mo_all = matches.merge(outcomes, on="match_id")
close_ref = market[market.is_reference & (market.snapshot == "CLOSING")].set_index("match_id")
pre_quotes = {}
for r in quotes[quotes.snapshot == "PRE_CLOSING"].itertuples():
    pre_quotes.setdefault(r.match_id, []).append({
        "quote_id": r.quote_id, "bookmaker": r.bookmaker, "odds": [r.odds_home, r.odds_draw, r.odds_away],
        "is_valid": bool(r.is_valid), "is_outlier": False, "timestamp_precision": r.timestamp_precision,
        "observed_at_utc": None if pd.isna(r.observed_at_utc) else r.observed_at_utc})


def exec_quote(match_id, season, pol):
    qs = pre_quotes.get(match_id, [])
    if pol["price_source"] == "BEST_AVAILABLE":
        return best_price_quote(qs, season, pol["best_price_max_prob_gap"]) or reference_quote(qs)
    return reference_quote(qs)


pred_by = preds.set_index(["match_id", "model_id"])

ledger = []
for model_id in ("elo", "lgbm"):
    for policy_id, pol in POLICIES.items():
        settled, open_bets = pol["initial_bankroll"], []
        cfg_hash = hashlib.sha256(json.dumps({**pol, "model": model_id}, sort_keys=True).encode()).hexdigest()[:12]
        scored = mo_all[mo_all.season.map(SPLIT_OF) != "warmup"].sort_values(["feature_cutoff_utc", "match_id"])
        for dday, batch in scored.groupby(scored.feature_cutoff_utc.dt.floor("D")):
            # settle bets whose kickoff date + 1 day <= this batch date
            still = []
            for ob in open_bets:
                if ob["settle_at"] <= dday:
                    settled += ob["pnl"]
                else:
                    still.append(ob)
            open_bets = still
            open_stake = sum(ob["stake"] for ob in open_bets)
            available = settled - open_stake
            rows = []
            for r in batch.itertuples():
                qdict = exec_quote(r.match_id, r.season, pol)
                pr = pred_by.loc[(r.match_id, model_id)] if (r.match_id, model_id) in pred_by.index else None
                probs = None if pr is None else [pr.p_home, pr.p_draw, pr.p_away]
                d = decide_match(probs, qdict, r.kickoff_utc, pol, available_bankroll=available)
                rows.append((r, d))
            rows.sort(key=lambda t: (-(t[1]["ev_stressed"] or -9), t[0].match_id))
            cap_total = pol["max_open_exposure_frac"] * settled
            for r, d in rows:
                stake, code, codes = d["proposed_stake"], d["reason_code"], list(d["reason_codes_all"])
                if d["action"] == "BET" and open_stake + stake > cap_total + 1e-9:
                    stake, code, codes = 0.0, "EXPOSURE_LIMIT_TOTAL", ["EXPOSURE_LIMIT_TOTAL"]
                action = "BET" if code == "ACCEPTED" else "NO_BET"
                pnl = clv_v = cfp = None
                if action == "BET":
                    open_stake += stake
                    k = M.OUTCOMES.index(d["selection"])
                    pnl = stake * (d["odds_offered"] - 1) if r.result == d["selection"] else -stake
                    open_bets.append({"stake": stake, "pnl": pnl, "settle_at": r.kickoff_utc.floor("D") + timedelta(days=1)})
                    if r.match_id in close_ref.index:
                        c = close_ref.loc[r.match_id]
                        cfp = float([c.q_home, c.q_draw, c.q_away][k])
                        clv_v = M.clv(d["odds_offered"], cfp)
                ledger.append(dict(
                    decision_id=f"{policy_id}-{model_id}-{r.match_id}", run_id="golden", policy_id=policy_id,
                    match_id=r.match_id, model_id=model_id, split=SPLIT_OF[r.season],
                    decision_time_utc=r.feature_cutoff_utc, batch_key=r.matchweek_key,
                    quote_id=d["quote_id"], action=action, selection=d["selection"] if action == "BET" else None,
                    bookmaker=d["bookmaker"] if action == "BET" else None,
                    p_model=d["p_model"], p_stressed=d["p_stressed"], odds_offered=d["odds_offered"],
                    odds_adjusted=d["odds_adjusted"], ev_raw=d["ev_raw"], ev_stressed=d["ev_stressed"],
                    stake=float(stake), reason_code=code, reason_codes_all="|".join(codes),
                    explanation=d["explanation"] if code == d["reason_code"] else "No bet: the total open exposure limit is already used.",
                    settled_pnl=pnl, closing_fair_prob=cfp, clv=clv_v, config_hash=cfg_hash, is_exploratory=False))
ledger = pd.DataFrame(ledger)
for c in ("p_model", "p_stressed", "odds_offered", "odds_adjusted", "ev_raw", "ev_stressed",
          "settled_pnl", "closing_fair_prob", "clv"):
    ledger[c] = ledger[c].astype(float)
S.write_table(ledger, TAB / "decisions.parquet", "decisions")
ledger.head(200).to_csv(TAB / "decisions_sample.csv", index=False)

# ------------------------------------------------------------ golden policy metrics (Part 2 output shape)
def policy_summary(g, label):
    bets = g[g.action == "BET"]
    turnover = float(bets.stake.sum())
    profit = float(bets.settled_pnl.sum())
    by_week = bets.groupby("batch_key").agg(stake=("stake", "sum"), pnl=("settled_pnl", "sum"))
    # bootstrap ROI by matchweek blocks
    brng = np.random.default_rng(0)
    rois = []
    if len(by_week) > 1:
        for _ in range(2000):
            s_ = by_week.sample(len(by_week), replace=True, random_state=int(brng.integers(1e9)))
            if s_.stake.sum() > 0:
                rois.append(s_.pnl.sum() / s_.stake.sum())
    ci = [float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))] if rois else [None, None]
    rets = (by_week.pnl / 1000.0)
    sharpe = float(rets.mean() / rets.std()) if len(rets) > 1 and rets.std() > 0 else None
    return {"policy_id": g.policy_id.iloc[0], "label": label,
            "n_decisions": int(len(g)), "n_bets": int(len(bets)), "n_abstain": int((g.action == "NO_BET").sum()),
            "turnover": turnover, "profit": profit, "roi": profit / turnover if turnover else None,
            "max_drawdown": None, "mean_clv": float(bets.clv.mean()) if bets.clv.notna().any() else None,
            "sharpe_per_matchweek": sharpe, "deflated_sharpe": None,
            "reason_counts": g.reason_code.value_counts().to_dict(),
            "bootstrap_ci_95": {"roi": ci}}


labels = {"naive": "EV > 0 at Bet365 odds, no checks, flat stakes",
          "controlled": "Bet365 prices; threshold + stress + data checks + exposure caps",
          "controlled_best_price": "Best price across bookmakers; same checks and caps"}
pm_rows = []
for (model_id, split), g in ledger.groupby(["model_id", "split"]):
    pols, curve = [], []
    for pid, gp in g.groupby("policy_id"):
        s_ = policy_summary(gp, labels[pid])
        bets = gp[gp.action == "BET"].groupby("batch_key").agg(t=("decision_time_utc", "min"), pnl=("settled_pnl", "sum")).sort_values("t")
        bank, peak, mdd = 1000.0, 1000.0, 0.0
        for bk, row in bets.iterrows():
            bank += row.pnl
            peak = max(peak, bank)
            mdd = max(mdd, peak - bank)
            curve.append({"batch_key": bk, "time_utc": row.t.isoformat().replace("+00:00", "Z"),
                          "policy_id": pid, "bankroll": round(bank, 4), "drawdown": round(peak - bank, 4)})
        s_["max_drawdown"] = mdd
        pols.append(s_)
    pm_rows.append({"model_id": model_id, "split": split, "is_locked_holdout": False,
                    "policies": pols, "equity_curve": curve})
policy_metrics = {"dataset_mode": "SYNTHETIC", "contract_version": S.CONTRACT_VERSION,
                  "note": "Golden example. Bankroll curve is the settled bankroll per batch.",
                  "rows": pm_rows}
(ART / "policy_metrics.json").write_text(json.dumps(policy_metrics, indent=2, default=str))

# ------------------------------------------------------------ golden ML explainability (Part 2 output shape)
FEATS = ["dc_logratio_home", "dc_logratio_away", "rest_days_diff", "elo_diff", "q_draw", "fav_log_q", "league_E0"]
feature_importance = {"dataset_mode": "SYNTHETIC", "rows": [
    {"model_id": "lgbm", "model_version": "golden-stub-1", "method": "mean_abs_shap",
     "split": "validation", "features": [{"feature": f, "importance": float(v)}
                                         for f, v in zip(FEATS, sorted(rng.uniform(0.001, 0.05, len(FEATS)), reverse=True))]},
    {"model_id": "marketnet", "model_version": "golden-stub-1", "method": "mean_abs_adjustment_logit",
     "split": "validation", "features": [{"feature": f, "importance": float(v)}
                                         for f, v in zip(FEATS, sorted(rng.uniform(0.0, 0.02, len(FEATS)), reverse=True))]}]}
(ART / "feature_importance.json").write_text(json.dumps(feature_importance, indent=2))

expl = []
for mid_ in preds[preds.model_id == "lgbm"].match_id.head(60):
    for lab in M.OUTCOMES:
        vals = rng.normal(0, 0.02, 3)
        for rank, (f, v) in enumerate(sorted(zip(rng.choice(FEATS, 3, replace=False), vals), key=lambda t: -abs(t[1])), 1):
            expl.append(dict(match_id=mid_, model_id="lgbm", model_version="golden-stub-1", outcome=lab,
                             rank=rank, feature=str(f), feature_value=float(rng.normal()), contribution=float(v)))
expl = pd.DataFrame(expl)
S.write_table(expl, TAB / "prediction_explanations.parquet", "prediction_explanations")
expl.to_csv(TAB / "prediction_explanations.csv", index=False)

# ------------------------------------------------------------ frozen config (Part 1 writes the real one)
frozen = {
    "contract_version": S.CONTRACT_VERSION, "frozen_at_utc": None, "model_id": "lgbm",
    "model_version": "golden-stub-1", "policy": dict(DEFAULT_POLICY),
    "chosen_using": ["validation"], "n_trials_logged": 1,
}
frozen["config_hash"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:12]
(ART / "frozen_config.json").write_text(json.dumps(frozen, indent=2))

# ------------------------------------------------------------ golden holdout_locked (Part 2 output shape)
hl_row = next(r for r in pm_rows if r["model_id"] == frozen["model_id"] and r["split"] == "holdout")
holdout_locked = {"dataset_mode": "SYNTHETIC", "contract_version": S.CONTRACT_VERSION,
                  "config_hash": frozen["config_hash"], "locked_at_utc": None,
                  "model_id": frozen["model_id"], "model_version": frozen["model_version"],
                  "split": "holdout", "is_locked_holdout": True,
                  "forecast": [r for r in fm if r["split"] == "holdout"],
                  "policies": hl_row["policies"], "equity_curve": hl_row["equity_curve"],
                  "note": "Golden example only. The real file is written once at the freeze."}
(ART / "holdout_locked.json").write_text(json.dumps(holdout_locked, indent=2, default=str))

# ------------------------------------------------------------ API examples for the UI (Part 3)
iso = lambda t: None if t is None or pd.isna(t) else pd.Timestamp(t).isoformat().replace("+00:00", "Z")
ex = mo_all[mo_all.season == "2024-25"].iloc[40]
mid = ex.match_id
qrow = quotes[(quotes.match_id == mid) & (quotes.snapshot == "PRE_CLOSING") & (quotes.bookmaker == S.REFERENCE_BOOKMAKER)].iloc[0]
qdict = reference_quote(pre_quotes[mid])
pr = pred_by.loc[(mid, "lgbm")]
probs = [pr.p_home, pr.p_draw, pr.p_away]
d = decide_match(probs, qdict, ex.kickoff_utc, DEFAULT_POLICY)
r6 = lambda x: None if x is None else round(float(x), 6)
meta = {"contract_version": S.CONTRACT_VERSION, "dataset_mode": "SYNTHETIC",
        "storage_backend": "local", "gemini_enabled": False,
        "leagues": sorted(LEAGUE_TEAMS), "seasons": SEASONS, "models": ["market", "elo", "lgbm"],
        "frozen_config_hash": frozen["config_hash"], "holdout_locked": False}
fixtures = [{"match_id": r.match_id, "league": r.league, "season": r.season,
             "kickoff_utc": iso(r.kickoff_utc), "home_team": r.home_team, "away_team": r.away_team,
             "split": SPLIT_OF[r.season], "has_closing": True}
            for r in matches[matches.season == "2024-25"].head(8).itertuples()]
replay = {
    "dataset_mode": "SYNTHETIC", "is_exploratory": False, "config_hash": frozen["config_hash"],
    "match": {"match_id": mid, "league": ex.league, "season": ex.season, "kickoff_utc": iso(ex.kickoff_utc),
              "home_team": ex.home_team, "away_team": ex.away_team, "split": SPLIT_OF[ex.season]},
    "information_cutoff_utc": iso(ex.feature_cutoff_utc),
    "offered_quote": {"quote_id": qrow.quote_id, "bookmaker": qrow.bookmaker, "snapshot": "PRE_CLOSING",
                      "observed_at_utc": iso(qrow.observed_at_utc), "timestamp_precision": qrow.timestamp_precision,
                      "source": qrow.source, "odds": qdict["odds"], "bookmakers": qdict["bookmakers"]},
    "price_source": "REFERENCE", "policy_id": "controlled",
    "market_probs": [r6(x) for x in M.implied_probs(qdict["odds"])[0]],
    "model": {"model_id": "lgbm", "model_version": "golden-stub-1", "probs": [r6(x) for x in probs]},
    "per_outcome": [{k: (r6(v) if isinstance(v, float) else v) for k, v in row.items()} for row in d["per_outcome"]],
    "decision": {"action": d["action"], "selection": d["selection"], "stake": d["proposed_stake"], "bookmaker": d["bookmaker"],
                 "reason_code": d["reason_code"], "reason_codes_all": d["reason_codes_all"],
                 "explanation": d["explanation"]},
    "top_features": [{"outcome": e.outcome, "rank": int(e.rank), "feature": e.feature,
                      "feature_value": r6(e.feature_value), "contribution": r6(e.contribution)}
                     for e in expl[expl.match_id == mid].itertuples()],
    "config_used": dict(DEFAULT_POLICY),
}
cl = quotes[(quotes.match_id == mid) & (quotes.snapshot == "CLOSING") & (quotes.bookmaker == qrow.bookmaker)].iloc[0]
lrow = ledger[(ledger.match_id == mid) & (ledger.model_id == "lgbm") & (ledger.policy_id == "controlled")].iloc[0]
reveal = {"match_id": mid,
          "closing_quote": {"quote_id": cl.quote_id, "bookmaker": cl.bookmaker, "snapshot": "CLOSING",
                            "observed_at_utc": iso(cl.observed_at_utc), "timestamp_precision": cl.timestamp_precision,
                            "source": cl.source, "odds": [cl.odds_home, cl.odds_draw, cl.odds_away]},
          "closing_fair_probs": [r6(x) for x in M.implied_probs([cl.odds_home, cl.odds_draw, cl.odds_away])[0]],
          "result": {"home_goals": int(ex.home_goals), "away_goals": int(ex.away_goals), "result": ex.result},
          "clv": r6(lrow.clv) if pd.notna(lrow.clv) else None,
          "settled_pnl": r6(lrow.settled_pnl) if pd.notna(lrow.settled_pnl) else None}
eff_api = {"dataset_mode": "SYNTHETIC",
           "sign_conventions": {"fav_longshot_gap": "negative = longshots win less often than priced (classic favourite-longshot bias)",
                                "draw_bias": "positive = draws happen more often than priced",
                                "calib_slope": "1 = calibrated, below 1 = market overconfident"},
           "rows": json.loads(eff.to_json(orient="records"))}
pm_val = next(r for r in pm_rows if r["model_id"] == "lgbm" and r["split"] == "validation")
evidence = {"dataset_mode": "SYNTHETIC", "split": "validation", "model_id": "lgbm", "is_locked_holdout": False,
            "forecast": [r for r in fm if r["split"] == "validation"],
            "policies": pm_val["policies"], "equity_curve": pm_val["equity_curve"],
            "feature_importance": feature_importance["rows"],
            "limitations": ["SYNTHETIC golden data: these numbers are placeholders, not results."]}
scenario_defs = [
    ("likely_winner_negative_ev", "Likely winner, but the price is too short", [0.52, 0.28, 0.20], [1.90, 3.60, 4.80], "WINDOW"),
    ("stale_quote", "Apparent value, unverified quote time", [0.30, 0.28, 0.42], [3.10, 3.50, 2.80], "UNKNOWN"),
    ("survives_stress", "Edge survives the stress test", [0.30, 0.28, 0.42], [3.10, 3.50, 2.80], "WINDOW"),
]
scenarios = {"label": "DEMONSTRATION SCENARIOS (constructed inputs, not historical findings)", "scenarios": []}
for sid, title, p_, o_, prec in scenario_defs:
    dd = decide_match(p_, qd(o_, precision=prec), KO, DEFAULT_POLICY)
    scenarios["scenarios"].append({"id": sid, "title": title, "p_model": p_, "odds": o_,
                                   "timestamp_precision": prec,
                                   "decision": {"action": dd["action"], "selection": dd["selection"],
                                                "stake": dd["proposed_stake"], "reason_code": dd["reason_code"],
                                                "reason_codes_all": dd["reason_codes_all"],
                                                "explanation": dd["explanation"]},
                                   "per_outcome": [{k: (r6(v) if isinstance(v, float) else v) for k, v in row.items()}
                                                   for row in dd["per_outcome"]]})
explain_api = {"available": False, "text": None,
               "reason": "Gemini not configured. The UI must still work and must not show it as active."}
for name, obj in (("meta", meta), ("fixtures", fixtures), ("replay", replay), ("reveal", reveal),
                  ("efficiency_map", eff_api), ("evidence", evidence), ("scenarios", scenarios),
                  ("audit", audit), ("explain", explain_api)):
    (API / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))

print("golden files written:", len(matches), "matches,", len(quotes), "quotes,", len(preds),
      "predictions,", len(ledger), "decisions")
