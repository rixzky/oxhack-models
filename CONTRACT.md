# CONTRACT.md (shared by all three repos, contract v1.2.0)

This file is identical in all three repos. It is the only place where cross-part
decisions live. If your repo's spec and this file disagree, THIS FILE WINS.
If this file is silent on something that affects another part, do not guess:
ask the contract owner, who updates this file, bumps the version and reposts it.

Project name: **not decided**. Use "the project" everywhere. Do not invent a name,
logo text or package name containing a product name. Repo names are
`oxhack-data`, `oxhack-models`, `oxhack-app`.

---

## 1. What we are building (the agreed plan, do not change scope)

**Question:** Where are bookmaker odds wrong, and does an apparent edge survive
realistic prices, data checks and errors in our own model?

**Target:** HRT "Best use of Data for Predictions & Decision Making" at OxHack '26.
Side prizes: Tiger Data, Gemini, DEV Writing Challenge. Paper trading only.

**Data:** football-data.co.uk, 5 to 10 leagues (section 4.1). Decision time is the
pre-closing snapshot. Bet365 is the reference market for every season; a second
policy executes at the best valid price across individual bookmakers (line shopping). Closing odds are used only for CLV and evaluation, never as
features. Locked holdout: the 2025-26 season.

**Findings and models**
1. Efficiency map (hero): league x season calibration of margin-free market
   probabilities, favourite-longshot bias, draw mispricing. Past seasons only when
   used for decisions.
2. Baselines: market, Elo.
3. Team strength: time-decayed Dixon-Coles.
4. MarketNet: a correction to the market in log-odds, starting at zero, with an L2
   penalty. Inputs: Dixon-Coles output, rest days, league, favourite-longshot gap.
   No closing odds.
5. LightGBM: a planned ML challenger on the same leakage-safe features, with SHAP
   feature importance (global and per match) for the UI. MarketNet is also ML (a learned,
   penalised correction trained by gradient descent). Whichever model wins on
   validation is frozen for the holdout. ML must earn its place: if no model beats the
   market on validation, we say so.

**Decisions:** EV at the offered odds, minimum EV threshold, probability stress test
(p minus delta), freshness and price checks, flat stakes first, then quarter Kelly
with exposure caps. Visible "no bet" reasons.

**Evaluation:** log loss, Brier, calibration plot, CLV (only where closing odds
exist), ROI, drawdown, deflated Sharpe, bootstrap by matchweek. Evaluate the holdout
once. Report negative results.

**Demo (3 screens):** efficiency map; match replay with model vs market and the
decision explained; evidence (calibration, equity curve, baselines). A data-audit
panel may sit inside the evidence screen. Gemini explains an existing decision record
only. Tiger Data stores quotes and the ledger, with a local fallback. Frontend: React.

**Cut (do not build):** state-space model, market-maker flip, Understat xG (stretch
only, Part 2 decides after core is done), injury news, in-play, arbitrage, Polymarket.

**Timeline (all parts)**
- Sat 16:00: clean data loaded; market and Elo predictions scored; decision engine
  passing the test vectors; UI shows all 3 screens from golden files.
- Sat 22:00: efficiency map done; Dixon-Coles working; first real backtest and policy
  metrics on validation; UI on real files.
- Sun 03:00: MarketNet and LightGBM trained walk-forward; full evaluation running.
- Sun morning: freeze config (section 8), run the holdout exactly once.
- After that: demo, pitch, write-up only. No new features.

---

## 2. Parts, ownership and data flow

| Part | Repo | Owns | Produces (into `artifacts/`) | Consumes |
|---|---|---|---|---|
| 1 Data and Decisions | `oxhack-data` | ingestion, cleaning, team names, timestamps, market probabilities, efficiency map, data audit, splits, Tiger loader; backtest (batching, bankroll, settlement) using `contract/decisions.py`; freeze file | `matches`, `outcomes`, `quotes`, `market_probs`, `efficiency_map`, `decisions` (parquet); `manifest.json`, `splits.json`, `data_audit.json`, `data_manifest.md`, `frozen_config.json` | raw CSVs, Part 2 predictions |
| 2 Models and Evaluation (statistics and ML) | `oxhack-models` | Elo, Dixon-Coles, MarketNet, LightGBM, calibration, SHAP, forecast metrics, policy metrics (ROI, drawdown, CLV, Sharpe, deflated Sharpe, bootstrap), locked holdout result | `predictions`, `prediction_explanations` (parquet); `forecast_metrics.json`, `feature_importance.json`, `policy_metrics.json`, `holdout_locked.json`, `model_card.md`, `trials_log.jsonl` | Part 1 artifacts |
| 3 App (UI) | `oxhack-app` | FastAPI that READS the artifact files, React UI, Gemini explainer, demo script | the app (no analysis files) | all artifacts |

```
raw football-data CSVs
   -> [Part 1] matches, outcomes, quotes, market_probs, efficiency_map, splits, audit
        -> [Part 2] predictions, explanations, forecast metrics
             -> [Part 1] decisions ledger (backtest, naive + controlled)
                  -> [Part 2] policy metrics, holdout_locked
                       -> [Part 3] FastAPI reads files -> React UI
```

Part 3 does **no** statistics and **no** data processing. It reads files and displays
them. The only calculation it runs is `contract.decisions.decide_match` when a user
moves a slider in the replay screen (exploratory, clearly labelled).

Parts exchange **files, not code**. Nobody imports another part's code. The only
shared code is the `contract/` folder.

**Contract owner:** one named person (write their name here: `__________`).
Only they edit `contract/` or this file.

---

## 3. Environment (pinned, all parts)

- Python 3.11. Core: `pandas>=2.2`, `numpy>=1.26`, `pyarrow>=15`, `pytest`.
- Part-specific extras go in that part's `requirements.txt` (e.g. scipy, scikit-learn,
  torch, lightgbm, fastapi, uvicorn, psycopg). The merged repo uses the union.
- Node 20+, React 18, TypeScript, Vite. Styling and chart libraries are free choice.
- File format between parts: **parquet** (pyarrow). CSV copies only for eyeballing.
- Paths: every script reads `ARTIFACTS_DIR` (default `./artifacts`) and never hardcodes
  absolute paths. Every part keeps the same top-level layout (section 10).
- Secrets only via environment variables: `DATABASE_URL`, `GEMINI_API_KEY`.
  Never commit them. Each repo ships `.env.example`.

---

## 4. Global conventions

- **Outcome order is always H, D, A** (home win, draw, away win), in arrays, columns
  and UI. Result codes are exactly `"H"`, `"D"`, `"A"`.
- **Odds** are decimal (stake included). A valid quote has all three odds finite and
  > 1.0.
- **Probabilities** are floats in [0, 1]; each triple sums to 1 within 1e-6.
- **Times** are tz-aware UTC (`datetime64[us, UTC]` in parquet, ISO 8601 with `Z`
  in JSON). Never naive datetimes, never local time after ingestion.
- **Season** strings: `"2023-24"`. **League** codes: football-data codes
  (`E0`, `E1`, `SP1`, `D1`, `I1`, `F1`, `N1`, `P1`).
- **Team names**: one canonical name per team, chosen by Part 1 in
  `data_pipeline/team_aliases.csv`. Everyone uses the canonical names in the tables.
- **match_id**: `{league}_{season}_{YYYYMMDD of kickoff UTC}_{home_slug}_{away_slug}`,
  built ONLY with `contract.schemas.make_match_id`.
- **matchweek_key**: league + ISO year-week of kickoff (`E0_2023-W32`), built ONLY
  with `contract.schemas.matchweek_key`. It is the bootstrap unit and batch unit.
- **Money**: paper "units". Initial bankroll 1000 units.
- **Shared maths**: implied probabilities, EV, stress, adverse odds, Kelly, log loss,
  Brier, CLV all come from `contract/maths.py`. Do not reimplement them.
- **Extra columns** in any table must be prefixed `x_` (validators allow them).

### 4.1 Source facts (football-data.co.uk) and how we map them

Verify these against the site's `notes.txt` and terms on Saturday morning (Part 1).
Until verified, treat them as assumptions.

- No opening odds exist. The early snapshot is a **pre-closing** capture (believed to
  be collected around Friday afternoon for weekend games and Tuesday for midweek).
  We call it `PRE_CLOSING`. Never call anything "opening".
- Closing odds columns (e.g. `PSCH/PSCD/PSCA`, `B365CH/B365CD/B365CA`) exist only for
  recent seasons (believed 2019-20 onwards). CLV is computed only where they exist.
- Pinnacle pre-closing `PSH/PSD/PSA` exist from roughly 2012-13. Bet365 `B365H/D/A`
  go back further. Other individual bookmakers (`BW`, `IW`, `WH`, `VC`) appear for
  some seasons; Part 1 ingests every individual bookmaker present, per season.
  Aggregate columns (`Max*`, `Avg*`, older `BbMx*`, `BbAv*`) are stored as `MAX`/`AVG`
  for the efficiency map only and are never executable.
- The provider warns that Pinnacle prices are systematically stale from 2025-07-23.
  Pinnacle rows are kept for audit and the efficiency map but are never executable in
  2025-26 (`schemas.EXECUTION_EXCLUDE`).
- Leagues: start with `E0` only for the first slice, then add up to the full list.
  Seasons: `2005-06` to `2025-26`. Do not ingest 2026-27.
- **Season assignment**: the source file's season wins. Date sanity bounds come from
  `schemas.season_date_bounds` (Jul 1 to Jun 30, except 2019-20 which runs to
  2020-08-31 because of fixtures replayed after the COVID suspension). Count the
  exception rows in the manifest; reject any other out-of-bounds row.
- **Reference quote rule** (one per match per snapshot, `is_reference = True`):
  **Bet365 (`B365`) for every season** (`schemas.REFERENCE_BOOKMAKER`). It is the
  probability benchmark, the model input for market features, the executable price
  for the `naive` and `controlled` policies, and the CLV benchmark (Bet365 closing).
  If the Bet365 quote is missing or invalid there is no reference for that match.
- **Best available price** (policy `controlled_best_price` only): built by
  `contract.decisions.best_price_quote` from the match's valid PRE_CLOSING quotes of
  individual, non-excluded, non-outlier bookmakers, per outcome, with a stale-price
  guard (a price is ignored if its margin-free probability differs from Bet365's by
  more than `best_price_max_prob_gap`, default 0.10). Never used as a probability
  benchmark: `q_market` is always Bet365.
- **Timestamp precision** of football-data quotes:
  `PRE_CLOSING` -> `observed_at_utc` = the **latest plausible capture time** (the end of
  the documented capture window, e.g. Friday evening for weekend games), precision
  `WINDOW`; if no window can be justified, `observed_at_utc` null, precision `UNKNOWN`.
  `CLOSING` -> `observed_at_utc` = kickoff_utc, precision `WINDOW`.
  Kickoff with a time column -> `EXACT`; date only -> 00:00 UTC of that date, `DATE`.
- **Kick-off timezone (assumption)**: the `Time` column is treated as `Europe/London`
  for all leagues. This only affects date-level cutoffs and batch dates; the 2-day
  buffer absorbs small shifts. Part 1 reports in the audit how many kick-offs fall
  within 2 hours of midnight UTC (the only ones a wrong timezone could move).

---

## 5. Time and leakage rules (the part judges will attack first)

- **feature_cutoff_utc** = kickoff date floored to 00:00 UTC, minus 2 days
  (`contract.schemas.feature_cutoff`). A feature for a match may only use matches with
  `kickoff_utc < feature_cutoff_utc`. This is deliberately conservative so that it is
  safe for both weekend and midweek capture times.
- **decision_time_utc** in `predictions` = `feature_cutoff_utc`. It is the
  **information time**: the forecast uses information up to that time only. The bet is
  **executed later** at the PRE_CLOSING quote. Ignoring information between the cutoff
  and the quote makes us more conservative, not less, so this is not leakage. What
  would be leakage, and is blocked: features using anything after the cutoff (validator
  and Part 2 tests), and executing at a price that may have been captured at or after
  kickoff (`decide_match` rejects `observed_at_utc >= kickoff_utc` as `STALE_QUOTE`,
  using the conservative latest-capture time from 4.1).
- **training_cutoff_utc** <= `feature_cutoff_utc` for every prediction row
  (validator enforces it). Walk-forward: a model may only be fitted on matches that
  kicked off before its training cutoff.
- **Closing odds and the match result never enter features, model selection inputs
  for that match, or the replay payload before "reveal".**
- Opening-to-closing movement is NOT a feature (it needs closing odds).
- The efficiency map may feed decisions (league gate) only using seasons strictly
  before the match's season.
- Preprocessing, calibration and penalty strength are fitted on earlier data only.
- Outcomes live in their own table. Exclusions in the audit are decided without
  looking at outcomes.

---

## 6. Decision rules (everyone must agree)

Sections 6.1, 6.2 and 6.4 are already implemented in `contract/decisions.py`
(`decide_match`), tested against `contract/golden/test_vectors.json`. Part 1 calls it
per match and implements 6.3 (batching, bankroll, caps, settlement) on top. Part 3
calls it only for exploratory slider changes. Nobody reimplements it.

### 6.1 Definitions
For outcome k with model probability `p_k` and offered decimal odds `O_k` from the
executable `PRE_CLOSING` quote (Bet365 for `price_source = REFERENCE`, best available
for `BEST_AVAILABLE`, see 4.1):
- adjusted odds `O'_k = 1 + (O_k - 1) * (1 - price_haircut)` (adverse-price assumption)
- stressed probability `p'_k = max(0, p_k - prob_stress_delta)` (a **sensitivity
  assumption**, not a confidence interval; label it that way in the UI)
- raw EV `ev_raw_k = p_k * O'_k - 1`; stressed EV `ev_stressed_k = p'_k * O'_k - 1`
- "edge" (`p_k - q_k`) is never a decision input. Regression test: p = 0.52 at
  O = 1.90 gives EV = -0.012 and must be rejected even though q = 0.50.

### 6.2 Per-match checks, in this order (first failure is `reason_code`; all failures go in `reason_codes_all`)
1. `MISSING_PREDICTION`: no prediction row for the chosen model.
2. `MISSING_QUOTE`: no reference `PRE_CLOSING` quote.
3. `INVALID_PRICE`: reference quote `is_valid = False`.
4. `STALE_QUOTE`: precision not in `allow_precisions` (default EXACT, DATE, WINDOW,
   so `UNKNOWN` fails), or `observed_at_utc >= kickoff_utc`, or
   `kickoff_utc - observed_at_utc > max_quote_age_hours`. For a best-price quote the
   worst precision and latest observed time of its components are used.
5. `LEAGUE_NOT_ELIGIBLE`: only if `league_gate = True` (past-season efficiency only).
6. Select the single outcome with the highest `ev_raw` (one position per match max).
   `EV_BELOW_THRESHOLD` if `ev_raw < min_ev`.
7. `FAILS_STRESS` if `ev_stressed < min_ev`.
8. Otherwise the match is a candidate (`ACCEPTED` unless sizing rejects it).

### 6.3 Sizing and bankroll
- Batches: all candidates sharing the same decision date (date of `decision_time_utc`)
  form one batch and are sized against the same **available bankroll** = settled
  bankroll at batch start minus stakes of every bet still open (from any earlier
  batch). Unsettled winnings are never reused. A bet settles at its kickoff date + 1 day,
  so settlement events are processed before any batch dated later than that.
- Order inside a batch: `ev_stressed` descending, then `match_id` ascending.
- `FLAT`: stake = `flat_stake_frac * initial_bankroll`.
- `QUARTER_KELLY` (only after flat accounting is tested): stake =
  `available * kelly_multiplier * full_kelly(p'_k, O'_k)`.
- Caps: stake <= `max_stake_frac_per_match * available` (else reduce to the cap; if the
  cap is 0 -> `EXPOSURE_LIMIT_MATCH`); total open stakes (earlier unsettled bets plus
  this batch so far) <= `max_open_exposure_frac * settled bankroll at batch start` (bets that would exceed it get
  `EXPOSURE_LIMIT_TOTAL`); stake > remaining available -> `INSUFFICIENT_BANKROLL`.
- Settlement after the match: pnl = stake * (O_k - 1) if win, else -stake, at the
  executed bookmaker's OFFERED odds (column `bookmaker` in the ledger), not the adjusted
  odds (the haircut is a decision assumption).
- Defaults (`frozen_config.json` policy block): `min_ev 0.02`, `prob_stress_delta 0.02`,
  `price_haircut 0.0`, `FLAT`, `flat_stake_frac 0.01`, `kelly_multiplier 0.25`,
  `max_stake_frac_per_match 0.02`, `max_open_exposure_frac 0.20`,
  `initial_bankroll 1000`, `max_quote_age_hours 120`, `league_gate false`.

### 6.4 Explanations
`explanation` is built from reason codes and numbers by the deterministic templates
in `contract/decisions.py` (`explain`). Part 1 overwrites it only for
`EXPOSURE_LIMIT_TOTAL` and `INSUFFICIENT_BANKROLL`, which are decided in 6.3. Gemini may rephrase an existing decision record but must never produce
probabilities, stakes, odds or news.

---

## 7. Evaluation definitions

- **Log loss**: mean natural-log loss of the realised outcome (`maths.log_loss_row`).
- **Brier**: SUM convention over 3 outcomes, range 0 to 2 (`maths.brier_row`).
  Always label it "Brier (sum over 3 outcomes)".
- **Calibration bins**: pool all 3 outcome probabilities, 10 equal-width bins on
  [0, 1], report `n`, mean predicted, observed frequency per bin.
- **Forecast comparisons** use the SAME eligible matches for every model (inner join on
  match_id within the split).
- **CLV** (per bet, only where a Bet365 closing quote exists):
  `O_offered * q_close - 1`, q_close = margin-free Bet365 closing probability of the
  same outcome (`maths.clv`), for every policy including best price. A benchmark, not
  proof.
- **Policy metrics**: n decisions, n bets, n abstentions, reason counts, turnover
  (sum of stakes), profit, ROI = profit / turnover, max drawdown of the settled
  bankroll curve, mean CLV.
- **Sharpe per matchweek**: mean / std of per-batch returns (batch profit / available
  bankroll at batch start), not annualised. **Deflated Sharpe** (Bailey and
  Lopez de Prado) using the number of trials in `trials_log.jsonl` (models and policy
  settings tried, logged by Parts 1 and 2).
- **Uncertainty**: 95% bootstrap intervals resampling **matchweek_key** blocks, 2000
  resamples, seed 0, for ROI and for log-loss differences vs the market.
- Three policies are always compared on the same matches (`contract.decisions.POLICIES`):
  `naive` (EV > 0 at raw Bet365 odds, no checks, flat stakes), `controlled` (Bet365
  prices, all of section 6) and `controlled_best_price` (same checks and caps, best
  available price). **Line-shopping value** = ROI of `controlled_best_price` minus
  `controlled`, with a matchweek-block bootstrap interval, reported next to the
  difference in number of bets and turnover. Best-of-many prices are biased upwards by
  stale or wrong quotes; say so wherever the number is shown.
- **Who computes what:** Part 2 computes forecast metrics (from predictions and
  outcomes) and policy metrics (from Part 1's `decisions.parquet`). Part 3 never
  recomputes either. Report
  turnover next to every return; abstaining less is not a success by itself.

---

## 8. Splits, freeze and holdout protocol

`splits.json` (Part 1) assigns every season a role:
`warmup` 2005-06 to 2009-10 (rating burn-in, never scored), `train` 2010-11 to
2022-23, `calibration` 2023-24, `validation` 2024-25, `holdout` 2025-26.

- Everything (features, model choice, MarketNet penalty, calibration, policy
  thresholds) is chosen using seasons up to and including `validation`.
- Before the freeze, nobody computes or looks at metrics on the holdout season:
  Part 1 leaves 2025-26 out of `efficiency_map.parquet` and does not backtest it,
  Part 2 does not write holdout predictions or metrics, Part 3 shows nothing from it. Holdout matches, quotes and outcomes
  are still ingested and validated (data checks only).
- Every configuration tried is appended to `trials_log.jsonl` (one JSON per line:
  time, part, what changed, validation metrics). Part 1 logs policy settings, Part 2
  logs models. Merge both into one file at the freeze. This feeds the deflated Sharpe.
- **Freeze (Sun morning, whole team present):** Part 2 names the winning model_id and
  version on validation; Part 1 writes `frozen_config.json` (that model, the policy
  block, `config_hash` = first 12 hex chars of sha256 of the sorted JSON without the
  hash field). Commit it in all repos.
- **Holdout run, once, in order:** Part 1 regenerates the efficiency map including
  2025-26; Part 2 writes holdout predictions with the frozen model; Part 1 runs the
  frozen policy on the holdout (`run_id = "holdout_locked"`); Part 2 computes holdout
  forecast and policy metrics and writes `holdout_locked.json` (metrics + config_hash).
  It is never overwritten. Any later holdout run with different settings is
  `is_exploratory = True`, stored separately and labelled "exploratory" in the UI.

---

## 9. Honesty rules (all parts, all outputs)

- Never invent results, timestamps, sources, integrations or profitable performance.
- `dataset_mode` (`REAL` or `SYNTHETIC`) is carried in `manifest.json` and every API
  payload. Synthetic data is shown with a prominent "Synthetic demonstration" banner and
  is never mixed into real metrics.
- An integration (Tiger Data, Gemini) is shown as active only if it is configured AND
  working in that run.
- Demonstration scenarios are labelled as constructed inputs, never as findings.
- Negative or inconclusive results are reported as they are.

---

## 10. Repo layout and the `artifacts/` directory

Every repo has the same top level so that merging is copying folders:

```
<repo>/
  CONTRACT.md  INTEGRATION.md  SPEC.md  AGENTS.md  CLAUDE.md  pytest.ini
  contract/            identical in all repos (do not edit)
  artifacts/           gitignored; inputs from other parts are dropped here
  <part folders>/      Part 1: data_pipeline/ decisions/ | Part 2: modelling/ evaluation/ | Part 3: app/
  tests/               part-specific tests
  requirements.txt     .env.example     README.md
```

Artifact files (exact names, all in `artifacts/`):

| File | Producer | Schema |
|---|---|---|
| `matches.parquet` | 1 | `SCHEMAS["matches"]` |
| `outcomes.parquet` | 1 | `SCHEMAS["outcomes"]` |
| `quotes.parquet` | 1 | `SCHEMAS["quotes"]` |
| `market_probs.parquet` | 1 | `SCHEMAS["market_probs"]` |
| `efficiency_map.parquet` | 1 | `SCHEMAS["efficiency_map"]` |
| `manifest.json`, `splits.json`, `data_audit.json`, `data_manifest.md` | 1 | see golden examples |
| `decisions.parquet` | 1 | `SCHEMAS["decisions"]` |
| `frozen_config.json` | 1 | golden example |
| `predictions.parquet` | 2 | `SCHEMAS["predictions"]` |
| `prediction_explanations.parquet` | 2 (optional) | `SCHEMAS["prediction_explanations"]` |
| `forecast_metrics.json`, `feature_importance.json`, `policy_metrics.json` | 2 | golden examples |
| `holdout_locked.json` | 2 | same shape as one `policy_metrics.json` row plus forecast rows, `config_hash`, `locked_at_utc` |
| `model_card.md`, `trials_log.jsonl` | 2 (Part 1 appends policy trials) | free text / one JSON per line |

Golden (synthetic) examples of every file are in `contract/golden/`. Build and test
against them first; swap in real artifacts when they arrive.

Sharing: zip `artifacts/` and post it in the team channel or shared drive. Raw
football-data files are not committed to any repo.

---

## 11. API (Part 3 backend to Part 3 frontend)

Shapes: `contract/api_types.ts`. Examples: `contract/golden/api/*.json`. The backend
builds every response by reading artifact files (Part 3 SPEC has the exact
file-to-endpoint mapping) and, for slider changes only, calling `decide_match`.
Endpoints: `GET /api/meta`, `GET /api/efficiency-map`, `GET /api/fixtures`,
`GET /api/replay/{match_id}`, `GET /api/replay/{match_id}/reveal`,
`GET /api/evidence`, `GET /api/audit`, `GET /api/scenarios`,
`POST /api/explain/{match_id}`.

---

## 12. Change control

1. Need a change? Message the contract owner with the exact field or rule.
2. Owner edits `contract/` and/or this file, bumps `CONTRACT_VERSION` (patch for
   clarifications, minor for new optional columns, major for renamed or removed
   fields), regenerates golden files, reposts the zip.
3. Everyone replaces their `contract/` folder and this file wholesale and re-runs
   `python -m pytest contract -q`.
4. Log the change in section 13.

## 13. Change log
- 1.0.0: initial contract.
- 1.2.0: Bet365 is the reference for all seasons; best-price policy `controlled_best_price`
  with `best_price_quote`; `bookmaker` column in decisions; more bookmaker codes;
  Pinnacle not executable in 2025-26; quotes at or after kickoff rejected; WINDOW times
  store the latest plausible capture; 2019-20 season exception; timezone assumption
  documented (answers Q1 to Q4).
- 1.1.0: parts re-split (Part 1 Data and Decisions, Part 2 Models and Evaluation, Part 3 App/UI only); `contract/decisions.py` added; `policy_id` column in decisions; `prediction_explanations` table; LightGBM with SHAP is a planned model.

## 14. Open items
- football-data.co.uk terms permit our use; record in `data_manifest.md` (Part 1).
- Closing-odds season coverage per bookmaker from `notes.txt` (Part 1).
- HRT judging criteria and submission deadline (whole team).
- Resolved in 1.2.0: 2019-20 boundary, quote time vs decision time, stale Pinnacle,
  kick-off timezone.
