# SPEC: Part 2, Models and Evaluation (`oxhack-models`)

Read `CONTRACT.md` first. It wins over this file. You work in your own repo; you hand
off **artifact files**, not code (see `INTEGRATION.md`). Until Part 1's real files
arrive, build and test against `contract/golden/` (use `--golden`).

## 1. Mission

You own all the statistics and ML.
1. **Models:** leakage-free, calibrated home/draw/away probabilities from a ladder of
   models (market, Elo, Dixon-Coles, MarketNet, LightGBM), compared honestly against
   the market on the same matches. Two of them are ML: **MarketNet** (a learned,
   penalised correction to the market) and **LightGBM** (gradient-boosted trees with
   SHAP explanations).
2. **Evaluation:** every number the app shows: forecast metrics, and policy metrics
   computed from Part 1's decisions ledger (ROI, drawdown, CLV, Sharpe, deflated
   Sharpe, bootstrap intervals), and the locked holdout result.

"No model beats the market" is an acceptable, reportable result.

## 2. Inputs

From Part 1: `matches`, `outcomes`, `quotes`, `market_probs`, `splits.json`, and later
`decisions.parquet` and policy trials in `trials_log.jsonl`. Read with
`contract.schemas.read_table`. Model inputs use only `is_reference=True`,
`snapshot="PRE_CLOSING"` market rows. **Never read CLOSING rows or the outcome of the
match being predicted** when building its features.

## 3. You produce (into `artifacts/`)

| File | Notes |
|---|---|
| `predictions.parquet` | one row per match x model_id x model_version; splits train, calibration, validation (holdout only at the freeze) |
| `prediction_explanations.parquet` | top 3 contributions per match x outcome for `lgbm` (SHAP) and `marketnet` (W_k x_k terms), validation (and holdout at the freeze) |
| `feature_importance.json` | per model: global importance (mean abs SHAP for lgbm, mean abs adjustment for marketnet); golden example shape |
| `forecast_metrics.json` | per model x split: n, log_loss, brier_sum3, calibration_bins, same matches for every model in a split; golden example shape |
| `policy_metrics.json` | per model x split: the two policies' summaries and equity curve; golden example shape (= `PolicySummary` in `contract/api_types.ts`) |
| `holdout_locked.json` | once, at the freeze (CONTRACT 8) |
| `model_card.md` | per model: inputs, fitting, cutoffs, hyperparameters and how chosen, weaknesses |
| `trials_log.jsonl` | one line per model configuration tried (`"part": 2`) |

Code: `modelling/` (`run.py` entry point, one module per model, `features.py`) and
`evaluation/` (`run.py` entry point, `forecast.py`, `policy.py`, `bootstrap.py`,
`lock_holdout.py`), `tests/`.

## 4. Feature table (build once, used by MarketNet and LightGBM)

One row per match, all values known at `feature_cutoff_utc`:
market reference q (and log q) for H, D, A; Elo rating difference and home Elo;
Dixon-Coles probabilities, log-ratio to the market, expected goals; rest days home and
away (capped at 14); league one-hot; favourite-longshot features from q (log q of the
favourite, q_draw, overround); lagged form (points per game over last 5 before the
cutoff). **No closing odds, no odds movement, no post-cutoff information.**
Store it as `artifacts/x_features.parquet` if useful for debugging (not shared).

## 5. Model ladder (build in this order; each complete before the next)

1. **`market`** (P0, minutes): reference PRE_CLOSING q as the prediction. The benchmark.
2. **`elo`** (P0, by Sat 16:00): ratings with home advantage, goal-difference K
   multiplier, regression to the mean between seasons; ratings use only matches with
   `kickoff_utc < feature_cutoff_utc`. Map rating difference to H, D, A with a
   multinomial (or ordered) logistic regression fitted on `train`.
3. **`dixon_coles`** (P0, by Sat 22:00): Poisson goals with the Dixon-Coles low-score
   correction and exponential time decay (half-life chosen on `calibration`), per
   league, refit at least per matchweek batch (or monthly if slow) on matches before
   the training cutoff. Scoreline grid 0 to 10 summed to H, D, A. Fill `xg_home`,
   `xg_away`.
4. **`marketnet`** (P0, by Sun 03:00, ML): `p = softmax(log q + delta)`,
   `delta = W x` with the draw logit pinned to 0. W initialised at exactly zero (so the
   untrained model is the market), L2 penalty `lambda * ||W||^2`, lambda chosen on
   `calibration`, confirmed on `validation`. Train walk-forward by season (fit on all
   seasons before the predicted one). PyTorch, scipy or scikit-learn are all fine.
   Report the size of the learned adjustments by league.
5. **`lgbm`** (P0, by Sun 03:00, ML): LightGBM multiclass on the feature table,
   walk-forward by season, early stopping on `calibration`, conservative settings
   (shallow trees, strong regularisation; small data). Compare with and without the
   market features. SHAP values (TreeExplainer) for global importance and per-match
   top contributions.
6. Calibration (isotonic or Platt per outcome, renormalised) only if fitted on
   `calibration` and shown to help on `validation`. Log it as a trial.

At the freeze, recommend the model with the best validation log loss (ties: simpler
model) to Part 1, who writes `frozen_config.json`.

## 6. Time rules

- `decision_time_utc = feature_cutoff_utc` = the match's value from `matches`.
- `training_cutoff_utc` = latest kickoff any fitted parameter could have seen, and
  `<= feature_cutoff_utc`. For season-level refits use
  `min(first feature_cutoff of the season, feature_cutoff of the match)`.
- `model_version`: short string you bump when the model changes, e.g. `lgbm-1.2`.
- Warm-up seasons may be predicted with `split="warmup"`; never scored.

## 7. Evaluation definitions (CONTRACT section 7 is binding)

- **Forecast** (`evaluation/forecast.py`): log loss, Brier (sum convention),
  calibration bins, per split, all models on the inner-joined matches; also per league
  on validation; bootstrap 95% interval of the log-loss difference vs `market`
  (matchweek blocks, 2000 resamples, seed 0).
- **Policy** (`evaluation/policy.py`), from Part 1's `decisions.parquet`, per model x
  split x policy: n decisions, bets, abstentions, reason counts, turnover, profit,
  ROI, max drawdown of the settled bankroll curve, mean CLV (bets with closing quotes),
  Sharpe per matchweek (batch profit / available bankroll at batch start), deflated
  Sharpe (Bailey and Lopez de Prado, number of trials = lines in the merged
  `trials_log.jsonl`), bootstrap 95% interval for ROI by matchweek blocks. Equity
  curve: settled bankroll per batch, with drawdown.
- Report turnover next to returns. Do not call a lower-turnover policy "better" by
  itself.

## 8. Holdout

Before the freeze: no holdout predictions, no holdout metrics. At the freeze, in the
order of CONTRACT section 8: run `python -m modelling.run --include-holdout` for the
frozen model only; after Part 1 posts the holdout ledger, run
`python -m evaluation.lock_holdout`, which writes `holdout_locked.json` once and
refuses to overwrite it.

## 9. Tests you must write (`tests/`)

- Contract tests pass.
- Leakage: changing the result of a match (and of any match after its cutoff) leaves
  its prediction unchanged, for every model.
- Closing quotes are never read by feature code.
- MarketNet with W = 0 returns exactly the market probabilities.
- Probabilities valid for every model (validator).
- Metrics reproduce the `brier`, `log_loss` and `clv` vectors; policy metrics on the
  golden ledger match a hand calculation for one small split.
- `lock_holdout` refuses to overwrite an existing `holdout_locked.json`.

## 10. Do not

- Do not edit `contract/`. Extra columns need an `x_` prefix and the contract owner's
  agreement.
- Do not use closing odds, odds movement or post-cutoff information as features.
- Do not tune on `validation` and present it as untouched; the holdout is the only
  untouched set.
- Do not build the state-space model, injury features or anything on the cut list.
  No decision logic (Part 1) and no UI (Part 3).

## 11. Hand-off checklist

- [ ] `predictions.parquet` passes `read_table`; lists every model in the post.
- [ ] Metrics files cover the same matches for every model in a split.
- [ ] `trials_log.jsonl`, `model_card.md` updated.
- [ ] Channel message: models, versions, splits, headline validation numbers.
