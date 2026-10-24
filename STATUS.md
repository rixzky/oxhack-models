# STATUS.md

Contract version: 1.2.0
Last updated: 2026-10-24

## Works

- **market** baseline: PRE_CLOSING Bet365 implied probs; passes contract schema validation.
- **elo**: Walk-forward per season; home-advantage offset (100 pts); goal-diff K multiplier; season regression (0.3); season-level multinomial logistic regression → H/D/A probs.
- **dixon_coles**: Poisson goals, DC low-score correction (rho fixed -0.13), exponential time decay, half-life tuned on calibration split (candidates: 30/60/90/180 days). Vectorised NLL and score matrix. Walk-forward by (league, season) cache — fast on 8-league data.
- **marketnet**: PyTorch log-odds correction over market; draw logit pinned to 0 (W=0 reproduces market); L2 penalty tuned on calibration; walk-forward by season.
- **lgbm**: LightGBM multiclass; walk-forward by season; early stopping on calibration split; SHAP TreeExplainer explanations for validation/holdout; 34 features.
- **features**: 34-column feature table — market q/log-q, Elo diff, DC log-ratios, rest days, PPG form (last 5), league one-hot, fav/longshot signals.
- **evaluation/forecast**: log_loss, Brier, calibration bins, per-league breakdown, bootstrap CI for ll-diff vs market (matchweek blocks, 2000 resamples).
- **evaluation/policy**: ROI, max drawdown, CLV, Sharpe per matchweek, deflated Sharpe (Bailey-LdP), bootstrap 95% CI for ROI, line-shopping value (controlled_best_price vs controlled).
- **evaluation/bootstrap**: Matchweek-block bootstrap for ROI CI and paired diff CI.
- **evaluation/lock_holdout**: Writes holdout_locked.json once, refuses overwrite.
- **Real artifacts**: Marlon's 8-league data (62k matches, 2005-06 to 2025-26) integrated. All 5 models run successfully.
- **policy_metrics.json**: Complete — all 5 models × 3 policies × validation (43,740 decisions). Ready for Armaan.
- **contract tests + model tests + evaluation tests**: 59/59 pass (`python -m pytest -q`).
- Golden parquet regeneration works (`python -m contract.golden.generate_golden`).

## Validation headline numbers (2024-25, 2916 common matches)

| Model | Log loss | Brier | vs market |
|---|---|---|---|
| marketnet | **0.9705** | 0.5780 | −0.0021 (CI [−0.0033, −0.0008]) ✓ sig |
| lgbm | 0.9709 | 0.5782 | −0.0016 (CI [−0.0034, +0.0002]) borderline |
| market | 0.9725 | 0.5790 | baseline |
| elo | 0.9990 | 0.5974 | +0.0265 worse |
| dixon_coles | 1.0008 | 0.5984 | +0.0283 worse |

**Recommended model for freeze: marketnet** (lowest log loss, statistically significant edge over market).

## 4-year OOS robustness test (2021-22 to 2024-25, 11,809 matches)

Training window: 2005-06 to 2020-21 (calibration: 2020-21). Models never saw 2021-25 data.

| Model | Log loss | vs market |
|---|---|---|
| marketnet | **0.9710** | −0.0005 ✓ edge holds |
| market | 0.9715 | baseline |
| lgbm | 0.9741 | +0.0026 ✗ worse than market over 4 years |

**Key finding:** marketnet's edge is robust across 4 unseen seasons. lgbm overfits to recent patterns and loses its edge. v1.1 exponential weighting confirmed as the right choice for robustness.

**Betting simulation** (bet when model prob > market fair prob, pay offered odds):
- marketnet: 17,047 bets, −3.7% ROI — edge does not cover Bet365's ~6.1% overround
- 1-year naive policy (2024-25, production split): lgbm +1.0% ROI [−8.9%, +10.8%]
- Profit requires exchange betting (no margin) or multi-bookmaker line shopping

## Stubbed or partial

- **holdout_locked.json**: Written at the freeze only (`python -m evaluation.lock_holdout`). Refuses to overwrite.
- **Holdout predictions**: Excluded until team freeze. Run `python -m modelling.run --include-holdout` for frozen model only after freeze.

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Regenerate golden files (if parquets are missing/corrupt)
python -m contract.golden.generate_golden

# Run all models on golden data
python -m modelling.run --golden --models market,elo,dixon_coles,marketnet,lgbm

# Run on real artifacts (once Marlon's artifacts/ is available)
python -m modelling.run --models market,elo,dixon_coles,marketnet,lgbm

# Evaluate forecast metrics
python -m evaluation.run --golden --split validation

# Run all tests
python -m pytest -q
```

## Known issues / assumptions taken

- Half-life tuning for DC uses a single calibration cutoff (min of calibration split's feature_cutoff_utc); this is faster than multi-cutoff tuning and avoids over-optimising.
- DC seasonal refitting (not monthly) trades some recency for 10-20x speed on multi-league data.
- MarketNet imputes NaN features with column median at normalisation time; scaler is fit on training data only.
- LightGBM `multi_class` kwarg removed in sklearn 1.5+; elo.py uses sklearn without the deprecated kwarg.
- Golden parquets were regenerated 2026-10-24 (old ones had a PyArrow format mismatch).
- torch CPU-only installed (`torch>=2.2` in requirements.txt covers this).
