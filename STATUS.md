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
- **evaluation/forecast**: log_loss, Brier, calibration bins; inner-join across models on shared match set.
- **contract tests**: 43/43 pass (`python -m pytest -q`).
- Golden parquet regeneration works (`python -m contract.golden.generate_golden`).

## Stubbed or partial

- **evaluation/policy**: ROI, CLV, Sharpe, drawdown, three-policy comparison — not yet implemented (Phase 4).
- **evaluation/bootstrap**: Bootstrap CIs for ROI — not yet implemented (Phase 4).
- **evaluation/lock_holdout**: Holdout lock — not yet implemented (Phase 4).
- **policy_metrics.json**: Not yet written (needs evaluation/policy.py).
- **holdout_locked.json**: Not yet written (needs evaluation/lock_holdout.py).
- **model_card.md**: Not yet written.
- **Real artifacts**: Using golden synthetic data. Real artifacts from Marlon's Part 1 not yet integrated.

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
