# Model Card — Part 2: Models & Evaluation

Contract version: 1.2.0  
Dataset: 8 leagues (E0 E1 SP1 D1 I1 F1 N1 P1), seasons 2005-06 to 2025-26  
Reference bookmaker: Bet365 (B365) — all splits, PRE_CLOSING snapshot  
Holdout season: 2025-26 (locked at the freeze)

---

## 1. market

**What it is:** Reference bookmaker (Bet365) PRE_CLOSING implied probabilities, after margin removal. The zero-skill baseline every other model must beat.

**Inputs:** `market_probs` where `is_reference=True` and `snapshot="PRE_CLOSING"`.  
**Fitting:** None. No parameters.  
**Leakage controls:** Only uses PRE_CLOSING odds known at `feature_cutoff_utc`. Never reads CLOSING rows.  
**Hyperparameters:** None.  
**Known weaknesses:** Embeds the bookmaker's margin assumptions. Favourite-longshot bias is baked in.

---

## 2. elo

**What it is:** Walk-forward Elo ratings mapped to H/D/A via multinomial logistic regression.

**Inputs:** Match schedule, kickoff times, outcomes (goals and result) up to `feature_cutoff_utc`.  
**Fitting:** Elo ratings updated after every match (chronological). Season regression factor 0.3 applied at season start (ratings pulled toward mean). Logistic regression fitted once per season on all prior warmup+train seasons.  
**Cutoff:** `training_cutoff_utc` = latest kickoff in prior training seasons.  
**Hyperparameters:**
- K base = 20, goal-diff multiplier `1 + 0.5 * log1p(|gd|)`
- Home advantage offset = 100 Elo points
- Season regression factor = 0.3
- Initial rating = 1500

**Known weaknesses:** Elo does not model draw probability directly; the logistic layer is the draw model. Sensitive to early-season data scarcity.

---

## 3. dixon_coles

**What it is:** Poisson goals model with Dixon-Coles low-score correction, exponential time decay, per-league attack/defence parameters.

**Inputs:** Home/away goals per match, kickoffs, league, up to `feature_cutoff_utc`.  
**Fitting:** L-BFGS-B minimisation of the weighted negative log-likelihood. Refit once per (league, season). Scoreline grid 0–10 → H/D/A by summing the 11×11 probability matrix.  
**Cutoff:** `training_cutoff_utc` = latest training kickoff before the season's first `feature_cutoff_utc`.  
**Hyperparameters:**
- Rho (DC correction) = −0.13 (fixed; typical football value)
- Half-life: tuned on calibration split from {30, 60, 90, 180} days (selected = 180 days on golden data)
- `MAX_GOALS` = 10 for grid

**Known weaknesses:** Seasonal refitting means mid-season form changes propagate slowly. Ignores market signal.

---

## 4. marketnet

**What it is:** Shallow PyTorch model learning a log-odds correction over the Bet365 market. `p = softmax(log_q + W·x)`, draw logit pinned to 0 so W=0 reproduces the market exactly.

**Inputs:** Feature table (34 features) + Bet365 PRE_CLOSING implied log-probabilities.  
**Fitting:** Adam optimiser, walk-forward by season (fit on prior warmup+train seasons). Early stopping with patience 50. L2 penalty lambda tuned on calibration split from {1e-3, 1e-2, 1e-1, 1.0, 10.0}.  
**Cutoff:** `training_cutoff_utc` = latest training kickoff.  
**Hyperparameters:**
- LR = 5e-3, max epochs = 2000, patience = 50
- Lambda (L2): tuned per run on calibration split

**Known weaknesses:** Linear correction only — no non-linear market inefficiencies captured. Upward bias if lambda is too small (tends to memorise market noise). W is a 2 × 34 matrix so explanations are approximate (linear attribution).

**Line-shopping note:** `controlled_best_price` uses the best price across all bookmakers. ROI comparisons with `controlled` have an inherent upward bias because best-of-many odds are selected ex-post per outcome.

---

## 5. lgbm

**What it is:** LightGBM multiclass gradient-boosted trees on the 34-feature table.

**Inputs:** Feature table (34 features: market q, Elo diff, DC log-ratios, rest days, form, league one-hot, longshot features).  
**Fitting:** Walk-forward by season (fit on prior warmup+train). Early stopping on calibration split (or 80/20 train split if calibration is unavailable). Shallow trees, strong regularisation to prevent overfitting on ~300–600 training matches per initial season.  
**Cutoff:** `training_cutoff_utc` = latest training kickoff.  
**Hyperparameters (fixed, conservative):**
- `num_leaves` = 15, `max_depth` = 4, `min_child_samples` = 20
- `learning_rate` = 0.05, `feature_fraction` / `bagging_fraction` = 0.8
- `reg_alpha` = 1.0, `reg_lambda` = 5.0
- Early stopping rounds = 30, max rounds = 500

**Explanations:** SHAP TreeExplainer, top-3 contributions per match × outcome on validation (and holdout at the freeze). Global importance: mean absolute SHAP across all validation outcomes.

**Known weaknesses:** Small training sets in early walk-forward windows → stubs (1/3) returned until enough training data accumulates. Feature leakage is prevented by design (no post-cutoff data in the feature table) but must be verified per deployment.

---

## Evaluation summary (validation split, real data — 2024-25, 2916 matches)

### Forecast metrics

| Model | Log loss | Brier | vs market (bootstrap CI) |
|---|---|---|---|
| **marketnet** | **0.9705** | 0.5780 | −0.0021 [−0.0033, −0.0008] ✓ significant |
| lgbm | 0.9709 | 0.5782 | −0.0016 [−0.0034, +0.0002] borderline |
| market | 0.9725 | 0.5790 | baseline |
| elo | 0.9990 | 0.5974 | +0.0265 worse |
| dixon_coles | 1.0008 | 0.5984 | +0.0283 worse |

### Policy metrics (naive policy, 2916 validation matches per model)

| Model | Bets | ROI | Bootstrap CI 95% | Mean CLV |
|---|---|---|---|---|
| lgbm | 1254 | +1.0% | [−8.9%, +10.8%] | −4.4% |
| marketnet | 453 | +0.6% | [−12.7%, +13.9%] | −4.5% |
| dixon_coles | 889 | −11.3% | [−18.8%, −4.5%] | −5.0% |
| elo | 466 | −21.6% | [−36.0%, −4.6%] | −4.3% |

Policy ROI CIs overlap zero for lgbm and marketnet — no model shows statistically significant betting profitability. Negative CLV across the board confirms the market is well-priced.

**Recommended model for freeze: marketnet** — lowest log loss, only model with a statistically significant forecasting edge over the market baseline.

---

## Holdout protocol

- Holdout season 2025-26 is excluded from all model fitting and hyperparameter tuning.
- `holdout_locked.json` is written once at the freeze by `python -m evaluation.lock_holdout`.
- The file refuses to overwrite itself.
- At the freeze: run `python -m modelling.run --include-holdout` for the frozen model only.
