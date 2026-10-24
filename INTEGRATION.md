# INTEGRATION.md (shared by all three repos, contract v1.2.0)

How three separate repos become one working project. Read with `CONTRACT.md`.

## 1. The model: three workers, files in between

- Each person works in their own repo with their own AI agent (Claude Code or Codex).
- Parts never import each other's code. They exchange **artifact files** (CONTRACT
  section 10) through a zip in the team channel or shared drive. The only shared code
  is `contract/` (schemas, maths, the single-match decision function).
- Every part starts on the **golden synthetic files** in `contract/golden/`, so all
  three can work from minute one. Real artifacts replace golden ones as they arrive,
  with no code changes (same names, same schemas).
- The end merge is mostly copying folders, because all repos share one layout.

Who does what:
- **Part 1, Data and Decisions:** turns raw CSVs into clean tables and the efficiency
  map; later runs the backtest that turns predictions into a decisions ledger.
- **Part 2, Models and Evaluation (statistics and ML):** builds the models (Elo,
  Dixon-Coles, MarketNet, LightGBM with SHAP); computes every metric shown in the app.
- **Part 3, App (UI):** a FastAPI backend that only reads files, and the React UI.
  No statistics, no data processing.

Hand-off order (repeat as things improve):
`Part 1 data -> Part 2 predictions -> Part 1 decisions -> Part 2 metrics -> Part 3 shows it`.

## 2. Fixed entry points

All commands run from the repo root. Each reads `ARTIFACTS_DIR` (default `./artifacts`).
`--golden` reads inputs from `contract/golden/` instead.

| Part | Command | Effect |
|---|---|---|
| 1 | `python -m data_pipeline.run --leagues E0 --seasons 2005-06:2025-26` | raw CSVs to Part 1 data artifacts |
| 1 | `python -m data_pipeline.load_tiger` | optional: load artifacts into Tiger Data if `DATABASE_URL` is set |
| 2 | `python -m modelling.run --models market,elo,dixon_coles,marketnet,lgbm` | `predictions.parquet`, `prediction_explanations.parquet`, `feature_importance.json` |
| 1 | `python -m decisions.run_backtest --split validation --models market,elo,dixon_coles,marketnet,lgbm` | `decisions.parquet` (naive, controlled, controlled_best_price) |
| 2 | `python -m evaluation.run --split validation` | `forecast_metrics.json`, `policy_metrics.json` |
| 1 | `python -m decisions.freeze --model lgbm --model-version X` | `frozen_config.json` (team present) |
| 1, 2 | holdout sequence, CONTRACT section 8 | `holdout_locked.json` (Part 2 writes it) |
| 3 | `uvicorn app.backend.main:app --port 8000` | API reading `artifacts/` |
| 3 | `cd app/frontend && npm install && npm run dev` | UI (proxies `/api` to port 8000) |
| all | `python -m pytest -q` | contract tests + part tests |

## 3. Sync points (integrate early, not only at the end)

**Sat ~13:30, contract review (30 to 45 min, everyone).** Read `CONTRACT.md`
together. Fill in names and the contract owner in `START_HERE.md`.

**Sat 16:00, first real hand-off.**
- Part 1 posts `artifacts.zip` with real E0 data (matches, outcomes, quotes,
  market_probs, splits, manifest, audit). All validate. Backtest code passes the
  vectors on golden data.
- Part 2 posts `predictions.parquet` with `market` and `elo` on real E0, plus
  `forecast_metrics.json`.
- Part 3 shows all three screens working from golden files via the API, then points
  the API at the real `artifacts/` folder and checks it still renders.

**Sat 22:00.** Part 1 posts the efficiency map (all leagues) and the first real
`decisions.parquet` (validation, market and elo). Part 2 posts Dixon-Coles
predictions and the first real `policy_metrics.json`. Part 3 shows real data on all
screens.

**Sun 03:00.** MarketNet and LightGBM (with SHAP) predictions; backtest and metrics
re-run for all models. Part 3 shows the model selector, feature importance and
per-match top features.

**Sun morning, freeze and holdout (everyone present).** CONTRACT section 8.

## 4. Final merge procedure (Sunday)

1. Create the final repo (name decided later). Copy in:
   - `contract/`, `CONTRACT.md`, `INTEGRATION.md` (latest, from the contract owner)
   - `data_pipeline/` and `decisions/` (Part 1), `modelling/` and `evaluation/`
     (Part 2), `app/` (Part 3)
   - each part's `tests/` into `tests/data/`, `tests/models/`, `tests/app/`
   - `requirements.txt`: union of the three; resolve clashes to the highest pin
   - `.env.example`: union
   - final `artifacts/` (gitignored; keep the zip for the demo)
   - the three `STATUS.md` files as `docs/STATUS_part1.md` etc.
2. Run `python -m pytest -q`, start the API and the UI, run `npm run build`.
3. Hand to Claude Code or Codex with the prompt in section 5 for remaining glue.
4. Hand-check section 6. Rehearse the demo.

## 5. Merge prompt for Claude Code / Codex

```
You are integrating three parts built separately against CONTRACT.md (read it first,
then INTEGRATION.md and docs/STATUS_part*.md). Folders: data_pipeline/ and decisions/
(Part 1), modelling/ and evaluation/ (Part 2), app/ (Part 3), contract/ (shared, do
not edit). Goal: the API and UI run on the real artifacts/ folder, all tests pass, and
the pipeline commands in INTEGRATION.md section 2 still run. Rules: do not change
contract/ or any artifact schema; fix mismatches on the consumer side and list every
change; do not add features; do not alter frozen_config.json or holdout_locked.json;
keep every honesty rule in CONTRACT.md section 9. Start by running the tests and the
app and report what fails before changing anything.
```

## 6. Integration checklist (hand-check)

- [ ] All parquet files pass `contract.schemas.read_table` with real data.
- [ ] `manifest.json` says `REAL`; UI shows no synthetic banner. With golden files it
      says `SYNTHETIC` and the banner appears.
- [ ] Probabilities in every view are in H, D, A order (spot-check one match against
      the raw CSV by hand).
- [ ] The replay payload contains no closing odds or result until reveal.
- [ ] One match: offered odds in the UI = raw CSV odds for the reference bookmaker.
- [ ] EV regression case rejected (p 0.52 at 1.90) in the scenarios screen.
- [ ] `frozen_config.json` hash matches the UI and `holdout_locked.json`.
- [ ] Tiger Data and Gemini badges show inactive when their env vars are missing.
- [ ] Turnover is shown next to every ROI. Exploratory views are labelled.

## 7. Known friction points (pre-empted)

| Risk | Rule that prevents it |
|---|---|
| Different team names across seasons or sources | Part 1 canonical names only; `make_match_id` |
| Timezone drift | tz-aware UTC everywhere; validators reject naive timestamps |
| Outcome order mixups | H, D, A everywhere |
| Two "market" definitions | reference quote rule (CONTRACT 4.1), `is_reference` flag |
| Leakage via closing odds | not in prediction inputs; replay hides until reveal |
| Decision logic drift between backtest and UI | both call `contract.decisions.decide_match` |
| Metric drift | only Part 2 computes metrics; Part 3 only displays them |
| Silent schema drift | every write goes through `write_table`; version bumps only by owner |
| Dependency clash at merge | pinned Python and core libs in CONTRACT section 3 |
| Path assumptions | `ARTIFACTS_DIR`, run from repo root, `python -m` entry points |
