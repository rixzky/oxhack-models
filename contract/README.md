# contract/ (identical in all three repos, do not edit)

| File | What it is |
|---|---|
| `schemas.py` | enums, ID helpers (`make_match_id`, `feature_cutoff`, `matchweek_key`), table schemas, `validate`, `read_table`, `write_table` |
| `maths.py` | the only implementation of implied probabilities, EV, stress, adverse odds, Kelly, log loss, Brier, CLV |
| `decisions.py` | `decide_match` (single-match decision, CONTRACT 6.1, 6.2, 6.4), `DEFAULT_POLICY`, `NAIVE_POLICY`, `explain` |
| `api_types.ts` | API response shapes for the React app |
| `tiger_schema.sql` | optional Tiger Data / Postgres tables and the replay query |
| `test_contract.py` | contract tests every repo runs |
| `golden/test_vectors.json` | numeric cases every part must reproduce |
| `golden/tables/` | SYNTHETIC example of every table: data, predictions (market, elo and lgbm stubs), prediction_explanations, decisions ledger (parquet + csv) |
| `golden/artifacts_json/` | SYNTHETIC manifest, splits, audit, forecast metrics, policy metrics, feature importance, frozen config, holdout_locked |
| `golden/api/` | SYNTHETIC example API responses (frontend mock mode) |
| `golden/generate_golden.py` | regenerates all golden files (`python -m contract.golden.generate_golden`) |

The golden data uses invented teams and prices. It exists so every part can build and
test from minute one. It is not a result and must never appear in real metrics.
Unlike real artifacts, the golden files include the holdout season, purely so code
paths can be exercised; the real holdout rules in CONTRACT section 8 still apply.
