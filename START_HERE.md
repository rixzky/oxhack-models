# START HERE (OxHack '26, team of 3)

Three repos, three people, one contract. Name of the project: to be decided later.

| Person | Repo | Part |
|---|---|---|
| Marlon | `oxhack-data` | 1 Data and Decisions: ingestion, cleaning, market probabilities, efficiency map, audit, Tiger loader; backtest (bankroll, caps, settlement) using the shared decision function |
| Ricky | `oxhack-models` | 2 Models and Evaluation (statistics and ML): Elo, Dixon-Coles, MarketNet, LightGBM with SHAP; forecast metrics, policy metrics, holdout result |
| Armaan| `oxhack-app` | 3 App (UI): FastAPI that reads the files, React UI, Gemini explainer, demo. No statistics or data processing |

Contract owner (only person who edits `contract/` and `CONTRACT.md`): Ricky 

## Next 45 minutes
1. Each person unzips their kit, `git init`, pushes to their own GitHub repo.
2. Everyone reads `CONTRACT.md` together (it holds the whole agreed plan). Fill in the
   names above and in CONTRACT section 2. Raise anything ambiguous now.
3. Each person: `pip install -r requirements.txt` then `python -m pytest -q`
   (contract tests should pass immediately on the golden data).
4. Open the repo in Claude Code or Codex and say: "Read AGENTS.md, CONTRACT.md and
   SPEC.md, then give me a short plan for the first milestone and start building it."

## How it fits together
- Everyone starts on the synthetic golden files in `contract/golden/`, so nobody waits.
- Hand-off order: Part 1 data -> Part 2 predictions -> Part 1 decisions ledger ->
  Part 2 metrics -> Part 3 shows it. Drop received files into your `artifacts/` folder.
- First real hand-off at Sat 16:00 (see `INTEGRATION.md`).
- Sync points and the final merge: `INTEGRATION.md`.

## Each kit contains
`CONTRACT.md`, `INTEGRATION.md`, `SPEC.md` (part-specific), `AGENTS.md` (Codex) and
`CLAUDE.md` (Claude Code, imports AGENTS.md), `contract/` (shared code, golden data,
test vectors), `STATUS.md`, `QUESTIONS.md`, `requirements.txt`, `.env.example`,
`pytest.ini`, `.gitignore`, and an empty part folder.
