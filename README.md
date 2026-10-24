# oxhack-models: Part 2, Models and Evaluation (statistics and ML)

One of three repos for our OxHack '26 project (name to be decided). See `CONTRACT.md`
for the shared plan and rules, `SPEC.md` for this part, `INTEGRATION.md` for hand-offs.

## Setup
```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

## Run
```
python -m modelling.run --golden --models market,elo
python -m modelling.run --models market,elo,dixon_coles,marketnet,lgbm
python -m evaluation.run --golden --split validation
```

## Getting started with an AI agent
Open this folder in Claude Code or Codex and say:
"Read AGENTS.md, CONTRACT.md and SPEC.md, then give me a short plan for the first
milestone and start building it."
