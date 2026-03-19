# Auto Data Scientist

A Claude Code plugin that runs an autonomous, multi-agent ML research loop. Drop it into any folder with train/test CSVs and it will iteratively form hypotheses, write training code, execute experiments, learn from results, and produce a final ensemble submission — all without human intervention after initial setup.

## Quick Start

```bash
cd your-project-with-data/
cp -r path/to/auto-data-scientist/.claude .claude/
cp path/to/auto-data-scientist/launch_research.sh .
cp path/to/auto-data-scientist/requirements.txt .
pip install -r requirements.txt
```

Then from inside Claude Code:

```
/auto-research
```

That's it. The skill will:
1. Scan your data files, ask for the target column and time budget, run full EDA
2. Generate a `feature_template.py` with dataset-specific preprocessing
3. Write the research plan to `research_state.md`
4. Launch the autonomous loop — no further input needed until the budget runs out

To resume or start a new run later, invoke `/auto-research` again.

## Architecture

```
launch_research.sh                    Outer bash loop (owns the clock)
  │
  │  each iteration calls claude CLI
  ▼
.claude/skills/auto-research/         Master Agent (Supervisor)
  │                                   Reads/writes research_state.md
  │                                   Decides phase, picks hypothesis
  │  spawns via Agent tool
  ▼
.claude/agents/experiment-worker.md   Worker Agent (Executor)
                                      Imports feature_template.py
                                      Writes Python, executes, self-debugs
                                      Returns structured results
```

### Why Two Agents?

| Concern | Master Agent | Worker Agent |
|---------|-------------|-------------|
| **Context** | Strategy, history, phase logic | One experiment in isolation |
| **Freedom** | Constrained to planning rules | Full freedom to write any Python |
| **Persistence** | Reads/writes `research_state.md` | Writes to `logs/iteration_NNN/` |
| **Interaction** | Init: interactive. Loop: fully autonomous | Never interacts with user |

### Why an Outer Bash Loop?

Each iteration invokes `claude` CLI fresh — no context window bloat across a 24-hour run. The Master Agent is stateless between invocations; all memory lives in `research_state.md`.

## File Structure

```
your-project/
├── .claude/
│   ├── skills/
│   │   └── auto-research/
│   │       └── SKILL.md              # Master Agent — skill definition
│   └── agents/
│       └── experiment-worker.md      # Worker Agent — agent definition
├── launch_research.sh                # Launcher script (entry point)
├── requirements.txt                  # Python dependencies
├── feature_template.py               # Created at init — shared preprocessing
├── research_state.md                 # Created at init — persistent state
├── research_state.md.bak             # Auto-backup before each state update
├── submission.csv                    # Created at end — final predictions
├── data/
│   ├── train.csv                     # Your training data
│   └── test.csv                      # Your test data
└── logs/
    ├── .elapsed_seconds              # Cumulative time tracker (for resume)
    ├── eda_report.json               # EDA results from init
    ├── iteration_001/
    │   ├── code.py                   # Self-contained training script
    │   ├── results.json              # Experiment metrics
    │   ├── oof_preds.csv             # Out-of-fold predictions
    │   └── test_preds.csv            # Test set predictions
    ├── iteration_002/
    │   └── ...
    └── ...
```

## Usage

### First Run

From inside Claude Code, run `/auto-research`. The skill will:
- Scan for CSV/parquet files in the directory
- Ask for the **target column** (only required input)
- Auto-detect task type (binary classification, multiclass, regression) and metric
- Validate data integrity (column alignment, ID uniqueness, target existence)
- Run full EDA (shape, distributions, correlations, cardinality, missing values)
- Generate `feature_template.py` with dataset-specific preprocessing
- Discuss strategy with you
- Write `research_state.md` with initial hypothesis queue
- Ask for budget hours and launch the autonomous loop

### Resume a Stopped Run

Invoke `/auto-research` again from Claude Code. It will show your current progress and let you start a new run. **Elapsed time from prior sessions is tracked** — the budget accounts for all time spent, not just the current session.

### Check Progress

```bash
./launch_research.sh --status    # Quick summary from terminal
cat research_state.md             # Full state
ls logs/                          # Per-iteration outputs
```

### Stop Early

Press `Ctrl+C` in the terminal where the loop is running. State is preserved — resume anytime via `/auto-research`.

## Three-Phase Research Strategy

### Phase 1: Bold Bets (Exploration)

High-risk, high-reward experiments. New model types, radical feature engineering, diverse architectures. Prioritized by estimated speed (fastest first) to maximize learning rate.

### Phase 2: Fine-Tuning (Exploitation)

Triggered automatically when a Phase 1 result scores in the top percentile. Hyperparameter sweeps, feature pruning, regularization tuning on proven winners.

**Adaptive promotion threshold:**
- First 5 experiments: within 10% relative of global best
- 5-19 experiments: top 25th percentile
- 20+ experiments: top 15th percentile

### Phase 3: Meta-Stacking (Endgame)

**Auto-triggered at 90% of the time budget** (configurable). Collects OOF predictions from top-K iterations, trains a Ridge/shallow-tree meta-learner, and produces the final `submission.csv`.

## Autonomy & Self-Correction

Once the init interview is complete, the Master Agent operates with **zero human input**:

| Scenario | Autonomous Response |
|----------|-------------------|
| Worker fails 3 times in a row | Pivot to minimal baseline (default LightGBM, numeric features only) |
| All hypothesis queues empty | Brainstorm new hypotheses using structured templates |
| 90% of time budget elapsed | Force Phase 3 meta-stacking and produce `submission.csv` |
| Worker script crashes | Worker self-debugs up to 3 attempts before reporting terminal failure |
| Worker script hangs | Killed by timeout (default 30 min) |
| State file corrupted | Auto-restore from research_state.md.bak |
| Disk space low (<1GB) | Loop stops gracefully |

## Supported Task Types

| Task | Metric | Calibration | CV Strategy |
|------|--------|-------------|-------------|
| Binary classification | log_loss | Isotonic regression | Stratified K-Fold |
| Multiclass classification | log_loss | Isotonic regression | Stratified K-Fold |
| Regression | RMSE | None | K-Fold |

## Configuration

Default settings are in the **Config Overrides** section of `SKILL.md`. Key tunables:

| Setting | Default | Description |
|---------|---------|-------------|
| `n_folds` | 5 | CV fold count (reduce to 3 for small datasets) |
| `worker_timeout_seconds` | 1800 | Max time per experiment |
| `early_stopping_patience` | 50 | Early stopping rounds for GBDT |
| `calibration_method` | isotonic | Probability calibration (classification only) |
| `phase3_budget_pct` | 90 | When to trigger meta-stacking |
| `max_consecutive_failures` | 3 | Failures before forcing minimal baseline |

## Output Contracts

### results.json (per iteration)

```json
{
  "cv_score": 0.02215,
  "cv_std": 0.00010,
  "fold_scores": [0.0220, 0.0223, 0.0219, 0.0225, 0.0221],
  "duration_seconds": 708.5,
  "model_type": "lightgbm",
  "feature_count": 50,
  "feature_names": ["col1", "col2", "..."],
  "top_features": [{"name": "col1", "importance": 1523.4}],
  "calibration_applied": true,
  "calibration_method": "isotonic",
  "n_train_rows": 100000,
  "n_folds": 5,
  "predictions_mean": 0.0054,
  "predictions_std": 0.0312
}
```

### oof_preds.csv (per iteration)

```
id,pred_prob,fold
12345,0.0034,0
12346,0.0012,1
```

### test_preds.csv (per iteration)

```
id,pred_prob
12345,0.0034
```

## Requirements

- **Claude Code CLI** (`claude`) installed and authenticated
- **Python 3.8+** with dependencies from `requirements.txt`
- **bash** shell (Git Bash on Windows works)

## Generalizing to a New Dataset

1. Create a new project folder
2. Drop your `train.csv` and `test.csv` in it (or `data/` subfolder)
3. Copy `.claude/`, `launch_research.sh`, and `requirements.txt` into the folder
4. Run `pip install -r requirements.txt`
5. Open Claude Code in the project folder
6. Run `/auto-research` — answer the init questions (target column, time budget)
7. Walk away

The plugin auto-detects task type (classification or regression), metric, ID column, and adapts its strategy to the dataset's characteristics.
