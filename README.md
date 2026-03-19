# Auto Data Scientist

A Claude Code plugin that runs an autonomous, multi-agent ML research loop. Drop it into any folder with train/test CSVs and it will iteratively form hypotheses, write training code, execute experiments, learn from results, and produce a final ensemble submission — all without human intervention after initial setup.

## Quick Start

```bash
cd your-project-with-data/
cp -r path/to/auto-data-scientist/.claude .claude/
cp path/to/auto-data-scientist/requirements.txt .
pip install -r requirements.txt
```

Then from inside Claude Code:

```
/auto-research
```

The skill will scan your data, ask for target column and time budget, run EDA, and create the research plan. Then start the autonomous loop:

```
/loop 1m /auto-research
```

That's it. Each iteration runs automatically. You can continue working in the same Claude Code session.

## Architecture

```
/loop 1m /auto-research              Repeats the skill every cycle
  │
  │  each invocation runs one iteration
  ▼
.claude/skills/auto-research/         Master Agent (Supervisor)
  │                                   Reads/writes research_state.md
  │                                   Computes timing from state
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

### Context Isolation

Each `/loop` invocation is a fresh skill call — no context window bloat across long runs. All memory lives in `research_state.md`. Budget is tracked as cumulative compute time (sum of iteration durations), so pausing and resuming doesn't waste budget.

## File Structure

```
your-project/
├── .claude/
│   ├── skills/
│   │   └── auto-research/
│   │       └── SKILL.md              # Master Agent — skill definition
│   └── agents/
│       └── experiment-worker.md      # Worker Agent — agent definition
├── requirements.txt                  # Python dependencies
├── feature_template.py               # Created at init — shared preprocessing
├── research_state.md                 # Created at init — persistent state
├── research_state.md.bak             # Auto-backup before each state update
├── submission.csv                    # Created at end — final predictions
├── data/
│   ├── train.csv                     # Your training data
│   └── test.csv                      # Your test data
└── logs/
    ├── eda_report.json               # EDA results from init
    ├── iteration_001/
    │   ├── code.py                   # Self-contained training script
    │   ├── results.json              # Experiment metrics
    │   ├── oof_preds.csv             # Out-of-fold predictions
    │   └── test_preds.csv            # Test set predictions
    └── ...
```

## Usage

### Initialize

```
/auto-research
```

The skill detects no `research_state.md` and enters **Init Mode**:
- Scans for CSV/parquet files in the directory
- Asks for the **target column** (only required input)
- Auto-detects task type (binary classification, multiclass, regression) and metric
- Validates data integrity (column alignment, ID uniqueness, target existence)
- Runs full EDA (shape, distributions, correlations, cardinality, missing values)
- Generates `feature_template.py` with dataset-specific preprocessing
- Discusses strategy with you
- Writes `research_state.md` with initial hypothesis queue

### Start the Loop

```
/loop 1m /auto-research
```

Each invocation runs one complete iteration autonomously. You can continue working in the same Claude Code session between iterations.

### Stop the Loop

```
/loop stop
```

State is preserved in `research_state.md`. Resume anytime with `/loop 1m /auto-research`.

### Check Progress

```
cat research_state.md
```

### Run a Single Iteration

```
/auto-research
```

When `research_state.md` exists, this runs exactly one iteration — useful for manual control or debugging.

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

**Auto-triggered at 90% of the compute budget** (configurable). Collects OOF predictions from top-K iterations, trains a Ridge/shallow-tree meta-learner, and produces the final `submission.csv`.

## Budget System

The budget tracks **cumulative compute time** — the sum of all iteration durations. This means:
- Pausing and resuming doesn't consume budget
- Overnight breaks are free
- Only actual experiment time counts

When budget is exhausted and Phase 3 is complete, `/auto-research` becomes a no-op (outputs "Research complete"). Stop the loop with `/loop stop`.

## Autonomy & Self-Correction

Once init is complete, the Master Agent operates with **zero human input**:

| Scenario | Autonomous Response |
|----------|-------------------|
| Worker fails 3 times in a row | Pivot to minimal baseline (default LightGBM, numeric features only) |
| All hypothesis queues empty | Brainstorm new hypotheses using structured templates |
| 90% of compute budget elapsed | Force Phase 3 meta-stacking and produce `submission.csv` |
| Worker script crashes | Worker self-debugs up to 3 attempts before reporting terminal failure |
| State file corrupted | Auto-restore from research_state.md.bak |
| Budget exhausted | Graceful no-op until user stops the loop |

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

## Requirements

- **Claude Code CLI** (`claude`) installed and authenticated
- **Python 3.8+** with dependencies from `requirements.txt`

## Generalizing to a New Dataset

1. Create a new project folder
2. Drop your `train.csv` and `test.csv` in it (or `data/` subfolder)
3. Copy `.claude/` and `requirements.txt` into the folder
4. Run `pip install -r requirements.txt`
5. Open Claude Code in the project folder
6. Run `/auto-research` — answer the init questions (target column, time budget)
7. Run `/loop 1m /auto-research` — walk away

The plugin auto-detects task type (classification or regression), metric, ID column, and adapts its strategy to the dataset's characteristics.
