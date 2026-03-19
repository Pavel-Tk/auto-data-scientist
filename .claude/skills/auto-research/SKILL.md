---
name: auto-research
description: Autonomous ML research loop. Handles initialization (interactive dataset scan, EDA, user interview, strategy planning) and autonomous iteration (hypothesis selection, experiment execution via worker agent, state updates). Use when the user says "start research", "auto research", "run ML research", "initialize research", or invokes /auto-research.
---

# Auto Research — Master Agent

You are a Lead ML Research Supervisor. You handle interactive initialization and autonomous iteration depending on mode. This skill runs entirely within Claude Code — no external scripts needed.

## Mode Detection

Check for `research_state.md` in the current working directory:
- **If it does NOT exist** → enter **Init Mode** (interactive, one-time setup)
- **If it DOES exist** → enter **Iteration Mode** (autonomous, one cycle)

---

## Config Overrides

These are the default settings. They can be overridden per-project by editing this section.

- **n_folds**: 5 (reduce to 3 for small datasets <10K rows)
- **worker_timeout_seconds**: 1800 (30 minutes per experiment)
- **early_stopping_patience**: 50
- **calibration_method**: isotonic (options: isotonic, platt, none; ignored for regression)
- **min_learning_rate**: 0.03 (below this, large datasets tend to timeout)
- **phase3_budget_pct**: 90 (trigger meta-stacking at this % of budget)
- **max_consecutive_failures**: 3 (force minimal baseline after this many)

---

# INIT MODE — Interactive Setup

This runs once to create `research_state.md`. This is the ONLY mode where you interact with the user.

## Step 1: Environment Scan

Scan the current working directory and subdirectories for data files:

```
Glob patterns to search:
- **/*.csv
- **/*.parquet
- **/*.tsv
- **/*.feather
```

For each file found, report: filename, path, file size. Identify likely train/test splits by name convention (train/test, train/val, etc.) or ask the user if ambiguous.

## Step 2: User Interview

Ask the user these questions (use AskUserQuestion tool):

1. **Target column**: "Which column is the prediction target?" (Show column names from the identified training file)
2. **Time budget**: "How many hours of compute budget for the research loop?" (Options: 2h, 4h, 8h, 12h, 24h, custom). This is cumulative experiment time, not wall-clock — pausing and resuming won't consume budget.
3. **Any domain context?**: Free text — anything the user knows about the problem that would help. This is optional.

## Step 3: Auto-Detection

Based on the target column, automatically infer:

- **Task type**:
  - Binary classification: target has exactly 2 unique values
  - Multiclass classification: target has 3-20 unique categorical values
  - Regression: target is continuous numeric
  - If ambiguous, ask the user
- **Evaluation metric** (inferred from task type):
  - Binary classification → `log_loss` (lower is better)
  - Multiclass classification → `log_loss` (lower is better)
  - Regression → `rmse` (lower is better)
  - Ask user if they prefer a different metric
- **ID column**: Look for columns named `id`, `*_id`, `index`, or the first column. Confirm with user if ambiguous.
- **Metric direction**: `lower_is_better` or `higher_is_better`

## Step 3.5: Data Validation

Before proceeding, verify:

1. **Target column exists** in the training data
2. **Train and test columns match** (same columns except target in train only). Warn if they differ.
3. **ID column has no duplicates** in train or test
4. **No constant columns** (same value in every row) — flag these for exclusion
5. **Basic shape check**: report train and test row/column counts
6. **Warn about potential leakage** if any non-target column has suspiciously high correlation with target (>0.95)

If any critical check fails (target missing, ID not unique), report to user and ask how to proceed.

## Step 4: Full EDA

Spawn the `experiment-worker` agent to run a comprehensive EDA script. The script should compute and save to `logs/eda_report.json`:

```json
{
  "shape": {"train_rows": int, "train_cols": int, "test_rows": int, "test_cols": int},
  "target": {
    "name": "column_name",
    "dtype": "type",
    "n_unique": int,
    "value_counts": {"val1": count1, "val2": count2},
    "positive_rate": float_or_null
  },
  "columns": [
    {
      "name": "col_name",
      "dtype": "type",
      "n_unique": int,
      "missing_pct": float,
      "mean": float_or_null,
      "std": float_or_null,
      "min": float_or_null,
      "max": float_or_null,
      "skew": float_or_null,
      "top_values": ["val1", "val2", "val3"]
    }
  ],
  "correlations_with_target": [
    {"column": "name", "correlation": float}
  ],
  "numeric_columns": ["list"],
  "categorical_columns": ["list"],
  "high_cardinality_columns": [{"name": "col", "n_unique": int}],
  "datetime_columns": ["list"],
  "constant_columns": ["list"],
  "duplicate_rows_pct": float
}
```

Tell the Worker to write and execute a Python script that computes all of the above, handling large datasets efficiently (sample for correlations if >1M rows).

## Step 4.5: Validate EDA Output

After the Worker returns, verify:

1. `logs/eda_report.json` exists
2. It contains the required top-level keys: `shape`, `target`, `columns`
3. `shape.train_rows > 0` and `shape.train_cols > 0`

If the file is missing or malformed, re-run EDA with a simpler fallback: just compute shape, target distribution, and column dtypes. Do not proceed to Step 5 without at least basic EDA data.

## Step 5: Strategy Discussion

Present the EDA findings to the user in a clear summary:

- Dataset dimensions and target distribution
- Notable features (high correlation, high cardinality, many missing values)
- Potential challenges (class imbalance, data leakage risks, size/speed concerns)
- Your proposed initial strategy

Discuss with the user. Incorporate their feedback.

## Step 5.5: Generate Feature Template

Write a `feature_template.py` in the working directory that encodes the dataset-specific preprocessing pipeline. This file will be imported by Worker scripts to avoid re-implementing preprocessing every iteration.

The template should contain a `load_and_preprocess(train_path, test_path, target_col, id_col)` function that:

1. Loads train and test data
2. Separates target and ID columns
3. Handles missing values (strategy based on EDA findings)
4. Creates base feature engineering:
   - Frequency encoding for high-cardinality categorical columns
   - Date/time features if datetime columns exist
   - Basic numeric transformations (log1p for skewed columns, ratios between related columns)
5. Returns: `X_train, y_train, X_test, id_train, id_test, feature_names`

The template should be dataset-aware (column names, dtypes) but model-agnostic. Worker scripts will import it and add hypothesis-specific features on top.

Also include a `get_data_info()` function that returns a dict with: train_path, test_path, target_col, id_col, task_type, n_rows, feature_names — so workers can inspect dataset metadata without re-reading files.

## Step 6: Write research_state.md

Create `research_state.md` in the current working directory with this exact structure:

```markdown
# Research State

## Config
- **Target**: {target_column}
- **Task**: {binary_classification|multiclass_classification|regression}
- **Metric**: {metric_name} ({lower_is_better|higher_is_better})
- **Time Budget**: {budget_seconds}s ({budget_hours}h)
- **Elapsed Compute**: 0s
- **N Folds**: {n_folds from Config Overrides}
- **Data**:
  - Train: {train_path} ({n_rows} rows x {n_cols} cols)
  - Test: {test_path} ({n_rows} rows x {n_cols} cols)
  - Enriched: {list any additional data files found, or "none"}
- **ID Column**: {id_column}
- **Positive Rate**: {rate}% (only for classification)
- **Created**: {ISO timestamp}

## EDA Summary
{Concise summary of key EDA findings — 10-20 bullet points max}

## Strategy & Reflections
- {User's domain context if provided}
- {Initial strategic observations from EDA}
- {Key risks identified (imbalance, cardinality, speed constraints)}

## Global Best
- **Score**: none (no experiments yet)
- **Model**: none
- **Consecutive Failures**: 0
- **Total Experiments**: 0
- **Successful Experiments**: 0

## Hypothesis Queue

### Phase 1 (Bold Bets)
1. {First hypothesis — always a simple baseline: default LightGBM/LogisticRegression with only numeric features, no engineering} — est. 5min
2. {Second hypothesis — add basic feature engineering: date features, frequency encoding for categoricals} — est. 10min
3. {Third hypothesis — try a different model family} — est. 15min
4. {Fourth hypothesis — creative feature engineering based on EDA insights} — est. 20min
5. {Fifth hypothesis — bold bet} — est. 15min

### Phase 2 (Fine-Tuning)
{empty — populated after Phase 1 results}

### Phase 3 (Meta-Stacking)
1. Ridge meta-learner on top-K OOF predictions
2. Shallow LightGBM stacker on OOF predictions

## Experiment History (Rolling: last 10 + top 5)

### Top 5 by Score
| Iter | Phase | Model | CV Score | CV Std | Features | Duration | Key Insight |
|------|-------|-------|----------|--------|----------|----------|-------------|
{empty}

### Recent 10
| Iter | Phase | Model | CV Score | Status | Duration | Key Insight |
|------|-------|-------|----------|--------|----------|-------------|
{empty}

## Completed Experiments (one-line summaries)
{empty}

## Learnings
{empty — will be populated as experiments run}
```

## Init Mode — Final Step

After writing research_state.md:
- The first hypothesis MUST be a trivial baseline (default model, numeric features only)
- Hypotheses 2-5 should be informed by EDA findings
- Always create the `logs/` directory if it doesn't exist
- Save the EDA report to `logs/eda_report.json`
- List all additional data files in Config so the autonomous loop knows they exist
- Tell the user: "Setup complete! Starting the autonomous research loop. To stop: `/loop stop`. To check progress: `cat research_state.md`"
- **Start the loop automatically** by invoking the Skill tool with skill `loop` and args `1m /auto-research`

---

# ITERATION MODE — Autonomous Cycle

This mode runs one complete experiment cycle. It is designed to be invoked repeatedly by `/loop` or manually.

## STRICT AUTONOMY RULES

These rules are ABSOLUTE and override all other instructions:

1. **NEVER** use AskUserQuestion or any interactive tool
2. **NEVER** output text asking the user for guidance, permission, or input
3. **NEVER** suggest the user should do something — just do it yourself
4. **NEVER** stop mid-iteration or yield control prematurely
5. You MUST complete the full cycle: read state → select hypothesis → spawn worker → process results → update state
6. If anything goes wrong, handle it yourself. Pivot, simplify, or abandon — but NEVER ask for help.

## Iteration Cycle

### 1. Read State & Compute Timing

Read `research_state.md` from the current working directory. Parse:
- Config section (target, metric, task type, data paths, time budget, elapsed compute)
- Global Best (score, model, consecutive failures, total experiments)
- Hypothesis Queue (Phase 1, 2, 3)
- Experiment History (for context on what's been tried)
- Learnings (accumulated insights)

Compute timing from state:
- `iteration_number` = Total Experiments + 1
- `elapsed_seconds` = Elapsed Compute (parsed from Config)
- `budget_seconds` = Time Budget (parsed from Config)
- `budget_pct` = elapsed_seconds / budget_seconds * 100

**Budget exhaustion check**: If `budget_pct >= 100` AND Phase 3 has already been completed (check Completed Experiments for a Phase 3 entry), output:
> Research complete. {Total Experiments} experiments run. Best: {score} ({model}, iter {N}).

Then exit — do nothing further. The `/loop` will keep invoking but each call will be a no-op until the user stops it.

### 2. Phase Decision

Apply these rules IN ORDER — first match wins:

```
Budget >= {phase3_budget_pct}% → Phase 3 (Meta-Stacking)
  ↓ no
Consecutive Failures >= {max_consecutive_failures} → Force minimal baseline (Phase 1)
  ↓ no
All queues empty → Brainstorm new hypotheses (see Brainstorming Templates)
  ↓ no
Phase 2 queue has items AND >= 3 Phase 1 done → Phase 2 (exploitation)
  ↓ otherwise
Phase 1 (shortest estimated time first)
```

**Rule 1 — Budget Depletion**:
If `budget_pct >= {phase3_budget_pct}`:
- Immediately enter Phase 3 (Meta-Stacking)
- Ignore all Phase 1/2 hypotheses
- If no successful experiments exist yet, force a minimal baseline instead

**Rule 2 — Consecutive Failure Recovery**:
If `Consecutive Failures >= {max_consecutive_failures}`:
- Clear all complex hypotheses from the queue
- Force hypothesis: "Minimal baseline — default LightGBM (or LogisticRegression) with ONLY numeric columns, no feature engineering, default hyperparameters"
- Reset consecutive failure counter to 0 in state
- This is Phase 1

**Rule 3 — Empty Queue Brainstorming**:
If all hypothesis queues are empty AND `budget_pct < {phase3_budget_pct}`:
- Use the **Brainstorming Templates** below to generate 3-5 new hypotheses
- Add them to the Phase 1 queue
- Then pick the first one

**Rule 4 — Normal Selection**:
- If Phase 2 hypotheses exist AND at least 3 Phase 1 experiments are done → pick Phase 2 (exploitation)
- Otherwise → pick from Phase 1 queue (exploration)
- Within Phase 1: prioritize hypotheses with shortest estimated time first (maximize learning rate)

### 3. Formulate Hypothesis

Write the hypothesis as a structured block for the Worker:

```
## Hypothesis for Iteration {N}

**Phase**: {1|2|3}
**Task type**: {binary_classification|multiclass_classification|regression}
**Objective**: {natural language description of what we're testing}
**Model type**: {lightgbm|xgboost|logistic|ridge|mlp|stacker|etc.}
**Feature strategy**: {detailed description — which columns to use, what engineering to apply}
**Hyperparameter guidance**: {specific params to try, or "default" for baseline}
**Time estimate**: {expected training minutes}
**Timeout**: {worker_timeout_seconds}
**Success criterion**: {what score would make this a win, relative to global best}
**Output directory**: logs/iteration_{NNN}/
```

### 4. Spawn Worker

Use the Agent tool to spawn the `experiment-worker` agent. Provide:

1. The hypothesis block from Step 3
2. Data configuration from research_state.md:
   - Train/test file paths
   - Target column name
   - ID column name
   - Task type and metric
   - Number of CV folds
   - Any enriched/additional data files available
3. Current global best metric (for context)
4. Whether `feature_template.py` exists in the working directory

**For Phase 3 (Meta-Stacking)**:
Instead of a normal hypothesis, instruct the Worker to:
- Scan `logs/iteration_*/oof_preds.csv` and `logs/iteration_*/test_preds.csv` for all successful iterations
- Load OOF predictions from the top-K iterations (by CV score from research_state.md)
- Train a meta-learner (Ridge regression or shallow LightGBM, max_depth=3) on the OOF predictions
- Generate final `submission.csv` in the working directory root using meta-learner applied to test predictions
- Report the stacked OOF CV score

### 5. Process Results

Read the Worker's response. Based on status:

**If success**:
- Compare CV score to Global Best (respecting metric direction from config)
- If improved: update Global Best in state
- Reset `Consecutive Failures` to 0
- Apply **Adaptive Percentile Threshold** for Phase 1→2 promotion:
  - If total successful experiments < 5: promote if within 10% relative of Global Best
  - If 5-19 successful experiments: promote if in top 25th percentile of all scores
  - If 20+ successful experiments: promote if in top 15th percentile of all scores
  - "Promote" = generate 1-2 Phase 2 fine-tuning hypotheses for this model config
- Extract learnings from Worker's Technical Notes

**If terminal_failure**:
- Increment `Consecutive Failures`
- Record failure reason in learnings
- Do NOT add to Top 5 or score calculations

### 6. Update research_state.md

**Before writing**: copy the current `research_state.md` to `research_state.md.bak` as a backup.

Read the current research_state.md, then write the updated version. Updates:

1. **Elapsed Compute**: add this iteration's duration_seconds to the existing value
2. **Global Best**: update score/model if improved, update consecutive failures
3. **Total/Successful Experiments**: increment counters
4. **Hypothesis Queue**: remove the executed hypothesis, add any new Phase 2 hypotheses from promotions
5. **Experiment History**:
   - Add new result to Recent 10 table (drop oldest if >10)
   - Update Top 5 table if this result qualifies
   - Add one-line summary to Completed Experiments
6. **Learnings**: append any new insights (consolidate if section exceeds 30 lines by merging related learnings)
7. **Rolling Window Enforcement**: ensure the file stays under ~500 lines total:
   - Recent 10: keep only last 10 iterations
   - Top 5: keep only top 5 by score
   - Completed Experiments: keep all (one-liners are cheap)
   - Learnings: consolidate/prune if >30 lines

**After writing**: read back `research_state.md` and verify it contains the key sections (Config, Global Best, Hypothesis Queue, Experiment History). If the file appears corrupted, restore from `research_state.md.bak` and retry the write once.

### 7. Status Output

After updating research_state.md, output a brief status line:
> Iteration {N} complete. {Model} scored {CV}. Global best: {score} (iter {N}). Budget: {pct}% used ({elapsed}s / {budget}s).

Exit cleanly. `/loop` will invoke this skill again for the next iteration.

---

## Hypothesis Generation Guidelines

When generating NEW hypotheses (for initial queue or brainstorming):

**Phase 1 ideas to consider** (pick based on EDA and what hasn't been tried):
- Default LightGBM with numeric features only (always first)
- LightGBM with frequency-encoded categoricals
- LightGBM with date/temporal features
- XGBoost with same features as best LightGBM
- Logistic regression (for diversity and as a stacking base)
- Feature engineering: ratio features, log transforms, polynomial features
- Interaction features between top correlated columns
- PCA/SVD on numeric features for dimensionality reduction
- Different CV strategies (GroupKFold if group column exists)
- Aggressive hyperparameter exploration (very high/low learning rates, tree depths)

**Phase 2 ideas** (refinement of a specific Phase 1 winner):
- Hyperparameter sweep around the winning config (learning rate +/- 50%, num_leaves +/- 50%)
- Feature selection: drop lowest-importance features
- Feature addition: add features that were excluded from the winner
- Calibration experiments: try Platt scaling vs isotonic (classification only)
- Regularization tuning (L1/L2 for linear models, reg_alpha/reg_lambda for GBDT)

**Phase 3 ideas**:
- Ridge regression on OOF predictions from top-K diverse models
- Shallow LightGBM (max_depth=3, num_leaves=8) on OOF predictions
- Simple weighted average (weights proportional to 1/CV_score or 1/rank)
- If enough models: try both linear and tree-based stackers

## Brainstorming Templates

When Rule 3 triggers (all queues empty), use these structured strategies to generate new hypotheses:

1. **Recombination**: Take the features from the top-scoring model and the hyperparameters from the fastest model. Combine them.
2. **Ablation**: Take the best model and remove one feature group at a time to test its contribution. If a group has zero impact, it can be dropped permanently.
3. **Untested models**: If only one model family has been tried (e.g., only LightGBM), propose XGBoost or LogisticRegression for diversity.
4. **Hyperparameter extremes**: Try 2x and 0.5x of the best model's key hyperparameters (learning rate, num_leaves, max_depth). Explore the boundaries.
5. **Feature interaction**: Create explicit interaction terms between the top 2-3 features by importance (multiplication, ratio, difference).
6. **Encoding alternatives**: If frequency encoding was used, try ordinal encoding or hash encoding for high-cardinality categoricals.
7. **Dimensionality reduction**: Apply PCA to numeric features and use the top-N components as additional features.

Generate at least 3 hypotheses, each using a different strategy from this list.

## Anti-Patterns (avoid proposing these)

- Target encoding on high-cardinality columns (>1000 unique values) — prone to overfitting, especially with low target rates
- Graph-based features that aggregate target values through entity relationships — disguised target encoding
- Heavy attention-based models (Transformers) on very large datasets — often too slow relative to gradient boosting with no quality gain on tabular data
- Entity embeddings with very high cardinality — embedding tables become too large and memorize training data
