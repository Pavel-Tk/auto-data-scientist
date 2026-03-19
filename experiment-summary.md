# CLAUDE.md — Project Guide for AI Assistants

## Project Overview

This is **Assignment 2** for the SFMBA S5/S6 Product and Marketing Analytics course. The task is a **Kaggle-style binary classification competition**: predict whether an e-commerce order will be cancelled (`is_cancelled`).

- **Evaluation metric**: Log-loss (lower is better)
- **Submission format**: `order_item_id`, `is_cancelled` (probability)
- **Data**: Train/test CSVs from an online retailer with order-level features

---

## Architecture: Auto-Recursive ML Pipeline

The project implements a **Karpathy-style automated research loop**:

1. `auto_research.py` — main loop orchestrator
2. `llm_interface.py` — calls OpenRouter API (Claude Opus 4.6) to propose experiments
3. `utils.py` — data loading, CV scoring, feature engineering, checkpointing
4. `config.py` — all configuration (paths, API keys, hyperparameter ranges)
5. `check_progress.py` — utility to display pipeline status and iteration summary

### Loop Behaviour

Each iteration:
1. LLM receives: data summary + full history of past experiments (top 15 by CV score + most recent 5)
2. LLM proposes: model type, features, hyperparameters, reasoning (JSON format)
3. `llm_interface.generate_training_code()` produces a self-contained Python script
4. Code is written to `logs/iteration_NNN/code.py` and executed as a subprocess (30-min timeout)
5. Results (CV score, std, fold details, feature importance, predictions) saved to `logs/iteration_NNN/results.json`
6. Checkpoint is updated; loop continues

### Boldness Cycling

The LLM receives a different exploration directive every iteration (cycle of 4):
- **Cycle 1 (CONSERVATIVE)**: Refine the best-performing approach with small tweaks
- **Cycle 2 (MODERATE)**: Meaningful variation on a promising approach
- **Cycle 3 (ADVENTUROUS)**: Push boundaries, try combining ideas
- **Cycle 0 (BOLD)**: Force novelty — different model type, radical feature changes, extreme hyperparameters (temp=0.9 vs 0.7 for others)

### Objective Function (see `specific_idea.md` for full rationale)

- **Primary signal**: 5-fold stratified CV log-loss on training data only
- **Stability penalty**: add `alpha * CV_std` (alpha=0.1)
- **Complexity penalty**: add `beta * n_features` (beta=0.01)
- **Adjusted score formula**: `CV_mean + 0.1*CV_std + 0.01*n_features` (lower is better)
- **NOT used for loop decisions**: 50% visible test score (logged separately)
- **Final output**: softmax-weighted ensemble of top models by raw CV score (adjusted score over-penalizes feature count)

---

## Experiment Results Summary (as of 64 iterations)

### Overall Statistics
- **Total iterations**: 64 (59 LLM/manual LightGBM + 5 neural network)
- **Successful**: 44 (69%)
- **Failed/Timed out/Stopped**: 20 (31%)
- **Best single-model CV log-loss**: **0.02215** (iteration 41, LightGBM — PCA30 embeddings) / **0.02216** (iteration 45, PCA20 + tuned params)
- **Best NN CV log-loss**: **0.02660** (iteration 61, Regularized MLP with BN+Dropout+OneCycleLR)
- **Best ensemble OOF log-loss**: **0.02126** (10-model ensemble with isotonic calibration, iters 40-52 excluding pseudo-labeling)
- **Previous best (pre-enrichment)**: 0.02608 (iteration 39, XGBoost)
- **Total improvement from baseline**: 0.0961 → 0.02126 = **78% reduction in log-loss** across all phases

> [!CAUTION]
> **WARNING: ITERATIONS 51-58 ARE BROKEN AND SHOULD NOT BE USED FOR AN ENSEMBLE.**
> These iterations used pseudo-labeling that caused severe training-on-test data leakage, leading to artificially low CV scores (0.02176) but catastrophic test scores (0.705). The folder names have been marked with `-BROKEN`.
> 
> Iteration 59 implemented leak-free pseudo-labeling. While it fixed the leakage, the resulting 50% Kaggle test log-loss was **0.07860 (very bad)**, proving that pseudo-labeling is fundamentally ineffective/unstable for this highly imbalanced dataset.

### Phase 2 — Pipeline Improvements (Iterations 36–39)

Four improvements were implemented after iteration 35: early stopping, frequency encoding, isotonic calibration, product name features.

| Iter | Model | CV Score | Features | Duration | Notes |
|------|-------|----------|----------|----------|-------|
| 36 | xgboost | timeout | 19 | >1800s | lr=0.01 too slow even with early stopping |
| 37 | lightgbm | 0.02839 | 19 | 1262s | Early stopping + calibration |
| 38 | lightgbm | timeout | 22 | >1800s | lr=0.015 with 22 features too slow |
| 39 | xgboost | 0.02608 | 24 | 1885s | All 4 improvements; best pre-enrichment score |

### Phase 3 — Data Enrichment (Iterations 40–41)

Product data enriched with LLM classification (Gemini 2.0 Flash) and sentence embeddings (MiniLM). See `data/enriched/walkthrough.md`.

| Iter | Model | CV Score | Features | Duration | Notes |
|------|-------|----------|----------|----------|-------|
| 40 | lightgbm | 0.02250 | 40 | 1164s | LLM features + PCA10 embeddings |
| **41** | **lightgbm** | **0.02215** | **60** | **3049s** | **PCA30 embeddings; best single-model CV** |

### Phase 4 — Manual Optimization (Iterations 42–46)

Five manually-designed iterations with cross-iteration learning:

| Iter | Model | CV Score | Features | Duration | Notes |
|------|-------|----------|----------|----------|-------|
| 42 | xgboost | 0.02449 | 40 | 2258s | XGBoost with enriched data; 2x worse than LightGBM |
| 43 | lightgbm | 0.02248 | 40 | 960s | MPNet ≈ MiniLM at same PCA dims |
| 44 | lightgbm | 0.02252 | 50 | 990s | Dual embeddings; no gain, depth > breadth |
| **45** | **lightgbm** | **0.02216** | **50** | **708s** | **Tuned params (127 leaves, lr=0.07); matched iter 41 at 4.3x speed** |
| 46 | lightgbm | 0.02227 | 40 | 582s | Pruned 10 low-importance features |

### Phase 5 — Exploration Experiments (Iterations 47–52)

Six targeted experiments testing distinct ideas against iter 46 baseline (CV=0.02227):

| Iter | Model | CV Score | Features | Duration | Notes |
|------|-------|----------|----------|----------|-------|
| 47 | lightgbm | 0.02233 | 44 | 652s | User behavioral features (order count, spend, avg GMV); marginal gain |
| 48 | lightgbm | 0.02227 | 40 | 1055s | Target encoding (alpha=500); TE features excluded by bug, ran as baseline |
| 49 | lightgbm | 0.02228 | 45 | 630s | Order composition features (53% multi-item orders, mean 3.45 items); no gain |
| 50 | lightgbm | 0.02448 | 40 | 1143s | is_unbalance=True (scale_pos_weight=186); worse AUC (0.864 vs 0.905), calibration can't recover |
| 51 | lightgbm | 0.02187 | 40 | 1072s | **[BROKEN - LEAKAGE]** Pseudo-labeling. CV is fake. Kaggle test = 0.705 |
| 52 | lightgbm | 0.02208 | 41 | 1041s | **[BROKEN - LEAKAGE]** Isolation Forest anomaly score + pseudo-labeling |

**Key findings:**
- **Pseudo-labeling caused catastrophic data leakage**: Iteration 51 trained fold models on pseudo-labeled test rows, then evaluated test predictions using those same models. This train-on-test leakage inflated CV to 0.02187 but destroyed the Kaggle test score (0.705).
- **Isolation Forest anomaly score** (iter 52) used the same broken template.
- **User behavioral features provided marginal signal** — user_order_count/spend/avg_gmv are mostly redundant with user_id_freq.
- **Target encoding caused severe overfitting** — even with alpha=50-500 smoothing, OOF target encoding on high-cardinality entities causes the model to overfit to the encoding and crash AUC. Dangerous technique for this dataset.
- **Order composition features are noise** — despite 53% multi-item orders, order structure doesn't predict cancellation.
- **Class reweighting hurts** — is_unbalance=True degrades AUC from 0.905 to 0.864. The model learns better without artificial class weighting; isotonic calibration handles probability correction.

### Phase 6 — Advanced Techniques (Iterations 53–58)

> [!CAUTION]
> **ALL MODELS IN THIS PHASE ARE BROKEN.** They inherit the train-on-test pseudo-labeling leakage from iteration 51. While the techniques can be analyzed for insights, the models CANNOT be used in an ensemble. The iter folders have been marked with `-BROKEN`.

| Iter | Model | CV Score | Features | Duration | Notes |
|------|-------|----------|----------|----------|-------|
| 53 | lightgbm | 0.02327 | 40 | ~900s | **[BROKEN]** Negative downsampling (1:10) + base rate correction |
| 54 | lightgbm | 0.02187 | 40 | ~1100s | **[BROKEN]** Beta calibration (3-param) |
| 55 | lightgbm | 0.02186 | 44 | ~1200s | **[BROKEN]** Temporal velocity features (user recency, node stress) |
| 56 | lightgbm | 0.02816 | 44 | ~1300s | **[BROKEN]** Graph neighborhood features (OOF-safe) |
| 57 | lightgbm | 0.02204 | 40 | ~1400s | **[BROKEN]** Two-stage cascade (bouncer + detective) |
| 58 | lightgbm | 0.02176 | 44 | ~1500s | **[BROKEN]** Temporal features + tuned hyperparams |

**Key findings (for insight only):**
- **Temporal features add marginal but real signal** — user_recency_days and node_stress features (iter 55) did show small but consistent signal.
- **Graph neighborhood features are target encoding in disguise** — Computing cancellation rates through user-SKU-node bipartite graph (iter 56) is equivalent to smoothed target encoding through intermediary entities. CV crashed to 0.02816.
- **Negative downsampling loses information** — 1:10 downsampling with analytical base rate correction (iter 53) produced CV=0.02327. Discarding 90% of negatives loses subtle patterns in this dataset.
- **Two-stage cascade creates discontinuity** — Shallow bouncer + deep detective (iter 57) achieved CV=0.02204, worse than single-stage. The prediction discontinuity damages log-loss.
- **Ensemble scale mismatch problem** — Phase 6 models predicted mean ~0.17 after isotonic calibration (due to applying non-test-calibrated leaked probabilities). This was a major red flag that pseudo-labeling was leaking.

### Phase 7 — Leakage Fixes & Verification (Iteration 59)

| Iter | Model | CV Score | Features | Duration | Notes |
|------|-------|----------|----------|----------|-------|
| 59 | lightgbm | 0.02312 | 44 | 1751s | Leakage-free pseudo-labeling. Kaggle test = **0.07860** (terrible) |

**Key findings:**
- Iteration 59 fixed the pseudo-label leakage completely (Phase 1 generates clean pseudo-labels from iter 46; Phase 2 trains final model; proper calibration).
- The resulting 50% Kaggle test log-loss was **0.07860**. This confirms that pseudo-labeling (even when done correctly) causes the model's predictive distributions to distort irrecoverably on test data for this specific imbalanced problem.
- **Conclusion**: Abandon pseudo-labeling. Revert to standard supervised ensembles.

### Phase 8 — Neural Network Experiments (Iterations 60–64)

Five progressive NN iterations exploring MLP architectures for ensemble diversity with existing LightGBM models. All used 44 features from pre-computed parquet, bias init to log(pos_rate/neg_rate) ≈ -5.3, isotonic calibration, and no pos_weight (learned from iter 50 that class reweighting hurts).

| Iter | Model | CV Score | AUC | Features | Duration | Notes |
|------|-------|----------|-----|----------|----------|-------|
| 60 | MLP (vanilla) | 0.02682 | 0.852 | 44 | 4457s | 2-layer MLP (256-128), Adam lr=1e-3, no regularization |
| **61** | **MLP (regularized)** | **0.02660** | **0.858** | **44** | **4042s** | **BN+Dropout(0.3/0.2/0.1)+OneCycleLR(3e-3); best NN** |
| 62 | Entity Embedding | 0.02746 | 0.839 | 41+emb | 3592s | user/sku/node embeddings (4.8M params); severe overfitting by epoch 6 |
| 63 | FT-Transformer | ~0.02894 | 0.825 | 44 | ~14000s (est) | Stopped after fold 1 — 46 min/fold, worst CV, self-attention too slow |
| 64 | Snapshot+Mixup | ~0.02735 | ~0.851 | 44 | ~4000s (est) | Stopped after 2 folds — snapshot averaging hurt (early snapshots dilute) |

**Key findings:**
- **Regularized MLP (iter 61) is the best NN**: BN + progressive dropout + OneCycleLR improved all metrics over vanilla MLP — CV 0.02660 (vs 0.02682), AUC 0.858 (vs 0.852), CV std 0.00007 (vs 0.00017), and 10% faster.
- **Entity embeddings overfit catastrophically**: 100K user + 50K SKU hash buckets (4.8M embedding params) memorized training data within 3-6 epochs. Train loss crashed from 0.031 to 0.014 while val loss spiked from 0.028 to 0.045. Mirrors the project's earlier finding that high-cardinality entity features (target encoding, graph features) cause overfitting. Frequency encoding captures entity signal more robustly.
- **FT-Transformer was too slow and underperformed**: Self-attention over 45 feature tokens is O(n^2) per sample. 76K params but 46 min/fold (vs 13 min for MLP). CV 0.02894, AUC 0.825 — worst of all architectures.
- **Snapshot ensemble averaging hurt**: Averaging 4 snapshots (at LR restart points) diluted the best snapshot with weaker earlier ones. Fold 1: ensemble 0.02745 vs best single snapshot 0.02726.
- **NNs provide ensemble diversity but not competitive CV**: Best NN (0.02660) is 20% worse than best LightGBM (0.02215). AUC gap is 0.858 vs 0.907. However, NN predictions may add diversity value in the ensemble due to fundamentally different decision boundaries.
- **pos_weight removal was critical**: Original scripts used pos_weight~186 (same as is_unbalance=True), producing catastrophic fold 1 logloss of 0.415. Removing it immediately fixed predictions (0.027). This confirms iter 50's finding that class reweighting destroys probability calibration.

### Top 10 Successful Models (CLEAN MODELS ONLY)

*(Excludes iterations 51-59 completely to avoid ANY contamination from pseudo-labeling)*

| Iter | Model | CV Score | CV Std | Features | Duration |
|------|-------|----------|--------|----------|----------|
| 41 | lightgbm | 0.02215 | 0.00010 | 60 | 3049s |
| 45 | lightgbm | 0.02216 | 0.00009 | 50 | 708s |
| 46 | lightgbm | 0.02227 | 0.00010 | 40 | 582s |
| 48 | lightgbm | 0.02227 | 0.00000 | 40 | 1055s |
| 49 | lightgbm | 0.02228 | 0.00006 | 45 | 630s |
| 47 | lightgbm | 0.02233 | 0.00011 | 44 | 652s |
| 43 | lightgbm | 0.02248 | 0.00008 | 40 | 960s |
| 40 | lightgbm | 0.02250 | 0.00007 | 40 | 1164s |
| 44 | lightgbm | 0.02252 | 0.00010 | 50 | 990s |
| 42 | xgboost | 0.02449 | 0.00016 | 40 | 2258s |

### Best Clean Configurations

**Iteration 41 — Best CV (0.02215)**:
- **Model**: LightGBM
- **Features (60)**: 8 raw numeric + 6 engineered ratios + 5 date features + 3 freq-encoded + 2 product name + 2 LLM category + 4 LLM ordinal + 30 MiniLM PCA
- **Key hyperparameters**: num_leaves=63, learning_rate=0.05, early_stopping=50
- **Calibration**: Isotonic regression applied

**Iteration 45 — Practical Best (0.02216, 4.3x faster)**:
- **Model**: LightGBM (tuned hyperparameters)
- **Features (50)**: Same non-embedding features + 20 MiniLM PCA embeddings
- **Key hyperparameters**: num_leaves=127, learning_rate=0.07, n_estimators=2000
- **Why practical best**: Same CV as iter 41 but trains in 12 min vs 51 min. Higher tree capacity (127 leaves) + faster learning rate + less regularization matched the default model's quality with 10 fewer features.

### Best Model Configurations



**Iteration 41 — Previous Best CV (0.02215)**:
- **Model**: LightGBM
- **Features (60)**: 8 raw numeric + 6 engineered ratios + 5 date features + 3 frequency-encoded originals + 2 product name features + 2 LLM category freq-encoded + 4 LLM ordinal-encoded + 30 MiniLM PCA embeddings
- **Key hyperparameters**: num_leaves=63, learning_rate=0.05, n_estimators=2000, max_depth=-1, min_child_samples=100, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, early_stopping_rounds=50
- **OOF metrics**: log-loss=0.02215, AUC=0.907, Brier=0.0043, Accuracy=99.54%
- **Top features by importance**: margin_ratio, fulfillment_node_freq, emb_minilm_pca_4, emb_minilm_pca_8, emb_minilm_pca_6 (embeddings dominate the top features)
- **Calibration**: Isotonic regression improved CV from 0.02239 to 0.02215
- **Drawback**: 51-minute training time

**Iteration 45 — Practical Best (0.02216, 4.3x faster)**:
- **Model**: LightGBM (tuned hyperparameters)
- **Features (50)**: Same non-embedding features as iter 41 + 20 MiniLM PCA embeddings (PCA20 sweet spot)
- **Key hyperparameters**: num_leaves=127, learning_rate=0.07, n_estimators=2000, max_depth=-1, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.5, early_stopping_rounds=50
- **Why practical best**: Same CV as iter 41 but trains in 12 min vs 51 min. Higher tree capacity (127 leaves) + faster learning rate + less regularization matched the default model's quality with 10 fewer features.

### Model Type Performance
- **LightGBM**: Dominant model type throughout. Best clean CV=0.02215 (iter 41) with 60 features (PCA30 embeddings). Significantly faster than XGBoost — iter 45 trains in 12 min vs XGBoost's 38 min on same feature count.
- **XGBoost**: Best CV=0.02449 (iter 42) with enriched data (40 features, PCA10); 0.02608 (iter 39) pre-enrichment. Consistently ~2x worse log-loss than LightGBM on enriched data, and ~2.5x slower. Not competitive for this dataset.
- **Neural Networks (PyTorch)**: Best CV=0.02660 (iter 61, regularized MLP with BN+Dropout+OneCycleLR). 20% worse than LightGBM (0.02215) and AUC 0.858 vs 0.907. Entity embeddings overfit catastrophically; FT-Transformer too slow. NNs struggle on this tabular dataset but offer ensemble diversity.
- **Logistic regression**: Best CV=0.03234 (iter 28). Serves as diversity baseline in ensemble.
- **CatBoost**: CV=0.03407 (iter 12). LLM proposed it but it's not in ALLOWED_MODELS — the generated code ran it anyway.
- **Random Forest**: Both attempts timed out (5.4M rows + many trees = too slow).

### Failure Analysis
- **Timeouts (15/17 failures)**: Caused by high `n_estimators` (2000+) combined with slow learning rates on 5.4M rows. Even with early stopping, XGBoost with lr=0.01 (iter 36) and LightGBM with lr=0.015 + 22 features (iter 38) timed out — early stopping helps but doesn't eliminate the issue for very slow configurations.
- **Errors (2/17)**: Iterations 34–35 failed silently (empty error messages in results.json) — likely LLM response parsing issues or code generation problems.
- **Pattern**: Timeouts correlate with low learning rates (< 0.03) more than feature count. A learning rate of 0.05 appears to be the safe minimum for 5.4M rows within 30 minutes.

---

## Data Details

### Dataset Size
- **Train**: 5,420,537 rows x 15 columns
- **Test**: 1,576,463 rows x 14 columns
- **Cancellation rate**: 0.54% (highly imbalanced — only ~29,000 positive cases in 5.4M)

### Raw Columns
`order_date`, `order_item_id`, `user_id`, `fulfillment_node`, `retail_sku`, `order_id`, `gmv_price`, `effective_price`, `total_gmv`, `units`, `total_margin`, `quantity_savings`, `smart_cart`, `total_savings`, `is_cancelled`

### Engineered Features (computed in generated code)
- `price_diff` = gmv_price - effective_price
- `price_ratio` = effective_price / gmv_price
- `savings_ratio` = (quantity_savings + smart_cart) / total_gmv
- `margin_ratio` = total_margin / total_gmv
- `log_units` = log1p(units)
- `price_per_unit` = total_gmv / units
- Date features: year, month, day, dayofweek, quarter, is_weekend

### Frequency-Encoded Features (auto-computed in code template)
- `user_id_freq` = frequency of user_id in training data (normalized count)
- `fulfillment_node_freq` = frequency of fulfillment_node in training data
- `retail_sku_freq` = frequency of retail_sku in training data
- Note: `order_id` excluded — nearly unique per row, so frequency ≈ constant

### Product Name Features (auto-computed in code template)
- `product_name_len` = character length of product name (joined from product_names.csv via retail_sku)
- `product_name_words` = word count of product name

### Enriched Features — LLM Classification (from `data/enriched/product_llm_features.csv`)
Gemini 2.0 Flash classified all 686K unique products into 6 categorical fields:
- `category_freq` = frequency-encoded product category (15 categories: Food & Beverage, Health & Wellness, Household, etc.)
- `subcategory_freq` = frequency-encoded product subcategory (115K unique values)
- `demographic_ord` = ordinal-encoded target demographic (General=0, Baby/Infant=1, Children=2, Teen=3, Adult=4, Senior=5, Pet=6)
- `price_tier_ord` = ordinal-encoded price tier (Budget=0, Mid-Range=1, Premium=2, Luxury=3)
- `durability_ord` = ordinal-encoded durability (Consumable=0, Semi-Durable=1, Durable=2)
- `seasonality_ord` = ordinal-encoded seasonality (Year-Round=0, Seasonal=1, Holiday=2)

### Enriched Features — PCA Embeddings (from `data/enriched/product_embeddings_minilm_pca*.csv`)
MiniLM sentence-transformer (all-MiniLM-L6-v2) encoded product names into 384-dim vectors, PCA-reduced:
- `emb_minilm_pca_0` through `emb_minilm_pca_29` = 30 PCA components (PCA30 in iter 41; PCA20 in iter 45; PCA10 in iter 40)
- **PCA20 is the sweet spot**: Matches PCA30 CV (0.02216 vs 0.02215) at 4.3x speed. PCA10 slightly worse (0.02250).
- **These dominate feature importance**: 8 of the top 13 features in iter 40 were embedding components
- MPNet embeddings also available (`product_embeddings_mpnet_pca*.csv`) — tested in iter 43, virtually identical to MiniLM at same PCA dims

### Unused Columns (remaining potential)
- `order_id` — not unique per row (53% multi-item orders, mean 3.45 items/order), but order composition features didn't improve CV (iter 49)

### Tested but Not Beneficial
- **MPNet embeddings** — tested in iter 43 (PCA10). CV=0.02248 vs MiniLM's 0.02250 — virtually identical. Higher-dimensional base model (768 vs 384) provided no advantage. Dual MiniLM+MPNet tested in iter 44 — no gain either, confirming the two models capture redundant semantic signal.
- **User behavioral features** (iter 47) — user_order_count, user_total_spend, user_avg_gmv, user_is_first_order. CV=0.02233 vs baseline 0.02227 — marginal, mostly redundant with user_id_freq.
- **Order composition features** (iter 49) — order_item_count, order_total_value, is_most_expensive_in_order, item_value_share, order_unique_skus. CV=0.02228 — no improvement despite 53% multi-item orders.
- **Target encoding** (iter 48) — OOF target encoding for fulfillment_node/retail_sku caused severe overfitting with alpha=50. With alpha=500, the features were too smooth to add signal. Dangerous technique for this dataset's cardinality.
- **Class reweighting / is_unbalance** (iter 50) — is_unbalance=True (scale_pos_weight≈186). Degraded AUC from 0.905 to 0.864. Calibration partially recovered log-loss to 0.02448 but still 10% worse than baseline.
- **Isolation Forest anomaly score** (iter 52) — IsolationForest trained on negative-class only (5.39M rows, max_samples=10000). Anomaly score ranked #2 by feature importance (6280.6), but CV=0.02208 vs iter 51's 0.02187. Cancelled orders had slightly higher anomaly scores (mean -0.102 vs -0.116 for negatives) but the signal was redundant with features LightGBM already uses. Still useful for ensemble diversity.
- **Negative majority downsampling** (iter 53) — 1:10 downsampling of negatives with analytical base rate correction (p_corrected = p / (p + (1-p)*w)). CV=0.02327, 6.4% worse than baseline. Even with recalibration, discarding 90% of negatives loses subtle patterns critical for this low-positive-rate (0.54%) dataset.
- **Beta calibration** (iter 54) — 3-parameter model (a*log(p) - b*log(1-p) + c) fitted via scipy L-BFGS-B. CV=0.02190 vs isotonic's 0.02187. Tied with isotonic — the extra parametric flexibility doesn't help when isotonic already handles the well-separated probability distribution.
- **Graph neighborhood features** (iter 56) — OOF-safe cancellation rate aggregation through user-SKU-node bipartite graph (user_cancel_neighbor_rate, node_cancel_sku_rate, sku_cancel_node_rate, user_sku_diversity). CV=0.02816, AUC dropped from 0.913 to 0.806. These features are target encoding through intermediary entities — even with Bayesian smoothing (alpha=100), the signal leaks through high-cardinality SKU/node joins.
- **Two-stage cascade** (iter 57) — Shallow screener (15 leaves) + deep classifier (255 leaves) on suspicious subset (~32% of rows where Stage 1 > threshold). CV=0.02204, worse than single-stage. The prediction discontinuity at the Stage 1 threshold damages log-loss. Isotonic calibration partially smooths it but can't fully recover.

### Enriched Features — Anomaly Scores (from `data/enriched/anomaly_scores_*.csv`)
Isolation Forest (200 estimators, max_samples=10000) trained on negative-class training data only:
- `anomaly_score` = negated decision_function (higher = more anomalous)
- Pre-computed for all train and test rows, keyed by `order_item_id`
- Cancelled orders: mean=-0.102, std=0.038; Non-cancelled: mean=-0.116, std=0.023

### Enriched Features — Pseudo-Labels (from `data/enriched/pseudo_labels_iter51.csv`)
High-confidence predictions from iter 51's Phase 1 model on test data:
- `pseudo_label` = 0 or 1, `pred_prob` = raw probability, `is_confident` = meets threshold, `weight` = 0.5 for confident rows
- Thresholds: p < 0.001 (negative), p > 0.03 (positive)

---

## Current Submission

- **File**: `submission.csv` (1,576,463 rows)
- **Ensemble**: Softmax-weighted ensemble of clean models (iters 40, 41, 43, 44, 45, 46, 47, 48, 49) by raw CV score
- **Ensemble OOF log-loss**: 0.02126
- **Calibration**: Ensemble-level isotonic regression applied on OOF predictions
- **Note**: This ensemble explicitly excludes iterations 51-59 to guarantee no contamination from the catastrophic pseudo-labeling methodology.

---

## Key Files

| File | Purpose |
|------|---------|
| `auto_research.py` | Main pipeline loop + ensemble creation |
| `llm_interface.py` | LLM prompt building, API calls, response parsing |
| `utils.py` | Data loading, CV scoring, feature engineering, checkpointing |
| `config.py` | All configuration — paths, API key, model list, hyperparams |
| `check_progress.py` | Quick progress summary (iterations, scores, failures) |
| `rules.txt` | Original competition description and evaluation metric |
| `specific_idea.md` | Design rationale for the 3-tier objective function |
| `submission.csv` | Current best submission file |
| `data/train.csv` | Training data (5.4M rows) |
| `data/test.csv` | Test data (1.6M rows, 50% visible / 50% hidden) |
| `data/product_names.csv` | Product name lookup table (joined via retail_sku for name length/word count features) |
| `data/enriched/` | Enriched product data (LLM features, PCA embeddings) — see walkthrough.md |
| `data/enriched/product_llm_features.csv` | LLM-classified product categories (6 fields, 686K products) |
| `data/enriched/product_embeddings_minilm_pca*.csv` | MiniLM PCA-reduced embeddings (10/20/30 dims) |
| `data/enriched/product_embeddings_mpnet_pca*.csv` | MPNet PCA-reduced embeddings (10/20/30 dims) |
| `data/enriched/anomaly_scores_train.csv` | Isolation Forest anomaly scores for training data (5.4M rows) |
| `data/enriched/anomaly_scores_test.csv` | Isolation Forest anomaly scores for test data (1.6M rows) |
| `data/enriched/pseudo_labels_iter51.csv` | Pseudo-labels from iter 51 Phase 1 predictions on test data |
| `data/enriched/walkthrough.md` | Full enrichment methodology documentation |
| `data/enriched/train_NN_features.parquet` | Pre-computed 44-feature train dataset for NN iterations (fast loading) |
| `data/enriched/test_NN_features.parquet` | Pre-computed 44-feature test dataset for NN iterations |
| `logs/iteration_060/train_iter.py` | Vanilla MLP baseline (CV=0.02682) |
| `logs/iteration_061/train_iter.py` | Regularized MLP — best NN (CV=0.02660) |
| `logs/iteration_062/train_iter.py` | Entity Embedding Network (CV=0.02746, overfitting) |
| `logs/iteration_063/train_iter.py` | Mini FT-Transformer (stopped — too slow) |
| `logs/iteration_064/train_iter.py` | Snapshot Ensemble + Mixup (stopped — averaging hurts) |
| `logs/` | Per-iteration outputs: code, prompt, response, results |
| `logs/checkpoint.json` | Resume state: last completed iteration + full history |
| `logs/history.json` | All experiment results |

---

## Running the Pipeline

```bash
# Run with default max iterations (from config.py)
python auto_research.py

# Run with custom iteration count
python auto_research.py --iterations 20

# Check progress without running
python check_progress.py
```

The pipeline **auto-resumes** from the last checkpoint if one exists in `logs/`.

---

## Configuration (`config.py`)

Key settings to adjust:

```python
MAX_ITERATIONS = 50        # How many experiments to run
N_FOLDS = 5                # Cross-validation folds
RANDOM_SEED = 42
OPENROUTER_MODEL = "anthropic/claude-opus-4.6"  # LLM brain
TOP_K_MODELS = 10          # Ensemble size
TIMEOUT_SECONDS = 1800     # Per-iteration timeout (30 min)
```

**Important**: `OPENROUTER_API_KEY` is stored in `config.py`. Keep this file private — do not commit to public repos.

---

## Allowed Models (`config.ALLOWED_MODELS`)

- Linear: `logistic`, `logistic_l1`, `logistic_l2`, `logistic_elasticnet`, `lasso`, `ridge`, `elasticnet`
- Tree/Boosting: `lightgbm`, `xgboost`, `random_forest`, `gradient_boosting`

---

## What Worked Well

### Feature & Data Improvements
1. **Feature engineering drove early gains**: Ratio features (margin_ratio, savings_ratio) + date features improved CV 3.4x (0.096→0.028).
2. **Frequency encoding was transformative**: `fulfillment_node_freq` became the #1 feature at 15.4% importance, driving most of the 0.0284→0.0261 improvement. The fulfillment node is a strong proxy for product category, region, and logistics risk.
3. **Product name embeddings dominated**: MiniLM PCA embeddings were the single most important feature group. All 10 PCA dims ranked in top 13 features. PCA20 identified as the sweet spot — matching PCA30 quality at 4.3x speed.
4. **LLM product classification added signal**: `subcategory_freq` (115K unique values) ranked #14 by importance, contributing meaningfully alongside embeddings.
5. **Isotonic calibration consistently helped**: Free improvement of 0.0005–0.002 per iteration with zero risk of degradation.

### Model & Training
6. **LightGBM dominated throughout**: Handled 5.4M rows efficiently, consistently outperformed XGBoost by ~2x in both CV and training speed on enriched data.
7. **Hyperparameter tuning yielded major efficiency gains**: 127 leaves + lr=0.07 + less regularization matched the default config's CV while training 4.3x faster (12 min vs 51 min). Default params were under-utilizing model capacity.
8. **Feature pruning maintained quality**: Removing 10 low-importance features only increased CV by 0.001 while reducing features by 20%.
9. **10-model ensemble improved over best single model**: OOF log-loss 0.02126 vs 0.02187 — a further 2.8% improvement from diversity.
10. **AUC jumped from 0.824 to 0.907**: Recall improved dramatically across all phases.

### Semi-Supervised & Advanced Techniques (Post-Mortem)
11. **Pseudo-labeling FAILED SPECTACULARLY**: While it inflated CV to 0.02187, the actual Kaggle 50% test score was 0.705 due to train-on-test leakage. Even when properly implemented without leakage (iteration 59), the pseudo-labels caused the test distribution to skew badly, resulting in a 0.07860 Kaggle test score. Do not use pseudo-labeling.
12. **Temporal features + hyperparameter diversity**: Iter 55 tested temporal velocity features (user_recency_days, node_stress). Sadly, this was tested in a broken Phase 6 model (iter 58). However, the signal from these temporal features was real and consistent, so this feature engineering approach should be applied to a CLEAN model without pseudo-labeling.

### Pipeline & Process
12. **Cross-iteration learning was highly effective**: Each Phase 4 iteration's results directly informed the next — 100% success rate vs 37% failure rate for LLM-driven iterations.
13. **CV-only optimization prevented overfitting**: Using training CV as the only signal over 51 iterations avoided the multiple-comparisons problem.
14. **Auto-resume checkpointing**: Pipeline survived multiple restarts without losing progress.

## What Did Not Work Well

### LLM-Driven Pipeline Limitations
1. **High timeout rate (37%)**: The LLM frequently proposes slow configurations. Learning rate < 0.03 on 5.4M rows almost always times out, even with early stopping.
2. **Template rigidity**: The LLM proposes advanced features (target encoding, interactions) but `generate_training_code()` ignores them — it only uses the feature list and hyperparameters.
3. **LLM didn't use available features**: Despite being listed in the prompt, LLM-driven iterations often omitted new features. Manually-crafted iterations consistently outperformed.
4. **Diminishing returns from hyperparameter tuning alone**: 29 iterations on a fixed 19-feature set improved CV by only 0.0002. New features were needed to break through.

### Model & Feature Experiments That Failed
5. **XGBoost not competitive on enriched data**: CV=0.02449 vs LightGBM's 0.02248 on identical features — 9% worse and 2.5x slower (iter 42).
6. **MPNet ≈ MiniLM**: Despite higher dimensionality (768 vs 384), MPNet provided no advantage at same PCA dims (iter 43). Dual embeddings also added no value (iter 44) — models too correlated.
7. **Adjusted score formula over-penalized features**: The complexity penalty (0.01/feature) ranked old 19-feature models above enriched 50-feature models. Required switching to raw CV for ensemble selection.
8. **Target encoding caused catastrophic overfitting** (iter 48): Even with Bayesian smoothing (alpha=50-500), out-of-fold target encoding on high-cardinality entities (user_id, fulfillment_node, retail_sku) causes LightGBM to latch onto the encoding and destroy generalization. With alpha=50, CV inflated to 0.032 and AUC dropped to 0.74. With alpha=500, the encoding was too smooth to add signal. Dangerous for datasets with high cardinality and low positive rates.
9. **Class reweighting degraded performance** (iter 50): `is_unbalance=True` (scale_pos_weight≈186) degraded AUC from 0.905 to 0.864. With 0.54% positive rate, the extreme weighting distorts the model's decision surface. Isotonic calibration can't fully recover — CV=0.02448 vs baseline 0.02227.
10. **User behavioral features mostly redundant** (iter 47): user_order_count, user_total_spend, user_avg_gmv largely duplicated information already captured by user_id_freq. CV=0.02233, marginal improvement not worth the complexity.
11. **Isolation Forest anomaly score redundant** (iter 52): Despite ranking #2 by feature importance (6280.6), the anomaly score didn't improve CV (0.02208 vs 0.02187). The IF was trained on the same 40 features LightGBM already uses — the anomaly signal is already captured implicitly by the tree model. Positive class had only slightly higher scores (mean -0.102 vs -0.116). Still useful for ensemble diversity.

### Phase 6 Failures
12. **Graph neighborhood features are disguised target encoding** (iter 56): Computing cancellation rates through user-SKU-node bipartite graph — even with OOF-safe computation and Bayesian smoothing (alpha=100) — is equivalent to target encoding through intermediary entities. CV crashed to 0.02816, AUC dropped from 0.913 to 0.806. The high-cardinality SKU joins allow target information to leak through even with smoothing.
13. **Negative downsampling loses critical patterns** (iter 53): 1:10 downsampling with analytical base rate correction produced CV=0.02327, 6.4% worse. With only 0.54% positive rate, the boundary between positive and negative classes is subtle — discarding 90% of negatives removes important hard-negative examples.
14. **Two-stage cascade creates harmful discontinuity** (iter 57): Shallow screener (15 leaves) + deep classifier (255 leaves) on suspicious subset achieved CV=0.02204. The prediction discontinuity at the Stage 1 threshold damages log-loss. Single-stage models with sufficient capacity handle the full distribution better.
15. **Beta calibration offers no advantage over isotonic** (iter 54): 3-parameter parametric calibration matched but couldn't beat isotonic's non-parametric flexibility. For well-separated probability distributions (0.54% positive rate), isotonic regression is already near-optimal.
16. **Ensemble prediction scale mismatch**: Phase 6 models (using iter 51's pseudo-labeling template) produce predictions with mean ~0.17 after isotonic calibration, while Phase 4-5 models predict mean ~0.007. Weighted averaging + final isotonic calibration couldn't reconcile these scales — adding Phase 6 models degraded ensemble OOF from 0.02126 to 0.02173+. Root cause: different isotonic calibration curves learned per-phase due to different training data compositions (pseudo-label augmentation changes the target distribution seen by isotonic).

### Phase 8 NN Failures
17. **Neural networks 20% worse than LightGBM on this tabular data**: Best NN (iter 61, regularized MLP) achieved CV=0.02660 vs LightGBM's 0.02215. AUC gap was 0.858 vs 0.907. LightGBM's native handling of categorical splits, missing values, and feature interactions gives it a fundamental advantage on structured tabular data with 5.4M rows.
18. **Entity embeddings overfit catastrophically** (iter 62): 100K user + 50K SKU hash buckets with dim=32 created 4.8M embedding parameters. Train loss crashed from 0.031 to 0.014 by epoch 6 while val loss spiked from 0.028 to 0.045. High-cardinality entity embeddings are a fundamentally broken approach for this dataset — consistent with target encoding (iter 48) and graph features (iter 56) also overfitting.
19. **FT-Transformer too slow for 5.4M rows** (iter 63): Self-attention over 45 feature tokens made each fold take 46 min (vs 13 min for MLP). CV=0.02894, AUC=0.825 — worst of all architectures. Stopped after fold 1. Transformer attention is O(n^2) in sequence length and provides no benefit when features don't have natural ordering or token-like semantics.
20. **Snapshot ensemble averaging hurts** (iter 64): Averaging 4 snapshots at CosineAnnealingWarmRestarts minima diluted the best snapshot (0.02726) with weaker earlier ones (0.02822). Ensemble val_logloss 0.02745 was worse than best single snapshot. The snapshots aren't diverse enough — they're all from the same training trajectory, just at different training stages.
21. **pos_weight/class reweighting confirmed harmful for NNs too**: Original NN scripts used BCEWithLogitsLoss(pos_weight=186), producing catastrophic fold 1 logloss of 0.415. Removing pos_weight immediately fixed predictions to 0.027. This is the NN equivalent of iter 50's is_unbalance=True disaster — extreme class reweighting destroys probability calibration regardless of model family.

---

## Ideas for Future Improvements

### Implemented ✓
1. ~~**Early stopping for LightGBM/XGBoost**~~ — ✓ (Iter 36–39). Default 50 rounds. Reduced LightGBM time ~25%.
2. ~~**Frequency encoding for categoricals**~~ — ✓ (Iter 36–39). user_id, fulfillment_node, retail_sku. `fulfillment_node_freq` became #1 feature.
3. ~~**Probability calibration**~~ — ✓ (Iter 36–39). Isotonic regression per-iteration. Consistently improves log-loss by 0.0005–0.002.
4. ~~**Product name features**~~ — ✓ (Iter 36–39). Name length and word count from product_names.csv.
5. ~~**LLM product classification**~~ — ✓ (Iter 40–41). Gemini Flash classified 686K products into category, subcategory, demographic, price_tier, durability, seasonality.
6. ~~**Product name embeddings**~~ — ✓ (Iter 40–41). MiniLM sentence embeddings, PCA-reduced to 10/30 dims. Dominated feature importance.
7. ~~**Run LightGBM with full feature set**~~ — ✓ (Iter 40–41). LightGBM with enriched data achieved 0.02215, beating XGBoost's 0.02608.
8. ~~**XGBoost with enriched data**~~ — ✓ (Iter 42). CV=0.02449, significantly worse than LightGBM (0.02248). Not competitive for this dataset.
9. ~~**MPNet embeddings**~~ — ✓ (Iter 43). CV=0.02248, virtually identical to MiniLM at same PCA dims. No benefit from switching.
10. ~~**Both embedding models**~~ — ✓ (Iter 44). Dual MiniLM+MPNet PCA10 = no gain. Models too correlated; depth > breadth.
11. ~~**Hyperparameter tuning on enriched features**~~ — ✓ (Iter 45). 127 leaves + lr=0.07 matched iter 41 CV at 4.3x speed. PCA20 identified as sweet spot.
12. ~~**Feature selection**~~ — ✓ (Iter 46). Pruned 10 low-importance features. CV only 0.001 worse, best adjusted score.
13. ~~**Regenerate ensemble submission**~~ — ✓. 9-model ensemble with isotonic calibration, OOF log-loss=0.02193.
14. ~~**User behavioral features**~~ — ✓ (Iter 47). order_count, total_spend, avg_gmv, is_first_order. CV=0.02233 — mostly redundant with user_id_freq.
15. ~~**Target encoding for categoricals**~~ — ✓ (Iter 48). OOF target encoding for fulfillment_node/retail_sku. Caused severe overfitting — dangerous for high-cardinality + low positive rate.
16. ~~**Order composition features**~~ — ✓ (Iter 49). Multi-item order structure (item count, value share, etc.). No improvement despite 53% multi-item orders.
17. ~~**Class reweighting / focal loss**~~ — ✓ (Iter 50). is_unbalance=True degraded AUC from 0.905 to 0.864. Custom focal loss timed out (Python objective too slow on 5.4M rows).
18. ~~**Pseudo-labeling from test data**~~ — ✓ (Iter 51, 59). **FAILURE.** Both naive (0.705) and leak-free (0.07860) implementations resulted in catastrophic Kaggle test scores. Do not use.
19. ~~**Anomaly score via Isolation Forest**~~ — ✓ (Iter 52). IsolationForest on negative-class only → anomaly_score feature. Ranked #2 by importance but CV=0.02208 (worse than iter 51). Redundant with existing features. Added ensemble diversity → OOF improved from 0.02135 to 0.02126.
20. ~~**Regenerate ensemble with iters 51+52**~~ — ✓. 10-model ensemble with isotonic calibration, OOF log-loss=0.02126 (from 0.02135).
21. ~~**Negative majority downsampling**~~ — ✓ (Iter 53). 1:10 downsampling with base rate correction. CV=0.02327 — lost too much information from discarding 90% of negatives.
22. ~~**Beta calibration**~~ — ✓ (Iter 54). 3-parameter parametric calibration. CV=0.02190 vs isotonic's 0.02187 — tied, not worth the complexity.
23. ~~**Temporal velocity features**~~ — ✓ (Iter 55). user_recency_days, node_stress_7d, node_stress_ratio, user_velocity_72h. CV=0.02186 — marginal but consistent improvement.
24. ~~**Graph neighborhood features**~~ — ✓ (Iter 56). OOF-safe bipartite graph cancellation rate aggregation. CV=0.02816 — essentially target encoding through intermediary entities, severe overfitting.
25. ~~**Two-stage cascade model**~~ — ✓ (Iter 57). Shallow screener + deep classifier on suspicious subset. CV=0.02204 — threshold discontinuity hurts log-loss.
26. ~~**Temporal features + hyperparameter diversity**~~ — ✓ (Iter 58). Extracted temporal features (user_recency_days, node_stress_7d, etc). Must be re-tested without pseudo-labeling.

### High-Impact (likely to improve score further)
27. **Stacking/blending**: Train a meta-learner on OOF predictions from diverse base models. OOF predictions now available for iters 37, 39, 40–52, 60–62.
28. ~~**Neural network approaches**~~ — Implemented (Iters 60-64). Best NN CV=0.02660 (iter 61, regularized MLP). 20% worse than LightGBM but provides ensemble diversity. Entity embeddings and FT-Transformer failed. See Phase 8 results.

### Medium-Impact
31. **Bayesian optimization**: Use Optuna for systematic fine-tuning. Manual tuning in iter 45 showed significant gains — systematic search may find even better configurations.
32. **Interaction features**: Explicitly model interactions between top features (e.g., margin_ratio × fulfillment_node_freq, embedding components × price features).

### Operational
33. **Parallel fold training**: Train CV folds concurrently to reduce wall-clock time.
34. **Better error logging**: Capture full stderr/stdout on failure.
35. **Prompt compression**: Summarize older experiments more aggressively to reduce token waste.

---

## Anti-Overfitting Design

The loop uses CV on training data as the **only** optimisation signal. The 50% visible test set is logged but never fed back as a selection criterion. This prevents the multiple-comparisons / adaptive overfitting problem that would arise from optimising against a fixed held-out set over many iterations.

See `specific_idea.md` for the full 3-tier evaluation architecture rationale.

---

## Resuming / Debugging

- Check `logs/checkpoint.json` for current iteration count and history
- Run `python check_progress.py` for a quick summary
- Per-iteration code is in `logs/iteration_NNN/code.py` — runnable standalone
- LLM prompts/responses in `logs/iteration_NNN/prompt.txt` / `response.txt`
- If a run fails mid-iteration, just restart — it will resume from the checkpoint
- Each results.json contains fold-level details, feature importance, and prediction statistics

---

## Security Note

`config.py` contains an OpenRouter API key. Rotate it if this repo is ever made public.
