# American Express Offer Ranking — README

This README documents the **AmericanExpress.ipynb** pipeline that achieved stronger leaderboard results on the Campus Challenge 2025 “Amex Offerings” task.

> **Goal**: Predict the probability a customer will **click** an offer when it’s shown, then **rank** offers **per user** (evaluation: **MAP@7**).  
> **Why this notebook performs better**: it is **temporally honest** (no look‑ahead), uses **smoothed target encodings** for high‑signal IDs/categories, and trains a **calibrated XGBoost classifier** with **time‑based validation** that mirrors the competition’s out‑of‑time test split.

---

## 1) Environment

- Python 3.10+
- Key packages
  - `pandas`, `numpy`, `pyarrow`
  - `xgboost` (GPU supported; falls back to CPU if needed)
  - `matplotlib`/`seaborn` (optional, for EDA)
  - `catboost` (optional experiments)
- Recommended GPU: any CUDA‑capable card (the notebook uses `tree_method='gpu_hist'` / `predictor='gpu_predictor'` when available).

Create/activate an environment and install minimal deps:
```bash
conda create -n amex python=3.10 -y
conda activate amex
pip install pandas numpy pyarrow xgboost catboost matplotlib
```

---

## 2) Data & Files

Place these (parquet/CSV) files in the working directory (paths are configurable via CLI args; defaults shown below):

- `train_data.parquet` — training impressions with target `y`  
- `test_data.parquet` — test impressions (no `y`)  
- `add_event.parquet` — event logs per user/offer over time  
- `add_trans.parquet` — transaction records per user over time  
- `offer_metadata.parquet` — brand/category/dates for offers  
- `submission_template.csv` — template with `id1` to attach predictions

Optional: intermediate EDA plots/figures.

> **IDs**: `id2`=user, `id3`=offer, `id5`=timestamp (converted to datetime in the notebook).

---

## 3) Reproducible Run (CLI)

The notebook contains a script‑like cell guarded by `if __name__ == "__main__":` so you can run it end‑to‑end as a script using default paths/params or overrides.

**Defaults (inside notebook):**
```text
--train_file train_data.parquet
--test_file test_data.parquet
--offer_meta_file offer_metadata.parquet
--trans_file add_trans.parquet
--submission_file submission_template.csv
--preds_out submission.csv
--n_estimators 1000
--max_depth 6
--learning_rate 0.03
--subsample 0.8
--colsample_bytree 0.8
--min_child_weight 1.0
--reg_lambda 1.0
```
**Example run from a terminal:**
```bash
python AmericanExpress.ipynb  # if using Jupytext/nbconvert script, or run all cells in the notebook UI
```
Or convert to a `.py` first (optional):
```bash
jupyter nbconvert --to script AmericanExpress.ipynb && python AmericanExpress.py
```

> Output: a `submission.csv` with `id1,pred` where `pred` is the click probability used for per‑user ranking (top‑7).

---

## 4) Core Approach

### 4.1 Temporal validation (no leakage)
- Convert `id5` to datetime.
- **Hold out the last day** (or a chosen final time window) as **validation**; train on all earlier rows.
- This mirrors the competition’s **out‑of‑time** evaluation and prevents **look‑ahead**.

### 4.2 Feature set (compact, high‑SNR)
The notebook starts from base `f1…f366` features and prunes very sparse columns (>90–95% missing). It then enriches with **row‑local (right‑censored)** features:

- **Offer metadata**: `offer_category`, `brand`, `offer_duration_days`  
- **Temporal**: `impression_hour`, `impression_dayofweek`  
- **User dynamics**: `time_since_last_impression` (per `id2` diff), `user_offer_impression_count` (cumulative count per `id2,id3`)  
- **Transactions (light)**: `user_trans_count`, `user_total_spend`, `user_avg_spend`

#### Bayesian Target Encoding (smoothed CTR)
To boost categorical/ID signal while avoiding overfit, the notebook builds **smoothed CTR** features on the **train‑only** slice and merges them back:

- `id3_bayes_ctr` (offer), `id2_bayes_ctr` (user), `brand_bayes_ctr`, `offer_category_bayes_ctr`
- Smoothing toward the **global CTR** using a factor like **20** (blend by category count).

> These features are computed **only from past/seen training rows** and then applied to validation/test, preserving temporal integrity.

### 4.3 Model: XGBoost classifier
- Objective: `binary:logistic`
- Class imbalance: compute `scale_pos_weight = (neg / pos)` on the **train** split.
- GPU acceleration: `tree_method='gpu_hist'`, `predictor='gpu_predictor'` (fallback to CPU works).
- Early stopping on the time‑based validation set.

### 4.4 Prediction & ranking
- Predict **P(click)** for each (user, offer, impression).
- **Within each user (`id2`)**, sort offers by `pred` **descending** and take **top‑7** for MAP@7 evaluation / submission.

---

## 5) End‑to‑End Flow

1. **Load data** (`train`, `test`, `events`, `transactions`, `offer_metadata`).
2. **Coerce types** for IDs/timestamps; ensure merge keys align (`id2`, `id3`, `id5`).
3. **Engineer features**:
   - temporal + user×offer recency/depth
   - smoothed CTR encodings (train‑only)
   - light spend aggregates
4. **Assemble training matrix**; drop ultra‑sparse base columns.
5. **Time split**: last day → validation; earlier days → train.
6. **Train XGBoost** with `scale_pos_weight`, early stopping.
7. **Predict** on validation/test; **rank per user**.
8. **Export** `submission.csv` (`id1,pred`).

---

## 6) Parameters You Can Tune

- **Validation window**: length of the final time slice (e.g., last day vs. last N days).
- **Smoothing factor** for Bayesian encodings (default ~20). Larger → more shrinkage to global CTR.
- **XGBoost hyperparameters**: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`.
- **Feature pruning threshold**: missingness cutoff for dropping base features.

---

## 7) Tips to Maintain “Temporal Honesty”

- When creating any aggregate (CTR, counts, recency), compute it **only from data up to the impression time** in train; never use future rows from the same user/offer.
- Keep validation strictly **later in time** than training.
- Prefer **user‑ and user×offer‑level** signals over global aggregates that don’t personalize ranking.

---

## 8) Troubleshooting

- **CV looks great, LB drops** → check for **leakage** (are encodings/aggregates built on all rows at once?). Verify your validation is **out‑of‑time**.
- **Inconsistent merges** → ensure `id2`, `id3` dtypes match across tables; after merges, assert row counts and null rates.
- **Class imbalance** → re‑compute `scale_pos_weight` after each new train split.
- **Memory** → drop unused columns early; prefer `parquet` and dtype down‑casting where safe.

---

## 9) Repo / Notebook Structure

- `AmericanExpress.ipynb` — main pipeline (can be run top‑to‑bottom)
- `data/*.parquet` — input data (not included)
- `submission.csv` — output predictions for the competition portal

---

## 10) License & Attribution

This notebook README is provided for the Amex Offerings challenge context. Please adapt paths and parameters to your environment and data access constraints.
