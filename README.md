# Delhivery Last-Mile Delay Intelligence Pipeline

A production-grade, end-to-end delay intelligence and operational dispatch system for Delhivery logistics network. This repository integrates data cleaning, exploratory delay diagnosis, point-estimate delivery hours regression, proactive delay classification with operations-tuned alerting thresholds, and K-Means hub segmentation.

---

## 1. How to Run

### Environment Setup
All models and scripts run offline using free and open-source Python libraries pinned in `requirements.txt`.
To install dependencies in your virtual environment:

```bash
pip install -r requirements.txt
```

> **Note on Model Serialization:**
> `models/delay_model.pkl` is serialized using `joblib` on `scikit-learn==1.7.0`. The pickle loads reliably only on the pinned scikit-learn version in `requirements.txt`. If your environment has a different scikit-learn version, regenerate the pickle by running:
> ```bash
> python package_model.py
> ```

### Execution Order
From a fresh clone, run the scripts in repository root in the following sequential order:

```bash
# 1. Regenerate synthetic data (verifies 24 hubs, 1224 raw shipments, 48 new shipments)
python generate_data.py

# 2. Part A: Cleaning, feature engineering, and delay checks
python part_a_clean_eda.py

# 3. Part B: Delivery-hours regression and model selection
python part_b_regression.py

# 4. Part C: Delay classification, feature selection, and alert threshold optimization
python part_c_classification.py

# 5. Part D: Hub segmentation and clustering
python part_d_segments.py

# 6. Part D: Model artifact packaging and reload verification
python package_model.py

# 7. Final Scoring: Batch scoring of new parcels with operational action assignment
python predict_delay.py
```

---

## 2. Data Findings

All metrics are computed in [`part_a_clean_eda.py`](file:///e:/Masai_CapStone_Delhivery/Masai_CapStone_Project_DelhiveryPipeline/part_a_clean_eda.py) and recorded in `outputs/part_a_checks.csv`, `outputs/part_a_delay_by_group.csv`, and `outputs/part_a_correlations.csv`.

### Key Data Cleaning Metrics
- **Dataset:** 1,224 raw shipments deduplicated to 1,200 records (24 duplicates removed; 0 nulls after merge with hubs).
- **Service Types:** Standardized 12 irregular spellings down to 3 (`Express`, `Standard`, `Economy`).
- **Sentinels & Missing Values:** 5 scanner-fault sentinels (>100 kg) treated as missing. Filled with stored medians: `distance_km` = **314.95 km**, `parcel_weight_kg` = **1.83 kg**, `declared_value_inr` = **INR 1,121.50**.
- **Delays:** 278 out of 1,200 shipments delayed (**23.17%** overall delay rate).

### Delay Rate by Segment (`outputs/part_a_delay_by_group.csv`)
| Dimension | Group | Shipments | Delayed | Delay Rate | Mean Hours |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Service Type** | Economy | 315 | 27 | **0.0857** | 82.0931 |
| **Service Type** | Express | 243 | 119 | **0.4897** | 36.8056 |
| **Service Type** | Standard | 642 | 132 | **0.2056** | 52.2569 |
| **Hub Tier** | Tier 1 | 400 | 16 | **0.0400** | 48.5664 |
| **Hub Tier** | Tier 2 | 400 | 86 | **0.2150** | 58.8194 |
| **Hub Tier** | Tier 3 | 400 | 176 | **0.4400** | 63.4943 |
| **Distance Band** | 0-200 km | 274 | 6 | **0.0219** | 46.7559 |
| **Distance Band** | 201-400 km | 563 | 77 | **0.1368** | 55.9660 |
| **Distance Band** | 401-600 km | 275 | 120 | **0.4364** | 64.2660 |
| **Distance Band** | 601+ km | 88 | 75 | **0.8523** | 72.2608 |

### Five Core Findings
1. **Service Type Failure Rate:** **Express** is delayed most often, with a failure rate of **0.4897** (119 / 243 parcels delayed) compared to **0.2056** for Standard and **0.0857** for Economy, driven by its tight 36-hour promise SLA.
2. **Hub Tier Delay Escalation:** Delay rate increases 11-fold from Tier 1 to Tier 3: **0.0400** in Tier 1 $\rightarrow$ **0.2150** in Tier 2 $\rightarrow$ **0.4400** in Tier 3 (+40.00 percentage points).
3. **Distance Band Vulnerability:** Delay rate increases monotonically across distance bands: **0.0219** (0–200 km), **0.1368** (201–400 km), **0.4364** (401–600 km), and **0.8523** for long-haul routes (601+ km).
4. **Top Correlated Features with `delivery_hours`:** The three columns most strongly correlated (ranked by absolute Pearson correlation from `part_a_correlations.csv`, excluding `delivery_hours` itself) are:
   - `promised_hours`: **+0.8372**
   - `distance_km`: **+0.4013**
   - `n_stops`: **+0.3247**
5. **Log Transformation of Declared Value:** Raw `declared_value_inr` is heavily right-skewed with a mean of **INR 1,533.0542** versus a median of **INR 1,121.50**; applying natural log normalizes the distribution, aligning the post-transform mean of **7.0277** with the median of **7.0224**.

---

## 3. Model Results (Regression)

All metrics below are drawn directly from `outputs/part_b_checks.csv`, `outputs/part_b_tree_depth.csv`, `outputs/part_b_metrics.csv`, `outputs/part_b_gb_learning_rate.csv`, and `outputs/part_b_cv.csv`.

### Baseline and Model Benchmarking
- **Mean Baseline:** Predicting the training target mean (49.88 hours) on the validation set yields a validation $R^2$ of **-0.0212** (MAE 15.7952 hours, RMSE 18.8552 hours). A negative $R^2$ confirms that an unconditioned mean baseline performs worse than predicting the validation mean itself and explains none of the variance.
- **Validation Ranking:**
  - **Linear Regression:** MAE 3.9222 hrs, MSE 23.9235, RMSE **4.8912 hrs**, $R^2$ = **0.9313** (Validation Winner).
  - **Gradient Boosting (`lr=0.1`):** MAE 3.9166 hrs, MSE 24.1722, RMSE **4.9165 hrs**, $R^2$ = **0.9306**.
  - **Random Forest (`n=100`):** MAE 4.2882 hrs, MSE 30.3550, RMSE **5.5095 hrs**, $R^2$ = **0.9128**.
  - **Decision Tree (`depth=6`):** MAE 5.0971 hrs, MSE 39.4338, RMSE **6.2796 hrs**, $R^2$ = **0.8867**.

### Tree-Depth Diagnostic (Bias vs. Variance)
- **High Bias (Underfitting):** At depth 1, `train_rmse` is 11.6334 hours and `val_rmse` is 12.1324 hours (gap = **0.4990 hours**). While the gap is small, both errors are severely elevated: `val_rmse` is **5.8528 hours** worse than the optimal depth, indicating extreme high bias.
- **Optimal Depth:** Depth 6 minimizes validation RMSE at **6.2796 hours** (`train_rmse` 4.7114 hours, gap = 1.5682 hours).
- **High Variance (Overfitting):** With no depth limit (`unlimited`), `train_rmse` collapses to **0.0000** (perfect memorization) while `val_rmse` inflates to **7.8368 hours**, generating a generalization gap of **7.8368 hours** (far exceeding the 5.0 hour threshold).

### Learning-Rate Sensitivity (Gradient Boosting)
- **`learning_rate = 0.1` (Optimal):** Validation RMSE = **4.9165 hrs**, $R^2$ = **0.9306**. Balanced step size enables steady descent to the global error basin.
- **`learning_rate = 0.01` (Too Slow):** Validation RMSE = **10.0018 hrs**, $R^2$ = **0.7127**. Steps are too small for 100 trees, leaving the ensemble underfit.
- **`learning_rate = 1.0` (Overshooting):** Validation RMSE = **7.3701 hrs**, $R^2$ = **0.8440**. Aggressive steps overshoot minima and bounce unstably across the loss surface.

### Cross-Validation and Final Production Regression Model
- Running 5-fold cross validation on the combined 1,020 training + validation samples yields:
  - **Gradient Boosting:** Mean $R^2$ = **0.9353** (std = **0.0045**).
  - **Linear Regression:** Mean $R^2$ = **0.9261** (std = **0.0048**).
- **Model Choice:** While Gradient Boosting shows a marginal $R^2$ edge in 5-fold CV (+0.0092), Linear Regression won on validation RMSE (**4.8912 hrs** vs **4.9165 hrs**). Linear Regression was scored once on the held-out 180 test rows, achieving test RMSE of **4.5235 hours** and test $R^2$ of **0.9493** (MAE 3.5671 hours).
- **Deployment Rationale:** We recommend Linear Regression for point-estimate dispatch: it computes in sub-millisecond inference time, requires no tree traversal, provides transparent feature coefficients, and avoids piecewise step-function anomalies on unseen route combinations.

---

## 4. Model Results (Classification)

All metrics below are drawn directly from `outputs/part_c_models.csv`, `outputs/part_c_feature_scores.csv`, `outputs/part_c_selectk.csv`, `outputs/part_c_tuning.csv`, `outputs/part_c_thresholds.csv`, `outputs/part_c_final_config.csv`, and `outputs/part_c_final_test.csv`.

### Baseline and Default Classifiers
- **Blind Baseline:** Predicts "not delayed" for every parcel, yielding an accuracy of **0.7667** (138 TN, 0 FP, 42 FN, 0 TP). This baseline is completely useless operationally: its precision is **0.0**, recall is **0.0**, and F1 is **0.0**, missing 100% of delayed shipments.
- **Trained Classifiers (default 0.5 threshold on validation):**
  - **Logistic Regression:** Accuracy 0.9056, Precision 0.7778, Recall 0.8333, F1 = **0.8046**.
  - **Decision Tree (`depth=6`):** Accuracy 0.8833, Precision 0.7561, Recall 0.7381, F1 = **0.7470**.
  - **Random Forest (default, 17 features):** Accuracy 0.9056, Precision 0.8378, Recall 0.7381, F1 = **0.7848**.

### Feature Signal and Noise Elimination
- The five bookkeeping noise columns (`label_print_batch`, `scanner_build_no`, `tape_roll_no`, `sticker_serial_mod`, `shelf_slot_no`) command **0.2245** (22.45%) of the default Random Forest's feature importance purely through spurious tree splits.
- Univariate ANOVA F-classification tests confirm they are noise:
  - All 5 noise columns have F-scores below 5.0 (ranging from **0.0230** to **4.5847**) and hold ranks 8, 12, 14, 16, and 17.
  - In contrast, the top 7 signal features (`distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag`, `sale_event_flag`) all exhibit F-scores above 35.0 (up to **389.2131** for `distance_km`).
- **Feature Selection ($k$ Search):**
  - $k=5$: validation F1 = 0.6988
  - **$k=7$**: validation F1 = **0.8333** (Peak performance)
  - $k=10$: validation F1 = 0.7857
  - $k=12$: validation F1 = 0.8049
  - $k=17$: validation F1 = 0.7848
  Restricting the feature set to the top 7 verified signal features eliminates noise overfitting and lifts validation F1 from 0.7848 to 0.8333.

### Hyperparameter Tuning Diagnostics
- Evaluated on the 7 selected features:
  - **Default Forest ($k=7$):** Validation F1 = **0.8333** (Winner).
  - **Randomized Search (45 fits):** Best CV accuracy = 0.8905, Validation F1 = **0.7848**.
  - **Grid Search (135 fits):** Best CV accuracy = 0.8952, Validation F1 = **0.7750**.
- **Observation:** Tuning on 5-fold CV accuracy penalized depth (`max_depth=6`), which reduced recall on the validation set. On a 180-row validation set with 42 positive cases, the unconstrained default forest with $k=7$ features captured finer non-linear boundary interactions without overfitting noise, winning the selection.

### Alert Threshold Selection (The Operations Trade-Off)
Because a missed delayed parcel causes customer churn and contract breach while a proactive alert costs only a quick call, the operations policy mandates selecting the highest threshold $t$ that achieves a validation recall of at least **0.80**:
- Threshold 0.1: Precision 0.4824, Recall 0.9762, F1 0.6457 (44 false alarms)
- Threshold 0.3: Precision 0.6964, Recall 0.9286, F1 0.7959 (17 false alarms)
- Threshold 0.4: Precision 0.7400, Recall 0.8810, F1 0.8043 (13 false alarms)
- **Threshold 0.5 (Chosen):** Precision **0.7955**, Recall **0.8333**, F1 **0.8140** (9 false alarms, 7 missed delays)
- Threshold 0.6: Precision 0.8889, Recall **0.7619** (violates the $\ge 0.80$ recall requirement)

### Final Test Set Evaluation
Scoring the final pipeline on the unseen 180 test rows at threshold **0.5**:
- **Test Precision:** **0.6889** (31 TP / 45 total flagged)
- **Test Recall:** **0.7561** (31 TP / 41 actual delayed parcels)
- **Test F1 Score:** **0.7209** (Accuracy: 0.8667, 125 TN, 14 FP, 10 FN, 31 TP)
- **Honest Assessment:** Test recall (0.7561) is slightly lower than validation recall (0.8333). With only 41 delayed shipments in the test set, each missed parcel impacts recall by 2.44 percentage points. The model successfully intercepts over three-quarters of all delivery failures with a low false alarm footprint.

---

## 5. Operations Recommendation

### 1. Hub Clustering and Escalate Playbook
K-Means clustering ($K=3$) on scaled hub profiles (`mean_delivery_hours`, `delay_rate`, `mean_distance_km`, `mean_n_stops`) segments the 24 hubs into three distinct operational profiles:
- **Standard (7 hubs):** Avg delay rate **0.0257** (2.57%), delivery time 47.24 hrs, distance 188.93 km, stops 55.37.
- **Watch (8 hubs):** Avg delay rate **0.1550** (15.50%), delivery time 56.81 hrs, distance 314.18 km, stops 70.72.
- **Escalate (9 hubs):** Avg delay rate **0.4600** (46.00%), delivery time 64.65 hrs, distance 470.48 km, stops 84.35.

The **Escalate** segment contains 9 high-risk hubs requiring immediate managerial interventions:
- `HUB13` (Nagpur): Delay rate **0.4600**
- `HUB15` (Kochi): Delay rate **0.4600**
- `HUB17` (Patna): Delay rate **0.4600**
- `HUB18` (Ranchi): Delay rate **0.4400**
- `HUB19` (Guwahati): Delay rate **0.5600**
- `HUB20` (Raipur): Delay rate **0.4600**
- `HUB21` (Bhubaneswar): Delay rate **0.4800**
- `HUB23` (Siliguri): Delay rate **0.4400**
- `HUB24` (Agartala): Delay rate **0.3800**

### 2. Tier vs. Segment Mismatches
Four hubs deviate significantly from their official tier designations (`outputs/tier_vs_segment_mismatch.csv`):
1. **`HUB07` Kolkata:** Labelled **Tier 1**, but clustered into **Watch** (delay rate 0.2000, avg 57.07 hrs, lane distance 334.8 km). Kolkata operates as a congested regional depot rather than a streamlined metro hub; SLA promises should be extended by 12 hours.
2. **`HUB13` Nagpur:** Labelled **Tier 2**, but clustered into **Escalate** (delay rate 0.4600, avg 66.07 hrs, lane distance 473.3 km). It serves extensive rural corridors and must receive Tier 3 resource buffers and dedicated line-haul capacity.
3. **`HUB15` Kochi:** Labelled **Tier 2**, but clustered into **Escalate** (delay rate 0.4600, avg 63.66 hrs, lane distance 480.9 km). Route density and geography push delivery times into the stretched category.
4. **`HUB22` Dehradun:** Labelled **Tier 3**, but clustered into **Watch** (delay rate 0.2000, avg 55.43 hrs, lane distance 309.8 km). Dehradun consistently outperforms other Tier 3 stations; capacity can be expanded without risk of operational breakdown.

**Action:** Regional ops must decouple planning rules and SLA commitments from nominal tier labels and replace them with these behavior-derived segment profiles.

### 3. Dispatch Alert Reliability and Action Allocation
- **Alert Reliability:** Operating at threshold **0.5**, approximately **7 out of every 10 flagged parcels** (Precision = 0.6889) are confirmed delays, and **7.6 out of every 10 late parcels** (Recall = 0.7561) are intercepted before breach.
- **Scoring the 48 New Parcels (`outputs/new_shipment_actions.csv`):**
  - **8 parcels flagged** for delivery risk (16.67%).
  - **6 parcels assigned "Escalate to hub manager":** High-probability delayed parcels originating in or routing through Escalate hubs (`SHP90002` at HUB24, `SHP90003` at HUB18, `SHP90011` at HUB17, `SHP90021` at HUB23, `SHP90035` at HUB15, and `SHP90048` at HUB21). Managers must reassign riders, authorize split delivery, or expedite cross-dock handling.
  - **2 parcels assigned "Proactive customer call":** Flagged parcels in Watch hubs (`SHP90005` at HUB22, `SHP90007` at HUB09). Operations places a proactive call to inform the consignee, protecting NPS.
  - **40 parcels assigned "No action":** Expected to meet promised SLA on regular dispatch.

### 4. Rationale for $K=3$ Clusters
In `outputs/part_d_separation.csv`, the across-to-within distance ratio increases with $K$ (2.8940 at $K=2$, 5.3928 at $K=3$, 6.3047 at $K=4$, and 6.8866 at $K=5$). However, $K=4$ and $K=5$ create micro-clusters of 4, 3, or 2 hubs (e.g. cluster sizes 8-7-5-4 and 8-7-4-3-2). Having fewer than 6 hubs in a segment cannot sustain distinct operational playbooks or dedicated management oversight. $K=3$ achieves strong mathematical separation (ratio 5.3928) while forming three well-populated, operationally viable segments (7, 8, and 9 hubs).

---

## 6. Production Plan & Limitations

### Production Plan

#### (a) Repository Production-Readiness Mapping
1. **Consistency:** Centralized preprocessing functions `clean_shipments` and `build_features` in `src/shipment_prep.py` are imported by training, packaging, and scoring scripts (`part_a_clean_eda.py`, `package_model.py`, `predict_delay.py`). Median imputation values and feature order are serialized and enforced, eliminating training-serving skew.
2. **Reproducibility:** All scripts enforce fixed random seeds (`random_state=42`) across train-test splits, cross-validation folds, tree ensembles, and K-Means initializations (`n_init=10`). Running `predict_delay.py` consecutively yields byte-identical output hashes.
3. **Maintainability:** Modular pipeline architecture isolates data preparation (`src/`), modeling (`part_b`, `part_c`), operational segmentation (`part_d`), packaging (`package_model.py`), and real-time scoring (`predict_delay.py`).
4. **Reliability:** Inputs are validated with scanner-fault sentinel filtering ($>100\text{ kg}$ weight treated as missing), title-casing of service types, and programmatic currency string stripping.
5. **Monitorability:** System emits granular check tables (`part_a_checks.csv`, `part_b_checks.csv`, `part_d_reload_check.csv`) logging missing rates, scaling moments, and inference verification diffs ($0.0 \le 10^{-9}$).

#### (b) Operational Monitoring Metrics & Trigger Thresholds
1. **Weekly Delayed Parcel Share:**
   - *Baseline:* 23.17% (historical overall delay rate).
   - *Trigger Threshold:* Exceeding **28.0%** overall weekly network delay rate (or $>55\%$ in Escalate hubs) triggers an operational alert to review line-haul fleet availability and regional weather disruptions.
2. **Weekly Flagged Dispatch Ratio:**
   - *Baseline:* ~16.7%–20.0% flagged rate.
   - *Trigger Threshold:* Flagged rate rising above **25.0%** triggers a data validation audit for upstream sensor faults or dispatch scheduling shifts.
3. **Rolling Precision & Recall (Lagged Verification):**
   - *Trigger Threshold:* Rolling 30-day recall dropping below **0.70** or precision dropping below **0.60** on parcels whose delivery outcome is confirmed triggers an automated retraining and threshold recalibration pipeline.

#### (c) Retraining Schedule & Sale Event Window Handling
- **Retraining Cadence:** Monthly retraining on a rolling 90-day window, incorporating confirmed delivery timestamps.
- **Handling Sale Windows (`sale_event_flag`):** The annual festive sale window (September 15–28) features non-linear spikes in volume and Express transit times (+12 extra hours). Retraining or validating performance during sale periods must compare against past festival sale windows, rather than against normal operating weeks.
- **Data Drift Detection:** Data drift would manifest as significant shifts in distance distributions (e.g. median distance shifting $>15\%$ due to new regional routing), payment mode shifts (e.g., COD share exceeding 50%), or rider route stop counts increasing beyond 95 stops per tour in Tier 2/3 hubs.

---

### Limitations

1. **Synthetic Data Constraints:** The dataset is generated through controlled synthetic distributions. While it simulates non-linear interactions (e.g., long-distance rain delays, festive Express surges), real-world logistics data contains messy GPS anomalies, failed delivery attempts, address quality ambiguities, and localized micro-hub disruptions that require continuous feature enrichment.
2. **Small Delayed Sample Size in Test Partition:** The held-out test set contains 180 total parcels with only 41 delayed shipments. A swing of just 2 parcels changes recall by ~4.9 percentage points, meaning confidence intervals around test metrics are moderately wide.
3. **Threshold Calibration on a Single Validation Split:** The alert threshold (0.5) was optimized on a single 180-row stratified validation slice. In high-stakes production, threshold optimization should be evaluated across multiple temporal out-of-time folds to ensure stability against seasonal volume swings.
