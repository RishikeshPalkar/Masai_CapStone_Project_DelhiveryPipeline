# Delhivery Last-Mile Delay Intelligence Pipeline

A unified operational intelligence pipeline for Delhivery to estimate delivery hours, predict shipment delays with operations-tuned alerting, and segment hub performance.

---

## 1. How to Run

Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

> **Model Packaging Note:** `models/delay_model.pkl` is serialized with `scikit-learn==1.7.0`. The pickle loads reliably on the pinned version. To regenerate the model pickle on any environment, run `python package_model.py`.

Run the full pipeline in order from repository root:
```bash
python generate_data.py         # 1. Generate synthetic data (24 hubs, 1224 raw, 48 new)
python part_a_clean_eda.py      # 2. Clean data, engineer features, export checks
python part_b_regression.py     # 3. Fit and evaluate delivery-hours regression
python part_c_classification.py # 4. Train delay classifier and select alert threshold
python part_d_segments.py       # 5. Segment 24 hubs with K-Means (K=3)
python package_model.py         # 6. Package final model artifacts and verify reload
python predict_delay.py         # 7. Batch score 48 new parcels and assign actions
```

---

## Part A — Data Findings

All numbers are derived from `outputs/part_a_checks.csv`, `outputs/part_a_delay_by_group.csv`, and `outputs/part_a_correlations.csv`.

### Key Data Cleaning Metrics
- **Dataset:** 1,224 raw rows deduplicated to 1,200 records (24 duplicates removed; 0 nulls across all 17 features after hub merge).
- **Text & Sentinels:** 12 irregular `service_type` spellings cleaned to 3 (`Express`, `Standard`, `Economy`). 5 scanner-fault sentinels (>100 kg) treated as missing.
- **Imputed Medians:** `distance_km` = **314.95 km**, `parcel_weight_kg` = **1.83 kg**, `declared_value_inr` = **INR 1,121.50**.
- **Delays:** 278 out of 1,200 shipments delayed (**23.17%** network delay rate).

### Delay Rate by Group (`outputs/part_a_delay_by_group.csv`)
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
1. **Service Type Delay:** **Express** is delayed most often at **0.4897** (119 / 243 delayed) vs **0.2056** for Standard and **0.0857** for Economy, driven by its tight 36-hour promise SLA.
2. **Hub Tier Delay Escalation:** Delay rate increases 11-fold from Tier 1 to Tier 3: **0.0400** in Tier 1 $\rightarrow$ **0.2150** in Tier 2 $\rightarrow$ **0.4400** in Tier 3 (+40.00 percentage points).
3. **Distance Band Vulnerability:** Delay rate increases monotonically across distance bands: **0.0219** (0–200 km), **0.1368** (201–400 km), **0.4364** (401–600 km), and **0.8523** for long routes (601+ km).
4. **Top 3 Correlated Features with `delivery_hours`:** Ranking by absolute Pearson correlation (excluding `delivery_hours` itself):
   - `promised_hours`: **+0.8372**
   - `distance_km`: **+0.4013**
   - `n_stops`: **+0.3247**
5. **Declared Value Log Transformation:** Raw `declared_value_inr` is heavily right-skewed with a mean of **INR 1,533.0542** pulling far above the median of **INR 1,121.50**; taking the natural log normalizes the distribution, aligning post-transform mean (**7.0277**) with median (**7.0224**).

---

## Part B — Model Results (Regression)

All metrics are recorded in `outputs/part_b_checks.csv`, `outputs/part_b_tree_depth.csv`, `outputs/part_b_metrics.csv`, `outputs/part_b_gb_learning_rate.csv`, and `outputs/part_b_cv.csv`.

- **Leakage-Safe Split (70% / 15% / 15%):** 840 train / 180 val / 180 test. Scaler fit on training rows only (`train_scaled_max_abs_mean` = 0.0, `test_scaled_max_abs_mean` = 0.2187).

### Model Performance Summary (`outputs/part_b_metrics.csv`)
| Model | Split | MAE | MSE | RMSE | R2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| mean_baseline | val | 15.7952 | 355.5204 | 18.8552 | -0.0212 |
| linear_regression | val | 3.9222 | 23.9235 | **4.8912** | **0.9313** |
| decision_tree (d=6) | val | 5.0971 | 39.4338 | 6.2796 | 0.8867 |
| random_forest | val | 4.2882 | 30.3550 | 5.5095 | 0.9128 |
| gradient_boosting | val | 3.9166 | 24.1722 | 4.9165 | 0.9306 |
| **linear_regression (final)** | **test** | **3.5671** | **20.4624** | **4.5235** | **0.9493** |

### Key Regression Takeaways
- **Baseline R² (-0.0212):** Predicting the training target mean (49.88 hrs) performs worse than predicting the validation mean itself and explains 0% variance.
- **Validation Winner:** **Linear Regression** achieved the lowest validation RMSE of **4.8912 hours** ($R^2$ = 0.9313), closely followed by Gradient Boosting (4.9165 hours, $R^2$ = 0.9306).
- **Tree-Depth Bias vs. Variance:**
  - *High Bias:* Depth 1 has train RMSE 11.6334 and val RMSE 12.1324 (gap 0.4990 hrs, but val error is 5.8528 hrs worse than best depth).
  - *Optimal Depth:* Depth 6 minimizes validation RMSE at **6.2796 hours** (train RMSE 4.7114 hrs).
  - *High Variance:* Unlimited depth memorizes training data (train RMSE 0.0000) while val RMSE spikes to 7.8368 hours (gap = **7.8368 hrs**).
- **Learning Rate Experiment:** `lr=0.1` is optimal (val RMSE 4.9165 hrs, $R^2$ = 0.9306); `lr=0.01` underfits due to slow convergence (val RMSE 10.0018 hrs, $R^2$ = 0.7127); `lr=1.0` overshoots minima (val RMSE 7.3701 hrs, $R^2$ = 0.8440).
- **Cross-Validation & Shipping Decision:** 5-fold CV mean $R^2$ on 1,020 rows gave 0.9353 (std 0.0045) for Gradient Boosting vs 0.9261 (std 0.0048) for Linear Regression. We ship **Linear Regression** because it achieved the lowest validation RMSE, runs in sub-millisecond inference time, and achieved **test RMSE of 4.5235 hours** and **test $R^2$ of 0.9493**.

---

## Part C — Model Results (Classification)

All metrics are recorded in `outputs/part_c_models.csv`, `outputs/part_c_feature_scores.csv`, `outputs/part_c_selectk.csv`, `outputs/part_c_tuning.csv`, `outputs/part_c_thresholds.csv`, `outputs/part_c_final_config.csv`, and `outputs/part_c_final_test.csv`.

- **Stratified Split:** 840 train / 180 val / 180 test (195 / 42 / 41 delayed shipments).

### Baseline and Model Comparison (`outputs/part_c_models.csv`)
| Model | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| blind_all_on_time | 138 | 0 | 42 | 0 | 0.7667 | 0.0 | 0.0 | 0.0 |
| logistic_regression | 128 | 10 | 7 | 35 | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| decision_tree_depth_6 | 128 | 10 | 11 | 31 | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| random_forest_default | 132 | 6 | 11 | 31 | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

### Key Classification Takeaways
- **Blind Baseline (Accuracy 0.7667):** Completely useless operationally with 0.0 recall, precision, and F1, missing all 42 delays.
- **Noise Elimination:** The 5 bookkeeping noise columns accounted for 0.2245 of forest importance, but ANOVA F-scores were all $<5.0$ (ranks 8, 12, 14, 16, 17). Top 7 features all scored $>35.0$ (led by `distance_km` at 389.2131).
- **Feature Selection ($k$):** Validation F1 peaked at **$k=7$** (**0.8333**) vs 0.6988 ($k=5$), 0.7857 ($k=10$), 0.8049 ($k=12$), and 0.7848 ($k=17$).
- **Hyperparameter Tuning:** On 7 features, the untuned default forest (F1 = **0.8333**) beat Grid Search (F1 = 0.7750, CV accuracy 0.8952) and Randomized Search (F1 = 0.7848, CV accuracy 0.8905). Tuning for CV accuracy on 180 validation rows restricted tree depth, reducing recall.
- **Alert Threshold Selection:** Operations mandates the highest threshold with validation recall $\ge 0.80$. Threshold **0.5** was chosen (Recall = **0.8333**, Precision = **0.7955**, F1 = **0.8140**; at 0.6, recall falls to 0.7619).
- **Final Test Performance (at 0.5):** **Precision = 0.6889, Recall = 0.7561, F1 = 0.7209** (31 TP, 14 FP, 10 FN, 125 TN). Each missed parcel represents ~2.44% recall on 41 delayed test shipments.

---

## Part D — Operations Recommendation & Hub Segments

All numbers are derived from `outputs/hub_profile.csv`, `outputs/part_d_separation.csv`, `outputs/hub_segments.csv`, `outputs/segment_profile.csv`, `outputs/tier_vs_segment_mismatch.csv`, and `outputs/new_shipment_actions.csv`.

### Hub Segment Profiles ($K=3$ K-Means)
- **Standard (7 hubs):** Delay rate **0.0257** (2.57%), mean delivery time 47.24 hrs, distance 188.93 km, stops 55.37.
- **Watch (8 hubs):** Delay rate **0.1550** (15.50%), mean delivery time 56.81 hrs, distance 314.18 km, stops 70.72.
- **Escalate (9 hubs):** Delay rate **0.4600** (46.00%), mean delivery time 64.65 hrs, distance 470.48 km, stops 84.34.
  - *Hubs in Escalate:* HUB13 (0.46), HUB15 (0.46), HUB17 (0.46), HUB18 (0.44), HUB19 (0.56), HUB20 (0.46), HUB21 (0.48), HUB23 (0.44), HUB24 (0.38).

### Tier vs. Segment Mismatches
Four hubs deviate significantly from their official tier labels:
1. **`HUB07` Kolkata:** Labelled **Tier 1**, operates as **Watch** (delay rate 0.2000, 57.07 hrs). Needs SLA buffer extension.
2. **`HUB13` Nagpur:** Labelled **Tier 2**, operates as **Escalate** (delay rate 0.4600, 66.07 hrs). Needs Tier 3 line-haul resources.
3. **`HUB15` Kochi:** Labelled **Tier 2**, operates as **Escalate** (delay rate 0.4600, 63.66 hrs). Geography and route distance cause high delays.
4. **`HUB22` Dehradun:** Labelled **Tier 3**, operates as **Watch** (delay rate 0.2000, 55.43 hrs). Outperforms its tier label.
- *Recommendation:* Operations must base SLA promises and fleet allocation on empirical segment profiles rather than nominal tier labels.

### Alert Operational Precision & Batch Scoring
- **Dispatch Alert Metrics:** At threshold 0.5, **~7 out of 10 flagged parcels are really delayed** (Precision = 0.6889), and **~7.6 out of 10 delayed parcels are caught** (Recall = 0.7561).
- **New Shipments Action Allocation (48 parcels):**
  - **8 parcels flagged** (16.67%).
  - **6 Escalate to hub manager:** High-risk parcels originating in Escalate hubs (`SHP90002`, `SHP90003`, `SHP90011`, `SHP90021`, `SHP90035`, `SHP90048`).
  - **2 Proactive customer call:** Flagged parcels in Watch hubs (`SHP90005`, `SHP90007`).
  - **40 No action:** On-time delivery expected.
- **Why $K=3$ Was Kept:** Although across-to-within distance ratio increases at $K=4$ (6.3047) and $K=5$ (6.8866) in `part_d_separation.csv`, they create micro-clusters of 2–4 hubs (cluster sizes 8-7-5-4 and 8-7-4-3-2). Having fewer than 6 hubs cannot support a distinct operational playbook; $K=3$ provides 3 balanced, actionable segments (7, 8, 9 hubs).

---

## 6. Production Plan & Limitations

### Production Plan
- **Production Readiness:**
  - *Consistency:* Centralized data preprocessing in `src/shipment_prep.py` used identically in training, packaging, and scoring scripts.
  - *Reproducibility:* Fixed `random_state=42` across all splits, cross-validation, and models. Scoring script produces byte-identical hash outputs across runs.
  - *Maintainability:* Modular codebase with separated concerns (`src/`, `models/`, `outputs/`).
  - *Reliability:* Robust input cleaning (handling sentinel weights $>100\text{ kg}$, currency symbols, and whitespace).
  - *Monitorability:* Automated check files (`part_a_checks.csv`, `part_b_checks.csv`, `part_d_reload_check.csv` showing 0.0 reload probability diff).
- **Monitoring Metrics & Alert Triggers:**
  1. *Weekly Network Delay Rate:* Baseline 23.17%; alert trigger at **> 28.0%** overall (or $> 55\%$ in Escalate hubs).
  2. *Weekly Flagged Dispatch Ratio:* Baseline ~16.7%; alert trigger at **> 25.0%**.
  3. *Rolling 30-Day Lagged Recall:* Alert trigger at **< 0.70** recall on confirmed deliveries initiates retraining.
- **Retraining & Sale Events:** Monthly retraining on a rolling 90-day window. The September 15–28 festive sale window (`sale_event_flag`) introduces non-linear Express transit spikes (+12 hrs); performance must be benchmarked against past festive sales, not ordinary weeks.

### Limitations
1. **Synthetic Data:** Real-world logistics data contains messy GPS pings, address resolution errors, and local weather micro-events not captured in synthetic distributions.
2. **Small Test Delay Count:** Only 41 delayed shipments in the test partition; each missed parcel represents a 2.44% swing in recall.
3. **Single Validation Threshold Tuning:** The alert threshold was calibrated on a single 180-row validation slice; temporal out-of-time walk-forward validation is recommended for production deployment.
