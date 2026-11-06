"""Part C: Delay Classifier and Alert Threshold."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS


def get_cm_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return int(tn), int(fp), int(fn), int(tp), round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)


def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")

    X = df[FEATURES]
    y = df["is_delayed"]

    # Task 1: Stratified split 70% / 15% / 15%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Delayed counts: train={y_train.sum()}, val={y_val.sum()}, test={y_test.sum()}")

    # Task 2: Blind baseline and three classifiers
    # Baseline: predicts 0 for all
    y_val_blind = np.zeros_like(y_val)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_val, y_val_blind)
    # prompt: "write precision as 0"
    prec = 0.0

    models_metrics = [
        {
            "model": "blind_all_on_time",
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
        }
    ]

    # Logistic Regression Pipeline
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])
    lr_pipe.fit(X_train, y_train)
    y_val_lr = lr_pipe.predict(X_val)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_val, y_val_lr)
    models_metrics.append({
        "model": "logistic_regression",
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    })

    # Decision Tree diagnostic: choose d from [2, 3, 4, 5, 6, 8, 10] by highest validation F1
    dt_candidates = [2, 3, 4, 5, 6, 8, 10]
    best_dt_d = None
    best_dt_f1 = -1.0
    best_dt_metrics = None

    for d in dt_candidates:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        y_val_dt = dt.predict(X_val)
        m = get_cm_metrics(y_val, y_val_dt)
        f1_val = m[7]
        if f1_val > best_dt_f1:
            best_dt_f1 = f1_val
            best_dt_d = d
            best_dt_metrics = m

    tn, fp, fn, tp, acc, prec, rec, f1 = best_dt_metrics
    models_metrics.append({
        "model": f"decision_tree_depth_{best_dt_d}",
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    })

    # Default Random Forest
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def.fit(X_train, y_train)
    y_val_rf = rf_def.predict(X_val)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_val, y_val_rf)
    models_metrics.append({
        "model": "random_forest_default",
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    })

    part_c_models = pd.DataFrame(models_metrics)
    part_c_models.to_csv("outputs/part_c_models.csv", index=False)

    # Task 3: Signal columns
    rf_signal = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_signal.fit(X_train, y_train)
    rf_importances = rf_signal.feature_importances_

    selector_all = SelectKBest(f_classif, k="all")
    selector_all.fit(X_train, y_train)
    f_scores = selector_all.scores_

    feature_scores_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": [round(float(s), 4) for s in f_scores],
        "rf_importance": [round(float(imp), 4) for imp in rf_importances],
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES],
    })
    # f_rank 1 = highest F-score
    feature_scores_df["f_rank"] = feature_scores_df["f_score"].rank(ascending=False, method="min").astype(int)
    # Arrange columns: feature, f_score, f_rank, rf_importance, is_noise
    feature_scores_df = feature_scores_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
    feature_scores_df = feature_scores_df.sort_values(by="f_rank", ascending=True)
    feature_scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

    # SelectK experiments: k in 5, 7, 10, 12, 17
    k_values = [5, 7, 10, 12, 17]
    selectk_rows = []
    for k in k_values:
        pipe_k = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipe_k.fit(X_train, y_train)
        pred_k = pipe_k.predict(X_val)
        val_f1_k = f1_score(y_val, pred_k, zero_division=0)
        selectk_rows.append({
            "k": k,
            "val_f1": round(val_f1_k, 4)
        })
    pd.DataFrame(selectk_rows).to_csv("outputs/part_c_selectk.csv", index=False)

    # Task 4: Tuning with k=7
    selector_7 = SelectKBest(f_classif, k=7)
    X_train_k7 = selector_7.fit_transform(X_train, y_train)
    X_val_k7 = selector_7.transform(X_val)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }

    # Grid search
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5)
    grid.fit(X_train_k7, y_train)
    grid_best_pred = grid.predict(X_val_k7)
    grid_val_f1 = f1_score(y_val, grid_best_pred, zero_division=0)

    # Randomized search
    rnd = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_grid, n_iter=9, cv=5, random_state=42)
    rnd.fit(X_train_k7, y_train)
    rnd_best_pred = rnd.predict(X_val_k7)
    rnd_val_f1 = f1_score(y_val, rnd_best_pred, zero_division=0)

    # Default forest on same 7 features (n_estimators=100, no depth limit, min_samples_leaf=1)
    rf_k7_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_k7_def.fit(X_train_k7, y_train)
    rf_k7_pred = rf_k7_def.predict(X_val_k7)
    rf_k7_val_f1 = f1_score(y_val, rf_k7_pred, zero_division=0)

    tuning_rows = [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(grid.best_params_),
            "best_cv_accuracy": round(float(grid.best_score_), 4),
            "val_f1": round(grid_val_f1, 4),
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(rnd.best_params_),
            "best_cv_accuracy": round(float(rnd.best_score_), 4),
            "val_f1": round(rnd_val_f1, 4),
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": "",
            "val_f1": round(rf_k7_val_f1, 4),
        },
    ]

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)

    # Task 5: Choosing final configuration
    # Ties -> default_forest_same_7_features, then grid_search, then randomized_search
    # Rank by val_f1 descending
    f1_dict = {
        "default_forest_same_7_features": rf_k7_val_f1,
        "grid_search": grid_val_f1,
        "randomized_search": rnd_val_f1,
    }
    # Find max val_f1
    max_f1 = max(f1_dict.values())
    if f1_dict["default_forest_same_7_features"] == max_f1:
        chosen_from = "default_forest_same_7_features"
        chosen_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
    elif f1_dict["grid_search"] == max_f1:
        chosen_from = "grid_search"
        chosen_params = grid.best_params_
    else:
        chosen_from = "randomized_search"
        chosen_params = rnd.best_params_

    final_rf_kwargs = {"random_state": 42}
    final_rf_kwargs.update(chosen_params)
    final_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(**final_rf_kwargs))
    ])
    final_model.fit(X_train, y_train)

    # Task 6: Alert threshold
    val_probs = final_model.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_rows = []
    chosen_threshold = 0.1

    # Find highest t with recall >= 0.80
    valid_thresholds = []
    for t in thresholds:
        y_val_t = (val_probs >= t).astype(int)
        tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_val, y_val_t)
        thresh_rows.append({
            "threshold": t,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
        })
        if rec >= 0.80:
            valid_thresholds.append(t)

    if valid_thresholds:
        chosen_threshold = max(valid_thresholds)
    else:
        chosen_threshold = 0.1

    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv("outputs/part_c_thresholds.csv", index=False)

    # Task 7: One test evaluation at chosen threshold
    test_probs = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = (test_probs >= chosen_threshold).astype(int)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_cm_metrics(y_test, y_test_pred)

    final_test_df = pd.DataFrame([{
        "threshold": chosen_threshold,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    }])
    final_test_df.to_csv("outputs/part_c_final_test.csv", index=False)

    # outputs/part_c_final_config.csv
    # selector_k, n_estimators, max_depth, min_samples_leaf, threshold, chosen_from
    max_depth_val = "unlimited" if chosen_params.get("max_depth") is None else chosen_params.get("max_depth")
    final_config_df = pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": chosen_params.get("n_estimators", 100),
        "max_depth": max_depth_val,
        "min_samples_leaf": chosen_params.get("min_samples_leaf", 1),
        "threshold": chosen_threshold,
        "chosen_from": chosen_from
    }])
    final_config_df.to_csv("outputs/part_c_final_config.csv", index=False)

    print(f"Part C completed. Chosen model from: {chosen_from}, threshold: {chosen_threshold}")
    print(f"Test performance: Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")


if __name__ == "__main__":
    main()
