"""Part B: Delivery-Hours Regression."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES


def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")

    X = df[FEATURES]
    y = df["delivery_hours"]

    # Task 1: Split into training (70%), validation (15%), test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Task 2: Scaling without leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_scaled_max_abs_mean = float(np.max(np.abs(X_train_scaled.mean(axis=0))))
    test_scaled_max_abs_mean = float(np.max(np.abs(X_test_scaled.mean(axis=0))))

    part_b_checks = pd.DataFrame([
        ("train_rows", str(len(X_train))),
        ("val_rows", str(len(X_val))),
        ("test_rows", str(len(X_test))),
        ("train_scaled_max_abs_mean", str(round(train_scaled_max_abs_mean, 4))),
        ("test_scaled_max_abs_mean", str(round(test_scaled_max_abs_mean, 4))),
    ], columns=["metric", "value"])
    part_b_checks.to_csv("outputs/part_b_checks.csv", index=False)

    # Task 3: Baseline on validation
    y_train_mean = y_train.mean()
    y_val_baseline_pred = np.full_like(y_val, fill_value=y_train_mean)
    mae_base = mean_absolute_error(y_val, y_val_baseline_pred)
    mse_base = mean_squared_error(y_val, y_val_baseline_pred)
    rmse_base = np.sqrt(mse_base)
    r2_base = r2_score(y_val, y_val_baseline_pred)

    # Task 4: Tree-depth diagnostic
    tree_depths = [1, 2, 3, 4, 6, 8, 12, None]
    tree_rows = []
    best_depth = None
    best_val_rmse = float("inf")

    for d in tree_depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        pred_tr = dt.predict(X_train)
        pred_va = dt.predict(X_val)
        tr_rmse = np.sqrt(mean_squared_error(y_train, pred_tr))
        va_rmse = np.sqrt(mean_squared_error(y_val, pred_va))

        depth_str = str(d) if d is not None else "unlimited"
        tree_rows.append({
            "max_depth": depth_str,
            "train_rmse": round(tr_rmse, 4),
            "val_rmse": round(va_rmse, 4),
        })

        if va_rmse < best_val_rmse:
            best_val_rmse = va_rmse
            best_depth = d

    tree_depth_df = pd.DataFrame(tree_rows)
    tree_depth_df.to_csv("outputs/part_b_tree_depth.csv", index=False)

    # Task 5: Four candidate models
    models = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    metrics_rows = [
        {
            "model": "mean_baseline",
            "split": "val",
            "MAE": round(mae_base, 4),
            "MSE": round(mse_base, 4),
            "RMSE": round(rmse_base, 4),
            "R2": round(r2_base, 4),
        }
    ]

    val_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_val_pred)
        mse = mean_squared_error(y_val, y_val_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_val_pred)
        val_scores[name] = rmse
        metrics_rows.append({
            "model": name,
            "split": "val",
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        })

    # Task 6: Learning-rate experiment
    learning_rates = [0.01, 0.1, 1.0]
    lr_rows = []
    for lr in learning_rates:
        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gbr.fit(X_train, y_train)
        pred_va = gbr.predict(X_val)
        va_rmse = np.sqrt(mean_squared_error(y_val, pred_va))
        va_r2 = r2_score(y_val, pred_va)
        lr_rows.append({
            "learning_rate": lr,
            "val_rmse": round(va_rmse, 4),
            "val_r2": round(va_r2, 4),
        })
    lr_df = pd.DataFrame(lr_rows)
    lr_df.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

    # Task 7: Cross-validated comparison and final model
    # Rank candidates by validation RMSE
    ranked_candidates = sorted(val_scores.items(), key=lambda x: x[1])
    top_two = [ranked_candidates[0][0], ranked_candidates[1][0]]

    # 1,020 train + val rows
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    for name in top_two:
        # Clone or fresh instance
        m = models[name]
        scores = cross_val_score(m, X_train_val, y_train_val, cv=cv, scoring="r2")
        cv_rows.append({
            "model": name,
            "mean_r2": round(float(scores.mean()), 4),
            "std_r2": round(float(scores.std()), 4),
        })
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv("outputs/part_b_cv.csv", index=False)

    # Final model is the candidate with lowest validation RMSE
    final_model_name = ranked_candidates[0][0]
    final_model = models[final_model_name]
    # Model already fitted on X_train
    y_test_pred = final_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    metrics_rows.append({
        "model": final_model_name,
        "split": "test",
        "MAE": round(test_mae, 4),
        "MSE": round(test_mse, 4),
        "RMSE": round(test_rmse, 4),
        "R2": round(test_r2, 4),
    })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)

    print(f"Part B completed. Final model: {final_model_name}, test RMSE: {test_rmse:.4f}, test R2: {test_r2:.4f}")


if __name__ == "__main__":
    main()
