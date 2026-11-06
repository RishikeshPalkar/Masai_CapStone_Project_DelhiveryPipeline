"""Part D: Package final model and artifacts."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES, clean_shipments


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Read configuration
    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    selector_k = int(config["selector_k"])
    n_estimators = int(config["n_estimators"])
    max_depth = None if str(config["max_depth"]).lower() == "unlimited" else int(config["max_depth"])
    min_samples_leaf = int(config["min_samples_leaf"])
    threshold = float(config["threshold"])

    # 2. Get medians from clean_shipments
    raw = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw)

    # 3. Read modelling table and rebuild stratified split
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 4. Fit final pipeline
    pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=selector_k)),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    # 5. Save artifacts
    # models/delay_model.pkl
    joblib.dump(pipeline, "models/delay_model.pkl")

    # models/delay_threshold.csv
    pd.DataFrame({"threshold": [round(threshold, 4)]}).to_csv("models/delay_threshold.csv", index=False)

    # models/feature_order.csv
    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

    # models/impute_values.csv (columns: column, median)
    impute_df = pd.DataFrame([
        {"column": "distance_km", "median": round(medians["distance_km"], 4)},
        {"column": "parcel_weight_kg", "median": round(medians["parcel_weight_kg"], 4)},
        {"column": "declared_value_inr", "median": round(medians["declared_value_inr"], 4)},
    ])
    impute_df.to_csv("models/impute_values.csv", index=False)

    # 6. Reload and check
    reloaded_pipeline = joblib.load("models/delay_model.pkl")
    probs_in_memory = pipeline.predict_proba(X_test)[:, 1]
    probs_reloaded = reloaded_pipeline.predict_proba(X_test)[:, 1]
    max_abs_diff = float(np.max(np.abs(probs_in_memory - probs_reloaded)))

    reload_checks = pd.DataFrame([
        {"metric": "test_rows_scored", "value": str(len(X_test))},
        {"metric": "reload_max_abs_probability_diff", "value": str(round(max_abs_diff, 4))},
        {"metric": "threshold", "value": str(round(threshold, 4))},
    ])
    reload_checks.to_csv("outputs/part_d_reload_check.csv", index=False)

    print("Model packaged successfully.")
    print(f"Reload diff: {max_abs_diff}, Threshold: {threshold}")


if __name__ == "__main__":
    main()
