"""Part A: Data cleaning, feature engineering, and exploratory data analysis."""
import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    raw_rows = len(raw)
    
    # Pre-clean inspection on deduplicated raw data
    dedup_raw = raw.drop_duplicates().copy()
    duplicate_rows_removed = raw_rows - len(dedup_raw)
    rows_after_dedup = len(dedup_raw)

    missing_distance_km_before = int(dedup_raw["distance_km"].isna().sum())
    
    # missing parcel weight before (does not include sentinel rows)
    missing_parcel_weight_before = int(dedup_raw["parcel_weight_kg"].isna().sum())
    sentinel_weight_rows = int((dedup_raw["parcel_weight_kg"] > 100).sum())

    missing_declared_value_before = int(dedup_raw["declared_value"].isna().sum())
    missing_payment_mode_before = int(dedup_raw["payment_mode"].isna().sum())

    service_type_spellings_before = int(dedup_raw["service_type"].nunique())

    # Now clean shipments
    clean_df, medians = clean_shipments(raw)
    service_type_spellings_after = int(clean_df["service_type"].nunique())

    # Build features
    modelling_df = build_features(clean_df, hubs)
    rows_after_hub_merge = len(modelling_df)

    # Add target
    modelling_df["is_delayed"] = (modelling_df["delivery_hours"] > modelling_df["promised_hours"]).astype(int)

    # Save data/modelling_table.csv
    modelling_df.to_csv("data/modelling_table.csv", index=False)

    # Calculate checks
    check_cols = FEATURES + ["delivery_hours", "is_delayed"]
    nulls_in_features_and_target = int(modelling_df[check_cols].isna().sum().sum())

    delayed_count = int(modelling_df["is_delayed"].sum())
    delay_rate = float(modelling_df["is_delayed"].mean())

    hours = modelling_df["delivery_hours"]
    hours_q1 = float(hours.quantile(0.25))
    hours_q3 = float(hours.quantile(0.75))
    iqr = hours_q3 - hours_q1
    hours_lower_bound = hours_q1 - 1.5 * iqr
    hours_upper_bound = hours_q3 + 1.5 * iqr
    hours_outlier_count = int(((hours < hours_lower_bound) | (hours > hours_upper_bound)).sum())

    declared_value_mean_inr = float(modelling_df["declared_value_inr"].mean())
    declared_value_median_inr = float(modelling_df["declared_value_inr"].median())
    log_declared_value_mean = float(modelling_df["log_declared_value"].mean())
    log_declared_value_median = float(modelling_df["log_declared_value"].median())

    def format_check_val(v):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(round(float(v), 4))

    checks_data = [
        ("raw_rows", format_check_val(raw_rows)),
        ("duplicate_rows_removed", format_check_val(duplicate_rows_removed)),
        ("rows_after_dedup", format_check_val(rows_after_dedup)),
        ("missing_distance_km_before", format_check_val(missing_distance_km_before)),
        ("missing_parcel_weight_before", format_check_val(missing_parcel_weight_before)),
        ("sentinel_weight_rows", format_check_val(sentinel_weight_rows)),
        ("missing_declared_value_before", format_check_val(missing_declared_value_before)),
        ("missing_payment_mode_before", format_check_val(missing_payment_mode_before)),
        ("service_type_spellings_before", format_check_val(service_type_spellings_before)),
        ("service_type_spellings_after", format_check_val(service_type_spellings_after)),
        ("median_distance_km", format_check_val(medians["distance_km"])),
        ("median_parcel_weight_kg", format_check_val(medians["parcel_weight_kg"])),
        ("median_declared_value_inr", format_check_val(medians["declared_value_inr"])),
        ("rows_after_hub_merge", format_check_val(rows_after_hub_merge)),
        ("nulls_in_features_and_target", format_check_val(nulls_in_features_and_target)),
        ("delayed_count", format_check_val(delayed_count)),
        ("delay_rate", format_check_val(delay_rate)),
        ("hours_q1", format_check_val(hours_q1)),
        ("hours_q3", format_check_val(hours_q3)),
        ("hours_lower_bound", format_check_val(hours_lower_bound)),
        ("hours_upper_bound", format_check_val(hours_upper_bound)),
        ("hours_outlier_count", format_check_val(hours_outlier_count)),
        ("declared_value_mean_inr", format_check_val(declared_value_mean_inr)),
        ("declared_value_median_inr", format_check_val(declared_value_median_inr)),
        ("log_declared_value_mean", format_check_val(log_declared_value_mean)),
        ("log_declared_value_median", format_check_val(log_declared_value_median)),
    ]

    checks_df = pd.DataFrame(checks_data, columns=["metric", "value"])
    checks_df["value"] = checks_df["value"].astype(object)
    checks_df.to_csv("outputs/part_a_checks.csv", index=False)

    # Delay evidence table
    # distance_band is distance_km cut into bins 0-200, 201-400, 401-600, 601+
    # edges 0, 200, 400, 600, 2000; right edge inclusive
    bins = [0, 200, 400, 600, 2000]
    labels = ["0-200", "201-400", "401-600", "601+"]
    modelling_df["distance_band"] = pd.cut(modelling_df["distance_km"], bins=bins, labels=labels, right=True)

    group_rows = []
    
    # 1. service_type
    for st in sorted(modelling_df["service_type"].unique()):
        sub = modelling_df[modelling_df["service_type"] == st]
        group_rows.append({
            "dimension": "service_type",
            "group": st,
            "shipments": len(sub),
            "delayed": int(sub["is_delayed"].sum()),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })

    # 2. hub_tier
    for tier in sorted(modelling_df["hub_tier"].unique()):
        sub = modelling_df[modelling_df["hub_tier"] == tier]
        group_rows.append({
            "dimension": "hub_tier",
            "group": str(tier),
            "shipments": len(sub),
            "delayed": int(sub["is_delayed"].sum()),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })

    # 3. distance_band
    for band in labels:
        sub = modelling_df[modelling_df["distance_band"] == band]
        group_rows.append({
            "dimension": "distance_band",
            "group": band,
            "shipments": len(sub),
            "delayed": int(sub["is_delayed"].sum()),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })

    delay_by_group_df = pd.DataFrame(group_rows)
    delay_by_group_df.to_csv("outputs/part_a_delay_by_group.csv", index=False)

    # Correlation matrix of numeric columns
    numeric_cols = modelling_df.select_dtypes(include=[np.number]).columns
    corr_df = modelling_df[numeric_cols].corr().round(4)
    corr_df.to_csv("outputs/part_a_correlations.csv", index=True)

    print("Part A completed successfully.")


if __name__ == "__main__":
    main()
