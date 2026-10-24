"""Clean shipment exports and generate Part A modelling and evidence tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, build_features, clean_shipments

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DISTANCE_BINS = [0, 200, 400, 600, 2000]
DISTANCE_LABELS = ["0-200", "201-400", "401-600", "601+"]


def main() -> None:
    raw = pd.read_csv(DATA_DIR / "raw_shipments.csv")
    hubs = pd.read_csv(DATA_DIR / "hubs.csv")
    deduplicated = raw.drop_duplicates()

    missing_distance = int(deduplicated["distance_km"].isna().sum())
    missing_weight = int(deduplicated["parcel_weight_kg"].isna().sum())
    missing_value = int(deduplicated["declared_value"].isna().sum())
    missing_payment = int(deduplicated["payment_mode"].isna().sum())
    sentinel_weights = int((deduplicated["parcel_weight_kg"] > 100).sum())
    service_spellings_before = int(deduplicated["service_type"].nunique())

    clean, medians = clean_shipments(raw)
    service_spellings_after = int(clean["service_type"].nunique())
    modelling = build_features(clean, hubs)
    if len(modelling) != len(clean):
        raise ValueError("Hub merge changed the shipment row count")
    modelling["is_delayed"] = (
        modelling["delivery_hours"] > modelling["promised_hours"]
    ).astype(int)
    modelling.to_csv(DATA_DIR / "modelling_table.csv", index=False)

    declared_value = modelling["declared_value_inr"]
    log_declared_value = np.log(declared_value)
    q1, q3 = modelling["delivery_hours"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    numeric_features_and_target = [*FEATURES, "delivery_hours", "is_delayed"]
    checks = [
        ("raw_rows", len(raw)),
        ("duplicate_rows_removed", len(raw) - len(deduplicated)),
        ("rows_after_dedup", len(deduplicated)),
        ("missing_distance_km_before", missing_distance),
        ("missing_parcel_weight_before", missing_weight),
        ("sentinel_weight_rows", sentinel_weights),
        ("missing_declared_value_before", missing_value),
        ("missing_payment_mode_before", missing_payment),
        ("service_type_spellings_before", service_spellings_before),
        ("service_type_spellings_after", service_spellings_after),
        ("median_distance_km", medians["distance_km"]),
        ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
        ("median_declared_value_inr", medians["declared_value_inr"]),
        ("rows_after_hub_merge", len(modelling)),
        (
            "nulls_in_features_and_target",
            int(modelling[numeric_features_and_target].isna().sum().sum()),
        ),
        ("delayed_count", int(modelling["is_delayed"].sum())),
        ("delay_rate", round(float(modelling["is_delayed"].mean()), 4)),
        ("hours_q1", round(float(q1), 2)),
        ("hours_q3", round(float(q3), 2)),
        ("hours_lower_bound", round(float(lower_bound), 2)),
        ("hours_upper_bound", round(float(upper_bound), 2)),
        (
            "hours_outlier_count",
            int(
                (
                    (modelling["delivery_hours"] < lower_bound)
                    | (modelling["delivery_hours"] > upper_bound)
                ).sum()
            ),
        ),
        ("declared_value_mean_inr", round(float(declared_value.mean()), 2)),
        ("declared_value_median_inr", round(float(declared_value.median()), 2)),
        ("log_declared_value_mean", round(float(log_declared_value.mean()), 3)),
        ("log_declared_value_median", round(float(log_declared_value.median()), 3)),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks, columns=["metric", "value"]).to_csv(
        OUTPUT_DIR / "part_a_checks.csv", index=False
    )

    evidence = modelling.copy()
    evidence["distance_band"] = pd.cut(
        evidence["distance_km"],
        bins=DISTANCE_BINS,
        labels=DISTANCE_LABELS,
        right=True,
        include_lowest=True,
    )
    if evidence["distance_band"].isna().any():
        raise ValueError("Some distances fall outside the configured distance bands")

    delay_groups = []
    for dimension in ("service_type", "hub_tier", "distance_band"):
        grouped = evidence.groupby(dimension, sort=True, observed=True).agg(
            shipments=("shipment_id", "size"),
            delayed=("is_delayed", "sum"),
            mean_hours=("delivery_hours", "mean"),
        )
        for group, row in grouped.iterrows():
            shipments = int(row["shipments"])
            delayed = int(row["delayed"])
            delay_groups.append(
                {
                    "dimension": dimension,
                    "group": str(group),
                    "shipments": shipments,
                    "delayed": delayed,
                    "delay_rate": round(delayed / shipments, 4),
                    "mean_hours": float(row["mean_hours"]),
                }
            )
    pd.DataFrame(
        delay_groups,
        columns=[
            "dimension",
            "group",
            "shipments",
            "delayed",
            "delay_rate",
            "mean_hours",
        ],
    ).to_csv(OUTPUT_DIR / "part_a_delay_by_group.csv", index=False)

    numeric_columns = modelling.select_dtypes(include="number")
    numeric_columns.corr().round(4).to_csv(
        OUTPUT_DIR / "part_a_correlations.csv", index_label="column"
    )


if __name__ == "__main__":
    main()
