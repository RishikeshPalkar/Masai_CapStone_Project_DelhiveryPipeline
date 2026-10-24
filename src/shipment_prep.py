"""Shared cleaning and feature engineering for shipment data."""

import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]
MEDIAN_COLUMNS = ("distance_km", "parcel_weight_kg", "declared_value_inr")
HUB_COLUMNS = ["hub_city", "region", "hub_tier", "hub_daily_capacity"]


def clean_shipments(
    raw: pd.DataFrame, medians: dict[str, float] | None = None
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Clean raw shipments and fill numeric gaps using historical medians."""
    clean = raw.drop_duplicates().copy()
    clean["service_type"] = clean["service_type"].str.strip().str.title()
    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")
    value_text = clean["declared_value"].str.replace("₹", "", regex=False)
    value_text = value_text.str.replace(",", "", regex=False)
    clean["declared_value_inr"] = pd.to_numeric(value_text, errors="raise")
    clean = clean.drop(columns=["declared_value"])
    clean["dispatch_time"] = pd.to_datetime(clean["dispatch_time"])
    clean.loc[clean["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    if medians is None:
        fill_values = {
            column: float(clean[column].median()) for column in MEDIAN_COLUMNS
        }
    else:
        fill_values = {column: float(medians[column]) for column in MEDIAN_COLUMNS}

    for column, value in fill_values.items():
        if not np.isfinite(value):
            raise ValueError(f"Median for {column} must be a finite number")
        clean[column] = clean[column].fillna(value)

    return clean, fill_values


def build_features(clean: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    """Attach hub attributes and derive features without using delivery outcomes."""
    features = clean.merge(
        hubs[["hub_id", *HUB_COLUMNS]],
        on="hub_id",
        how="left",
        validate="many_to_one",
    )
    features["promised_hours"] = features["service_type"].map(PROMISED_HOURS)
    if features["promised_hours"].isna().any():
        unknown_services = features.loc[
            features["promised_hours"].isna(), "service_type"
        ].unique()
        raise ValueError(f"Unrecognized service_type values: {unknown_services}")
    features["promised_hours"] = features["promised_hours"].astype(int)
    features["is_cod"] = features["payment_mode"].eq("COD").astype(int)
    features["is_weekend"] = (
        features["dispatch_time"].dt.dayofweek.isin([5, 6]).astype(int)
    )
    features["dispatch_hour"] = features["dispatch_time"].dt.hour.astype(int)
    features["log_declared_value"] = np.log(features["declared_value_inr"])
    return features
