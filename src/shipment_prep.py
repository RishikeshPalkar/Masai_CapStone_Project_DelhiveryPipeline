"""Shared data preparation, cleaning, and feature engineering for Delhivery pipeline."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def clean_shipments(raw, medians=None):
    """
    Clean raw shipments DataFrame.
    Returns (clean_df, medians).
    """
    df = raw.copy()
    
    # Step 1: drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Step 2: service_type: strip whitespace and convert to title case
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()
    
    # Step 3: payment_mode: replace missing values with placeholder 'Unknown'
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    
    # Step 4: declared_value: remove rupee sign and commas, convert to float in declared_value_inr, drop declared_value
    def _parse_declared_val(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).replace("\u20b9", "").replace(",", "").strip()
        return float(val_str) if val_str else np.nan
    
    df["declared_value_inr"] = df["declared_value"].apply(_parse_declared_val)
    df = df.drop(columns=["declared_value"])
    
    # Step 5: dispatch_time: convert to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])
    
    # Step 6: parcel_weight_kg: values > 100 are scanner-fault sentinels -> treat as missing
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    
    # Step 7: fill missing distance_km, parcel_weight_kg, declared_value_inr with column median
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median()),
        }
    else:
        medians = {
            "distance_km": float(medians["distance_km"]),
            "parcel_weight_kg": float(medians["parcel_weight_kg"]),
            "declared_value_inr": float(medians["declared_value_inr"]),
        }
    
    df["distance_km"] = df["distance_km"].fillna(medians["distance_km"])
    df["parcel_weight_kg"] = df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    df["declared_value_inr"] = df["declared_value_inr"].fillna(medians["declared_value_inr"])
    
    return df, medians


def build_features(clean, hubs):
    """
    Attach hub information and build engineered features.
    Does not use delivery_hours.
    """
    df = clean.copy()
    
    # Left merge on hub_id
    hub_cols = [c for c in hubs.columns if c == "hub_id" or c not in df.columns]
    df = df.merge(hubs[hub_cols], on="hub_id", how="left")
    
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"].astype(float))
    
    return df
