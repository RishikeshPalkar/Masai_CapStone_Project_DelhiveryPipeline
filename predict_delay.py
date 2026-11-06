"""Scoring pipeline: predict delay probabilities and map operational actions for new parcels."""
import os
import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features


def main():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load data
    new_shipments_raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # 2. Load models and artifacts
    model = joblib.load("models/delay_model.pkl")
    threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    
    impute_df = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute_df["column"], impute_df["median"]))

    hub_segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

    # 3. Clean and prepare features
    clean_new, _ = clean_shipments(new_shipments_raw, medians=medians)
    featured_new = build_features(clean_new, hubs)

    # 4. Score with model
    X_new = featured_new[feature_order]
    probs = model.predict_proba(X_new)[:, 1]
    featured_new["delay_probability"] = [round(float(p), 4) for p in probs]

    # 5. Determine flagged
    featured_new["flagged"] = (featured_new["delay_probability"] >= threshold).astype(int)

    # 6. Join hub segment (preserving new_shipments original row order)
    # Note: featured_new already has hub_id
    # Drop any existing segment column in featured_new if present, then merge
    if "segment" in featured_new.columns:
        featured_new = featured_new.drop(columns=["segment"])
    scored = featured_new.merge(hub_segments, on="hub_id", how="left")

    # 7. Map actions:
    # - 'Escalate to hub manager' when flagged == 1 and segment == 'Escalate'
    # - 'Proactive customer call' when flagged == 1 otherwise
    # - 'No action' when flagged == 0
    def determine_action(row):
        if row["flagged"] == 1:
            if row["segment"] == "Escalate":
                return "Escalate to hub manager"
            return "Proactive customer call"
        return "No action"

    scored["action"] = scored.apply(determine_action, axis=1)

    # Output columns: shipment_id, hub_id, segment, delay_probability, flagged, action
    output_cols = ["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]
    actions_df = scored[output_cols]
    actions_df.to_csv("outputs/new_shipment_actions.csv", index=False)

    flagged_count = int(actions_df["flagged"].sum())
    escalate_count = int((actions_df["action"] == "Escalate to hub manager").sum())
    call_count = int((actions_df["action"] == "Proactive customer call").sum())
    no_action_count = int((actions_df["action"] == "No action").sum())

    print(f"Scoring complete. Total: {len(actions_df)}, Flagged: {flagged_count}")
    print(f"Escalate: {escalate_count}, Proactive calls: {call_count}, No action: {no_action_count}")


if __name__ == "__main__":
    main()
