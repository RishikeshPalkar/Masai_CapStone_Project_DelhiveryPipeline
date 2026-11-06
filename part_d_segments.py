"""Part D: Hub Segments and Clustering."""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # Task 1: Hub profile
    # Group by hub_id and compute shipments, mean_delivery_hours, delay_rate, mean_distance_km, mean_n_stops
    # Sorted by ascending hub_id
    hub_summary = df.groupby("hub_id").agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    ).reset_index().sort_values("hub_id").reset_index(drop=True)

    hub_profile_df = hub_summary.copy()
    hub_profile_df["mean_delivery_hours"] = hub_profile_df["mean_delivery_hours"].round(4)
    hub_profile_df["delay_rate"] = hub_profile_df["delay_rate"].round(4)
    hub_profile_df["mean_distance_km"] = hub_profile_df["mean_distance_km"].round(4)
    hub_profile_df["mean_n_stops"] = hub_profile_df["mean_n_stops"].round(4)
    hub_profile_df.to_csv("outputs/hub_profile.csv", index=False)

    # Task 2 & 3: Scale profile columns and KMeans clustering
    profile_features = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(hub_profile_df[profile_features])

    # Separation evidence for K in 2, 3, 4, 5
    n_hubs = len(hub_profile_df)
    # Precompute pairwise distances
    dist_matrix = np.zeros((n_hubs, n_hubs))
    for i in range(n_hubs):
        for j in range(i + 1, n_hubs):
            d = np.linalg.norm(scaled_matrix[i] - scaled_matrix[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    sep_rows = []
    kmeans_models = {}

    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_matrix)
        kmeans_models[k] = labels

        # Cluster sizes in descending order
        sizes = sorted(pd.Series(labels).value_counts().tolist(), reverse=True)
        cluster_sizes_str = "-".join(map(str, sizes))

        # Compute across and within distances
        across_dists = []
        within_dists = []

        for i in range(n_hubs):
            for j in range(i + 1, n_hubs):
                if labels[i] == labels[j]:
                    within_dists.append(dist_matrix[i, j])
                else:
                    across_dists.append(dist_matrix[i, j])

        mean_across = np.mean(across_dists)
        mean_within = np.mean(within_dists)
        ratio = mean_across / mean_within

        sep_rows.append({
            "K": k,
            "across_over_within": round(float(ratio), 4),
            "cluster_sizes": cluster_sizes_str,
        })

    sep_df = pd.DataFrame(sep_rows)
    sep_df.to_csv("outputs/part_d_separation.csv", index=False)

    # Task 4: Segment names and tier check for K=3
    k3_labels = kmeans_models[3]
    hub_profile_df["cluster"] = k3_labels

    # Order the three clusters by the mean of their hubs' delay_rate
    cluster_delay_rates = hub_profile_df.groupby("cluster")["delay_rate"].mean().sort_values()
    # Lowest -> Standard, Middle -> Watch, Highest -> Escalate
    cluster_order = cluster_delay_rates.index.tolist()
    segment_map = {
        cluster_order[0]: "Standard",
        cluster_order[1]: "Watch",
        cluster_order[2]: "Escalate",
    }
    hub_profile_df["segment"] = hub_profile_df["cluster"].map(segment_map)

    # outputs/hub_segments.csv: hub_id, segment, hub_city, hub_tier
    hub_segments_df = hub_profile_df.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
    hub_segments_df = hub_segments_df[["hub_id", "segment", "hub_city", "hub_tier"]]
    hub_segments_df.to_csv("outputs/hub_segments.csv", index=False)

    # outputs/segment_profile.csv: segment, hubs, mean_delivery_hours, delay_rate, mean_distance_km, mean_n_stops
    # (means of the hub-level values)
    segment_profile_rows = []
    for seg in ["Standard", "Watch", "Escalate"]:
        sub = hub_profile_df[hub_profile_df["segment"] == seg]
        segment_profile_rows.append({
            "segment": seg,
            "hubs": len(sub),
            "mean_delivery_hours": round(float(sub["mean_delivery_hours"].mean()), 4),
            "delay_rate": round(float(sub["delay_rate"].mean()), 4),
            "mean_distance_km": round(float(sub["mean_distance_km"].mean()), 4),
            "mean_n_stops": round(float(sub["mean_n_stops"].mean()), 4),
        })
    segment_profile_df = pd.DataFrame(segment_profile_rows)
    segment_profile_df.to_csv("outputs/segment_profile.csv", index=False)

    # outputs/tier_vs_segment_mismatch.csv
    # Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate
    expected_segment = {1: "Standard", 2: "Watch", 3: "Escalate"}
    mismatches = hub_segments_df[hub_segments_df["segment"] != hub_segments_df["hub_tier"].map(expected_segment)].copy()
    mismatches.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

    print("Part D segmentation completed successfully.")
    print("Mismatched hubs:\n", mismatches)


if __name__ == "__main__":
    main()
