# Delhivery Last-Mile Delay Intelligence Pipeline

## Part A — Data foundation and delay evidence

From the repository root, regenerate the supplied data and produce the modelling and evidence tables:

```bash
python generate_data.py
python part_a_clean_eda.py
```

The preparation functions in `src/shipment_prep.py` clean the raw export, calculate historical median fill values, and build dispatch-time features. The script writes `data/modelling_table.csv` and the three evidence tables under `outputs/`.

### Data findings

- Express shipments have the highest delay rate: **0.4897** (119 delayed out of 243), compared with 0.2056 for Standard and 0.0857 for Economy ([part_a_delay_by_group.csv](outputs/part_a_delay_by_group.csv)).
- Delay rates rise across hub tiers, from **0.0400** at Tier 1 to **0.2150** at Tier 2 and **0.4400** at Tier 3.
- Delay rates also rise with distance: **0.0219** for 0–200 km, **0.1368** for 201–400 km, **0.4364** for 401–600 km, and **0.8523** for 601+ km.
- The three strongest correlations with `delivery_hours` by absolute value are `promised_hours` (**+0.8372**), `distance_km` (**+0.4013**), and `n_stops` (**+0.3247**) ([part_a_correlations.csv](outputs/part_a_correlations.csv)).
- On the median-filled declared values, the INR mean (**INR 1,533.05**) exceeds the median (**INR 1,121.50**); after the natural-log transform, the mean is **7.028** and the median is **7.022**, indicating that the transform reduces the effect of the long right tail.
