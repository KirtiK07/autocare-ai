# Data

Raw data is never committed. `data/raw/` is git-ignored; fetch it with:

```bash
python scripts/download_data.py              # Kaggle API (needs credentials)
python scripts/download_data.py --csv FILE   # a CSV you downloaded manually
python scripts/download_data.py --synthetic  # labelled synthetic stand-in (dev/CI only)
```

The script writes `data/raw/engine_data.csv` and an `engine_data.meta.json` that records where it came from. Training copies that provenance into `artifacts/model_meta.json`, and the UI shows a warning whenever the model was trained on synthetic data.

## Dataset

| Field | Value |
| --- | --- |
| Name | Automotive Vehicles Engine Health Dataset |
| Link | https://www.kaggle.com/datasets/parvmodi/automotive-vehicles-engine-health-dataset |
| Licence | _check on the Kaggle page and record here_ |
| Rows | _fill in after download (the script prints it)_ |
| Columns | Engine rpm, Lub oil pressure, Fuel pressure, Coolant pressure, lub oil temp, Coolant temp, Engine Condition |
| Label | `Engine Condition` (1 = needs attention; set `positive_label` in `configs/train.yaml` if your copy differs) |

The dataset doesn't state its units. The app labels pressures as bar and temperatures as °C, which matches the value ranges, but treat units as approximate.
