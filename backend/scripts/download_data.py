"""Fetch the engine-health dataset into data/raw/ (never committed to Git).

    python scripts/download_data.py              # Kaggle (needs ~/.kaggle/kaggle.json or KAGGLE_* env vars)
    python scripts/download_data.py --csv FILE   # use a CSV you downloaded by hand
    python scripts/download_data.py --synthetic  # clearly-labelled synthetic stand-in for offline dev / CI

The Kaggle dataset is "Automotive Vehicles Engine Health Dataset" by Parv Modi:
https://www.kaggle.com/datasets/parvmodi/automotive-vehicles-engine-health-dataset
Check its licence on that page and record it in data/README.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TARGET = RAW_DIR / "engine_data.csv"
KAGGLE_SLUG = "parvmodi/automotive-vehicles-engine-health-dataset"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_SLUG}"


def _write_meta(source: str, **extra: str) -> None:
    meta = {"source": source, "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"), **extra}
    TARGET.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def from_kaggle() -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("The kaggle package is not installed: pip install kaggle")
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001 - kaggle raises bare exceptions
        sys.exit(
            f"Kaggle authentication failed ({exc}). Put kaggle.json in ~/.kaggle/ or set "
            "KAGGLE_USERNAME / KAGGLE_KEY. Or pass --csv / --synthetic."
        )
    with tempfile.TemporaryDirectory() as tmp:
        api.dataset_download_files(KAGGLE_SLUG, path=tmp, unzip=False)
        for z in Path(tmp).glob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(tmp)
        csvs = sorted(Path(tmp).rglob("*.csv"))
        if not csvs:
            sys.exit("No CSV found in the Kaggle download")
        shutil.copy(csvs[0], TARGET)
    _write_meta("kaggle", name="Automotive Vehicles Engine Health Dataset", url=KAGGLE_URL)


def from_csv(path: Path) -> None:
    shutil.copy(path, TARGET)
    _write_meta("manual", name=path.name, url=KAGGLE_URL)


def synthetic(rows: int = 19_535, seed: int = 7) -> None:
    """Synthetic readings with ranges loosely matching the Kaggle data.

    Faulty engines skew toward low oil pressure, abnormal coolant temperature and
    fuel-pressure drift, with heavy overlap so the task is not trivial. These numbers
    are NOT real measurements; never report metrics trained on them as real results.
    """
    rng = np.random.default_rng(seed)
    fault = rng.random(rows) < 0.63
    n = rows
    rpm = np.where(fault, rng.gamma(4.5, 160, n), rng.gamma(6.0, 145, n)).clip(60, 2300)
    oil_p = np.where(fault, rng.normal(3.0, 1.1, n), rng.normal(3.7, 0.95, n)).clip(0.01, 7.3)
    fuel_p = np.where(fault, rng.gamma(3.2, 2.2, n), rng.gamma(4.2, 1.45, n)).clip(0.01, 21)
    cool_p = np.where(fault, rng.normal(2.25, 1.05, n), rng.normal(2.45, 0.95, n)).clip(0.01, 7.5)
    oil_t = np.where(fault, rng.normal(77.9, 3.3, n), rng.normal(77.0, 2.6, n)).clip(71, 90)
    cool_t = np.where(fault, rng.normal(78.8, 7.0, n), rng.normal(77.6, 5.0, n)).clip(61, 196)
    df = pd.DataFrame(
        {
            "Engine rpm": rpm.round(0).astype(int),
            "Lub oil pressure": oil_p.round(6),
            "Fuel pressure": fuel_p.round(6),
            "Coolant pressure": cool_p.round(6),
            "lub oil temp": oil_t.round(6),
            "Coolant temp": cool_t.round(6),
            "Engine Condition": fault.astype(int),
        }
    )
    df.to_csv(TARGET, index=False)
    _write_meta("synthetic", name="Synthetic stand-in (scripts/download_data.py --synthetic)", url="")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--csv", type=Path, help="Path to a manually downloaded engine_data.csv")
    group.add_argument("--synthetic", action="store_true", help="Generate a labelled synthetic dataset")
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.csv:
        from_csv(args.csv)
    elif args.synthetic:
        synthetic()
    else:
        from_kaggle()
    rows = sum(1 for _ in TARGET.open(encoding="utf-8")) - 1
    print(f"Wrote {TARGET} ({rows} rows)")


if __name__ == "__main__":
    main()
