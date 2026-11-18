"""
prepare_uci_dataset.py
-----------------------
Loads the real UCI "Air Quality" dataset (data/AirQualityUCI.csv), cleans it,
derives the Good / Moderate / Poor label, and provides the train/val/test split
used by every training script in this project.

Run directly to write the cleaned file:
    python prepare_uci_dataset.py            ->  dataset_uci.csv

Dataset facts (De Vito et al., 2008; UCI ML Repository "Air Quality"):
  9,357 hourly rows, Mar 2004 - Apr 2005, one multisensor device on a road in
  an Italian city, next to a certified reference analyzer.
  * PT08.S1..S5   = 5 low-cost metal-oxide sensor responses  -> model INPUTS
  * T, RH         = temperature (C), relative humidity (%)   -> model INPUTS
  * CO(GT), NO2(GT), C6H6(GT), NOx(GT), NMHC(GT) = reference analyzer
    "ground truth" concentrations                            -> used ONLY to make labels
  * -200 means "missing" in the raw file.

Design decisions (all editable below):
  1. Inputs are the 5 sensor responses + T + RH. AH (absolute humidity) is
     dropped because it is computed from T and RH (redundant).
  2. Labels come from the reference analyzers for CO and NO2, using India's
     CPCB National AQI health breakpoints (2014), worst pollutant wins:
        CO  (mg/m3):  <=2.0 Good/Satisfactory | 2.1-10 Moderate | >10 Poor+
        NO2 (ug/m3):  <=80  Good/Satisfactory | 81-180 Moderate | >180 Poor+
     CPCB defines these for 8-h (CO) / 24-h (NO2) averages; the dataset is
     hourly, so applying them per-hour is an approximation. Say so in your report.
  3. C6H6 (benzene) is NOT used in the label: C6H6(GT) correlates 0.98 with
     the PT08.S2 sensor in this dataset, which would make part of the label a
     near-copy of an input. NMHC(GT) is 90% missing, so it is unusable.
  4. Rows with any missing input or missing CO/NO2 reference are dropped
     (no imputation), leaving ~6,900 rows.
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "data", "AirQualityUCI.csv")
OUT_PATH = os.path.join(HERE, "dataset_uci.csv")

FEATURES = ["T", "RH", "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
FEATURE_LABELS = ["Temperature", "Humidity", "CO sensor (S1)", "NMHC sensor (S2)",
                  "NOx sensor (S3)", "NO2 sensor (S4)", "O3 sensor (S5)"]
LABEL_NAMES = ["Good", "Moderate", "Poor"]

# CPCB breakpoints: (upper bound of class 0, upper bound of class 1); above = class 2
CO_BREAKPOINTS = (2.0, 10.0)     # mg/m3
NO2_BREAKPOINTS = (80.0, 180.0)  # ug/m3

SPLIT_SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15   # remaining 0.15 = test


def _band(values, breakpoints):
    lo, hi = breakpoints
    return np.where(values <= lo, 0, np.where(values <= hi, 1, 2))


def load_clean(path: str = RAW_PATH) -> pd.DataFrame:
    """Read the raw semicolon/decimal-comma file, fix missing values, add label."""
    df = pd.read_csv(path, sep=";", decimal=",")
    df = df.dropna(axis=1, how="all").dropna(how="all")
    df = df.replace(-200, np.nan)
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"].str.replace(".", ":", regex=False),
                                     format="%d/%m/%Y %H:%M:%S")
    needed = FEATURES + ["CO(GT)", "NO2(GT)"]
    df = df.dropna(subset=needed).copy()
    df["label"] = np.maximum(_band(df["CO(GT)"].to_numpy(), CO_BREAKPOINTS),
                             _band(df["NO2(GT)"].to_numpy(), NO2_BREAKPOINTS)).astype(int)
    # calendar week index, used to keep neighbouring hours together when splitting
    df["week"] = (df["timestamp"] - df["timestamp"].min()).dt.days // 7
    return df.sort_values("timestamp").reset_index(drop=True)


def week_block_split(df: pd.DataFrame, seed: int = SPLIT_SEED):
    """
    Split by whole calendar weeks (70/15/15), not by random rows.

    Hourly air-quality readings are strongly autocorrelated: the hour before
    and after a test row looks almost identical. A random row split would put
    those near-duplicates in train and test and inflate accuracy. Assigning
    whole weeks to one side removes that leak while still covering all seasons.
    Returns boolean masks (train, val, test).
    """
    weeks = df["week"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(weeks)
    n = len(weeks)
    n_tr, n_va = int(TRAIN_FRAC * n), int(VAL_FRAC * n)
    groups = {"train": set(weeks[:n_tr]), "val": set(weeks[n_tr:n_tr + n_va]), "test": set(weeks[n_tr + n_va:])}
    return tuple(df["week"].isin(groups[k]).to_numpy() for k in ("train", "val", "test"))


def main():
    df = load_clean()
    cols = ["timestamp"] + FEATURES + ["CO(GT)", "NO2(GT)", "label"]
    df[cols].to_csv(OUT_PATH, index=False)
    counts = df["label"].value_counts().sort_index()
    print(f"Cleaned rows: {len(df)}  (from 9357 raw hourly rows)")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:9s}: {counts.get(i, 0):5d}  ({counts.get(i, 0) / len(df):.1%})")
    tr, va, te = week_block_split(df)
    print(f"Split by week -> train {tr.sum()}  val {va.sum()}  test {te.sum()}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
