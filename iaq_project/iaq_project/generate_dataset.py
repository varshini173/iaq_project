"""
generate_dataset.py
--------------------
Generates a SYNTHETIC indoor-air-quality dataset for bootstrapping the project.

IMPORTANT (read this before your viva):
This script exists so you have something to train/deploy with on day one,
and so you can see the *shape* of a working pipeline end to end. For a real
submission you should replace this with data actually collected from your
ESP32 + sensors (see esp32_firmware/data_logger/data_logger.ino), and your
labels should cite a real reference standard rather than the thresholds
hard-coded below.

Labeling basis used here (documented so you can defend it, or swap it out):
  Gas/VOC (MQ-135 analog 0-4095 on ESP32's 12-bit ADC, unitless proxy) is
  the main pollutant signal (no CO2 sensor in this build):
      < 800        -> "Good"
      800-1800     -> "Moderate"
      > 1800       -> "Poor"
  Temperature/Humidity are used as secondary modifiers (comfort band
  20-26 C, 30-60% RH) rather than primary pollutant indicators, which
  matches how most indoor-air-quality references treat them.

The final label is a weighted combination of these sub-scores, not a single
hard if/else rule, so the model actually has something non-trivial to learn.
Replace REPLACE_WITH_YOUR_STANDARD below with the citation you use in
Chapter 6 of your report.
"""

import numpy as np
import pandas as pd

REPLACE_WITH_YOUR_STANDARD = (
    "e.g. ASHRAE 62.1 ventilation guidance, WHO indoor air quality "
    "guidelines, or your institution's documented lab protocol"
)

RNG_SEED = 42
N_SAMPLES = 6000

LABELS = {0: "Good", 1: "Moderate", 2: "Poor"}


def score_band(value, good_max, moderate_max):
    """Return 0 (good), 1 (moderate), 2 (poor) for a rising-is-worse metric."""
    if value <= good_max:
        return 0
    elif value <= moderate_max:
        return 1
    else:
        return 2


def generate(n_samples: int = N_SAMPLES, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(n_samples):
        # Pick an underlying "true" pollution level first, then generate
        # correlated sensor readings around it. This mimics how a real room
        # behaves: CO2, gas and comfort drift together, not independently.
        base = rng.uniform(0, 1)  # 0 = clean room, 1 = very polluted room

        gas = rng.normal(150 + base * 2600, 150)
        gas = float(np.clip(gas, 50, 4095))

        temperature = rng.normal(23 + base * 6, 2.0)
        temperature = float(np.clip(temperature, 15, 38))

        humidity = rng.normal(40 + base * 30, 6.0)
        humidity = float(np.clip(humidity, 15, 95))

        # Sub-scores
        s_gas = score_band(gas, 800, 1800)
        s_temp = 0 if 20 <= temperature <= 26 else (1 if 17 <= temperature <= 30 else 2)
        s_hum = 0 if 30 <= humidity <= 60 else (1 if 20 <= humidity <= 70 else 2)

        # Weighted combination: gas (pollutant) matters most, comfort is secondary.
        combined = 0.8 * s_gas + 0.1 * s_temp + 0.1 * s_hum
        # small label noise to avoid a perfectly separable toy problem
        combined += rng.normal(0, 0.15)

        if combined < 0.66:
            label = 0
        elif combined < 1.5:
            label = 1
        else:
            label = 2

        rows.append([round(temperature, 2), round(humidity, 2),
                      round(gas, 1), label])

    df = pd.DataFrame(rows, columns=["temperature", "humidity", "gas", "label"])
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("dataset.csv", index=False)
    print(df["label"].map(LABELS).value_counts())
    print(f"\nWrote {len(df)} rows to dataset.csv")
    print(f"\nRemember to replace the labeling standard with: {REPLACE_WITH_YOUR_STANDARD}")
