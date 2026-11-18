# Indoor Air Quality Predictor — ESP32 + TinyML

A working scaffold for: *"Deployment of a Machine Learning Based Indoor Air
Quality Predictor on NodeMCU ESP32 Using EloquentTinyML."*

## Real-data model (UCI Air Quality) - start here

The project now trains on the real **UCI Air Quality** dataset (`data/AirQualityUCI.csv`, 9,357 hourly
rows, Mar 2004 - Apr 2005, sensors + a certified reference analyzer).

```
python prepare_uci_dataset.py            # clean + label -> dataset_uci.csv (also imported by the scripts below)
python train_model_uci.py                # tune + train + evaluate (scikit-learn, ~2 min, no TensorFlow)
python train_model_uci_tensorflow.py     # Keras -> int8 .tflite -> model.h  (needs: pip install tensorflow)
python build_dashboard.py                # bake the trained model into dashboard/dashboard.html
```
Outputs land in `models/uci/` (`report.md` has every table for your report chapters; also
`metrics.json`, `confusion_matrix.png`, `training_curve.png`, `model_uci.json`, `model_weights_uci.h`,
and `eloquenttinyml/` with `iaq_uci.tflite`, `model.h`, `normalization.h`).

**How the data is used**
- Inputs: temperature, humidity and the five PT08 metal-oxide sensor responses (7 features).
- Labels: from the reference analyzer's CO and NO2, using India's CPCB AQI breakpoints, worst pollutant
  wins: Good/Satisfactory -> **Good**, Moderate -> **Moderate**, Poor or worse -> **Poor**. CPCB defines these
  for 8-h/24-h averages; here they are applied hour by hour (an approximation - say so in your report).
- Benzene is left out of the label because C6H6(GT) correlates 0.98 with the S2 sensor input; NMHC(GT) is 90% missing.
- Rows with any missing value are dropped (9,357 -> 6,941). Class share: Good 24.8%, Moderate 65.9%, Poor 9.2%.
- Split **by calendar week** (70/15/15), not by random row. Hourly readings are near-duplicates of their
  neighbours, so a random split leaks test data into training and inflates accuracy.

**Results on the held-out test weeks (1,330 hours)** - network (32,16,8), 947 parameters:

| Metric | scikit-learn MLP | Keras -> int8 TFLite (5,976 bytes) |
|---|---|---|
| Accuracy | 0.881 | 0.880 |
| Macro precision / recall / F1 | 0.808 / 0.770 / 0.785 | 0.814 / 0.739 / 0.760 |

Confusion matrix (rows actual, cols predicted; Good / Moderate / Poor): `[[249,38,0],[56,876,21],[0,43,47]]`.
Per class F1: Good 0.84, Moderate 0.92, Poor 0.60.

**Read these caveats before your viva**
- A plain logistic regression scores 0.876 accuracy / 0.783 macro-F1 on the same split, so the neural network
  is not meaningfully better here. The limit comes from the data (sensor noise, labels cut from continuous
  concentrations), not from model size. Report both; do not claim the MLP "wins".
- The **Poor** class is the weak spot: only ~52% of truly Poor hours are caught (90 test examples, so this estimate is noisy).
- Trained on the past and tested on the future (Feb-Apr 2005), accuracy drops to 0.766 (macro-F1 0.68): these
  sensors drift over time. The random-week number above is the optimistic one.
- The Keras model and the scikit-learn model are trained separately, so their per-class numbers differ slightly.
- **This model does not run on the DHT22 + MQ-135 board.** Its inputs are the UCI dataset's own sensor array, whose
  readings mean something different from an MQ-135's raw ADC counts. The firmware in `esp32_firmware/` still uses
  the 3-input model (`dataset.csv`, synthetic). To deploy on your ESP32, collect and label your own readings with
  `data_logger.ino` (steps below) - the UCI model then serves as the real-data benchmark for the report.
- The dashboard's **Replay** mode streams the held-out test hours through the trained model and compares each
  prediction with the reference analyzer. **ESP32** mode still shows whatever model is flashed on the board.

## Dashboard (`dashboard/dashboard.html`)
Open the file in a browser. **Day / Dark** buttons (top right) switch the theme; the choice is remembered, and
first-time visitors get their system's light/dark setting. Regenerate the embedded model with `build_dashboard.py`
after any re-training.

## What's in here

```
prepare_uci_dataset.py                 UCI data loader/labeller/splitter (shared)
train_model_uci.py                     tuned MLP on the UCI data (scikit-learn)
train_model_uci_tensorflow.py          same data -> Keras -> int8 TFLite -> model.h
build_dashboard.py                     bakes model + test hours into the dashboard
data/AirQualityUCI.csv                 real dataset
models/uci/                            trained model, metrics, plots, TFLite files
generate_dataset.py                    synthetic dataset (bootstrap only)
train_model_quickstart.py              trains + exports a model TODAY (sklearn, no TF needed)
train_model_tensorflow.py              the real pipeline: Keras -> TFLite -> EloquentTinyML
dataset.csv                            generated sample dataset (6000 rows)

esp32_firmware/
  data_logger/data_logger.ino          flash first: logs real sensor data over Serial
  quickstart_version/
    quickstart_version.ino             flashable NOW, hand-coded NN forward pass in C
    model_weights.h                    generated weights (from the synthetic dataset)
  eloquenttinyml_version/
    eloquenttinyml_version.ino         the "real" EloquentTinyML/TFLite Micro sketch
    (model.h + normalization.h are generated by train_model_tensorflow.py —
     not included yet, see below)
```

## Why two firmware versions?

This sandbox has no internet access and no TensorFlow installed, so I
could not run the actual Keras → TensorFlow Lite → EloquentTinyML
conversion here. Rather than hand you an untested pipeline, I built and
**actually ran** a functionally equivalent path using scikit-learn (which
was available), verified it trains, evaluates, and exports correctly, and
wired that into a sketch you can flash immediately. That's
`quickstart_version` — 88.75% test accuracy on the synthetic dataset, real
numbers from an actual run, confusion matrix included below.

`eloquenttinyml_version` is the pipeline that matches your project title
word-for-word (Keras → `tf.lite.TFLiteConverter` with int8 quantization →
C array → `EloquentTinyML`'s `TfLite<>` wrapper). Run
`train_model_tensorflow.py` on your own machine or in Google Colab (both
have TensorFlow preinstalled/installable) to generate `model.h` and
`normalization.h`, drop them next to the `.ino`, and you have the exact
architecture your title describes. The sensor code, wiring, web dashboard
and Serial output are identical between both versions — swapping is just
swapping the inference backend.

**Recommendation:** get `quickstart_version` running on your bench this
week so you have something to demo and iterate on, then run
`train_model_tensorflow.py` on your laptop once and swap in
`eloquenttinyml_version` before your final submission, since your title
specifically names EloquentTinyML/TFLite. Both use the same 3 input
features (temperature, humidity, gas) and 3 output classes
(Good/Moderate/Poor), so the swap is mechanical.

## What you should actually do, in order

1. **Wire up DHT22 + MQ-135** (through your level shifter) per the pinout
   comment at the top of
   `esp32_firmware/quickstart_version/quickstart_version.ino`. No CO2
   sensor needed — this build uses only temperature, humidity and gas.
2. **Flash `data_logger.ino`** and collect real readings for a few hours
   across genuinely different conditions (see comments in that file for
   how to get variety, and why auto-labeling is deliberately not done for
   you).
3. **Label your real data** against a documented standard — replace
   `REPLACE_WITH_YOUR_STANDARD` in `generate_dataset.py`'s docstring with
   whatever you cite (ASHRAE 62.1, WHO IAQ guidelines, or your own
   protocol), and use the *same* thresholds to label your logged CSV.
   This is the single most commonly-questioned part of a project like this
   at viva — have a one-paragraph answer ready.
4. Replace `dataset.csv` with your real, labeled data (same 4 columns:
   `temperature,humidity,gas,label` with label 0/1/2).
5. Re-run `train_model_quickstart.py` (works right now) to get a model
   trained on your own data, flash `quickstart_version.ino`, confirm the
   whole loop works end to end.
6. Once that's solid, install TensorFlow locally/Colab, run
   `train_model_tensorflow.py`, and flash `eloquenttinyml_version.ino`
   for your final deployed system.
7. For the report, Chapters 6–8 map directly onto the printed output of
   whichever training script you last ran (accuracy, precision/recall/F1,
   confusion matrix) plus the `.tflite` file size before/after
   quantization (printed by `train_model_tensorflow.py`).

## Results from the run I actually performed (synthetic data, quickstart model, no CO2 — 3 features)

```
Test set accuracy: 0.8958
Confusion matrix (rows=true, cols=pred; order Good/Moderate/Poor):
[[300  47   0]
 [ 36 372  23]
 [  0  19 403]]

              precision    recall  f1-score
    Good          0.89      0.86      0.88
    Moderate      0.85      0.86      0.86
    Poor          0.95      0.95      0.95
```
These numbers are from the synthetic dataset and will change — likely
move around, possibly down — once you retrain on data your own sensors
actually produced in your own room. Report the numbers from your own run,
not these.

## Arduino library checklist

- `DHT sensor library` (Adafruit) + its `Adafruit Unified Sensor` dependency
- For `eloquenttinyml_version` only: `EloquentTinyML` and a TFLite Micro
  backend for ESP32 (`TensorFlowLite_ESP32` or your course's preferred
  fork — check version compatibility with EloquentTinyML's `TfLite<>` API,
  it has changed across releases)
- Board package: ESP32 by Espressif Systems (Boards Manager)

## Honesty check on what's "working" here

- Dataset generation, both training scripts' math, and the C-array/header
  export logic were written *and executed* in this session — the numbers
  above are real, not invented.
- The `.ino` sketches are written to compile against standard, documented
  library APIs (DHT, WebServer, EloquentTinyML) but were not compiled or
  flashed against real hardware here — I don't have an ESP32 or Arduino
  toolchain in this sandbox. Check the first build carefully against your
  exact board/library versions, especially the EloquentTinyML API, which
  has changed shape across versions.
