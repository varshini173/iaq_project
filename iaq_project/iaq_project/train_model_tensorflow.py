"""
train_model_tensorflow.py
---------------------------
THIS is the pipeline your project title actually refers to:
  Keras model -> TensorFlow Lite -> quantized .tflite -> C array -> ESP32
  via EloquentTinyML.

Run this on your own computer or in Google Colab, NOT inside a sandbox
that has no TensorFlow/internet (that's why this repo ships a separate
scikit-learn "quickstart" path you can flash immediately — see
train_model_quickstart.py). This script produces the artifacts your
report's Chapter 6/7 should discuss: accuracy, confusion matrix, model
size before/after quantization, and the final model.h for EloquentTinyML.

Install (once):
    pip install tensorflow scikit-learn pandas numpy

Usage:
    python3 generate_dataset.py     # or supply your own collected dataset.csv
    python3 train_model_tensorflow.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report,
)
import os

FEATURES = ["temperature", "humidity", "gas"]
LABEL_NAMES = ["Good", "Moderate", "Poor"]
OUT_DIR = "esp32_firmware/eloquenttinyml_version"


def build_model(n_features: int, n_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return model


def representative_dataset_gen(X_train):
    def gen():
        for i in range(min(300, len(X_train))):
            yield [X_train[i:i + 1].astype(np.float32)]
    return gen


def main():
    df = pd.read_csv("dataset.csv")
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    def normalize(X):
        return ((X - mean) / std).astype(np.float32)

    Xn_train, Xn_val, Xn_test = normalize(X_train), normalize(X_val), normalize(X_test)

    model = build_model(len(FEATURES), len(LABEL_NAMES))
    model.summary()

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
    model.fit(Xn_train, y_train, validation_data=(Xn_val, y_val),
              epochs=200, batch_size=32, callbacks=callbacks, verbose=2)

    # ---- Evaluation ----
    for name, Xs, ys in [("Validation", Xn_val, y_val), ("Test", Xn_test, y_test)]:
        preds = np.argmax(model.predict(Xs, verbose=0), axis=1)
        acc = accuracy_score(ys, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(ys, preds, average="macro", zero_division=0)
        print(f"\n=== {name} set ===")
        print(f"Accuracy : {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
        print(confusion_matrix(ys, preds))
        print(classification_report(ys, preds, target_names=LABEL_NAMES, zero_division=0))

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Save Keras model (for your own records / report) ----
    model.save(os.path.join(OUT_DIR, "iaq_model.keras"))

    # ---- Convert to TensorFlow Lite, int8-quantized ----
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(Xn_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32   # keep float I/O; EloquentTinyML handles the rest
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    tflite_path = os.path.join(OUT_DIR, "iaq_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"\nQuantized model size: {len(tflite_model)} bytes -> {tflite_path}")

    # ---- Export as a C header for EloquentTinyML ----
    header_path = os.path.join(OUT_DIR, "model.h")
    write_c_array(tflite_model, header_path, var_name="iaq_model_tflite")

    # ---- Also dump the normalization constants EloquentTinyML sketch needs ----
    norm_path = os.path.join(OUT_DIR, "normalization.h")
    with open(norm_path, "w") as f:
        f.write("#pragma once\n")
        f.write(f"#define IAQ_N_FEATURES {len(FEATURES)}\n")
        f.write(f"static const float iaq_feature_mean[IAQ_N_FEATURES] = {{{', '.join(f'{m:.6f}f' for m in mean)}}};\n")
        f.write(f"static const float iaq_feature_std[IAQ_N_FEATURES]  = {{{', '.join(f'{s:.6f}f' for s in std)}}};\n")
        f.write('static const char* iaq_class_names[3] = {"Good", "Moderate", "Poor"};\n')

    print(f"Wrote {header_path} and {norm_path}")
    print("\nNext step: open esp32_firmware/eloquenttinyml_version/eloquenttinyml_version.ino "
          "in Arduino IDE, install the EloquentTinyML + TensorFlowLite_ESP32 libraries, and flash.")


def write_c_array(data: bytes, out_path: str, var_name: str):
    with open(out_path, "w") as f:
        f.write("// AUTO-GENERATED by train_model_tensorflow.py — do not hand-edit.\n")
        f.write("#pragma once\n\n")
        f.write(f"const unsigned int {var_name}_len = {len(data)};\n")
        f.write(f"alignas(8) const unsigned char {var_name}[] = {{\n")
        for i in range(0, len(data), 12):
            chunk = data[i:i + 12]
            f.write("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
        f.write("};\n")


if __name__ == "__main__":
    main()
