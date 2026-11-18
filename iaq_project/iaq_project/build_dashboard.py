"""
build_dashboard.py
-------------------
Bakes the trained model and the held-out test hours into dashboard/dashboard.html
so the page stays a single file you can open by double-clicking (no server, no build tools).

Run after every re-training:
    python train_model_uci.py
    python build_dashboard.py

It rewrites only the block between the /*__DATA_START__*/ and /*__DATA_END__*/ markers.
"""
import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(HERE, "models", "uci", "model_uci.json")
REPLAY_CSV = os.path.join(HERE, "models", "uci", "test_replay.csv")
DASHBOARD = os.path.join(HERE, "dashboard", "dashboard.html")


def r(x, nd):
    return [round(v, nd) if isinstance(v, float) else v for v in x]


def main():
    with open(MODEL_JSON) as f:
        m = json.load(f)
    model = {
        "mean": r(m["mean"], 6), "std": r(m["std"], 6),
        "layers": [{"W": [r(row, 6) for row in L["W"]], "b": r(L["b"], 6)} for L in m["layers"]],
    }
    rp = pd.read_csv(REPLAY_CSV, parse_dates=["timestamp"])
    rows = []
    for _, x in rp.iterrows():
        rows.append([x["timestamp"].strftime("%Y-%m-%d %H:%M"), round(x["T"], 1), round(x["RH"], 1),
                     int(x["PT08.S1(CO)"]), int(x["PT08.S2(NMHC)"]), int(x["PT08.S3(NOx)"]),
                     int(x["PT08.S4(NO2)"]), int(x["PT08.S5(O3)"]),
                     round(x["CO(GT)"], 1), int(round(x["NO2(GT)"])), int(x["label"])])
    meta = {"testAccuracy": round(m["test_metrics"]["accuracy"], 4), "testF1": round(m["test_metrics"]["f1_macro"], 4),
            "testRows": len(rows)}

    block = ("/*__DATA_START__*/\n"
             f"const MODEL = {json.dumps(model, separators=(',', ':'))};\n"
             f"const REPLAY = {json.dumps(rows, separators=(',', ':'))};\n"
             f"const META = {json.dumps(meta)};\n"
             "/*__DATA_END__*/")
    with open(DASHBOARD, encoding="utf-8") as f:
        html = f.read()
    new, n = re.subn(r"/\*__DATA_START__\*/.*?/\*__DATA_END__\*/", lambda _: block, html, flags=re.S)
    if n != 1:
        raise SystemExit("markers not found in dashboard.html")
    with open(DASHBOARD, "w", encoding="utf-8") as f:
        f.write(new)
    print(f"Updated {DASHBOARD}: {len(rows)} replay hours, test accuracy {meta['testAccuracy']:.4f}")


if __name__ == "__main__":
    main()
