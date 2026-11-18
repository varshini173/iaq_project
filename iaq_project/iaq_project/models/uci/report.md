# UCI Air Quality model - results

Rows after cleaning: **6941** (train 4600 / val 1011 / test 1330, split by calendar week).
Class share: Good 24.8%, Moderate 65.9%, Poor 9.2%.

Chosen network: hidden layers (32, 16, 8), L2 alpha 0.01, class balancing off; **947 parameters**; best epoch 17.

## Held-out test weeks

| Metric | Value |
|---|---|
| Accuracy | 0.8812 |
| Precision (macro) | 0.8076 |
| Recall (macro) | 0.7697 |
| F1 (macro) | 0.7845 |

Confusion matrix (rows = actual, columns = predicted; Good / Moderate / Poor):

```
[249, 38, 0]
[56, 876, 21]
[0, 43, 47]
```

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Good | 0.816 | 0.868 | 0.841 | 287 |
| Moderate | 0.915 | 0.919 | 0.917 | 953 |
| Poor | 0.691 | 0.522 | 0.595 | 90 |

## Baselines on the same test weeks

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority class | 0.717 | 0.278 |
| Logistic regression | 0.876 | 0.783 |
| Gradient boosting (reference, too big for MCU) | 0.870 | 0.753 |
| **This MLP** | **0.881** | **0.784** |

## Stress test: train on the past, test on the future

train on the earliest 70% of hours, test on the latest 15% (Feb-Apr 2005). Sensors drift and seasons change, so this is harder than the week-block split.

Test period 2005-02-18 02:00:00 to 2005-04-04 14:00:00: accuracy **0.766**, macro-F1 **0.679**.

## Hyper-parameter search (5-fold, week-grouped CV on train+val)

| Hidden | alpha | Balanced | CV macro-F1 | CV accuracy |
|---|---|---|---|---|
| (32, 16, 8) | 0.01 | False | 0.812 | 0.869 |
| (32, 16, 8) | 0.0001 | False | 0.811 | 0.869 |
| (32, 16) | 0.0001 | False | 0.810 | 0.871 |
| (32, 16) | 0.01 | False | 0.803 | 0.868 |
| (32, 16, 8) | 0.01 | True | 0.800 | 0.841 |
| (32, 16, 8) | 0.0001 | True | 0.799 | 0.841 |
| (32, 16) | 0.01 | True | 0.796 | 0.837 |
| (32, 16) | 0.0001 | True | 0.794 | 0.835 |
| (16, 8) | 0.0001 | False | 0.793 | 0.860 |
| (16, 8) | 0.01 | False | 0.789 | 0.860 |
| (16, 8) | 0.01 | True | 0.785 | 0.825 |
| (16, 8) | 0.0001 | True | 0.784 | 0.823 |
| (8,) | 0.01 | False | 0.781 | 0.852 |
| (8,) | 0.0001 | False | 0.780 | 0.850 |
| (8,) | 0.01 | True | 0.774 | 0.813 |
| (8,) | 0.0001 | True | 0.772 | 0.811 |
