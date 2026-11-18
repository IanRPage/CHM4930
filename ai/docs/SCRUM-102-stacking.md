# SCRUM-102 · Does RF improve the deployed MuFu ensemble?

**Serving recommendation. Keep the five-seed MuFu ensemble as the backend default.** On the default scaffold holdout, adding RF lowered RMSE for both bioactivity targets and raised Tox21 and CT_TOX ROC-AUC. It lowered FDA_APPROVED ROC-AUC from 0.8599 to 0.8278, bringing ClinTox macro ROC-AUC down from 0.8530 to 0.8439. This supports further evaluation of bioactivity stacking. A switch across all endpoints would sacrifice performance on FDA_APPROVED.

## Held-out test results

Lower RMSE is better. Higher R² and ROC-AUC are better. The standalone ensemble has one point estimate. The other columns show the mean ± sample standard deviation of five evaluations.

| Test metric | Individual MuFu | RF | Individual MuFu + RF | MuFu ensemble | Ensemble + RF |
|---|---:|---:|---:|---:|---:|
| BACE1 RMSE | 0.8272 ± 0.0123 | 0.8130 ± 0.0014 | 0.7955 ± 0.0052 | 0.7915 | 0.7854 ± 0.0004 |
| BACE1 R² | 0.5778 ± 0.0125 | 0.5922 ± 0.0014 | 0.6096 ± 0.0051 | 0.6135 | 0.6195 ± 0.0003 |
| EGFR RMSE | 0.9190 ± 0.0205 | 0.8770 ± 0.0018 | 0.8699 ± 0.0083 | 0.8861 | 0.8646 ± 0.0007 |
| EGFR R² | 0.4530 ± 0.0244 | 0.5021 ± 0.0020 | 0.5100 ± 0.0093 | 0.4917 | 0.5160 ± 0.0007 |
| Tox21 macro AUC | 0.7328 ± 0.0094 | 0.7119 ± 0.0023 | 0.7495 ± 0.0035 | 0.7585 | 0.7630 ± 0.0011 |
| ClinTox macro AUC | 0.8372 ± 0.0350 | 0.6518 ± 0.0264 | 0.7915 ± 0.0433 | 0.8530 | 0.8439 ± 0.0032 |
| CT_TOX AUC | 0.8194 ± 0.0341 | 0.7264 ± 0.0289 | 0.8207 ± 0.0359 | 0.8461 | 0.8601 ± 0.0070 |
| FDA_APPROVED AUC | 0.8551 ± 0.0384 | 0.5772 ± 0.0309 | 0.7623 ± 0.0598 | 0.8599 | 0.8278 ± 0.0093 |

## Method and interpretation

The inputs are the per-compound CSVs for the promoted default runs in tag `v1.1.1`: `mufu_20261106_022713_default`, `rf_bioactivity_default`, and `rf_toxicity_default`. Run metadata must identify the requested model and split, match the frozen data and split hashes, and confirm completion of seeds 0–4. Each file is joined to the split by unique SMILES. Members must agree on endpoint names. [Run manifest](../data/baseline/promoted.json)

The individual comparison pairs MuFu seed *s* with RF seed *s*. The ensemble comparison uses SCRUM-104's arithmetic mean of all five MuFu point predictions for each molecule. Classification uses each member's Platt-calibrated probability before averaging. RF contributes its positive-class probability. The same ensemble is paired separately with RF seeds 0–4. Its standalone test score is computed once. [Production averaging code](../src/mufu/ensemble.py)

For each endpoint and RF seed, validation fits the MuFu weight *w* in `w * MuFu + (1 - w) * RF`. The regression fit minimizes RMSE on [0, 1]. Classification maximizes validation ROC-AUC over 0.00, 0.01, …, 1.00, preferring a weight closest to 0.5 on ties. All weights are fixed before test predictions are loaded. A blend requires finite labels and both scores. Single-class validation tasks cannot fit an AUC weight. All 16 endpoints had usable fits for all five pairs, and the real test comparisons used the same labeled molecules.

The ensemble-blend SD measures variation from changing the RF seed **and its validation-fitted weight** while holding the five MuFu members and test molecules fixed. The individual-blend SD also includes variation between MuFu members. These are single-scaffold-split comparisons. The SD does not estimate uncertainty over new compounds or new ensembles, and the results establish no statistical significance.

MuFu used validation data for early stopping and calibration, and RF used validation for hyperparameter selection. Reusing validation to fit weights can yield optimistic validation gains. The default test split has 9 CT_TOX positives and 6 FDA_APPROVED negatives among 144 labeled molecules. A few changes in ranking can affect ClinTox AUC substantially. MuFu's deployed pIC50 intervals were calibrated for MuFu alone. A serving change would also require interval validation for the blended predictor.

## Fitted weights

Each entry is the weight on MuFu. The remaining weight goes to RF with the indicated seed. Regression weights are rounded to four decimals for display.

### Individual MuFu + RF

| Endpoint | RF 0 | RF 1 | RF 2 | RF 3 | RF 4 |
|---|---:|---:|---:|---:|---:|
| pIC50_BACE1 | 0.4729 | 0.4730 | 0.6217 | 0.7001 | 0.5138 |
| pIC50_EGFR | 0.5474 | 0.4838 | 0.4518 | 0.3799 | 0.4771 |
| NR-AR | 0.61 | 0.34 | 0.83 | 0.01 | 0.82 |
| NR-AR-LBD | 0.00 | 0.28 | 0.00 | 0.86 | 0.45 |
| NR-AhR | 0.55 | 0.30 | 0.59 | 0.25 | 0.39 |
| NR-Aromatase | 0.46 | 0.36 | 0.46 | 0.32 | 0.39 |
| NR-ER | 1.00 | 0.61 | 0.98 | 0.47 | 0.73 |
| NR-ER-LBD | 1.00 | 0.82 | 0.78 | 0.73 | 0.83 |
| NR-PPAR-gamma | 0.37 | 0.98 | 0.50 | 0.48 | 0.48 |
| SR-ARE | 0.38 | 0.55 | 0.51 | 0.39 | 0.40 |
| SR-ATAD5 | 0.81 | 0.81 | 0.85 | 0.81 | 0.81 |
| SR-HSE | 0.83 | 1.00 | 0.76 | 0.95 | 0.60 |
| SR-MMP | 0.36 | 0.42 | 0.39 | 0.40 | 0.42 |
| SR-p53 | 0.16 | 0.19 | 0.14 | 0.28 | 0.14 |
| FDA_APPROVED | 0.20 | 0.66 | 0.34 | 0.32 | 0.15 |
| CT_TOX | 0.16 | 0.30 | 0.30 | 0.21 | 0.23 |

### Five-seed MuFu ensemble + RF

| Endpoint | RF 0 | RF 1 | RF 2 | RF 3 | RF 4 |
|---|---:|---:|---:|---:|---:|
| pIC50_BACE1 | 0.8220 | 0.8307 | 0.8499 | 0.8530 | 0.8343 |
| pIC50_EGFR | 0.6208 | 0.6281 | 0.6416 | 0.6380 | 0.6349 |
| NR-AR | 0.85 | 0.80 | 0.97 | 0.98 | 0.97 |
| NR-AR-LBD | 0.29 | 0.65 | 0.47 | 0.53 | 0.52 |
| NR-AhR | 0.61 | 0.62 | 0.58 | 0.55 | 0.50 |
| NR-Aromatase | 0.49 | 0.56 | 0.54 | 0.45 | 0.43 |
| NR-ER | 1.00 | 0.94 | 1.00 | 0.93 | 1.00 |
| NR-ER-LBD | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| NR-PPAR-gamma | 0.51 | 0.89 | 0.54 | 0.67 | 0.70 |
| SR-ARE | 0.57 | 0.57 | 0.58 | 0.71 | 0.59 |
| SR-ATAD5 | 0.84 | 0.83 | 0.85 | 0.84 | 0.85 |
| SR-HSE | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 |
| SR-MMP | 0.53 | 0.52 | 0.52 | 0.47 | 0.52 |
| SR-p53 | 0.26 | 0.32 | 0.22 | 0.41 | 0.23 |
| FDA_APPROVED | 0.56 | 0.58 | 0.48 | 0.50 | 0.42 |
| CT_TOX | 0.47 | 0.36 | 0.47 | 0.39 | 0.49 |

## Reproduce from saved predictions

From `ai/` in the project environment, run both commands. Each reads the same default-split artifacts and prints its test summary and validation fits. [Artifact recovery instructions](../README.md#evaluating-mufu--rf-stacking)

```sh
PYTHONPATH=src python -m baseline.stacking --split-id default
PYTHONPATH=src python -m baseline.stacking --split-id default --ensemble
```
