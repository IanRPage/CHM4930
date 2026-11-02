# CHM4930: AI

MuFu predicts bioactivity (BACE-1 and EGFR pIC50) and toxicity (12 Tox21 tasks plus
ClinTox `CT_TOX` and `FDA_APPROVED`) from a molecular structure. It is one multitask model
that encodes each molecule three ways (MolFormer over SMILES, an MLP over ECFP4, a GINE
over the graph) and fuses them before shared prediction heads. Design:
[`docs/MuFu-v1.md`](docs/MuFu-v1.md). Random forests on ECFP4 are the baselines every
model is compared against.

```
ai/
├── data/baseline/   # frozen combined.csv snapshot, promoted.json and recorded results (tracked)
├── docs/            # design docs
├── drivers/         # MuFu training, promotion, rebuilding, ensemble, publishing
├── experiments/     # sweeps and one-off controls
├── notebooks/       # exploration; benchmark-results.ipynb plots recorded results
├── runs/            # self-contained per-run checkpoints, predictions and logs (gitignored)
├── src/
│   ├── baseline/    # RF baselines, shared benchmark protocol, results recorder
│   ├── encoders/    # MuFu's SMILES, fingerprint and graph encoders
│   ├── featurize/   # SMILES → ECFP4 / PyG graph
│   ├── mufu/        # MuFu model, loss, training, ensemble
│   ├── mufu_server/ # HTTP inference server
│   ├── pipeline/    # dataset loaders and preprocessing
│   └── evaluation.py
└── tests/
```

## Setup

From `ai/`:

```
conda env create -f environment.yml
conda activate CHM4930
nbstripout --install
python -m ipykernel install --sys-prefix --name=CHM4930 --display-name="Python (CHM4930)"
python scripts/check_env.py
```

`nbstripout` keeps notebook outputs out of git. `check_env.py` imports every dependency
and prints its version.

## Data

```
PYTHONPATH=src python -m pipeline.bioactivity_loader   # ChEMBL BACE-1 + EGFR
PYTHONPATH=src python -m pipeline.toxicity_loader      # Tox21 + ClinTox
```

Every model trains on the frozen snapshot `data/baseline/combined.csv`. Changing it
(`python -m pipeline.combined_loader --refresh`) invalidates every recorded result, so
only do it together with a full rerun, and commit the snapshot and results together.

Inputs are standardized SMILES, from which ECFP4 and the graph are derived; a fingerprint
alone is not accepted. Salts are reduced to their largest organic fragment, and inorganic
compounds are far outside the training data, so their predictions are extrapolations.

## Evaluation

Every model goes through `src/baseline/benchmark.py`, so all are scored on the same
molecules, seeds and metrics:

- **Splits.** Bemis–Murcko scaffold splits, so test scaffolds never appear in train.
  `default` puts the largest scaffold groups in train, which makes its test set the
  hardest. `scaffold-0` ... `scaffold-4` are balanced, shuffled splits.
- **Seeds.** Each split trains on seeds 0–4. Seed std only measures training noise.
- **Uncertainty.** Test metrics carry a 95% bootstrap CI over test compounds (2000
  resamples), which is the honest error bar for one split. Split-to-split spread is
  reported separately from seed spread.
- **Hyperparameters** are selected on validation.

| Endpoint                | Metrics                                                              |
| ----------------------- | -------------------------------------------------------------------- |
| `pIC50_<T>`             | RMSE, R², Spearman, Kendall; ROC/PR-AUC treating pIC50 ≥ 6 as active |
| `active_<T>`, tox tasks | ROC-AUC, PR-AUC, prevalence                                          |
| `macro_<group>`         | mean ROC-AUC and PR-AUC over the group's evaluable tasks             |

PR-AUC sits next to ROC-AUC because several endpoints are rare (CT_TOX 9/144 positive,
NR-AR 11/704), where ROC-AUC looks good even when flagged compounds are mostly wrong.
Spearman reflects ranking, which is how predictions get used to prioritize compounds.

Promoted runs are listed in `data/baseline/promoted.json` and their metrics land in
`data/baseline/results.csv` (one row per model, endpoint, split, seed and metric),
`results_runs.csv` (one row per promoted run with `run_id`, `n_seeds`, `params`,
`data_sha256`, `split_sha256`, `base_commit`, `ai_tree`, `patch_sha256` and `date`),
`results_paired.csv` (validation and test paired bootstrap CIs), and
`results_by_similarity.csv`. [`notebooks/benchmark-results.ipynb`](notebooks/benchmark-results.ipynb)
plots them.

### Evaluating a MuFu change

Compare variants on validation and score only the one you keep on test: picking whichever
variant does best on test inflates its test score.

1. Make the change (optionally set `run_name = "<name>"` in `drivers/train_mufu.py`) and
   run `python drivers/train_mufu.py`. Each run writes its checkpoints, `val`/`test`
   predictions, `val_metrics.csv` and `val_paired.csv` into its own folder
   `runs/<run_id>/` without touching `data/baseline/`.
   (On a fresh clone, populate reference predictions once with
   `python drivers/download_mufu.py`, which unpacks both the promoted MuFu run and
   the bundled RF baseline runs into `runs/`.)
2. Compare the candidate's per-seed validation metrics and paired CIs against `mufu` and
   RF in the printed `val_paired.csv` or the Validation section of
   `notebooks/benchmark-results.ipynb` (`CANDIDATE_RUNS = ["<run_id>"]`). Repeat 1–2
   until you have a version worth keeping; discard losing runs by deleting
   `runs/<run_id>/`.
3. To promote the winner without retraining, set `run_id = "<run_id>"` (and `tag` if
   bumping the version) in `drivers/promote_mufu.py` and run
   `python drivers/promote_mufu.py`. This scores the saved test predictions once and
   regenerates `data/baseline/promoted.json`, `results.csv`, `results_runs.csv`,
   `results_paired.csv` and `results_by_similarity.csv`.
4. Commit the code and `data/baseline/` changes, then publish to Hugging Face:
   `hf auth login`, run `python drivers/publish_mufu.py` with `dry_run = True` to review
   `runs/<run_id>/README.md`, then set `dry_run = False`. Tags are `vMAJOR.MINOR.PATCH`:
   minor for a drop-in retrained model, major if the server or backend must change,
   patch for metadata only.

## Results

Test set. "Default" is the mean over 5 seeds with the 95% bootstrap CI; "6 splits" is
mean ± std over all splits. MuFu has only been trained on the default split so far.

| Endpoint                | Metric   | RF default        | RF 6 splits | MuFu default      |
| ----------------------- | -------- | ----------------- | ----------- | ----------------- |
| BACE-1 pIC50            | RMSE     | 0.81 [0.76, 0.86] | 0.74 ± 0.04 | 0.83 [0.77, 0.88] |
| BACE-1 pIC50            | Spearman | 0.78 [0.75, 0.81] | 0.80 ± 0.02 | 0.78 [0.74, 0.81] |
| EGFR pIC50              | RMSE     | 0.88 [0.83, 0.92] | 0.80 ± 0.06 | 0.90 [0.86, 0.95] |
| EGFR pIC50              | Spearman | 0.72 [0.68, 0.75] | 0.78 ± 0.04 | 0.68 [0.64, 0.72] |
| Tox21 macro             | ROC-AUC  | 0.71 [0.67, 0.75] | 0.79 ± 0.05 | 0.73 [0.69, 0.77] |
| Tox21 macro             | PR-AUC   | 0.31 [0.27, 0.36] | 0.41 ± 0.08 | 0.26 [0.23, 0.32] |
| CT_TOX (9 / 144)        | ROC-AUC  | 0.73 [0.49, 0.92] | 0.75 ± 0.07 | 0.81 [0.66, 0.93] |
| CT_TOX (9 / 144)        | PR-AUC   | 0.22 [0.08, 0.52] | 0.26 ± 0.07 | 0.26 [0.09, 0.54] |
| FDA_APPROVED (6 / 144¹) | ROC-AUC  | 0.58 [0.24, 0.88] | 0.79 ± 0.12 | 0.83 [0.65, 0.96] |

¹ 6 unapproved compounds, so this CI is very wide. `macro_clintox` averages CT_TOX and
FDA_APPROVED, which are near-complements, so report the two separately.

- Paired bootstrap of MuFu − RF (same resampled compounds for both, averaged over seeds;
  `data/baseline/results_paired.csv`):
  RF is better on EGFR Spearman (−0.035 [−0.054, −0.015]) and Tox21 PR-AUC (−0.044
  [−0.081, −0.007]), and borderline on EGFR RMSE (+0.028 [+0.0003, +0.055]). MuFu is
  better on FDA_APPROVED ROC-AUC (+0.26 [0.03, 0.47], from 6 negatives). BACE-1, Tox21
  ROC-AUC and CT_TOX are ties. This is one split; MuFu hasn't been run on the others yet.
- The split matters more than the seed: on BACE-1, EGFR and Tox21, RF's split std is
  about 20× its seed std (18–27×).
- Error grows with distance from the training set: BACE-1 RMSE is about 0.74 for test
  compounds with nearest-train Tanimoto ≥ 0.6 and 1.2-1.4 below 0.4
  (`data/baseline/results_by_similarity.csv`).
- Y-scrambling (`experiments/y_scramble.py`): RF refit on shuffled labels falls to
  R² < 0 and Spearman ≈ 0, so the pipeline isn't leaking test labels.

## Running models

**RF baselines** (CPU, all 6 splits; writes to `runs/rf_*_<split_id>/`):

```
PYTHONPATH=src python -m baseline.baseline_bioactivity
PYTHONPATH=src python -m baseline.baseline_toxicity
```

**MuFu** (GPU). Settings are at the top of each driver.

```
python drivers/train_mufu.py       # train 5 seeds + ensemble into runs/<run_id>/ (val only)
python drivers/promote_mufu.py     # promote a run_id and rebuild data/baseline/*.csv
python drivers/rebuild_results.py  # recompute data/baseline/*.csv from saved predictions
python drivers/rescore_mufu.py     # recompute predictions/*.csv from saved checkpoints
python drivers/fit_ensemble.py     # fit/evaluate ensemble from saved checkpoints
python drivers/fit_mc_dropout.py   # refit per-seed MC-Dropout q
```

Each run lives in `runs/<run_id>/` with `run.json`, `checkpoints/`, `predictions/` and
`logs/`. Set `split_id = "scaffold-<k>"` in `train_mufu.py` to train on another split.
The ensemble averages each seed's Platt-calibrated outputs; pIC50 intervals combine
MC-Dropout with between-seed spread ([`docs/MC-Dropout.md`](docs/MC-Dropout.md)). Test-set
ensemble metrics go to `runs/<run_id>/ensemble_metrics.csv`.

**Encoder sweeps** (`experiments/{ECFP4,gnn,SMILES}_encoder_sweep.py <stage>`) tune each
encoder alone on the BACE-1 rows of the default split and write to
`outputs/encoders/<encoder>/sweep/`.

## Inference server

`src/mufu_server/` serves the 5-seed ensemble. `POST /predict` with `{"smiles": "..."}`
returns per-target pIC50 with a 95% interval and `p_active`, plus all 14 toxicity
probabilities; `GET /health` reports `loading` or `ready`. The contract is in
[`backend/docs/predict-api.md`](../backend/docs/predict-api.md).

```sh
export MUFU_CHECKPOINT_DIR=runs/mufu_v1.0.0_default/checkpoints   # 5 seeds + mufu_ensemble.json
export MUFU_API_KEY="$(openssl rand -hex 32)"
PYTHONPATH=src python -m mufu_server

curl -s localhost:8000/predict -H "X-API-Key: $MUFU_API_KEY" \
  -H 'Content-Type: application/json' -d '{"smiles": "CCOc1ccc(N)cc1"}'
```

Per-person keys: `PYTHONPATH=src python -m mufu_server.keys add <name>` prints a key once
and saves its hash; point `MUFU_API_KEYS_FILE` at that file. Other settings
(`MUFU_HOST`, `MUFU_PORT`, `MUFU_DEVICE`, rate limit, timeout, `MUFU_HF_REPO`) are in
`src/mufu_server/config.py`. The server refuses to start if any checkpoint is missing or
the ensemble was fitted on different ones.

Published versions are on [`CHM4930/mufu`](https://huggingface.co/CHM4930/mufu). To
serve one, set `MUFU_HF_REPO=CHM4930/mufu` and `MUFU_MODEL_VERSION=<tag>`;
`drivers/download_mufu.py` just downloads one. Hosting on personal compute behind a
Cloudflare Tunnel: [`docs/hosting.md`](docs/hosting.md).

## Testing

```
pytest              # from anywhere in ai/
pytest -m network   # also hit ChEMBL and MoleculeNet
```
