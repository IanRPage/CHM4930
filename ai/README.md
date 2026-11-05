# CHM4930: AI

MuFu predicts bioactivity (BACE-1 and EGFR pIC50) and toxicity (12 Tox21 tasks plus
ClinTox `CT_TOX` and `FDA_APPROVED`) from a molecular structure. It encodes each molecule
three ways (MolFormer over SMILES, an MLP over ECFP4, a GINE over the graph) and fuses
them before shared prediction heads ([`docs/MuFu-v1.md`](docs/MuFu-v1.md)). Random
forests on ECFP4 are the baselines.

```
ai/
├── data/baseline/   # frozen combined.csv, promoted.json, recorded results (tracked)
├── docs/            # design docs
├── drivers/         # MuFu training, promotion, rebuilding, ensemble, publishing
├── experiments/     # sweeps and one-off controls
├── notebooks/       # exploration; benchmark-results.ipynb plots recorded results
├── runs/            # per-run checkpoints, predictions, logs (ignored)
├── src/
│   ├── baseline/    # RF baselines, shared benchmark protocol, results recorder
│   ├── encoders/    # SMILES, fingerprint, graph encoders
│   ├── featurize/   # SMILES → ECFP4 / PyG graph
│   ├── mufu/        # MuFu, loss, training, ensemble, Hub publishing
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

## Data

```
PYTHONPATH=src python -m pipeline.bioactivity_loader   # ChEMBL BACE-1 + EGFR
PYTHONPATH=src python -m pipeline.toxicity_loader      # Tox21 + ClinTox
```

Every model trains on the frozen snapshot `data/baseline/combined.csv`. Refreshing it
(`python -m pipeline.combined_loader --refresh`) invalidates recorded results (only do so
with full rerun, and commit both together).

## Evaluation

All models go through `src/baseline/benchmark.py` on identical splits, seeds, and metrics:

- **Splits.** Bemis–Murcko scaffold splits so test scaffolds never appear in train.
  `default` puts the largest scaffold groups in train (hardest test set); `scaffold-0` ...
  `scaffold-4` are balanced shuffled splits.
- **Seeds.** Seeds 0–4 per split; seed std measures training noise only.
- **Uncertainty.** Test metrics carry a 95% bootstrap CI over test compounds (2000
  resamples). Split-to-split spread is reported separately from seed spread.
- **Model selection.** Hyperparameters and model variants are chosen on validation only.

| Endpoint                | Metrics                                                              |
| ----------------------- | -------------------------------------------------------------------- |
| `pIC50_<T>`             | RMSE, R², Spearman, Kendall; ROC/PR-AUC treating pIC50 ≥ 6 as active |
| `active_<T>`, tox tasks | ROC-AUC, PR-AUC, prevalence                                          |
| `macro_<group>`         | mean ROC-AUC and PR-AUC over the group's evaluable tasks             |

PR-AUC is tracked alongside ROC-AUC for imbalanced endpoints (e.g. CT_TOX 9/144 positive,
NR-AR 11/704), and Spearman tracks compound ranking.

Promoted runs in `data/baseline/promoted.json` record metrics in `results.csv` and run
provenance in `results_runs.csv`, plotted in
[`notebooks/benchmark-results.ipynb`](notebooks/benchmark-results.ipynb).

### Evaluating a MuFu change

1. Make the change (optionally set `run_name = "<name>"` in `drivers/train_mufu.py`) and
   run `python drivers/train_mufu.py`. This writes checkpoints, predictions, and val
   metrics to `runs/<run_id>/` without touching `data/baseline/`. (On a fresh clone, run
   `python drivers/download_mufu.py` once to populate reference runs.)
2. Compare against `mufu` and RF in `val_paired.csv` or the Validation section of
   `notebooks/benchmark-results.ipynb` (`CANDIDATE_RUNS = ["<run_id>"]`). Delete losing
   runs (`runs/<run_id>/`).
3. Set `run_id = "<run_id>"` in `drivers/promote_mufu.py`, then run:
   ```
   python drivers/promote_mufu.py
   python drivers/fit_ensemble.py
   ```
   to score test predictions, update `data/baseline/`, and fit ensemble + training
   fingerprint sidecar.
4. Publish to Hugging Face (`hf auth login`): run
   `python drivers/publish_mufu.py <major|minor|patch>` with `dry_run = True` to review
   `runs/<run_id>/README.md`, then set `dry_run = False` to upload the incremented tag and
   update `promoted.json`. Use `major` for breaking contract changes, `minor` for a
   retrained model, and `patch` for metadata/sidecar updates. Commit `data/baseline/`.

## Results

Test set ("Default" = 5-seed mean with 95% bootstrap CI; "6 splits" = mean ± std across
splits).

| Endpoint               | Metric   | RF default        | RF 6 splits | MuFu default      |
| ---------------------- | -------- | ----------------- | ----------- | ----------------- |
| BACE-1 pIC50           | RMSE     | 0.81 [0.76, 0.86] | 0.74 ± 0.04 | 0.83 [0.77, 0.88] |
| BACE-1 pIC50           | Spearman | 0.78 [0.75, 0.81] | 0.80 ± 0.02 | 0.78 [0.74, 0.81] |
| EGFR pIC50             | RMSE     | 0.88 [0.83, 0.92] | 0.80 ± 0.06 | 0.90 [0.86, 0.95] |
| EGFR pIC50             | Spearman | 0.72 [0.68, 0.75] | 0.78 ± 0.04 | 0.68 [0.64, 0.72] |
| Tox21 macro            | ROC-AUC  | 0.71 [0.67, 0.75] | 0.79 ± 0.05 | 0.73 [0.69, 0.77] |
| Tox21 macro            | PR-AUC   | 0.31 [0.27, 0.36] | 0.41 ± 0.08 | 0.26 [0.23, 0.32] |
| CT_TOX (9 / 144)       | ROC-AUC  | 0.73 [0.49, 0.92] | 0.75 ± 0.07 | 0.81 [0.66, 0.93] |
| CT_TOX (9 / 144)       | PR-AUC   | 0.22 [0.08, 0.52] | 0.26 ± 0.07 | 0.26 [0.09, 0.54] |
| FDA_APPROVED (6 / 144) | ROC-AUC  | 0.58 [0.24, 0.88] | 0.79 ± 0.12 | 0.83 [0.65, 0.96] |

## Running models

**RF baselines** (CPU, all 6 splits → `runs/rf_*_<split_id>/`):

```
PYTHONPATH=src python -m baseline.baseline_bioactivity
PYTHONPATH=src python -m baseline.baseline_toxicity
```

**MuFu** (GPU, opts at top of each driver):

```
python drivers/train_mufu.py                        # train 5 seeds into runs/<run_id>/ (val only)
python drivers/promote_mufu.py                      # promote run and rebuild data/baseline/*.csv
python drivers/fit_ensemble.py                      # fit ensemble q and pack train fingerprints
python drivers/publish_mufu.py <major|minor|patch>  # publish to HF and bump promoted.json tag
python drivers/download_mufu.py                     # download promoted tag from HF into runs/
python drivers/rebuild_results.py                   # rebuild data/baseline/*.csv from saved predictions
python drivers/rescore_mufu.py                      # recompute predictions from saved checkpoints
python drivers/fit_mc_dropout.py                    # refit per-seed MC-Dropout q
```

Each run lives in `runs/<run_id>/` (`run.json`, `checkpoints/`, `predictions/`, `logs/`).
The ensemble averages Platt-calibrated outputs across seeds; pIC50 intervals combine
MC-Dropout and between-seed spread ([`docs/MC-Dropout.md`](docs/MC-Dropout.md)).

**Encoder sweeps:** `experiments/{ECFP4,gnn,SMILES}_encoder_sweep.py <stage>` tunes each
encoder on BACE-1 (`default` split) under `outputs/encoders/<encoder>/sweep/`.

## Inference server

`src/mufu_server/` serves the 5-seed ensemble (`GET /health`, `POST /predict`). Responses
include per-target pIC50 with a 95% interval and `p_active`, all 14 toxicity
probabilities, and applicability-domain similarity (`nearest_train_tanimoto`,
`applicability_domain`); full contract in
[`backend/docs/api.md`](../backend/docs/api.md#inference-server-contract).

```sh
export MUFU_CHECKPOINT_DIR=runs/mufu_v1.0.0_default/checkpoints   # 5 seeds + mufu_ensemble.json + mufu_train_fps.npy
export MUFU_API_KEY="$(openssl rand -hex 32)"
PYTHONPATH=src python -m mufu_server

curl -s localhost:8000/predict -H "X-API-Key: $MUFU_API_KEY" \
  -H 'Content-Type: application/json' -d '{"smiles": "CCOc1ccc(N)cc1"}'
```

- **API keys:** `PYTHONPATH=src python -m mufu_server.keys add <name>` prints a key once
  and saves its hash (`~/.config/mufu/api_keys` or `--file <path>`); point
  `MUFU_API_KEYS_FILE` at that file.
- **Settings:** `MUFU_HOST`, `MUFU_PORT`, `MUFU_DEVICE`, rate limit, timeout, and
  `MUFU_HF_REPO` are configured in `src/mufu_server/config.py`. Startup fails loudly if
  any checkpoint or fingerprint sidecar is missing or mismatched.
- **Serving from HF ([`CHM4930/mufu`](https://huggingface.co/CHM4930/mufu)):** set
  `MUFU_HF_REPO=CHM4930/mufu` and `MUFU_MODEL_VERSION=<tag>`. Tunnel setup:
  [`docs/hosting.md`](docs/hosting.md).

## Testing

```
pytest              # from anywhere in ai/
pytest -m network   # also hit ChEMBL and MoleculeNet
```
