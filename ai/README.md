# CHM4930: AI

This is where all code relating to training our deep learning models will go,
along with any datasets or checkpoints for saved model weights.

## Directory Structure

```
ai/
├── data/               # local data storage
│   └── baseline/       # frozen combined.csv baselines train on, and their results.csv (tracked)
├── docs/               # design docs (MuFu-v1.md)
├── notebooks/          # exploration / scratchpads
├── drivers/            # training, MC-Dropout fitting, etc.
├── experiments/        # sweeps and benchmarks
├── scripts/            # utilities (check_env.py)
├── src/                # main code
│   ├── baseline/       # random forest baselines, shared benchmark, and results recorder
│   ├── featurize/      # RDKit to SMILES, ECFP4 fingerprint, and PyG graph
│   ├── mufu/           # MuFu multimodal dataset, model, loss, and training components
│   ├── mufu_server/    # HTTP inference server for the web backend
│   ├── pipeline/       # dataset loaders and structure to model input preprocessing
│   ├── encoders/       # where each of MuFu's encoders are implemented
│   └── evaluation.py   # regression, classification, and multi-task evaluation metrics
├── tests/              # pytest suite (mirrors src/ and experiments/)
├── checkpoints/        # saved model weights
├── outputs/            # generated artifacts (figures, logs, metrics, etc.)
├── environment.yml     # conda environment config
├── pytest.ini          # pytest config (test paths, import path, markers)
└── README.md
```

## Configuring Environment

We chose to use Conda as it's the simplest for setting up an environment while
still be cross compatible with different OSes and different hardware
configurations. Make sure you have
[`conda`](https://www.anaconda.com/docs/getting-started/installation) installed
on your system.

To build the environment everything will be run in, make sure you're in the root
of this directory and run:

```
conda env create -f environment.yml
```

From there, just activate the environment.

```
conda activate CHM4930
```

Then register the `nbstripout` git filter so notebook output and execution
metadata never gets committed.

```
nbstripout --install
```

Next, register the jupyter kernel for this environment so everyone's notebooks
reference the same kernel. Run:

```
python -m ipykernel install --sys-prefix --name=CHM4930 --display-name="Python (CHM4930)"
```

Now when you open a notebook, select "Python (CHM4930)" as the kernel. To make
sure everything installed correctly, run:

```
python scripts/check_env.py
```

It imports each dependency and prints its version, so if something's missing or
broken, you'll know.

## Data Pipeline

Modules under `src/pipeline/` import from `src/` as packages, so run them as
modules from this directory with `src/` on the import path:

```
PYTHONPATH=src python -m pipeline.bioactivity_loader   # download/load ChEMBL BACE-1 + EGFR
PYTHONPATH=src python -m pipeline.toxicity_loader      # download/load Tox21 + ClinTox
PYTHONPATH=src python -m pipeline.combined_loader      # load frozen combined table (see Baselines)
```

Each loader's documentation is in its file docstring. They cache cleaned data as
CSVs in `data/` and return a Pandas `DataFrame` with a standardized `smiles`
column plus labels.

`pipeline.preprocess.featurize()` takes a SMILES string or RDKit `Mol` and
returns a PyG `Data` object with all three model inputs:

- graph (`x`, `edge_index`, `edge_attr`)
- ECFP4 fingerprint (`fp`)
- canonical SMILES (`smiles`)

Note that fingerprints are rejected as input since ECFP is a lossy hash. To
featurize a whole dataset, pass its `smiles` column to `featurize_many()`.

> NOTE: While any valid input structure is accepted, predictions for inorganic
> compounds are basically extrapolations. The training data is almost entirely
> organic, and anything outside what the graph explicitly one-hot encodes is
> treated as "other". Salts are reduced to their largest organic fragment when
> there is one, so inorganic salts that share an ion standardize to the same
> SMILES.

## Baselines

Two random forest baselines on 2048-bit ECFP4 fingerprints, both on the same
Bemis-Murcko scaffold split:

```
PYTHONPATH=src python -m baseline.baseline_bioactivity
PYTHONPATH=src python -m baseline.baseline_toxicity
```

They train on the frozen snapshot `data/baseline/combined.csv`. The only way to
change the snapshot is:

```
PYTHONPATH=src python -m pipeline.combined_loader --refresh
```

Don't run that casually. It invalidates every recorded result, so delete
`data/baseline/results.csv`, re-run both baselines, and commit the new snapshot
together with the new results.

Each run replaces its own `(model, encoder, seed)` rows in
`data/baseline/results.csv`, one row per model × endpoint × split × seed × metric
across `SEEDS = (0, 1, 2, 3, 4)`. What is constant per run (params, the snapshot's
SHA-256 taken with LF line endings so Windows checkouts match, the git commit and the
date) is stored once per `(model, encoder, seed)` in `data/baseline/results_runs.csv`.
Read both with `baseline.results.load_results()` and aggregate across seeds with
`baseline.results.summarize_seeds()`. Current test set performance (mean over 5 seeds):

| Model            | Task                  | Test result                                  |
| ---------------- | --------------------- | -------------------------------------------- |
| `rf_bioactivity` | BACE-1 pIC50          | RMSE 0.81, R² 0.59, ROC-AUC 0.88 (pIC50 ≥ 6) |
| `rf_bioactivity` | EGFR pIC50            | RMSE 0.88, R² 0.50, ROC-AUC 0.88 (pIC50 ≥ 6) |
| `rf_toxicity`    | Tox21 (12 endpoints)  | macro ROC-AUC 0.71                           |
| `rf_toxicity`    | ClinTox (2 endpoints) | macro ROC-AUC 0.65                           |

NOTE: See `data/baseline/results.csv` for per-task numbers. Some test sets have
very few examples of one class, so read their ROC-AUC loosely. For example,
NR-AR has 11 actives out of 704 (val 0.82, test 0.56), and FDA_APPROVED has only
6 unapproved compounds out of 144 (val 0.83, test 0.58).

## Benchmarks

Every model is scored on the same split, rows, seeds, and metrics as the RF
baselines through `src/baseline/benchmark.py`:

- `SEEDS = (0, 1, 2, 3, 4)`: the shared training seeds every model runs across
- `load_benchmark_split()`: loads `data/baseline/combined.csv` and applies the
  shared Bemis-Murcko scaffold split
- `seed_everything(seed)`: seeds Python, NumPy, and Torch and returns a seeded
  `torch.Generator` for DataLoaders
- `evaluate_split_predictions(df, predictions)`: scores aligned per-split
  predictions with the dataset label masks (recording pIC50-derived `roc_auc`
  under `pIC50_<T>`, classifier-head `roc_auc` under `active_<T>`, and checking
  `smiles` order when provided)
- `run_benchmark(fit_predict, model, encoder, params)`: runs
  `fit_predict(df, seed)` across `SEEDS`, evaluates `"val"` and `"test"`, writes
  each seed's rows to `data/baseline/results.csv` as it finishes, and returns the
  `summarize_seeds()` table. It refuses to drop other models' rows from an older
  snapshot unless `allow_drop_stale=True`

### MuFu training

From `ai/`, train MuFu on all five shared seeds:

```sh
python drivers/train_mufu.py
```

`drivers/fit_mc_dropout.py` refits the MC-Dropout `q` for the saved checkpoints.

Once seeds finish, `train_mufu.py` fits the seed ensemble
(`ensemble_after_training`) and skips it if any seed checkpoint is missing.
`drivers/fit_ensemble.py` refits it from saved checkpoints without retraining.
Members are averaged after each seed's own Platt calibration, and pIC50
intervals combine MC-Dropout with the spread between seeds (see
`docs/MC-Dropout.md`).

`notebooks/MuFu_Training.ipynb` shows loss plots and a MuFu vs RF comparison from saved
results, and can run the same training with `RUN_TRAINING = True`. Both use
`src/mufu/experiment.py`.

To change settings, edit `config` (`replace(DEFAULT_CONFIG, ...)`) and `model_name`
in the script or notebook. Commit your changes before a final run so `results.csv`
records a clean git commit.

Training stops early after `patience` epochs without a validation loss improvement
of at least `min_delta`; `epochs` is the maximum.

Outputs (a rerun replaces that seed's files):

- `checkpoints/mufu_seed_<seed>.pt`: best model, with its params, git commit, and
  data hash
- `outputs/mufu/seed_<seed>_loss.csv`: train and validation loss per epoch
- `data/baseline/results.csv`: validation and test metrics per seed
- `checkpoints/mufu_ensemble.json`: ensemble `q`, with each member's sha256
- `outputs/mufu/ensemble_metrics.csv`, `ensemble_coverage.csv`: ensemble val/test
  metrics and test interval coverage. Ensemble metrics are not written to `results.csv`.

### Encoder sweeps

`experiments/ECFP4_encoder_sweep.py`, `gnn_encoder_sweep.py`, and
`SMILES_encoder_sweep.py` run exploratory hyperparameter stages on the BACE-1
rows of the shared split (6,469 / 808 / 810), the same molecules as
`rf_bioactivity`:

```sh
python experiments/ECFP4_encoder_sweep.py 1 2
python experiments/gnn_encoder_sweep.py 1
python experiments/SMILES_encoder_sweep.py 1
```

Stage CSVs and logs go to `--output-dir` (default
`outputs/encoders/<encoder>/sweep/`), tagged with `data_sha256` and
`split_sha256` so interrupted stages verify the snapshot and split before
resuming.

## Inference server

`src/mufu_server/` serves the 5-seed MuFu ensemble over HTTP. The response format is
in [`backend/docs/predict-api.md`](../backend/docs/predict-api.md) ("Inference
server contract").

- `POST /predict` with `{"smiles": "..."}`: the ensemble mean pIC50 with its
  95% combined interval and `p_active` per target, plus all 14 toxicity
  probabilities (each seed's calibrated probabilities, averaged). The same
  SMILES always gets the same response.
- `GET /health`: `{"status": "loading" | "ready", "model_version": ...}`. No key
  needed.

Other requests need the `X-API-Key` header. Errors are `{"error": string}`:

| Status | When                                                         |
| ------ | ------------------------------------------------------------ |
| 401    | missing or wrong key                                         |
| 413    | body over 4 KiB                                              |
| 422    | invalid SMILES or malformed body                             |
| 429    | over the rate limit; see `Retry-After`                       |
| 503    | model still loading, another inference running, or timed out |
| 500    | anything else (details in the server log)                    |

### Running it

From `ai/`, with the five `mufu_seed_<seed>.pt` checkpoints and the
`mufu_ensemble.json` that `drivers/fit_ensemble.py` writes in `checkpoints/`:

```sh
export MUFU_CHECKPOINT_DIR=checkpoints
export MUFU_API_KEY="$(openssl rand -hex 32)"
PYTHONPATH=src python -m mufu_server
```

```sh
curl -s localhost:8000/predict -H "X-API-Key: $MUFU_API_KEY" \
  -H 'Content-Type: application/json' -d '{"smiles": "CCOc1ccc(N)cc1"}'
```

For a local run of the whole app, give the backend the same key and
`MUFU_API_URL=http://localhost:8000`.

| Variable                     | Default                                                      |
| ---------------------------- | ------------------------------------------------------------ |
| `MUFU_CHECKPOINT_DIR`        | required                                                     |
| `MUFU_API_KEY`               | required                                                     |
| `MUFU_MODEL_NAME`            | `mufu`                                                       |
| `MUFU_MODEL_VERSION`         | `<model_name>_ensemble@<sha256 prefix of the ensemble file>` |
| `MUFU_HF_REPO`               | unset                                                        |
| `MUFU_HOST` / `MUFU_PORT`    | `127.0.0.1` / 8000                                           |
| `MUFU_DEVICE`                | `cpu`                                                        |
| `MUFU_RATE_LIMIT_PER_MINUTE` | 30                                                           |
| `MUFU_REQUEST_TIMEOUT_S`     | 30                                                           |

The first start downloads MolFormer; after that, `HF_HUB_OFFLINE=1` lets it
start without the network. If any of the five checkpoints or the ensemble file
is missing, or the ensemble was fitted on different checkpoints, the process
exits with status 1 instead of serving fewer seeds.

### Published weights

Different versions are on [`CHM4930/mufu`](https://huggingface.co/CHM4930/mufu),
tagged `vMAJOR.MINOR.PATCH`. Each tag's `version.json` records the commit the
checkpoints were trained at.

| Bump  | When                                                                     |
| ----- | ------------------------------------------------------------------------ |
| major | the server or backend must change to use it (outputs, files, model kind) |
| minor | a retrained or improved model that's a drop-in replacement               |
| patch | same weights, metadata only (e.g. a model card fix)                      |

- To serve one, set `MUFU_HF_REPO=CHM4930/mufu` and `MUFU_MODEL_VERSION` to its
  tag. The first start downloads it into `MUFU_CHECKPOINT_DIR`; later starts
  load it from there offline.
- To just download one, set `tag` in `drivers/download_mufu.py` and run it.
- To publish, train with `drivers/train_mufu.py` on a clean commit and commit
  the `results.csv` rows it writes. Then run `hf auth login` (write token), set
  `tag` in `drivers/publish_mufu.py` and run it, first with `dry_run = True` to
  review the model card.

### Hosting on personal compute

The deployed backend reaches the server through a named Cloudflare Tunnel
(quick tunnels change URL on every restart).

1. Download a published version or copy the five `mufu_seed_<seed>.pt` files and
   `mufu_ensemble.json` into a folder only the server uses, e.g.
   `~/.local/share/mufu/<version>/`. Don't serve `checkpoints/` directly:
   training overwrites it, so a restart mid-training would fail to load, and one
   after training would switch models unannounced. To ship a new model, copy it
   to a new folder, update `MUFU_CHECKPOINT_DIR` and restart the server.
2. Put the env vars in `~/.config/mufu/server.env` (`chmod 600`), with
   `MUFU_CHECKPOINT_DIR` set to that folder's absolute path, and
   `HF_HUB_OFFLINE=1`.
3. Create the tunnel: `cloudflared tunnel login`,
   `cloudflared tunnel create mufu`, `cloudflared tunnel route dns mufu <hostname>`.
4. `~/.cloudflared/config.yml`:

   ```yaml
   tunnel: mufu
   credentials-file: /home/<user>/.cloudflared/<tunnel-id>.json
   ingress:
     - hostname: <hostname>
       service: http://127.0.0.1:8000
     - service: http_status:404
   ```

5. `~/.config/systemd/user/mufu-server.service`:

   ```ini
   [Service]
   WorkingDirectory=%h/<path to repo>/ai
   EnvironmentFile=%h/.config/mufu/server.env
   Environment=PYTHONPATH=src
   ExecStart=<conda env prefix>/bin/python -m mufu_server
   Restart=on-failure
   RestartSec=10

   [Install]
   WantedBy=default.target
   ```

   `mufu-tunnel.service` is the same, with
   `ExecStart=/usr/bin/cloudflared tunnel run mufu` and `Restart=always`.

6. `systemctl --user enable --now mufu-server mufu-tunnel`, and
   `loginctl enable-linger $USER` so they start at boot.
7. Disable suspend on AC power and add an uptime check on
   `https://<hostname>/health`.

Keep the key and `~/.cloudflared/*.json` out of the repo.

## Testing

Tests use [`pytest`](https://docs.pytest.org). From anywhere inside this
directory, run:

```
pytest
```

`pytest.ini` points pytest at `tests/` and adds `src/` and `experiments/` to the
import path, so tests can `import` modules from either directly.

Tests marked `network` call external services (ChEMBL and MoleculeNet), so
they're skipped by default. To run `network` tests, do:

```
pytest -m network
```

`tests/encoders/test_smiles_encoder.py` needs `transformers` to import. If
collection fails with `ModuleNotFoundError: transformers`, run
`conda env update -n CHM4930 -f environment.yml`.
