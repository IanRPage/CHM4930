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
├── scripts/            # helpful utility scripts
├── src/                # main code
│   ├── baseline/       # random forest baselines and their results recorder
│   ├── featurize/      # RDKit to SMILES, ECFP4 fingerprint, and PyG graph
│   ├── mufu/           # MuFu multimodal dataset, model, loss, and training components
│   ├── pipeline/       # dataset loaders and structure to model input preprocessing
│   ├── encoders/       # where each of MuFu's encoders and their benchmark runner live
│   ├── benchmark.py    # shared split, seeding, and scoring every model uses
│   └── evaluation.py   # regression, classification, and multi-task evaluation metrics
├── tests/              # pytest suite (mirrors src/)
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

Don't run that casually. It invalidates every recorded result, so re-run both
baselines and commit the new snapshot together with the new results.

Each run replaces its own rows in `data/baseline/results.csv`, one row per model
× endpoint × split × metric. Each row is tagged with the snapshot's SHA-256 (taken
with LF line endings, so Windows checkouts match) and the git commit it ran on.
Current test set performance:

| Model            | Task                  | Test result                                  |
| ---------------- | --------------------- | -------------------------------------------- |
| `rf_bioactivity` | BACE-1 pIC50          | RMSE 0.81, R² 0.59, ROC-AUC 0.88 (pIC50 ≥ 6) |
| `rf_bioactivity` | EGFR pIC50            | RMSE 0.87, R² 0.51, ROC-AUC 0.88 (pIC50 ≥ 6) |
| `rf_toxicity`    | Tox21 (12 endpoints)  | macro ROC-AUC 0.72                           |
| `rf_toxicity`    | ClinTox (2 endpoints) | macro ROC-AUC 0.65                           |

NOTE: See `data/baseline/results.csv` for per-task numbers. Some test sets have
very few examples of one class, so read their ROC-AUC loosely. For example,
NR-AR has 11 actives out of 704 (val 0.83, test 0.55), and FDA_APPROVED has only
6 unapproved compounds out of 144 (val 0.84, test 0.56).

## Benchmarks

Every model is scored on the same split, rows, and metrics as the RF baselines.
`src/benchmark.py` has the shared pieces:

- `load_benchmark()`: the frozen table with its scaffold split
- `set_training_seed()`: seeds Python, NumPy, and Torch (default 42) and returns a
  seeded generator for DataLoaders
- `evaluate_predictions()`: shared metrics, with missing labels masked
- `benchmark_metadata()`: hashes of the dataset and split assignments. Store it in
  result `params` so a result can be checked against the snapshot it came from

### Encoder sweeps

`scripts/ECFP4_encoder_sweep.py`, `gnn_encoder_sweep.py`, and
`SMILES_encoder_sweep.py` train each encoder with a pIC50 head on the BACE-1 rows
of the shared split (6,469 / 808 / 810), the same molecules as `rf_bioactivity`.

```sh
python scripts/gnn_encoder_sweep.py 1 2                                  # exploratory stages (train/val only)
python scripts/gnn_encoder_sweep.py --benchmark --config chosen.json     # canonical seed-42 run, prints val/test
python scripts/gnn_encoder_sweep.py --benchmark --select-from <stage.csv> --write-results
```

- stage CSVs and logs go to `--output-dir` (default `outputs/encoders/<encoder>/sweep/`)
  and resume where they left off. Old files without matching benchmark metadata
  can't be resumed, so point new runs at a new directory
- `--select-from` picks the config with the lowest mean val RMSE over
  `--selection-seeds` (default `0 1 2`). Test is never used for selection
- `results.csv` is only written with `--write-results`. `--dry-run` checks the
  config without training
- `PRETRAINED_REVISION` in the SMILES script defaults to `DEFAULT_REVISION` (override
  with a 40-character MolFormer commit SHA via `--stage-config` or `--config`)

### MuFu

`mufu.benchmark.evaluate_mufu()` scores a trained MuFu on val/test and records it in
`results.csv` as `mufu_full`. It needs to be told how the model was trained, or it
only runs val diagnostics:

```python
training = {
    "trained": True,
    "training_steps": steps,
    "training_seed": 42,
    "selection_split": "val",
    "benchmark": benchmark_metadata(full),
}
evaluate_mufu(model, training=training, splits=("val", "test"), write_results=True)
```

Or pass `checkpoint=Path(...)`, a `torch.save` of
`{"model_state_dict": ..., "training": training}`.

## Testing

Tests use [`pytest`](https://docs.pytest.org). From anywhere inside this
directory, run:

```
pytest
```

`pytest.ini` points pytest at `tests/` and adds `src/` to the import path, so
tests can `import` modules from `src` directly.

Tests marked `network` call external services (ChEMBL and MoleculeNet), so
they're skipped by default. To run `network` tests, do:

```
pytest -m network
```

`tests/encoders/test_smiles_encoder.py` needs `transformers` to import. If
collection fails with `ModuleNotFoundError: transformers`, run
`conda env update -n CHM4930 -f environment.yml`.
