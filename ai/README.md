# CHM4930: AI

This is where all code relating to training our deep learning models will go,
along with any datasets or checkpoints for saved model weights.

## Directory Structure

```
ai/
├── data/            # local data storage
│   └── baseline/    # frozen combined.csv baselines train on, and their results.csv (tracked)
├── docs/            # design docs (MuFu-v1.md)
├── notebooks/       # exploration / scratchpads
├── scripts/         # helpful utility scripts
├── src/             # main code
│   ├── baseline/    # random forest baselines and their results recorder
│   ├── featurize/   # RDKit to SMILES, ECFP4 fingerprint, and PyG graph
│   ├── mufu/        # MuFu interface contract (names, shapes, defaults)
│   ├── pipeline/    # dataset loaders and structure to model input preprocessing
│   ├── encoders/    # where each of MuFu's encoders are implemented
│   └── check_env.py # sanity check that dependencies are installed correctly
├── tests/           # pytest suite (mirrors src/)
├── checkpoints/     # saved model weights
├── outputs/         # generated artifacts (figures, logs, metrics, etc.)
├── environment.yml  # conda environment config
├── pytest.ini       # pytest config (test paths, import path, markers)
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
python src/check_env.py
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
× endpoint × split × metric. Each row is tagged with the snapshot's SHA-256 and
the git commit it ran on. Current test set performance:

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

## Shared Benchmark Foundation

`src/benchmark.py` provides the common dataset, protocol metadata, seed setup,
and masked evaluation API. Load and split the full frozen combined table before
filtering for an endpoint. The canonical protocol keeps the existing RF split:
80/10/10, split seed `None`, training seed `42`, and active labels at pIC50 ≥ 6.0.

```python
from benchmark import (
    benchmark_metadata,
    evaluate_predictions,
    load_benchmark,
    set_training_seed,
    verify_benchmark_metadata,
)

full = load_benchmark()
metadata = benchmark_metadata(full)
generator = set_training_seed()  # pass to a shuffled DataLoader
endpoint = full.loc[full["pIC50_BACE1"].notna()]
# evaluate_predictions(labels, aligned_scores, "regression")
# verify_benchmark_metadata(metadata, full) before explicit results export
```

Metadata includes the dataset checksum and a checksum of sorted SMILES/split
assignments, independent of row order and DataFrame indexes. Keep it in result
`params`; `benchmark.record_results` re-exports the existing CSV recorder and
does not write automatically. NumPy/Torch inputs are positional; pandas inputs
must also have matching row indexes and task columns. Missing labels are masked
automatically; zero-filled MuFu labels require their explicit label masks.

Dataset checksums normalize CRLF to LF in memory, preserving the historical
checksum without rewriting the snapshot. `.gitattributes` enforces LF for future
checkouts of `combined.csv`. Its Git index already contains LF, so no
renormalization is needed for this change. Do not rebuild the dataset or rewrite
existing results to correct a checkout's line endings.

## Encoder Benchmark Runs

Both encoder sweep scripts use `src/encoder_benchmark.py` on top of the shared
foundation. They split the entire frozen combined table first, then select
observed BACE-1 labels: 6,469 train, 808 validation, and 810 test molecules, exactly
as in the BACE-1 RF baseline. Featurization failures abort rather than silently
changing that cohort. Evaluation uses the original labels and shared metrics.

Exploratory stages keep their grids, seeds, wide CSV columns, and log paths:

```sh
python scripts/ECFP4_encoder_sweep.py 3 4 --output-dir outputs/encoders/ECFP4/shared-v1
python scripts/gnn_encoder_sweep.py 1 --output-dir outputs/encoders/GNN/shared-v1
```

These are noncanonical experiments. Only train/validation metrics are computed;
the existing `test_*` columns contain NaN. Each row adds dataset/split/protocol
metadata. Resume and automatic selection reject legacy files or mismatched
metadata. Existing CSVs/logs are preserved; use a new `--output-dir` for these
experiments instead of deleting old results. Run separate stages concurrently
if useful, but use one writer per stage file or `results.csv`.

A canonical run requires either an explicit configuration JSON (training keys
from the script's `DEFAULT`, with omitted keys retaining those defaults) or
compatible stage files. For example, after choosing a configuration using
validation only, put `{"HIDDEN_DIMS": [128, 128], "EPOCHS": 100}` in `chosen.json`:

```sh
python scripts/gnn_encoder_sweep.py --benchmark --config chosen.json --dry-run
python scripts/gnn_encoder_sweep.py --benchmark --config chosen.json
python scripts/gnn_encoder_sweep.py --benchmark --config chosen.json --write-results
```

The ECFP4 script accepts the same flags. Alternatively:

```sh
python scripts/ECFP4_encoder_sweep.py --benchmark --select-from outputs/encoders/ECFP4/shared-v1/3-training-dynamics.csv --selection-seeds 0 1 2 --write-results
```

Automatic selection minimizes mean validation RMSE over a complete, matching
seed set; it never reads test metrics for selection. The selected configuration
is retrained with seed 42, its best validation checkpoint is restored, and only
then is test evaluated. The scripts retain their optimizers, losses, epoch
limits, and GNN patience. Avoid repeatedly tuning against canonical test results.

Without `--write-results`, canonical metrics are displayed without exporting.
`--dry-run` loads/verifies metadata and configuration without featurization,
training, or writes, even when combined with `--write-results`. Explicit export
uses the existing recorder and unchanged 13-column `data/baseline/results.csv`
schema: six validation/test rows per encoder (`rmse`, `r2`, `roc_auc`), tagged as
`ecfp4_encoder` or `gnn_encoder`. Protocol, configuration, seed, selected epoch,
selection method, device, and determinism setting are stored in `params`.
Re-export replaces only that encoder's rows on the current snapshot. Export
refuses a snapshot mismatch that would discard existing RF results.

### Standalone SMILES Encoder

`scripts/SMILES_encoder_sweep.py` evaluates the existing MolFormer wrapper with
frozen pretrained weights and a trainable linear pIC50 head. It uses the same
6,469/808/810 BACE-1 cohort, seed-42 canonical run, validation checkpoint selection,
metrics, and results recorder as the other standalone encoders. It does not run
the full MuFu model.

Supply an immutable 40-character pretrained commit SHA in `pretrained.json`:

```json
{
  "MODEL_NAME": "ibm-research/MoLFormer-XL-both-10pct",
  "PRETRAINED_REVISION": "<40-character commit SHA from the model repository>",
  "FREEZE": true
}
```

The placeholder must be replaced with a real revision; a moving branch such as
`main` is rejected. The same revision is passed to the model and tokenizer. The
requested/resolved revision, model identity, freezing policy, selected epoch,
training configuration, seed, and benchmark metadata are retained in result
parameters. The wrapper's existing unpinned behavior remains available to other
callers that omit its new `revision` argument.

```sh
python scripts/SMILES_encoder_sweep.py --stage-config pretrained.json
python scripts/SMILES_encoder_sweep.py --benchmark --select-from outputs/encoders/SMILES/sweep/1-head.csv --write-results
python scripts/SMILES_encoder_sweep.py --benchmark --config pretrained.json --dry-run
```

The exploratory stage compares three head learning rates on seeds 0, 1, and 2
using validation only. `--stage-config` supplies overrides to those configurations;
it is separate from canonical `--config`. Canonical selection uses a complete
matching seed set, retrains the chosen head with seed 42, and evaluates test only
after restoring the best validation checkpoint. The encoder remains frozen and
in evaluation mode. Only training labels update the head; frozen train/validation
embeddings are reused, and test embeddings are requested after selection.

Results use model identifier `smiles_encoder` and the existing six-row
validation/test format. As with the other encoders, export requires
`--write-results`; dry runs load neither Transformers nor pretrained weights and
never export. Actual execution requires the declared Transformers dependency and
the pinned pretrained model/tokenizer, either locally available or obtainable
from their model repository. Offline tests replace pretrained loading with small
local mocks and do not download weights.

The GNN runner merges a final singleton training batch into its preceding batch
to satisfy BatchNorm. Normal batches keep their existing order and sizes, and no
training examples are dropped. Evaluation batching is unchanged and retains all
samples, including singleton final batches.

## Full MuFu Benchmark Evaluation

`mufu.benchmark.evaluate_mufu()` is an inference-only adapter. SCRUM-60 owns model
construction, training, and validation-based checkpoint selection. Pass the
selected model on its evaluation device; the adapter does not construct encoders,
download weights, train, select checkpoints, or move the model between devices.

After training, invoke it with the actual training provenance. `training_metadata`
below must be the benchmark metadata retained from the full frozen table used for
training, and `global_step` must be the actual number of completed training steps:

```python
from mufu.benchmark import evaluate_mufu

training = {
    "trained": True,
    "training_steps": global_step,
    "training_seed": 42,
    "selection_split": "val",
    "benchmark": training_metadata,
    # Optional: run ID, selected epoch/criterion, encoder configuration/revisions.
}
report = evaluate_mufu(
    selected_model,
    training=training,
    splits=("val", "test"),
    batch_size=64,
    write_results=True,
)
```

No trainer changes are needed beyond invoking the adapter with this information.
The adapter requires explicit training attestation, positive training steps, seed
42, validation selection, and matching dataset/split/protocol metadata for test
evaluation or export. This guards against accidentally reporting random initial
weights; it cannot prove a caller's training-history declaration. Do not fabricate
provenance for untrained or synthetic models.

For a saved checkpoint, construct a compatible MuFu model first. The optional
checkpoint format contains tensors and primitive metadata:

```python
from pathlib import Path
import torch

torch.save(
    {"model_state_dict": selected_model.state_dict(), "training": training},
    "selected.pt",
)
report = evaluate_mufu(
    constructed_model,
    checkpoint=Path("selected.pt"),
    splits=("val", "test"),
    write_results=True,
)
```

Checkpoint loading uses `weights_only=True`, validates provenance plus exact
state keys/shapes/dtypes and finite tensor values, and loads weights in-place.
Pass either checkpoint metadata or `training`, not both. Existing checkpoint
formats can instead be loaded by SCRUM-60 and passed through the in-memory model
interface. Checkpoint identity includes a SHA-256 of the evaluated state and, for
file loading, a raw file checksum.

The default call `evaluate_mufu(model)` returns validation diagnostics only, marked
`training_attested=False`. Without valid provenance, test evaluation and export
are refused. Results are never written by default. Export requires **both** val
and test, producing all 50 rows in one existing-recorder call under `mufu_full`:

- BACE-1/EGFR pIC50: RMSE, R2, and pIC50-derived activity ROC-AUC.
- `active_BACE1`/`active_EGFR`: separate classification-head ROC-AUC.
- Fourteen toxicity endpoints and overall/Tox21/ClinTox macro ROC-AUC groups.

Continuous logits score classification heads; pIC50-derived ROC-AUC retains its
existing `pIC50_*` identity. Endpoint counts use observed labels, undefined metrics
remain NaN, and macro counts follow the RF convention. The 13-column results
schema is unchanged. Parameters record score source, training provenance,
checkpoint identity, benchmark metadata, and evaluation configuration. Re-export
replaces the complete `mufu_full` result set while preserving compatible RF and
standalone encoder rows; incompatible RF snapshot checks abort before export.

Evaluation splits the full frozen table before building partition datasets. It
rejects molecule loss/reordering or label/mask mismatches, runs sequential batches
without dropping samples or using training samplers, and calls the model with only
the three `MuFuInputs` fields. All modules enter evaluation mode and gradients are
disabled, so normal MuFu modality dropout is off. Previous component modes are
restored, and the evaluation loader uses a private generator without reseeding
training RNGs. No test labels are used to select weights.

`report.rows` contains metric rows; `report.predictions[split][output]` contains
output DataFrames indexed by unique SMILES in evaluation order, with columns from
the MuFu contract. Mocked offline tests exercise this workflow without pretrained
downloads or publishing synthetic measurements. Real performance requires the
selected trained weights supplied by SCRUM-60.

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

`tests/encoders/test_smiles_encoder.py` imports `transformers` during collection, even
though its model downloads are mocked. If collection fails with
`ModuleNotFoundError: transformers`, the local environment is incomplete:
`environment.yml` already declares the pinned dependency in its `pip` section.
From `ai/`, synchronize the project environment with
`conda env update -n CHM4930 -f environment.yml`, or create it as described above.
Use `python src/check_env.py` to check installed versions. Encoder benchmark tests
run offline without Transformers; excluding that module lets the remaining
offline suite run while the environment is repaired.
