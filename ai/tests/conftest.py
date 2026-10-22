import random

import numpy as np
import pandas as pd
import pytest
import torch

import benchmark as bm
from baseline import results
from encoders import benchmark as eb
from pipeline.combined_loader import COMBINED_CSV_PATH

CYCLOALKANE_SMILES = ["C1" + "C" * (n - 1) + "1" for n in range(3, 23)]


@pytest.fixture
def cycloalkane_smiles():
    return list(CYCLOALKANE_SMILES)


@pytest.fixture(autouse=True)
def isolate_benchmark_runtime(monkeypatch):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)

    record = bm.record_results

    def guarded_record(*args, **kwargs):
        assert (
            kwargs.get("path", results.RESULTS_PATH).resolve()
            != results.RESULTS_PATH.resolve()
        )
        return record(*args, **kwargs)

    monkeypatch.setattr(bm, "record_results", guarded_record)
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.set_num_threads(threads)


@pytest.fixture(scope="session")
def frozen_bace1():
    before = {
        path: path.read_bytes() for path in (COMBINED_CSV_PATH, results.RESULTS_PATH)
    }
    loaded = eb.BACE1Benchmark.load()
    yield loaded
    for path, contents in before.items():
        assert path.read_bytes() == contents


@pytest.fixture(scope="session")
def frozen_benchmark(frozen_bace1):
    return frozen_bace1.full


@pytest.fixture
def snapshot_csv(tmp_path):
    df = pd.DataFrame(
        {
            "smiles": CYCLOALKANE_SMILES,
            "pIC50_BACE1": [
                5.0 + (i % 4) if i < 8 or i >= 16 else np.nan for i in range(20)
            ],
            "pIC50_EGFR": [6.0 + i / 10 for i in range(20)],
            "in_bace1": True,
            "in_egfr": True,
        }
    )
    path = tmp_path / "combined.csv"
    df.to_csv(path, index=False, lineterminator="\n")
    return path


@pytest.fixture
def bace1_data(snapshot_csv):
    return eb.BACE1Benchmark.load(snapshot_csv)
