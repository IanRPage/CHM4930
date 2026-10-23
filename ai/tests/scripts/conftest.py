import pytest

import encoder_benchmark as eb
from baseline import results
from pipeline.combined_loader import COMBINED_CSV_PATH


@pytest.fixture(scope="session")
def frozen_bace1():
    before = {
        path: path.read_bytes() for path in (COMBINED_CSV_PATH, results.RESULTS_PATH)
    }
    loaded = eb.BACE1Benchmark.load()
    yield loaded
    for path, contents in before.items():
        assert path.read_bytes() == contents


@pytest.fixture
def bace1_data(snapshot_csv):
    return eb.BACE1Benchmark.load(snapshot_csv)
