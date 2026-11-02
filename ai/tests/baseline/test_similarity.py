import json

import numpy as np
import pandas as pd
import pytest

from baseline import benchmark, similarity


@pytest.fixture
def split_df():
    smiles = ["c1ccccc1O", "CCCCCCCC", "c1ccccc1O", "C1CCNCC1", "CCCCCCCO", "N#N"]
    return pd.DataFrame(
        {
            "smiles": smiles,
            "split": ["train", "train", "test", "test", "test", "test"],
            "pIC50_BACE1": [5.0, 7.0, 5.0, 7.0, 6.5, 4.0],
            "active_BACE1": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "pIC50_EGFR": np.nan,
            "active_EGFR": np.nan,
            **{task: np.nan for task in benchmark.TOX_TASKS},
        }
    )


def test_nearest_train_similarity_is_one_for_a_train_duplicate(split_df):
    sim = similarity.nearest_train_similarity(split_df)
    assert sim.shape == (4,)
    assert sim[0] == pytest.approx(1.0)
    assert (sim[1:] < 1.0).all()


def test_bucket_labels_close_the_last_bin():
    labels = similarity.bucket_labels(np.array([0.0, 0.29, 0.3, 0.55, 0.6, 1.0]))
    assert labels.tolist() == [
        "[0.0, 0.3)",
        "[0.0, 0.3)",
        "[0.3, 0.4)",
        "[0.5, 0.6)",
        "[0.6, 1.0]",
        "[0.6, 1.0]",
    ]


def test_similarity_table_scores_each_bucket_and_averages_seeds(split_df, tmp_path):
    run_dir = tmp_path / "m_default"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "m_default",
                "model": "m",
                "encoder": "enc",
                "split_id": "default",
                "seeds": [0, 1],
            }
        )
    )
    test = split_df[split_df["split"] == "test"]
    for seed, offset in ((0, 0.0), (1, 0.2)):
        benchmark.save_split_predictions(
            split_df,
            "test",
            {"pIC50_BACE1": test["pIC50_BACE1"].to_numpy() + offset},
            run_dir / "predictions" / f"test_seed_{seed}.csv",
        )

    table = similarity.similarity_table(split_df, [run_dir])

    buckets = similarity.bucket_labels(similarity.nearest_train_similarity(split_df))
    rmse = table[table["metric"] == "rmse"].set_index("bucket")
    assert set(rmse.index) == set(buckets)
    assert rmse["n_compounds"].sum() == 4
    # rmse is 0.0 and 0.2 on the two seeds in every bucket
    assert rmse["value"].tolist() == pytest.approx([0.1] * len(rmse))
    assert (rmse["n_seeds"] == 2).all()
    assert set(table["metric"]) <= {"rmse", "spearman", "roc_auc"}
