import numpy as np
from rdkit import Chem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from bioactivity_loader import load_bace1
from evaluation import evaluation_metrics
from featurize.fingerprint import mol_to_ecfp4

RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    df = load_bace1()

    fingerprints = []

    for smiles in df["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        fingerprints.append(mol_to_ecfp4(mol))

    X = np.stack(fingerprints)
    y = df["active"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]

    metrics = evaluation_metrics(
        y_test,
        y_score,
        task_type="classification",
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
