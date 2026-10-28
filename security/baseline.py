
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_FILE = PROJECT_DIR / "data" / "training_pool.csv"
TEST_FILE = PROJECT_DIR / "data" / "independent_test.csv"
LABEL_COLUMN = "Label"
RANDOM_SEED = 42


def load_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing '{LABEL_COLUMN}' in {file_path.name}")

    y = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")

    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError(f"Invalid labels in {file_path.name}")

    X = df.drop(columns=[LABEL_COLUMN]).apply(
        pd.to_numeric, errors="coerce"
    )
    X = X.replace([np.inf, -np.inf], np.nan)

    valid_rows = X.notna().all(axis=1)
    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].astype(int).reset_index(drop=True)

    if X.empty:
        raise ValueError(f"No valid feature rows in {file_path.name}")

    return X, y


def main():
    X_train, y_train = load_data(TRAIN_FILE)
    X_test, y_test = load_data(TEST_FILE)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and test feature columns do not match."
        )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )

    print("Training centralized Random Forest model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nCentralized Baseline Results")
    print("----------------------------")
    print("Training records:", len(X_train))
    print("Independent test records:", len(X_test))
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(
        f"Precision: "
        f"{precision_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:    "
        f"{recall_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        f"F1-score:  "
        f"{f1_score(y_test, predictions, zero_division=0):.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "DDoS"],
            zero_division=0,
        )
    )

    print("Confusion Matrix (actual rows, predicted columns):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()
