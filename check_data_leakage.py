
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = PROJECT_DIR / "data" / "training_pool.csv"
TEST_FILE = PROJECT_DIR / "data" / "independent_test.csv"
LABEL = "Label"


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if LABEL not in df.columns:
        raise ValueError(f"Missing {LABEL} column in {path.name}")

    return df


def main():
    train = load_data(TRAIN_FILE)
    test = load_data(TEST_FILE)

    if list(train.columns) != list(test.columns):
        raise ValueError("Training and test columns do not match.")

    print("Training rows:", len(train))
    print("Test rows:", len(test))

    # Check exact duplicate rows across the two datasets.
    train_rows = set(
        pd.util.hash_pandas_object(train, index=False).tolist()
    )
    test_hashes = pd.util.hash_pandas_object(test, index=False)

    overlap = int(test_hashes.isin(train_rows).sum())

    print("\nExact duplicate rows shared by training and test:", overlap)

    # Check feature columns that have only one unique value.
    features = train.drop(columns=[LABEL])
    constant_columns = features.columns[
        features.nunique(dropna=False) <= 1
    ].tolist()

    print("\nConstant feature columns:", len(constant_columns))
    if constant_columns:
        print(constant_columns)

    # Display labels and class distributions.
    print("\nTraining class counts:")
    print(train[LABEL].value_counts().sort_index())

    print("\nTest class counts:")
    print(test[LABEL].value_counts().sort_index())

    print("\nCheck complete.")
    print(
        "Note: zero exact duplicates does not rule out related flows, "
        "label leakage, or capture/session overlap."
    )


if __name__ == "__main__":
    main()
