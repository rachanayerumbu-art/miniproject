
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "MERGED_CSV"
OUTPUT_DIR = PROJECT_DIR / "data"

LABEL_COLUMN = "Label"
TEST_SIZE = 0.20
RANDOM_SEED = 42


def main():
    csv_files = sorted(INPUT_DIR.glob("client_*.csv"))

    if len(csv_files) != 20:
        raise ValueError(
            f"Expected 20 CSV files in {INPUT_DIR}, "
            f"but found {len(csv_files)}."
        )

    frames = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        if LABEL_COLUMN not in df.columns:
            raise ValueError(
                f"Missing '{LABEL_COLUMN}' in {file_path.name}"
            )

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    labels = data[LABEL_COLUMN].astype(str).str.strip().str.lower()

    label_mapping = {
        "benign": 0,
        "normal": 0,
        "0": 0,
        "ddos": 1,
        "1": 1,
    }

    mapped_labels = labels.map(label_mapping)

    if mapped_labels.isna().any():
        unknown = sorted(labels[mapped_labels.isna()].unique())
        raise ValueError(f"Unknown labels found: {unknown}")

    data[LABEL_COLUMN] = mapped_labels.astype(int)

    data = data.drop_duplicates().reset_index(drop=True)

    if set(data[LABEL_COLUMN].unique()) != {0, 1}:
        raise ValueError("Dataset must contain both classes.")

    train_pool, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=data[LABEL_COLUMN],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_pool.to_csv(
        OUTPUT_DIR / "training_pool.csv",
        index=False,
    )

    test_data.to_csv(
        OUTPUT_DIR / "independent_test.csv",
        index=False,
    )

    print("Training records:", len(train_pool))
    print("Test records:", len(test_data))

    print("\nTraining class counts:")
    print(train_pool[LABEL_COLUMN].value_counts().sort_index())

    print("\nTest class counts:")
    print(test_data[LABEL_COLUMN].value_counts().sort_index())

    print("\nDatasets saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
