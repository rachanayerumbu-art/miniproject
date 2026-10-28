
from pathlib import Path

import pandas as pd
import numpy as np

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_DIR / "MERGED_CSV"
OUTPUT_DIR = PROJECT_DIR / "CLIENT_DATA"

NUMBER_OF_CLIENTS = 5
RANDOM_SEED = 42

# --------------------------------------------------
# STEP 1: LOAD ALL 20 DATASETS
# --------------------------------------------------

def main():
    csv_files = sorted(INPUT_DIR.glob("client_*.csv"))

    if len(csv_files) != 20:
        raise FileNotFoundError(
            f"Expected 20 CSV files in {INPUT_DIR}, "
            f"but found {len(csv_files)}."
        )

    dataframes = []

    for file in csv_files:
        print(f"Loading: {file.name}")
        df = pd.read_csv(file)

        if "Label" not in df.columns:
            raise ValueError(
                f"'Label' column missing in {file.name}"
            )

        dataframes.append(df)

    # Combine all records into one dataset.
    data = pd.concat(dataframes, ignore_index=True)

    print("\nAll datasets combined.")
    print("Total records:", len(data))
    print("Total features including Label:", len(data.columns))

    # --------------------------------------------------
    # STEP 2: CLEAN AND STANDARDIZE LABELS
    # --------------------------------------------------

    data["Label"] = (
        data["Label"].astype(str).str.strip().str.lower()
    )

    label_mapping = {
        "benign": 0,
        "normal": 0,
        "ddos": 1,
    }

    unknown_labels = set(data["Label"].unique()) - set(label_mapping)

    if unknown_labels:
        raise ValueError(
            f"Unexpected labels found: {unknown_labels}. "
            "Check the dataset before distributing."
        )

    # 0 = Normal traffic; 1 = DDoS traffic
    data["Label"] = data["Label"].map(label_mapping).astype("int8")

    print("\nLabel distribution:")
    print(data["Label"].value_counts().sort_index().rename(
        index={0: "Normal", 1: "DDoS"}
    ))

    # --------------------------------------------------
    # STEP 3: SHUFFLE AND DISTRIBUTE RECORDS
    # --------------------------------------------------

    rng = np.random.default_rng(RANDOM_SEED)

    # Create the output directory outside MERGED_CSV.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Distribute each class separately so that every client
    # receives approximately the same class proportions.
    client_parts = [[] for _ in range(NUMBER_OF_CLIENTS)]

    for label in sorted(data["Label"].unique()):
        class_data = data[data["Label"] == label]

        # Shuffle the rows of this class.
        class_data = class_data.sample(
            frac=1,
            random_state=RANDOM_SEED
        ).reset_index(drop=True)

        # Split the DataFrame using row positions.
        split_positions = np.array_split(
            np.arange(len(class_data)),
            NUMBER_OF_CLIENTS
        )

        for client_index, positions in enumerate(split_positions):
            partition = class_data.iloc[positions].copy()
            client_parts[client_index].append(partition)

    # --------------------------------------------------
    # STEP 4: SAVE FIVE CLIENT DATASETS
    # --------------------------------------------------

    print("\nCreating five IoT client datasets...")

    for i, parts in enumerate(client_parts, start=1):
        client_data = pd.concat(parts, ignore_index=True)

        # Shuffle normal and DDoS rows together within each client.
        client_data = client_data.sample(
            frac=1,
            random_state=RANDOM_SEED
        ).reset_index(drop=True)

        output_file = OUTPUT_DIR / f"iot_client_{i}.csv"
        client_data.to_csv(output_file, index=False)

        normal_count = int((client_data["Label"] == 0).sum())
        ddos_count = int((client_data["Label"] == 1).sum())

        print(f"\nClient {i}")
        print("  Total records:", len(client_data))
        print("  Normal records:", normal_count)
        print("  DDoS records:", ddos_count)
        print("  Saved to:", output_file)

    # --------------------------------------------------
    # STEP 5: VERIFY DISTRIBUTION
    # --------------------------------------------------

    total_distributed = sum(
        len(pd.read_csv(OUTPUT_DIR / f"iot_client_{i}.csv"))
        for i in range(1, NUMBER_OF_CLIENTS + 1)
    )

    print("\n-----------------------------------")
    print("DATA DISTRIBUTION COMPLETED")
    print("-----------------------------------")
    print("Original combined records:", len(data))
    print("Distributed records:", total_distributed)
    print("Number of IoT clients:", NUMBER_OF_CLIENTS)
    print("Output folder:", OUTPUT_DIR)

    if total_distributed != len(data):
        raise RuntimeError("Record count mismatch!")

    print("Verification successful: all records distributed.")


if __name__ == "__main__":
    main()
