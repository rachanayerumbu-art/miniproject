
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "data" / "training_pool.csv"
OUTPUT_DIR = PROJECT_DIR / "CLIENT_DATA"

RANDOM_SEED = 42
LABEL_COLUMN = "Label"

# Relative distribution: (Normal, DDoS)
CLIENT_DISTRIBUTION = {
    1: (90, 10),
    2: (75, 25),
    3: (50, 50),
    4: (25, 75),
    5: (10, 90),
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    if LABEL_COLUMN not in data.columns:
        raise ValueError(
            f"Missing '{LABEL_COLUMN}' column."
        )

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

    normal_data = data[data[LABEL_COLUMN] == 0].sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    ddos_data = data[data[LABEL_COLUMN] == 1].sample(
        frac=1, random_state=RANDOM_SEED + 1
    ).reset_index(drop=True)

    # Calculate class counts using the available training data.
    normal_weights = {
        client: values[0]
        for client, values in CLIENT_DISTRIBUTION.items()
    }
    ddos_weights = {
        client: values[1]
        for client, values in CLIENT_DISTRIBUTION.items()
    }

    def allocate(total, weights):
        weight_sum = sum(weights.values())
        exact = {
            client: total * weight / weight_sum
            for client, weight in weights.items()
        }
        counts = {
            client: int(value)
            for client, value in exact.items()
        }

        remainder = total - sum(counts.values())
        order = sorted(
            exact,
            key=lambda client: exact[client] - counts[client],
            reverse=True,
        )

        for client in order[:remainder]:
            counts[client] += 1

        return counts

    normal_counts = allocate(len(normal_data), normal_weights)
    ddos_counts = allocate(len(ddos_data), ddos_weights)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    normal_start = 0
    ddos_start = 0
    total_written = 0

    for client_id in CLIENT_DISTRIBUTION:
        normal_count = normal_counts[client_id]
        ddos_count = ddos_counts[client_id]

        client_normal = normal_data.iloc[
            normal_start:normal_start + normal_count
        ]
        client_ddos = ddos_data.iloc[
            ddos_start:ddos_start + ddos_count
        ]

        client_data = pd.concat(
            [client_normal, client_ddos],
            ignore_index=True,
        ).sample(
            frac=1,
            random_state=RANDOM_SEED + client_id,
        ).reset_index(drop=True)

        output_file = OUTPUT_DIR / f"iot_client_{client_id}.csv"
        client_data.to_csv(output_file, index=False)

        actual_normal = int(
            (client_data[LABEL_COLUMN] == 0).sum()
        )
        actual_ddos = int(
            (client_data[LABEL_COLUMN] == 1).sum()
        )

        print(f"\nClient {client_id}")
        print(f"  Normal: {actual_normal}")
        print(f"  DDoS:   {actual_ddos}")
        print(f"  Total:  {len(client_data)}")
        print(f"  Saved:  {output_file}")

        normal_start += normal_count
        ddos_start += ddos_count
        total_written += len(client_data)

    print("\nDistribution completed.")
    print("Training records:", len(data))
    print("Records distributed:", total_written)
    print("Normal records:", len(normal_data))
    print("DDoS records:", len(ddos_data))
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
