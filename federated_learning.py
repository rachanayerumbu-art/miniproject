
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from energy_config import CLIENT_CONFIG
from energy_aware_selection import select_clients
from energy_simulator import (
    charge_clients,
    create_energy_state,
)


PROJECT_DIR = Path(__file__).resolve().parent
CLIENT_DIR = PROJECT_DIR / "CLIENT_DATA"
TEST_FILE = PROJECT_DIR / "data" / "independent_test.csv"
ENERGY_LOG_FILE = PROJECT_DIR / "energy_federated_rounds.csv"
ENERGY_LOG_FIELDS = [
    "Round",
    "Client",
    "Participated",
    "Energy_Before",
    "Energy_Used",
    "Energy_Remaining",
]

NUMBER_OF_CLIENTS = 5
ROUNDS = 5
LOCAL_EPOCHS = 3
RANDOM_SEED = 42
LABEL_COLUMN = "Label"


def build_energy_round_rows(
    round_number,
    energy_before,
    energy_state,
    selected_clients,
    energy_used,
):
    """Create one energy and participation record for each client."""
    return [
        {
            "Round": round_number,
            "Client": client_id,
            "Participated": client_id in selected_clients,
            "Energy_Before": energy_before[client_id],
            "Energy_Used": energy_used.get(client_id, 0.0),
            "Energy_Remaining": energy_state[client_id],
        }
        for client_id in CLIENT_CONFIG
    ]


def save_energy_round_log(rows, output_path):
    """Write a new log file without replacing an earlier experiment."""
    with output_path.open("x", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=ENERGY_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_dataset(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing '{LABEL_COLUMN}' in {file_path.name}")

    labels = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")

    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(
            f"{file_path.name} must contain only labels 0 and 1."
        )

    features = df.drop(columns=[LABEL_COLUMN]).apply(
        pd.to_numeric, errors="coerce"
    )
    features = features.replace([np.inf, -np.inf], np.nan)

    valid_rows = features.notna().all(axis=1)
    features = features.loc[valid_rows].reset_index(drop=True)
    labels = labels.loc[valid_rows].astype(int).reset_index(drop=True)

    if features.empty:
        raise ValueError(f"No valid rows in {file_path.name}")

    if features.shape[1] == 0:
        raise ValueError(f"No feature columns in {file_path.name}")

    return features, labels


def average_client_models(client_models, client_sizes):
    if not client_models or len(client_models) != len(client_sizes):
        raise ValueError("Client models and sizes must match.")

    total_size = sum(client_sizes)

    if total_size <= 0:
        raise ValueError("Total client training size must be positive.")

    coefficients = sum(
        model.coef_ * size
        for model, size in zip(client_models, client_sizes)
    ) / total_size

    intercepts = sum(
        model.intercept_ * size
        for model, size in zip(client_models, client_sizes)
    ) / total_size

    return coefficients, intercepts


def prepare_data():
    client_frames = []
    feature_columns = None

    for client_id in range(1, NUMBER_OF_CLIENTS + 1):
        client_file = CLIENT_DIR / f"iot_client_{client_id}.csv"
        X, y = load_dataset(client_file)

        if feature_columns is None:
            feature_columns = list(X.columns)
        elif list(X.columns) != feature_columns:
            raise ValueError(
                f"Feature columns or order differ in "
                f"{client_file.name}."
            )

        if set(y.unique()) != {0, 1}:
            raise ValueError(
                f"{client_file.name} must contain both classes."
            )

        client_frames.append((X, y))

    X_test, y_test = load_dataset(TEST_FILE)

    if list(X_test.columns) != feature_columns:
        raise ValueError(
            "Independent test feature columns or order do not match "
            "the client training data."
        )

    # Fit preprocessing on training data only.
    X_all_train = pd.concat(
        [X for X, _ in client_frames],
        ignore_index=True,
    )

    scaler = StandardScaler()
    scaler.fit(X_all_train)

    scaled_clients = []

    for X, y in client_frames:
        scaled_clients.append(
            (scaler.transform(X), y.to_numpy(dtype=int))
        )

    X_test_scaled = scaler.transform(X_test)

    print("Training records:", len(X_all_train))
    print("Independent test records:", len(X_test))
    print("Scaler fitted on training records only.")

    print("\nIndependent test class counts:")
    print(y_test.value_counts().sort_index().rename(
        index={0: "Normal", 1: "DDoS"}
    ))

    return (
        scaled_clients,
        X_test_scaled,
        y_test.to_numpy(dtype=int),
    )


ROUND_METRIC_FIELDS = [
    "Strategy",
    "Round",
    "Participating_Clients",
    "Energy_Used",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "False_Positive_Rate",
]
PARTICIPATION_FIELDS = ["Strategy"] + ENERGY_LOG_FIELDS
COMPARISON_SUMMARY_FIELDS = [
    "Strategy",
    "Communication_Rounds",
    "Total_Participations",
    "Total_Simulated_Energy_Used",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "False_Positive_Rate",
]


def calculate_metrics(y_true, predictions):
    """Calculate binary classification metrics with label 1 as positive."""
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )
    false_positive_rate = (
        false_positive / (false_positive + true_negative)
        if false_positive + true_negative
        else 0.0
    )

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1_Score": f1_score(y_true, predictions, zero_division=0),
        "False_Positive_Rate": false_positive_rate,
    }


def run_federated_experiment(
    client_data,
    X_test,
    y_test,
    energy_aware,
    track_energy,
):
    """Run one policy from the same deterministic initial model."""
    first_X, first_y = client_data[0]
    global_model = SGDClassifier(
        loss="log_loss",
        random_state=RANDOM_SEED,
        max_iter=1000,
        tol=1e-3,
    )
    initial_size = min(100, len(first_X))
    global_model.partial_fit(
        first_X[:initial_size],
        first_y[:initial_size],
        classes=np.array([0, 1]),
    )

    strategy = (
        "Energy-aware FedAvg" if energy_aware else "Standard FedAvg"
    )
    energy_state = create_energy_state() if track_energy else None
    client_numbers = {
        f"client{number}": number
        for number in range(1, len(client_data) + 1)
    }
    if track_energy and set(client_numbers) != set(CLIENT_CONFIG):
        raise ValueError(
            "Energy configuration must match the loaded training clients."
        )

    participation_rows = []
    round_metrics = []
    print(f"\nStarting {strategy}...")

    for round_no in range(1, ROUNDS + 1):
        client_models = []
        client_sizes = []
        energy_used = {}
        energy_before = energy_state.copy() if track_energy else {}

        if energy_aware:
            selected_clients = select_clients(energy_state)
            selected_client_ids = [
                client_numbers[client_id]
                for client_id in selected_clients
            ]
        else:
            selected_client_ids = list(range(1, len(client_data) + 1))
            selected_clients = [
                f"client{client_id}"
                for client_id in selected_client_ids
            ]

        print(f"\nRound {round_no}/{ROUNDS}")
        print("  Participating clients:", selected_client_ids)

        if not selected_client_ids:
            print("  No clients meet the energy threshold; stopping.")
            if track_energy:
                participation_rows.extend(
                    build_energy_round_rows(
                        round_no,
                        energy_before,
                        energy_state,
                        selected_clients,
                        energy_used,
                    )
                )
            break

        for client_id in selected_client_ids:
            X_client, y_client = client_data[client_id - 1]
            local_model = SGDClassifier(
                loss="log_loss",
                random_state=RANDOM_SEED + client_id,
            )
            local_model.coef_ = global_model.coef_.copy()
            local_model.intercept_ = global_model.intercept_.copy()
            local_model.classes_ = global_model.classes_.copy()
            local_model.n_features_in_ = global_model.n_features_in_

            rng = np.random.default_rng(
                RANDOM_SEED + round_no * 100 + client_id
            )
            for _ in range(LOCAL_EPOCHS):
                indices = rng.permutation(len(X_client))
                local_model.partial_fit(
                    X_client[indices],
                    y_client[indices],
                )

            client_models.append(local_model)
            client_sizes.append(len(X_client))

            if track_energy:
                energy_used.update(
                    charge_clients(
                        [f"client{client_id}"],
                        energy_state,
                    )
                )

        if track_energy:
            participation_rows.extend(
                build_energy_round_rows(
                    round_no,
                    energy_before,
                    energy_state,
                    selected_clients,
                    energy_used,
                )
            )

        global_model.coef_, global_model.intercept_ = (
            average_client_models(client_models, client_sizes)
        )
        predictions = global_model.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)
        round_metrics.append(
            {
                "Strategy": strategy,
                "Round": round_no,
                "Participating_Clients": len(selected_client_ids),
                "Energy_Used": sum(energy_used.values()),
                **metrics,
            }
        )

        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1-score:  {metrics['F1_Score']:.4f}")
        print(f"  FPR:       {metrics['False_Positive_Rate']:.4f}")

    final_predictions = global_model.predict(X_test)
    final_metrics = calculate_metrics(y_test, final_predictions)

    print(f"\nFinal {strategy} test results")
    for metric_name, metric_value in final_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, final_predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            final_predictions,
            labels=[0, 1],
            target_names=["Normal", "DDoS"],
            zero_division=0,
        )
    )

    return {
        "strategy": strategy,
        "participation_rows": participation_rows,
        "round_metrics": round_metrics,
        "final_metrics": final_metrics,
    }


def write_new_csv(output_path, fieldnames, rows):
    """Create a CSV without replacing an earlier result."""
    with output_path.open("x", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_results(results, results_dir):
    """Save participation, round metrics, and final policy summaries."""
    results_dir.mkdir(parents=True, exist_ok=False)

    participation_rows = []
    round_metrics = []
    summary_rows = []

    for result in results:
        strategy = result["strategy"]
        participation_rows.extend(
            {"Strategy": strategy, **row}
            for row in result["participation_rows"]
        )
        round_metrics.extend(result["round_metrics"])

        total_energy = sum(
            row["Energy_Used"] for row in result["participation_rows"]
        )
        total_participations = sum(
            row["Participated"] for row in result["participation_rows"]
        )
        summary_rows.append(
            {
                "Strategy": strategy,
                "Communication_Rounds": len(result["round_metrics"]),
                "Total_Participations": total_participations,
                "Total_Simulated_Energy_Used": total_energy,
                **result["final_metrics"],
            }
        )

    write_new_csv(
        results_dir / "participation.csv",
        PARTICIPATION_FIELDS,
        participation_rows,
    )
    write_new_csv(
        results_dir / "round_metrics.csv",
        ROUND_METRIC_FIELDS,
        round_metrics,
    )
    write_new_csv(
        results_dir / "comparison_summary.csv",
        COMPARISON_SUMMARY_FIELDS,
        summary_rows,
    )
    print("\nComparison summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("Comparison files saved in:", results_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare standard and energy-aware FedAvg."
    )
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument(
        "--energy-aware",
        action="store_true",
        help="Train using simulated energy-aware client selection.",
    )
    policy_group.add_argument(
        "--compare-energy",
        action="store_true",
        help="Run standard and energy-aware FedAvg with matched inputs.",
    )
    parser.add_argument(
        "--energy-log",
        type=Path,
        default=ENERGY_LOG_FILE,
        help=(
            "New CSV path for an energy-aware run "
            f"(default: {ENERGY_LOG_FILE.name})."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_DIR / "week3_energy_comparison",
        help="New directory for comparison CSVs.",
    )
    args = parser.parse_args()

    energy_log_path = args.energy_log.resolve()
    results_dir = args.results_dir.resolve()

    if args.energy_aware and energy_log_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing energy log: {energy_log_path}. "
            "Choose a new path with --energy-log."
        )
    if args.compare_energy and results_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results directory: {results_dir}. "
            "Choose a new path with --results-dir."
        )

    print("Loading client training data and independent test data...")
    client_data, X_test, y_test = prepare_data()

    print("\nClient training distributions:")
    for client_id, (_, y_client) in enumerate(client_data, start=1):
        print(
            f"Client {client_id}: "
            f"Normal={(y_client == 0).sum()}, "
            f"DDoS={(y_client == 1).sum()}, "
            f"Total={len(y_client)}"
        )

    if args.compare_energy:
        results = [
            run_federated_experiment(
                client_data,
                X_test,
                y_test,
                energy_aware=False,
                track_energy=True,
            ),
            run_federated_experiment(
                client_data,
                X_test,
                y_test,
                energy_aware=True,
                track_energy=True,
            ),
        ]
        save_comparison_results(results, results_dir)
        return

    result = run_federated_experiment(
        client_data,
        X_test,
        y_test,
        energy_aware=args.energy_aware,
        track_energy=args.energy_aware,
    )

    print("\nFederated learning completed.")
    print("Training source: CLIENT_DATA")
    print("Evaluation source: data/independent_test.csv")
    print("Clients:", NUMBER_OF_CLIENTS)
    print("Communication rounds:", ROUNDS)
    print("Local epochs:", LOCAL_EPOCHS)

    if args.energy_aware:
        save_energy_round_log(
            result["participation_rows"],
            energy_log_path,
        )
        print("Energy round log:", energy_log_path)


if __name__ == "__main__":
    main()
