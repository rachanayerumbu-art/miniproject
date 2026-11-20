"""Build a fair comparison table from available experiment results."""

import argparse
import csv
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score

from security.baseline import (
    RANDOM_SEED,
    TEST_FILE,
    TRAIN_FILE,
    load_data,
)


PROJECT_DIR = Path(__file__).resolve().parent
WEEK3_RESULTS_DIR = PROJECT_DIR / "week3_energy_comparison"
RESULTS_DIR = PROJECT_DIR / "week4_results"
RESULT_FIELDS = [
    "Experiment",
    "Status",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "False_Positive_Rate",
    "Total_Simulated_Energy_Used",
    "Total_Participations",
    "Communication_Rounds",
    "Notes",
]


def calculate_metrics(y_true, predictions):
    """Return binary classification metrics with DDoS as the positive class."""
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    true_negative, false_positive, _, _ = matrix.ravel()
    false_positive_rate = (
        false_positive / (false_positive + true_negative)
        if false_positive + true_negative
        else 0.0
    )

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1_Score": f1_score(y_true, predictions, zero_division=0),
        "False_Positive_Rate": false_positive_rate,
    }


def run_centralized_baseline():
    """Train the existing centralized Random Forest on the same test split."""
    X_train, y_train = load_data(TRAIN_FILE)
    X_test, y_test = load_data(TEST_FILE)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Centralized training and test features do not match.")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return calculate_metrics(y_test, predictions)


def load_federated_results(week3_results_dir):
    """Load final metrics from the completed standard and energy-aware runs."""
    summary_path = week3_results_dir / "comparison_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Federated summary not found: {summary_path}")

    summary = pd.read_csv(summary_path)
    required_strategies = {
        "Standard FedAvg",
        "Energy-aware FedAvg",
    }
    found_strategies = set(summary["Strategy"])
    missing = required_strategies - found_strategies
    if missing:
        raise ValueError(
            "Federated summary is missing strategies: "
            + ", ".join(sorted(missing))
        )

    return {
        row["Strategy"]: row
        for _, row in summary.iterrows()
    }


def create_experiment_rows(federated_results, centralized_metrics):
    rows = [
        {
            "Experiment": "Centralized Random Forest",
            "Status": "completed",
            **centralized_metrics,
            "Total_Simulated_Energy_Used": "",
            "Total_Participations": "",
            "Communication_Rounds": "",
            "Notes": (
                "Existing centralized baseline; energy and client "
                "participation are not applicable."
            ),
        }
    ]

    for strategy in ("Standard FedAvg", "Energy-aware FedAvg"):
        result = federated_results[strategy]
        rows.append(
            {
                "Experiment": strategy,
                "Status": "completed",
                "Accuracy": result["Accuracy"],
                "Precision": result["Precision"],
                "Recall": result["Recall"],
                "F1_Score": result["F1_Score"],
                "False_Positive_Rate": result["False_Positive_Rate"],
                "Total_Simulated_Energy_Used": (
                    result["Total_Simulated_Energy_Used"]
                ),
                "Total_Participations": result["Total_Participations"],
                "Communication_Rounds": result["Communication_Rounds"],
                "Notes": "Reused the Week 3 run on the held-out test set.",
            }
        )

    unavailable_note = (
        "Unavailable: no federated security-defense implementation "
        "was present in the inspected project."
    )
    for experiment in (
        "Federated Learning with Security Defense",
        "Security Defense with Energy-Aware Selection",
    ):
        rows.append(
            {
                "Experiment": experiment,
                "Status": "unavailable",
                "Accuracy": "",
                "Precision": "",
                "Recall": "",
                "F1_Score": "",
                "False_Positive_Rate": "",
                "Total_Simulated_Energy_Used": "",
                "Total_Participations": "",
                "Communication_Rounds": "",
                "Notes": unavailable_note,
            }
        )

    return rows


def build_report(rows, report_path):
    """Write a concise report that distinguishes missing data from results."""
    completed = [row for row in rows if row["Status"] == "completed"]
    lines = [
        "# Experiment Comparison Report",
        "",
        "All completed models were evaluated on the existing independent "
        "test set. Federated experiment results reuse the Week 3 run; the "
        "centralized baseline uses the existing training pool.",
        "",
        "| Experiment | Status | Accuracy | Precision | Recall | F1 | FPR | Energy units |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        values = [
            row["Experiment"],
            row["Status"],
            _format_metric(row["Accuracy"]),
            _format_metric(row["Precision"]),
            _format_metric(row["Recall"]),
            _format_metric(row["F1_Score"]),
            _format_metric(row["False_Positive_Rate"]),
            _format_metric(row["Total_Simulated_Energy_Used"]),
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "Energy values are simulated estimates based on the configured "
            "per-client training and communication costs. They are not "
            "hardware battery measurements.",
            "",
            "Security-defense experiments are marked unavailable because "
            "the inspected project contains no federated security-defense "
            "implementation. No performance values were invented for them.",
            "",
            "The centralized model is a Random Forest; the existing "
            "federated trainer uses scikit-learn SGDClassifier.",
        ]
    )

    standard = next(
        row for row in completed if row["Experiment"] == "Standard FedAvg"
    )
    energy_aware = next(
        row for row in completed if row["Experiment"] == "Energy-aware FedAvg"
    )
    standard_energy = float(standard["Total_Simulated_Energy_Used"])
    aware_energy = float(energy_aware["Total_Simulated_Energy_Used"])
    if standard_energy > 0:
        savings = (standard_energy - aware_energy) / standard_energy * 100
        accuracy_delta = (
            float(standard["Accuracy"]) - float(energy_aware["Accuracy"])
        ) * 100
        lines.extend(
            [
                "",
                "## Standard vs. Energy-Aware FedAvg",
                "",
                f"The measured simulation totals are {standard_energy:.2f} "
                f"and {aware_energy:.2f} estimated units, respectively. "
                f"That is a {savings:.2f}% reduction under this simulation's "
                "cost assumptions, not a hardware energy claim.",
                f"Energy-aware accuracy differs by {accuracy_delta:.5f} "
                "percentage points (standard minus energy-aware).",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metric(value):
    if value == "":
        return "N/A"
    return f"{float(value):.4f}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare available centralized and federated results."
    )
    parser.add_argument(
        "--week3-results",
        type=Path,
        default=WEEK3_RESULTS_DIR,
        help="Directory containing Week 3 federated CSV results.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="New directory for Week 4 comparison results and report.",
    )
    args = parser.parse_args()

    week3_results_dir = args.week3_results.resolve()
    results_dir = args.results_dir.resolve()
    if results_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results directory: {results_dir}."
        )

    print("Training the existing centralized Random Forest baseline...")
    centralized_metrics = run_centralized_baseline()
    federated_results = load_federated_results(week3_results_dir)
    rows = create_experiment_rows(federated_results, centralized_metrics)

    results_dir.mkdir(parents=True, exist_ok=False)
    csv_path = results_dir / "experiment_results.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report_path = results_dir / "comparison_report.md"
    build_report(rows, report_path)
    print("Comparison results:", csv_path)
    print("Comparison report:", report_path)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
