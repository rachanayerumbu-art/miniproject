"""Generate presentation graphs from the completed experiment CSV files."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
WEEK3_RESULTS_DIR = PROJECT_DIR / "week3_energy_comparison"
RESULTS_DIR = PROJECT_DIR / "week4_results"
GRAPH_NAMES = [
    "energy_per_client.png",
    "energy_per_round.png",
    "participants_per_round.png",
    "accuracy_comparison.png",
    "energy_savings_comparison.png",
]
GRAPH_DIR_NAME = "presentation_graphs"


def load_results(week3_results_dir, results_dir):
    participation_path = week3_results_dir / "participation.csv"
    round_metrics_path = week3_results_dir / "round_metrics.csv"
    experiment_results_path = results_dir / "experiment_results.csv"

    for path in (
        participation_path,
        round_metrics_path,
        experiment_results_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required result file not found: {path}")

    participation = pd.read_csv(participation_path)
    participation["Participated"] = (
        participation["Participated"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    if participation["Participated"].isna().any():
        raise ValueError("Participation CSV contains invalid status values.")

    return (
        participation,
        pd.read_csv(round_metrics_path),
        pd.read_csv(experiment_results_path),
    )


def plot_energy_per_client(participation, output_path):
    client_energy = (
        participation.groupby(["Strategy", "Client"])["Energy_Used"]
        .sum()
        .unstack("Strategy")
        .fillna(0)
    )

    axis = client_energy.plot(kind="bar", figsize=(10, 5))
    axis.set_title("Estimated Simulated Energy Use per Client")
    axis.set_xlabel("Simulated IoT Client")
    axis.set_ylabel("Simulated Energy Units")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(title="Strategy")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_energy_per_round(round_metrics, output_path):
    figure, axis = plt.subplots(figsize=(9, 5))
    for strategy, rows in round_metrics.groupby("Strategy"):
        axis.plot(
            rows["Round"],
            rows["Energy_Used"],
            marker="o",
            label=strategy,
        )

    axis.set_title("Estimated Simulated Energy Use per Round")
    axis.set_xlabel("Federated Communication Round")
    axis.set_ylabel("Simulated Energy Units")
    axis.set_xticks(sorted(round_metrics["Round"].unique()))
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_participation_per_round(participation, output_path):
    counts = (
        participation.groupby(["Strategy", "Round"])["Participated"]
        .sum()
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    for strategy, rows in counts.groupby("Strategy"):
        axis.plot(
            rows["Round"],
            rows["Participated"],
            marker="o",
            label=strategy,
        )

    axis.set_title("Client Participation per Federated Round")
    axis.set_xlabel("Federated Communication Round")
    axis.set_ylabel("Participating Clients")
    axis.set_xticks(sorted(counts["Round"].unique()))
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_accuracy_comparison(experiment_results, output_path):
    completed = experiment_results[
        (experiment_results["Status"] == "completed")
        & experiment_results["Accuracy"].notna()
    ].copy()
    if completed.empty:
        raise ValueError("No completed accuracy results are available.")

    completed["Accuracy_Percent"] = (
        pd.to_numeric(completed["Accuracy"]) * 100
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(
        completed["Experiment"],
        completed["Accuracy_Percent"],
        color=plt.cm.tab10.colors[:len(completed)],
    )
    axis.set_title("Accuracy on the Independent Test Set")
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(0, 100)
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, completed["Accuracy_Percent"]):
        axis.annotate(
            f"{value:.3f}%",
            (
                bar.get_x() + bar.get_width() / 2,
                max(bar.get_height() - 0.8, 0),
            ),
            ha="center",
            va="top",
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_energy_savings(experiment_results, output_path):
    completed = experiment_results[
        (experiment_results["Status"] == "completed")
        & experiment_results["Total_Simulated_Energy_Used"].notna()
    ].copy()
    energy_by_strategy = {
        row["Experiment"]: float(row["Total_Simulated_Energy_Used"])
        for _, row in completed.iterrows()
    }

    standard_energy = energy_by_strategy.get("Standard FedAvg")
    aware_energy = energy_by_strategy.get("Energy-aware FedAvg")
    if standard_energy is None or aware_energy is None or standard_energy <= 0:
        raise ValueError(
            "Standard and energy-aware simulated energy results are required."
        )

    savings_percent = (
        (standard_energy - aware_energy) / standard_energy * 100
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    bar = axis.bar(
        ["Energy-aware FedAvg"],
        [savings_percent],
        color="#61DDAA",
    )[0]
    axis.set_title("Estimated Simulated Energy Savings vs Standard FedAvg")
    axis.set_ylabel("Estimated Reduction (%)")
    axis.set_ylim(0, max(10, savings_percent * 1.25))
    axis.grid(axis="y", alpha=0.3)
    axis.annotate(
        f"{savings_percent:.2f}%",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center",
        va="bottom",
    )
    axis.text(
        0.5,
        -0.18,
        "Simulation estimate only; not a hardware battery measurement.",
        transform=axis.transAxes,
        ha="center",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Create Week 4 graphs from experiment result CSV files."
    )
    parser.add_argument(
        "--week3-results",
        type=Path,
        default=WEEK3_RESULTS_DIR,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
    )
    args = parser.parse_args()

    week3_results_dir = args.week3_results.resolve()
    results_dir = args.results_dir.resolve()
    graphs_dir = results_dir / GRAPH_DIR_NAME
    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"Run compare_results.py first; results directory not found: "
            f"{results_dir}"
        )
    if graphs_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing graph directory: {graphs_dir}. "
            "Choose a new results directory."
        )

    participation, round_metrics, experiment_results = load_results(
        week3_results_dir,
        results_dir,
    )
    graphs_dir.mkdir()
    plot_energy_per_client(
        participation,
        graphs_dir / GRAPH_NAMES[0],
    )
    plot_energy_per_round(
        round_metrics,
        graphs_dir / GRAPH_NAMES[1],
    )
    plot_participation_per_round(
        participation,
        graphs_dir / GRAPH_NAMES[2],
    )
    plot_accuracy_comparison(
        experiment_results,
        graphs_dir / GRAPH_NAMES[3],
    )
    plot_energy_savings(
        experiment_results,
        graphs_dir / GRAPH_NAMES[4],
    )

    print("Graphs saved in:", graphs_dir)
    for graph_name in GRAPH_NAMES:
        print("  ", graph_name)


if __name__ == "__main__":
    main()
