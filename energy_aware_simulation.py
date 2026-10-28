
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# ENERGY-AWARE IOT CLIENT SIMULATION
# Rachitha - Week 1 and Week 2
# ==========================================

DEVICES = {
    "Smoke Detector": {
        "initial_energy": 100,
        "training_cost": 3.0,
        "communication_cost": 1.0
    },
    "Water-Level Monitor": {
        "initial_energy": 80,
        "training_cost": 2.5,
        "communication_cost": 1.0
    },
    "Motion Detector": {
        "initial_energy": 65,
        "training_cost": 2.0,
        "communication_cost": 0.8
    },
    "Smart Light": {
        "initial_energy": 55,
        "training_cost": 1.5,
        "communication_cost": 0.5
    },
    "Thermostat": {
        "initial_energy": 40,
        "training_cost": 2.0,
        "communication_cost": 0.7
    }
}

TOTAL_ROUNDS = 10
MINIMUM_CLIENTS = 3
ENERGY_RESERVE = 5.0


def create_energy_state():
    """Create a fresh energy budget for every experiment."""
    return {
        name: info["initial_energy"]
        for name, info in DEVICES.items()
    }


def get_round_cost(device_name):
    """Estimated energy cost of one FL training round."""
    info = DEVICES[device_name]
    return info["training_cost"] + info["communication_cost"]


def run_experiment(energy_aware):
    """
    Simulate FL client participation.

    energy_aware=False:
        Every client participates if it can afford the round.

    energy_aware=True:
        Select the eligible clients with the most remaining
        energy, subject to the minimum-client requirement.

    All energy values are illustrative simulation units.
    """

    energy = create_energy_state()
    history = []

    for round_number in range(1, TOTAL_ROUNDS + 1):

        # Clients must afford the cost and retain the reserve.
        eligible = []

        for name in DEVICES:
            cost = get_round_cost(name)

            if energy[name] >= cost + ENERGY_RESERVE:
                eligible.append(name)

        # Standard participation: every eligible client.
        if not energy_aware:
            selected = eligible.copy()

        else:
            # Prefer clients with more remaining energy.
            selected = sorted(
                eligible,
                key=lambda name: energy[name],
                reverse=True
            )[:max(MINIMUM_CLIENTS, 0)]

            # Do not invent participants if too few are eligible.
            selected = [
                name for name in selected
                if name in eligible
            ]

        selected_set = set(selected)

        # Record each client's status for this round.
        for name, info in DEVICES.items():

            cost = get_round_cost(name)
            participated = name in selected_set

            energy_before = energy[name]

            if participated:
                energy[name] -= cost

            history.append({
                "Strategy": (
                    "Energy-aware" if energy_aware
                    else "Standard"
                ),
                "Round": round_number,
                "Device": name,
                "Participated": participated,
                "Energy_Before": energy_before,
                "Energy_Used": cost if participated else 0.0,
                "Energy_Remaining": energy[name]
            })

        if len(selected) < MINIMUM_CLIENTS:
            print(
                f"Warning: only {len(selected)} clients "
                f"could participate in round {round_number}."
            )

    return pd.DataFrame(history)


def summarize_results(results):
    """Print summary metrics for each strategy."""

    print("\n========== ENERGY SUMMARY ==========")

    for strategy in results["Strategy"].unique():

        data = results[results["Strategy"] == strategy]

        total_energy = data["Energy_Used"].sum()

        total_participations = data["Participated"].sum()

        print(f"\nStrategy: {strategy}")
        print(f"Total simulated energy used: {total_energy:.2f}")
        print(f"Total client participations: {total_participations}")

        final_energy = (
            data.sort_values("Round")
            .groupby("Device")["Energy_Remaining"]
            .last()
        )

        print("Final device energy levels:")

        for device, value in final_energy.items():
            print(f"  {device}: {value:.2f}")


def main():

    print("Starting energy-aware IoT simulation...")

    # Both experiments start with identical energy budgets.
    standard = run_experiment(energy_aware=False)
    aware = run_experiment(energy_aware=True)

    results = pd.concat(
        [standard, aware],
        ignore_index=True
    )

    # Save all participation and energy records.
    results.to_csv(
        "energy_round_results.csv",
        index=False
    )

    # Create a summary table.
    summary = (
        results.groupby("Strategy")
        .agg(
            Total_Energy_Used=("Energy_Used", "sum"),
            Total_Participations=("Participated", "sum")
        )
        .reset_index()
    )

    summary.to_csv(
        "energy_summary.csv",
        index=False
    )

    print("\n========== COMPARISON ==========")
    print(summary.to_string(index=False))

    summarize_results(results)

    # Plot total energy used per round.
    energy_by_round = (
        results.groupby(["Strategy", "Round"])["Energy_Used"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(9, 5))

    for strategy in energy_by_round["Strategy"].unique():

        data = energy_by_round[
            energy_by_round["Strategy"] == strategy
        ]

        plt.plot(
            data["Round"],
            data["Energy_Used"],
            marker="o",
            label=strategy
        )

    plt.xlabel("Federated Learning Round")
    plt.ylabel("Simulated Energy Units")
    plt.title("Energy Consumption per Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("energy_per_round.png", dpi=150)
    plt.show()

    # Plot final energy remaining for each device.
    final = (
        results.sort_values("Round")
        .groupby(["Strategy", "Device"])["Energy_Remaining"]
        .last()
        .reset_index()
    )

    final_pivot = final.pivot(
        index="Device",
        columns="Strategy",
        values="Energy_Remaining"
    )

    final_pivot.plot(
        kind="bar",
        figsize=(10, 5)
    )

    plt.ylabel("Remaining Simulated Energy Units")
    plt.title("Final Energy Remaining per IoT Device")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("final_energy_comparison.png", dpi=150)
    plt.show()

    print("\nFiles saved:")
    print("1. energy_round_results.csv")
    print("2. energy_summary.csv")
    print("3. energy_per_round.png")
    print("4. final_energy_comparison.png")
    print("\nEnergy simulation completed.")


if __name__ == "__main__":
    main()
