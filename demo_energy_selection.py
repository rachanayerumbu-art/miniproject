"""Run a repeatable five-client demonstration of energy-aware client selection."""
from pathlib import Path
from energy_aware_fl import EnergyAwareSelector

HERE = Path(__file__).resolve().parent

def main():
    selector = EnergyAwareSelector(
        config_path=str(HERE / "energy_config.json"),
        client_csv_path=str(HERE / "client_energy_config.csv"),
    )

    # Replace these example sizes with len(train_df) for each client's TRAIN split.
    # These values are examples only and are not actual dataset row counts.
    client_sample_counts = {
        "client1": 2500,
        "client2": 1800,
        "client3": 900,
        "client4": 1200,
        "client5": 2200,
    }

    rounds = 8
    model_size_mb = 2.0  # placeholder estimate; replace with measured/estimated model size
    clients_per_round = 3

    for round_no in range(1, rounds + 1):
        selected = selector.select_clients(
            round_number=round_no,
            client_sample_counts=client_sample_counts,
            model_size_mb=model_size_mb,
            clients_per_round=clients_per_round,
        )
        print(f"Round {round_no}: selected {selected}")
        if not selected:
            print("No eligible clients remain; stopping the simulation.")
            break

        spent = selector.charge_selected(
            round_number=round_no,
            selected_clients=selected,
            client_sample_counts=client_sample_counts,
            model_size_mb=model_size_mb,
        )
        print(f"  simulated energy spent: {spent}")

    selector.save_outputs(str(HERE / "outputs"))
    selector.print_summary()

if __name__ == "__main__":
    main()
