
from pathlib import Path
from energy_aware_fl import EnergyAwareSelector

HERE = Path(__file__).resolve().parent

selector = EnergyAwareSelector(
    config_path=str(HERE / "energy_config.json"),
    client_csv_path=str(HERE / "client_energy_config.csv"),
)

# Replace with counts of TRAINING examples only, after train/test splitting.
client_sample_counts = {
    "client1": 2500,
    "client2": 1800,
    "client3": 900,
    "client4": 1200,
    "client5": 2200,
}
estimated_model_size_mb = 2.0
clients_per_round = 3
number_of_rounds = 5

global_model = None  # replace with your initialized model

for round_no in range(1, number_of_rounds + 1):
    selected = selector.select_clients(
        round_number=round_no,
        client_sample_counts=client_sample_counts,
        model_size_mb=estimated_model_size_mb,
        clients_per_round=clients_per_round,
    )
    if not selected:
        print(f"Round {round_no}: no clients have enough simulated energy.")
        break

    client_updates = []
    for client_id in selected:
        # TODO: replace with the team's actual local training function.
        # update = local_train(client_id, global_model)
        # client_updates.append((client_id, update, client_sample_counts[client_id]))
        pass

    # TODO: aggregate only successfully completed client updates:
    # global_model = fedavg_aggregate(client_updates)

    # Charge only clients whose local training and communication completed.
    # In a real integration, use completed_client_ids, not all selected clients.
    completed_client_ids = selected  # demo assumption; change if training can fail
    selector.charge_selected(
        round_number=round_no,
        selected_clients=completed_client_ids,
        client_sample_counts=client_sample_counts,
        model_size_mb=estimated_model_size_mb,
    )
    print(f"Round {round_no}: selected {selected}")

selector.save_outputs(str(HERE / "outputs"))
selector.print_summary()
