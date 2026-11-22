"""Track configurable simulated energy budgets for IoT clients."""

from typing import Dict, List, Mapping

from energy_config import CLIENT_CONFIG, ENERGY_UNIT


EnergyState = Dict[str, float]
ClientConfigs = Mapping[str, Mapping[str, float]]


def create_energy_state(
    client_configs: ClientConfigs = CLIENT_CONFIG,
) -> EnergyState:
    """Create a fresh copy of the configured initial client budgets."""
    return {
        client_id: info["initial_energy"]
        for client_id, info in client_configs.items()
    }


def get_round_cost(
    client_id: str,
    client_configs: ClientConfigs = CLIENT_CONFIG,
) -> float:
    """Estimate one client's local training and update communication cost."""
    if client_id not in client_configs:
        raise ValueError(f"Unknown client: {client_id}")

    info = client_configs[client_id]
    return info["training_cost"] + info["communication_cost"]


def charge_clients(
    selected_clients: List[str],
    energy_state: EnergyState,
    client_configs: ClientConfigs = CLIENT_CONFIG,
) -> Dict[str, float]:
    """Deduct each selected client's estimated cost after local training."""
    if len(selected_clients) != len(set(selected_clients)):
        raise ValueError("A client cannot be charged more than once per round.")

    costs = {}
    for client_id in selected_clients:
        cost = get_round_cost(client_id, client_configs)
        if client_id not in energy_state:
            raise ValueError(f"Energy state is missing {client_id}.")
        if energy_state[client_id] < cost:
            raise ValueError(
                f"{client_id} does not have enough energy for this round."
            )
        costs[client_id] = cost

    # Validate all clients before changing any budgets.
    for client_id, cost in costs.items():
        energy_state[client_id] -= cost

    return costs


def main() -> None:
    """Demonstrate the simulated cost of one participation per client."""
    energy_state = create_energy_state()
    selected_clients = list(CLIENT_CONFIG)

    print(f"Energy values are in {ENERGY_UNIT}.")
    print("Each client participates once in this demonstration.")
    energy_used = charge_clients(selected_clients, energy_state)

    for client_id in CLIENT_CONFIG:
        print(
            f"{client_id}: used {energy_used[client_id]:.2f}, "
            f"remaining {energy_state[client_id]:.2f}"
        )


if __name__ == "__main__":
    main()
