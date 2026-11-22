"""Select clients that can afford a training round and retain a reserve."""

from typing import List, Mapping, Optional

from energy_config import (
    CLIENT_CONFIG,
    ENERGY_UNIT,
    MAX_CLIENTS_PER_ROUND,
    MINIMUM_ENERGY_THRESHOLD,
)
from energy_simulator import (
    ClientConfigs,
    EnergyState,
    charge_clients,
    create_energy_state,
    get_round_cost,
)


def select_clients(
    energy_state: Mapping[str, float],
    minimum_energy_threshold: float = MINIMUM_ENERGY_THRESHOLD,
    max_clients: Optional[int] = MAX_CLIENTS_PER_ROUND,
    client_configs: ClientConfigs = CLIENT_CONFIG,
) -> List[str]:
    """Return eligible clients ranked by their available energy.

    A client must cover its estimated round cost and retain the configured
    threshold. Eligibility is recalculated at the start of every round.
    """
    if minimum_energy_threshold < 0:
        raise ValueError("Minimum energy threshold cannot be negative.")
    if max_clients is not None and max_clients < 0:
        raise ValueError("Maximum clients per round cannot be negative.")

    missing_clients = set(client_configs) - set(energy_state)
    if missing_clients:
        raise ValueError(
            "Energy state is missing clients: "
            + ", ".join(sorted(missing_clients))
        )

    eligible = [
        client_id
        for client_id in client_configs
        if energy_state[client_id]
        >= get_round_cost(client_id, client_configs)
        + minimum_energy_threshold
    ]

    # Stable sorting preserves configuration order when energy levels tie.
    eligible.sort(key=lambda client_id: energy_state[client_id], reverse=True)

    if max_clients is not None:
        eligible = eligible[:max_clients]

    return eligible


def main() -> None:
    """Demonstrate selection and energy updates over five rounds."""
    energy_state = create_energy_state()

    for round_number in range(1, 6):
        selected_clients = select_clients(energy_state)
        print(f"\nRound {round_number}")
        print("Selected clients:", selected_clients)

        if not selected_clients:
            print("No eligible clients remain.")
            break

        energy_used = charge_clients(selected_clients, energy_state)
        print(f"Energy used: {sum(energy_used.values()):.2f} {ENERGY_UNIT}")
        print("Remaining energy:")
        for client_id, remaining in energy_state.items():
            print(f"  {client_id}: {remaining:.2f} {ENERGY_UNIT}")


if __name__ == "__main__":
    main()
