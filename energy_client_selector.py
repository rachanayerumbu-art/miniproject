"""Compatibility entry point for the energy selection modules."""

from energy_config import (
    CLIENT_CONFIG,
    ENERGY_UNIT,
    MAX_CLIENTS_PER_ROUND,
    MINIMUM_ENERGY_THRESHOLD,
)
from energy_aware_selection import select_clients
from energy_simulator import (
    ClientConfigs,
    EnergyState,
    charge_clients,
    create_energy_state,
    get_round_cost,
)


def main() -> None:
    """Run a small, reproducible demonstration of client selection."""
    energy_state = create_energy_state()

    for round_number in range(1, 6):
        selected = select_clients(energy_state)

        print(f"\nRound {round_number}")
        print("Selected clients:", selected)

        if not selected:
            print("No eligible clients remain. Stopping.")
            break

        # In federated training, call this only after local training succeeds.
        energy_used = charge_clients(selected, energy_state)
        print(
            f"Simulated energy used: "
            f"{sum(energy_used.values()):.2f} {ENERGY_UNIT}"
        )
        print("Remaining energy:")
        for client_id, remaining in energy_state.items():
            print(f"  {client_id}: {remaining:.2f}")


if __name__ == "__main__":
    main()
