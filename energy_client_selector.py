
# Energy-aware client selector for Federated Learning

DEVICES = {
    "client1": {"energy": 100.0, "cost": 4.0},
    "client2": {"energy": 80.0, "cost": 3.5},
    "client3": {"energy": 65.0, "cost": 2.8},
    "client4": {"energy": 55.0, "cost": 2.0},
    "client5": {"energy": 40.0, "cost": 2.7},
}

MIN_CLIENTS = 3
ENERGY_RESERVE = 5.0


def select_clients():
    """Return eligible client IDs, ranked by remaining energy."""

    eligible = [
        client
        for client, info in DEVICES.items()
        if info["energy"] >= info["cost"] + ENERGY_RESERVE
    ]

    eligible.sort(
        key=lambda client: DEVICES[client]["energy"],
        reverse=True
    )

    return eligible[:MIN_CLIENTS]


def charge_clients(selected_clients):
    """Deduct estimated energy after a client participates."""

    for client in selected_clients:
        DEVICES[client]["energy"] -= DEVICES[client]["cost"]


if __name__ == "__main__":

    for round_number in range(1, 6):

        selected = select_clients()

        print(f"\nRound {round_number}")
        print("Selected clients:", selected)

        if len(selected) < MIN_CLIENTS:
            print("Not enough eligible clients. Stopping.")
            break

        charge_clients(selected)

        print("Remaining energy:")

        for client, info in DEVICES.items():
            print(f"{client}: {info['energy']:.2f}")
