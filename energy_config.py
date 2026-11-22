"""Shared configuration for the five-client energy simulation."""

from typing import Dict


# Energy values are illustrative simulation units, not battery measurements.
ENERGY_UNIT = "simulated units"
MINIMUM_ENERGY_THRESHOLD = 5.0
MAX_CLIENTS_PER_ROUND = 3

# Costs estimate one local-training run plus sending its model update.
CLIENT_CONFIG: Dict[str, Dict[str, float]] = {
    "client1": {
        "initial_energy": 100.0,
        "training_cost": 3.0,
        "communication_cost": 1.0,
    },
    "client2": {
        "initial_energy": 80.0,
        "training_cost": 2.5,
        "communication_cost": 1.0,
    },
    "client3": {
        "initial_energy": 65.0,
        "training_cost": 2.0,
        "communication_cost": 0.8,
    },
    "client4": {
        "initial_energy": 55.0,
        "training_cost": 1.5,
        "communication_cost": 0.5,
    },
    "client5": {
        "initial_energy": 40.0,
        "training_cost": 2.0,
        "communication_cost": 0.7,
    },
}
