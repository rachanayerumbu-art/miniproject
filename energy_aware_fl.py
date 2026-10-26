"""Simulation-only energy-aware client selection for a federated-learning prototype.

Uses Python standard library only. Energy values are abstract simulated units, not joules.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class ClientState:
    client_id: str
    initial_energy_units: float = 100.0
    remaining_energy_units: float = 100.0
    training_speed_factor: float = 1.0
    communication_factor: float = 1.0
    times_selected: int = 0
    total_energy_spent_units: float = 0.0


class EnergyAwareSelector:
    """Estimate costs and select clients with sufficient energy.

    The scoring rule favours:
      * a larger remaining-energy fraction,
      * a lower estimated round cost,
      * clients selected less often (small fairness term).
    """

    def __init__(self, config_path: str = "energy_config.json",
                 client_csv_path: Optional[str] = "client_energy_config.csv",
                 state_path: Optional[str] = None):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.state_path = Path(state_path) if state_path else None
        self.clients: Dict[str, ClientState] = self._load_clients(client_csv_path)
        if self.state_path and self.state_path.exists():
            self._load_saved_state(self.state_path)
        self.round_log: List[dict] = []

    def _load_clients(self, csv_path: Optional[str]) -> Dict[str, ClientState]:
        clients: Dict[str, ClientState] = {}
        if csv_path and Path(csv_path).exists():
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    cid = row["client_id"].strip()
                    initial = float(row.get("initial_energy_units") or self.config["initial_energy_units"])
                    clients[cid] = ClientState(
                        client_id=cid,
                        initial_energy_units=initial,
                        remaining_energy_units=initial,
                        training_speed_factor=float(row.get("training_speed_factor") or 1.0),
                        communication_factor=float(row.get("communication_factor") or 1.0),
                    )
        if not clients:
            for i in range(1, 6):
                cid = f"client{i}"
                initial = float(self.config["initial_energy_units"])
                clients[cid] = ClientState(cid, initial, initial)
        return clients

    def _load_saved_state(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        for cid, values in data.items():
            if cid in self.clients:
                for key, value in values.items():
                    if hasattr(self.clients[cid], key):
                        setattr(self.clients[cid], key, value)

    def estimate_training_cost(self, client_id: str, sample_count: int,
                               local_epochs: Optional[int] = None) -> float:
        """Estimated local compute cost; scales with samples, epochs and client factor."""
        epochs = int(local_epochs or self.config["local_epochs"])
        c = self.clients[client_id]
        base = float(self.config["training_base_cost_units"])
        per_1000 = float(self.config["training_cost_per_1000_samples_per_epoch"])
        return (base + (max(0, sample_count) / 1000.0) * per_1000 * epochs) * c.training_speed_factor

    def estimate_communication_cost(self, client_id: str, model_size_mb: Optional[float] = None) -> float:
        """Estimates upload + download cost. Model size is counted twice (send and receive)."""
        size_mb = float(model_size_mb if model_size_mb is not None else self.config["default_model_size_mb"])
        rate = float(self.config["communication_cost_units_per_mb"])
        return max(0.0, size_mb) * 2.0 * rate * self.clients[client_id].communication_factor

    def estimate_round_cost(self, client_id: str, sample_count: int,
                            model_size_mb: Optional[float] = None,
                            local_epochs: Optional[int] = None) -> float:
        return self.estimate_training_cost(client_id, sample_count, local_epochs) + \
               self.estimate_communication_cost(client_id, model_size_mb)

    def select_clients(self, round_number: int, client_sample_counts: Dict[str, int],
                       model_size_mb: Optional[float] = None,
                       clients_per_round: Optional[int] = None) -> List[str]:
        """Return selected client IDs; selection itself does not deduct energy."""
        target = int(clients_per_round or self.config["default_clients_per_round"])
        min_energy = float(self.config["minimum_energy_units"])
        candidates = []
        for cid, state in self.clients.items():
            if cid not in client_sample_counts:
                continue
            cost = self.estimate_round_cost(cid, int(client_sample_counts[cid]), model_size_mb)
            if state.remaining_energy_units >= cost and state.remaining_energy_units >= min_energy:
                energy_fraction = state.remaining_energy_units / max(state.initial_energy_units, 1e-9)
                # Lower cost is better; normalise relative to remaining energy.
                affordability = 1.0 - min(cost / max(state.remaining_energy_units, 1e-9), 1.0)
                # Fewer previous selections gives a higher fairness score.
                fairness = 1.0 / (1.0 + state.times_selected)
                score = (
                    float(self.config["energy_weight"]) * energy_fraction
                    + float(self.config["cost_weight"]) * affordability
                    + float(self.config["fairness_weight"]) * fairness
                )
                candidates.append((score, cid, cost))

        # Stable tie-break by client ID. Highest score first.
        candidates.sort(key=lambda item: (-item[0], item[1]))
        chosen = [cid for _, cid, _ in candidates[:max(0, target)]]
        chosen_costs = {cid: round(cost, 4) for _, cid, cost in candidates if cid in chosen}
        self.round_log.append({
            "round": int(round_number),
            "selected_clients": ";".join(chosen),
            "eligible_clients": len(candidates),
            "estimated_costs_units": json.dumps(chosen_costs, sort_keys=True),
            "status": "selected" if chosen else "no_eligible_clients",
        })
        return chosen

    def charge_selected(self, round_number: int, selected_clients: Iterable[str],
                        client_sample_counts: Dict[str, int],
                        model_size_mb: Optional[float] = None,
                        local_epochs: Optional[int] = None) -> Dict[str, float]:
        """Deduct estimated energy after selected clients actually complete local work."""
        spent: Dict[str, float] = {}
        for cid in selected_clients:
            if cid not in self.clients or cid not in client_sample_counts:
                continue
            cost = self.estimate_round_cost(
                cid, int(client_sample_counts[cid]), model_size_mb, local_epochs
            )
            state = self.clients[cid]
            # Guard against negative energy if settings changed between selection and charging.
            actual_charge = min(cost, state.remaining_energy_units)
            state.remaining_energy_units = round(max(0.0, state.remaining_energy_units - actual_charge), 4)
            state.total_energy_spent_units = round(state.total_energy_spent_units + actual_charge, 4)
            state.times_selected += 1
            spent[cid] = round(actual_charge, 4)
        for entry in reversed(self.round_log):
            if entry["round"] == int(round_number):
                entry["actual_simulated_costs_units"] = json.dumps(spent, sort_keys=True)
                entry["total_spent_this_round_units"] = round(sum(spent.values()), 4)
                break
        return spent

    def save_outputs(self, output_dir: str = "outputs") -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / "energy_round_log.csv"
        columns = [
            "round", "selected_clients", "eligible_clients", "estimated_costs_units",
            "status", "actual_simulated_costs_units", "total_spent_this_round_units"
        ]
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in self.round_log:
                writer.writerow({key: row.get(key, "") for key in columns})

        state = {cid: asdict(client) for cid, client in self.clients.items()}
        state_path = out / "energy_final_state.json"
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"Saved round log: {log_path}")
        print(f"Saved final client energy state: {state_path}")

    def print_summary(self) -> None:
        print("\nFinal simulated energy summary")
        print("-" * 72)
        print(f"{'Client':<10}{'Initial':>12}{'Remaining':>14}{'Spent':>12}{'Selected':>12}")
        for cid, c in sorted(self.clients.items()):
            print(f"{cid:<10}{c.initial_energy_units:>12.2f}{c.remaining_energy_units:>14.2f}"
                  f"{c.total_energy_spent_units:>12.2f}{c.times_selected:>12d}")
        print("All energy values are simulated units, not measured joules.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Energy-aware client selection demo helper")
    parser.add_argument("--config", default="energy_config.json")
    parser.add_argument("--clients", default="client_energy_config.csv")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    selector = EnergyAwareSelector(args.config, args.clients)
    counts = {cid: 2000 for cid in selector.clients}
    selected = selector.select_clients(1, counts)
    print("Selected:", selected)
    selector.charge_selected(1, selected, counts)
    selector.save_outputs(args.output)
    selector.print_summary()
