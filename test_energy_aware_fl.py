"""Small standard-library tests. Run: python -m unittest test_energy_aware_fl.py"""
import json
import tempfile
import unittest
from pathlib import Path
from energy_aware_fl import EnergyAwareSelector


class EnergySelectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / "energy_config.json").write_text(json.dumps({
            "initial_energy_units": 10.0,
            "training_base_cost_units": 1.0,
            "training_cost_per_1000_samples_per_epoch": 0.1,
            "communication_cost_units_per_mb": 0.1,
            "default_model_size_mb": 1.0,
            "default_clients_per_round": 2,
            "energy_weight": 0.65,
            "cost_weight": 0.25,
            "fairness_weight": 0.10,
            "minimum_energy_units": 0.5,
            "local_epochs": 1
        }), encoding="utf-8")
        (root / "clients.csv").write_text(
            "client_id,initial_energy_units,training_speed_factor,communication_factor\n"
            "client1,10,1,1\nclient2,10,1,1\nclient3,0.1,1,1\n",
            encoding="utf-8"
        )
        self.selector = EnergyAwareSelector(root / "energy_config.json", root / "clients.csv")

    def tearDown(self):
        self.temp.cleanup()

    def test_selects_only_affordable_clients(self):
        selected = self.selector.select_clients(1, {"client1": 100, "client2": 100, "client3": 100}, 1, 2)
        self.assertEqual(len(selected), 2)
        self.assertNotIn("client3", selected)

    def test_charging_never_goes_negative(self):
        selected = self.selector.select_clients(1, {"client1": 100}, 1, 1)
        self.selector.charge_selected(1, selected, {"client1": 100}, 1)
        self.assertGreaterEqual(self.selector.clients["client1"].remaining_energy_units, 0)

if __name__ == "__main__":
    unittest.main()
