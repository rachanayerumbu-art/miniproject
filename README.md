
# Security-Aware and Energy-Efficient Federated Learning for IoT Intrusion Detection

## Project Structure

```text

miniproject/
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── main.js
│   │   └── dashboard.js
│   └── assets/
│       ├── images/
│       └── icons/
│
├── backend/
│   ├── security/
│   │   ├── baseline.py
│   │   └── inspect_dataset.py
│   │
│   ├── federated_learning.py
│   ├── energy_aware_fl.py
│   ├── energy_client_selector.py
│   ├── demo_energy_selection.py
│   ├── integration_example.py
│   └── test_energy_aware_fl.py
│
├── data/
│   ├── MERGED_CSV/
│   │   ├── client_01.csv
│   │   ├── client_02.csv
│   │   └── ... (remaining original CSV files)
│   │
│   └── CLIENT_DATA/
│       ├── iot_client_1.csv
│       ├── iot_client_2.csv
│       ├── iot_client_3.csv
│       ├── iot_client_4.csv
│       └── iot_client_5.csv
│
├── config/
│   ├── client_energy_config.csv
│   └── energy_config.json
│
├── results/
│   └── sample_outputs/
│       ├── energy_final_state.json
│       └── energy_round_log.csv
│
├── scripts/
│   ├── distribute_iot_clients.py
│
│
├── docs/
│
├── README.md
├── requirements.txt
└── .gitignore

```

## Project Modules

- **Frontend:** Web interface built using HTML, CSS, and JavaScript.
- **Backend:** Python scripts for federated learning, energy-aware client selection, and security experiments.
- **Data:** CICIoT2023 dataset files used for intrusion detection.
- **Results:** Experiment outputs, including energy state and training-round logs.
- **Documentation:** Project reports and technical documentation.

## Week 1 Energy Design

Each client has a configurable initial energy budget and estimated local
training and communication costs in `energy_config.py`. For a participating
client, the simulation deducts:

```text
energy used per round = estimated training cost + estimated communication cost
```

The configured minimum threshold is retained after that deduction. These
abstract values are reproducible simulation units, not hardware measurements.
The current energy and training flow is shown in
[`energy_architecture.mmd`](./energy_architecture.mmd).

The existing trainer currently uses scikit-learn `SGDClassifier`; it is not a
neural-network implementation. The Week 1 energy design does not change the
trainer or teammates' dataset/security modules.

For the standalone Week 2 demonstrations, run:

```powershell
python energy_simulator.py
python energy_aware_selection.py
```

The simulator command demonstrates one participation per client. The
energy-aware selection command demonstrates five rounds, rechecking client
eligibility and updating the simulated budgets each round.

## Running Federated Training

Run these commands from the project root:

```powershell
python federated_learning.py
python federated_learning.py --energy-aware
python federated_learning.py --compare-energy
```

The first command keeps standard all-client participation. The second uses
simulated client energy budgets and writes each round's participation and
estimated energy use to `energy_federated_rounds.csv`. Energy values are
illustrative simulation units, not hardware battery measurements.

The comparison command runs both policies with the same prepared client data,
random seeds, and held-out test set. It writes participation, per-round metric,
and final summary CSVs to a new `week3_energy_comparison` directory. Existing
result directories are never overwritten; use `--results-dir` to choose a
different directory for another run.

The energy-aware command does not overwrite an existing log. To save another
run, provide a new path, for example:

```powershell
python federated_learning.py --energy-aware --energy-log energy_federated_rounds_run2.csv
```

## Week 4 Comparison, Graphs, and Dashboard

Install the project's Python dependencies, then run the comparison and graph
scripts:

```powershell
python -m pip install -r requirements.txt
python federated_learning.py --compare-energy
python compare_results.py
python visualize_results.py
streamlit run dashboard.py
```

The first command creates the Week 3 standard-versus-energy-aware experiment
CSVs used by the Week 4 scripts.
`compare_results.py` creates the `week4_results` directory and report.
`visualize_results.py` adds five PNG graphs under
`week4_results/presentation_graphs`. Both scripts refuse to replace existing
outputs; use their `--results-dir` options for a new run. The dashboard reads
the Week 3 and Week 4 CSVs and displays unavailable security-defense runs as
unavailable rather than filling in made-up results.

## Main Objectives

1. Detect BENIGN and DDoS network traffic.
2. Train a global model using federated learning and FedAvg.
3. Explore security-aware defenses against malicious client updates.
4. Implement energy-aware client selection.
5. Visualize model performance and energy-related metrics on a web dashboard.
