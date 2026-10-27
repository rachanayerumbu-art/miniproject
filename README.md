
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
│   ├── energy_aware_fl.py
│   ├── federated_learning.py
│   ├── demo_energy_selection.py
│   ├── integration_example.py
│   ├── client_energy_config.csv
│   ├── energy_config.json
│   └── test_energy_aware_fl.py
│
├── data/
│   └── MERGED_CSV/
│
├── results/
│   └── sample_outputs/
│       ├── energy_final_state.json
│       └── energy_round_log.csv
│
├── docs/
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

## Main Objectives

1. Detect BENIGN and DDoS network traffic.
2. Train a global model using federated learning and FedAvg.
3. Explore security-aware defenses against malicious client updates.
4. Implement energy-aware client selection.
5. Visualize model performance and energy-related metrics on a web dashboard.
