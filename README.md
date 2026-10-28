
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

## Main Objectives

1. Detect BENIGN and DDoS network traffic.
2. Train a global model using federated learning and FedAvg.
3. Explore security-aware defenses against malicious client updates.
4. Implement energy-aware client selection.
5. Visualize model performance and energy-related metrics on a web dashboard.
