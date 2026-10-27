# Security-Aware and Energy-Efficient Federated Learning for IoT Intrusion Detection

## Project Folder Structure

```text
miniproject/
│
├── frontend/                    # Your website
│   ├── index.html               # Main webpage
│   ├── style.css                # Website design
│   ├── script.js                # Buttons, charts, interactions
│   └── assets/                  # Images and icons
│
├── backend/                     # Python web server
│   ├── app.py                   # Connects webpage to Python code
│   └── routes/
│       └── results.py           # Results endpoints
│
├── data/
│   ├── raw/                     # Original CICIoT2023 CSV files
│   ├── processed/               # Cleaned BENIGN/DDoS data
│   └── README.md                # Dataset information
│
├── security/                    # Security-related modules
│
├── src/                         # Python project source code
│   ├── data_preprocessing.py    # Dataset preprocessing
│   ├── federated_learning.py    # Federated learning implementation
│   └── energy_aware_fl.py       # Energy-aware client selection
│
├── configs/
│   ├── energy_config.json       # Energy configuration
│   └── client_energy_config.csv # Client energy settings
│
├── tests/
│   └── test_energy_aware_fl.py  # Energy-aware FL tests
│
├── results/
│   ├── sample_outputs/          # Example experiment outputs
│   └── figures/                 # Graphs and visualizations
│
├── demo_energy_selection.py     # Energy selection demo
├── integration_example.py       # Integration example
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── .gitignore                   # Files excluded from Git
```

## Project Overview

This project aims to develop a security-aware and energy-efficient Federated Learning (FL) system for IoT intrusion detection using the CICIoT2023 dataset.

The initial detection task focuses on binary classification:

- **BENIGN:** Normal network traffic.
- **DDoS:** Distributed Denial-of-Service attack traffic.

## Main Objectives

1. Detect DDoS attacks in IoT network traffic.
2. Train a model using Federated Learning without centralizing all client training data.
3. Explore security mechanisms to reduce the impact of malicious clients.
4. Implement energy-aware client selection to reduce estimated training and communication costs.
5. Evaluate detection performance, energy consumption estimates, and communication overhead.

## Technologies

- Python
- HTML, CSS, and JavaScript
- TensorFlow / Keras
- Pandas and NumPy
- Scikit-learn
- Flask (planned backend, if used)

## Dataset

The project uses the CICIoT2023 dataset. Raw data should be stored in `data/raw/`, and cleaned data for the binary classification task should be stored in `data/processed/`.

Large dataset files should not be committed to Git unless appropriate storage and repository limits have been considered.

## Project Status

The folder structure is a starting point. Individual modules, frontend-backend integration, security defenses, and energy-aware experiments will be implemented and tested as the project progresses.
