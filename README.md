miniproject/
│
├── frontend/                    # Your website
│   ├── index.html                # Main webpage
│   ├── style.css                 # Website design
│   ├── script.js                 # Buttons, charts, interactions
│   └── assets/                   # Images and icons
│
├── backend/                     # Python web server (add when needed)
│   ├── app.py                    # Connects webpage to Python code
│   └── routes/
│       └── results.py            # Results endpoints
│
├── data/
│   ├── raw/                      # Original CICIoT2023 CSV files
│   ├── processed/                # Cleaned binary BENIGN/DDoS data
│   └── README.md
│
├── security/                    # Keep your existing folder
│
├── src/                         # Reusable Python project code
│   ├── data_preprocessing.py
│   ├── federated_learning.py
│   └── energy_aware_fl.py
│
├── configs/
│   ├── energy_config.json
│   └── client_energy_config.csv
│
├── tests/
│   └── test_energy_aware_fl.py
│
├── results/
│   ├── sample_outputs/
│   └── figures/
│
├── demo_energy_selection.py
├── integration_example.py
├── README.md
├── requirements.txt
└── .gitignore
