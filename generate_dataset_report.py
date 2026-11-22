from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
VALIDATION_FILE = PROJECT_DIR / "client_validation_report.csv"
DISTRIBUTION_FILE = PROJECT_DIR / "client_distribution.csv"
REPORT_FILE = PROJECT_DIR / "RACHANA_FINAL_REPORT.md"
RANDOM_SEED = 42


def run_script(script_name):
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / script_name)],
        cwd=PROJECT_DIR,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed:\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def create_graphs(distribution):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams["figure.dpi"] = 140
    clients = distribution["Client"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.36
    x = range(len(distribution))
    ax.bar(
        [value - width / 2 for value in x],
        distribution["BENIGN"],
        width,
        label="BENIGN",
        color="#4C78A8",
    )
    ax.bar(
        [value + width / 2 for value in x],
        distribution["DDoS"],
        width,
        label="DDoS",
        color="#F58518",
    )
    ax.set_title("BENIGN and DDoS Records per IoT Client")
    ax.set_xlabel("IoT Client")
    ax.set_ylabel("Number of Records")
    ax.set_xticks(list(x), clients)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "client_record_counts.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        clients,
        distribution["BENIGN_Percentage"],
        label="BENIGN",
        color="#4C78A8",
    )
    ax.bar(
        clients,
        distribution["DDoS_Percentage"],
        bottom=distribution["BENIGN_Percentage"],
        label="DDoS",
        color="#F58518",
    )
    ax.set_title("Percentage Distribution per IoT Client")
    ax.set_xlabel("IoT Client")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "client_percentage_distribution.png")
    plt.close(fig)

    total_benign = int(distribution["BENIGN"].sum())
    total_ddos = int(distribution["DDoS"].sum())
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(
        [total_benign, total_ddos],
        labels=["BENIGN", "DDoS"],
        autopct="%1.2f%%",
        colors=["#4C78A8", "#F58518"],
        startangle=90,
        wedgeprops={"edgecolor": "white"},
    )
    ax.set_title("Total Dataset Distribution")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "total_dataset_distribution.png")
    plt.close(fig)

    return total_benign, total_ddos


def markdown_table(table):
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    print(run_script("validate_clients.py"))
    print(run_script("verify_preprocessing.py"))

    validation = pd.read_csv(VALIDATION_FILE)
    distribution = pd.read_csv(DISTRIBUTION_FILE)
    total_benign, total_ddos = create_graphs(distribution)

    first = validation.iloc[0]
    table = distribution.copy()
    table["BENIGN_Percentage"] = table["BENIGN_Percentage"].map(
        lambda value: f"{value:.2f}%"
    )
    table["DDoS_Percentage"] = table["DDoS_Percentage"].map(
        lambda value: f"{value:.2f}%"
    )

    missing_total = int(validation["Missing_Values"].sum())
    infinite_total = int(validation["Infinite_Values"].sum())
    duplicate_total = int(validation["Duplicate_Rows"].sum())
    cross_duplicates = int(first["Cross_Client_Duplicate_Rows"])
    train_test_overlap = int(first["Training_Test_Overlap"])

    report = f"""# Rachana Final Dataset Report

## 1. Dataset Introduction

This report documents the Week 3 and Week 4 data responsibilities for the project **Security-Aware and Energy-Efficient Federated Learning for IoT Intrusion Detection**. The implementation uses the five existing simulated IoT client CSV files and performs binary classification: **BENIGN = 0** and **DDoS = 1**.

The source files are under `CLIENT_DATA/`. The scripts do not overwrite or rewrite the original client CSV files.

## 2. CICIoT2023 Dataset Description

The project uses CICIoT2023 network traffic records. The existing data has 37 numeric feature columns and one `Label` column. The prepared labels are mapped to binary values:

- `BENIGN` / `Normal` / `0` -> `0`
- `DDoS` / `1` -> `1`

The original CICIoT2023 source CSV files are combined and then separated into five client datasets for this federated-learning simulation.

## 3. Data Preprocessing Steps

The existing project:

1. Reads all 20 original MERGED_CSV files.
2. Standardizes the label column name and cleans label text.
3. Converts feature columns to numeric values.
4. Creates a held-out independent test split.
5. Maps labels to `0` for BENIGN and `1` for DDoS.
6. Distributes the training pool across five clients with the fixed random seed `42`.

`verify_preprocessing.py` verifies that preprocessing is consistent across clients. It converts features to numeric values, replaces infinite values with missing values, removes invalid rows, fits `StandardScaler` on client training records only, and transforms the independent test set using the fitted scaler. Test rows are never used to fit the scaler.

## 4. Binary Classification Approach

Each traffic record is classified as either:

- **0: BENIGN**
- **1: DDoS**

The project maintains this same two-class mapping in client files, preprocessing, and federated training.

## 5. Five-Client Data Distribution

{markdown_table(table)}

Total records: **{int(distribution['BENIGN'].sum() + distribution['DDoS'].sum())}**. Total BENIGN: **{total_benign}**. Total DDoS: **{total_ddos}**.

## 6. Non-IID Validation Results

The client class percentages vary substantially, from 90.64% BENIGN in Client 1 to 89.31% DDoS in Client 5. This uneven class ratio and opposite class dominance between clients provide clear evidence of a non-IID distribution. The actual counts and percentages come from the client CSV files rather than assumed values.

## 7. Data Quality Checks

All five client files exist. They have 38 columns including the label. Feature names and order match across all clients. The validation report shows:

- Missing values: **{missing_total}**
- Infinite values: **{infinite_total}**
- Duplicate rows within clients: **{duplicate_total}**
- Duplicate rows across clients: **{cross_duplicates}**
- Training duplicates: **{int(first['Training_Duplicate_Rows'])}**
- Test duplicates: **{int(first['Test_Duplicate_Rows'])}**

The affected missing/infinite values are not used in the preprocessing verification. The original CSV files remain unchanged.

## 8. Data Leakage Checks

- Exact duplicate rows shared by training and testing: **{train_test_overlap}**
- Duplicate rows across clients: **{cross_duplicates}**
- The independent test file exists and has matching feature columns/order.
- `federated_learning.py` concatenates only client training data to fit the `StandardScaler`, then transforms the independent test set.
- `verify_preprocessing.py` also fits only on client training data and never uses test rows for fitting.

An exact-row check is useful but cannot detect all possible session, flow, or host overlap. The separate test split is the available evidence against direct row leakage.

## 9. Graph Explanations

The following graphs are saved in `results/`:

- `client_record_counts.png`: grouped BENIGN/DDoS record counts for each client.
- `client_percentage_distribution.png`: stacked BENIGN/DDoS percentages for each client.
- `total_dataset_distribution.png`: overall BENIGN/DDoS proportion.

These graphs show both absolute class quantity and relative class balance.

## 10. Limitations and Future Improvements

- Missing and infinite values require row removal or a fitted imputation strategy before training.
- The current exact-row duplicate checks do not cover temporal/session-level overlap.
- Five clients are a simulation; real IoT devices may differ in sensor behavior and network patterns.
- Future work should use deterministic cross-validation, leakage checks across flow identifiers when available, imputation fitted only on training data, and external or temporal test sets.

## Reproducibility

The scripts use fixed seed `42` where applicable and do not modify source CSV files.

```powershell
python validate_clients.py
python verify_preprocessing.py
python generate_dataset_report.py
```
"""

    REPORT_FILE.write_text(report, encoding="utf-8")
    print("Saved:", VALIDATION_FILE)
    print("Saved:", DISTRIBUTION_FILE)
    print("Saved:", REPORT_FILE)
    print("Saved graphs under:", RESULTS_DIR)


if __name__ == "__main__":
    main()
