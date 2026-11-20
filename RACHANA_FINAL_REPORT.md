# Rachana Final Dataset Report

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

| Client | BENIGN | DDoS | BENIGN_Percentage | DDoS_Percentage |
| --- | --- | --- | --- | --- |
| Client 1 | 14386 | 1484 | 90.65% | 9.35% |
| Client 2 | 11989 | 3710 | 76.37% | 23.63% |
| Client 3 | 7992 | 7420 | 51.86% | 48.14% |
| Client 4 | 3996 | 11129 | 26.42% | 73.58% |
| Client 5 | 1599 | 13355 | 10.69% | 89.31% |

Total records: **77060**. Total BENIGN: **39962**. Total DDoS: **37098**.

## 6. Non-IID Validation Results

The client class percentages vary substantially, from 90.64% BENIGN in Client 1 to 89.31% DDoS in Client 5. This uneven class ratio and opposite class dominance between clients provide clear evidence of a non-IID distribution. The actual counts and percentages come from the client CSV files rather than assumed values.

## 7. Data Quality Checks

All five client files exist. They have 38 columns including the label. Feature names and order match across all clients. The validation report shows:

- Missing values: **6**
- Infinite values: **3**
- Duplicate rows within clients: **0**
- Duplicate rows across clients: **0**
- Training duplicates: **0**
- Test duplicates: **0**

The affected missing/infinite values are not used in the preprocessing verification. The original CSV files remain unchanged.

## 8. Data Leakage Checks

- Exact duplicate rows shared by training and testing: **0**
- Duplicate rows across clients: **0**
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
