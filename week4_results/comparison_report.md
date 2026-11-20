# Experiment Comparison Report

All completed models were evaluated on the existing independent test set. Federated experiment results reuse the Week 3 run; the centralized baseline uses the existing training pool.

| Experiment | Status | Accuracy | Precision | Recall | F1 | FPR | Energy units |
|---|---|---:|---:|---:|---:|---:|---:|
| Centralized Random Forest | completed | 0.9999 | 1.0000 | 0.9999 | 0.9999 | 0.0000 | N/A |
| Standard FedAvg | completed | 0.9999 | 1.0000 | 0.9998 | 0.9999 | 0.0000 | 75.0000 |
| Energy-aware FedAvg | completed | 0.9998 | 1.0000 | 0.9997 | 0.9998 | 0.0000 | 50.7000 |
| Federated Learning with Security Defense | unavailable | N/A | N/A | N/A | N/A | N/A | N/A |
| Security Defense with Energy-Aware Selection | unavailable | N/A | N/A | N/A | N/A | N/A | N/A |

Energy values are simulated estimates based on the configured per-client training and communication costs. They are not hardware battery measurements.

Security-defense experiments are marked unavailable because the inspected project contains no federated security-defense implementation. No performance values were invented for them.

The centralized model is a Random Forest; the existing federated trainer uses scikit-learn SGDClassifier.

## Standard vs. Energy-Aware FedAvg

The measured simulation totals are 75.00 and 50.70 estimated units, respectively. That is a 32.40% reduction under this simulation's cost assumptions, not a hardware energy claim.
Energy-aware accuracy differs by 0.00519 percentage points (standard minus energy-aware).
