
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
CLIENT_DIR = PROJECT_DIR / "CLIENT_DATA"
TEST_FILE = PROJECT_DIR / "data" / "independent_test.csv"

NUMBER_OF_CLIENTS = 5
ROUNDS = 5
LOCAL_EPOCHS = 3
RANDOM_SEED = 42
LABEL_COLUMN = "Label"


def load_dataset(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing '{LABEL_COLUMN}' in {file_path.name}")

    labels = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")

    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(
            f"{file_path.name} must contain only labels 0 and 1."
        )

    features = df.drop(columns=[LABEL_COLUMN]).apply(
        pd.to_numeric, errors="coerce"
    )
    features = features.replace([np.inf, -np.inf], np.nan)

    valid_rows = features.notna().all(axis=1)
    features = features.loc[valid_rows].reset_index(drop=True)
    labels = labels.loc[valid_rows].astype(int).reset_index(drop=True)

    if features.empty:
        raise ValueError(f"No valid rows in {file_path.name}")

    if features.shape[1] == 0:
        raise ValueError(f"No feature columns in {file_path.name}")

    return features, labels


def average_client_models(client_models, client_sizes):
    if not client_models or len(client_models) != len(client_sizes):
        raise ValueError("Client models and sizes must match.")

    total_size = sum(client_sizes)

    if total_size <= 0:
        raise ValueError("Total client training size must be positive.")

    coefficients = sum(
        model.coef_ * size
        for model, size in zip(client_models, client_sizes)
    ) / total_size

    intercepts = sum(
        model.intercept_ * size
        for model, size in zip(client_models, client_sizes)
    ) / total_size

    return coefficients, intercepts


def prepare_data():
    client_frames = []
    feature_columns = None

    for client_id in range(1, NUMBER_OF_CLIENTS + 1):
        client_file = CLIENT_DIR / f"iot_client_{client_id}.csv"
        X, y = load_dataset(client_file)

        if feature_columns is None:
            feature_columns = list(X.columns)
        elif list(X.columns) != feature_columns:
            raise ValueError(
                f"Feature columns or order differ in "
                f"{client_file.name}."
            )

        if set(y.unique()) != {0, 1}:
            raise ValueError(
                f"{client_file.name} must contain both classes."
            )

        client_frames.append((X, y))

    X_test, y_test = load_dataset(TEST_FILE)

    if list(X_test.columns) != feature_columns:
        raise ValueError(
            "Independent test feature columns or order do not match "
            "the client training data."
        )

    # Fit preprocessing on training data only.
    X_all_train = pd.concat(
        [X for X, _ in client_frames],
        ignore_index=True,
    )

    scaler = StandardScaler()
    scaler.fit(X_all_train)

    scaled_clients = []

    for X, y in client_frames:
        scaled_clients.append(
            (scaler.transform(X), y.to_numpy(dtype=int))
        )

    X_test_scaled = scaler.transform(X_test)

    print("Training records:", len(X_all_train))
    print("Independent test records:", len(X_test))
    print("Scaler fitted on training records only.")

    print("\nIndependent test class counts:")
    print(y_test.value_counts().sort_index().rename(
        index={0: "Normal", 1: "DDoS"}
    ))

    return (
        scaled_clients,
        X_test_scaled,
        y_test.to_numpy(dtype=int),
    )


def main():
    print("Loading client training data and independent test data...")

    client_data, X_test, y_test = prepare_data()

    print("\nClient training distributions:")

    for client_id, (_, y_client) in enumerate(client_data, start=1):
        print(
            f"Client {client_id}: "
            f"Normal={(y_client == 0).sum()}, "
            f"DDoS={(y_client == 1).sum()}, "
            f"Total={len(y_client)}"
        )

    # Initialize the global model.
    first_X, first_y = client_data[0]

    global_model = SGDClassifier(
        loss="log_loss",
        random_state=RANDOM_SEED,
        max_iter=1000,
        tol=1e-3,
    )

    initial_size = min(100, len(first_X))

    global_model.partial_fit(
        first_X[:initial_size],
        first_y[:initial_size],
        classes=np.array([0, 1]),
    )

    print("\nStarting federated training...")

    for round_no in range(1, ROUNDS + 1):
        client_models = []
        client_sizes = []

        for client_id, (X_client, y_client) in enumerate(
            client_data, start=1
        ):
            local_model = SGDClassifier(
                loss="log_loss",
                random_state=RANDOM_SEED + client_id,
            )

            local_model.coef_ = global_model.coef_.copy()
            local_model.intercept_ = global_model.intercept_.copy()
            local_model.classes_ = global_model.classes_.copy()
            local_model.n_features_in_ = global_model.n_features_in_

            rng = np.random.default_rng(
                RANDOM_SEED + round_no * 100 + client_id
            )

            for _ in range(LOCAL_EPOCHS):
                indices = rng.permutation(len(X_client))
                local_model.partial_fit(
                    X_client[indices],
                    y_client[indices],
                )

            client_models.append(local_model)
            client_sizes.append(len(X_client))

        global_model.coef_, global_model.intercept_ = (
            average_client_models(client_models, client_sizes)
        )

        # Monitoring uses the independent test set.
        # For strict experimental practice, use a separate validation
        # set for round-by-round monitoring and test only at the end.
        predictions = global_model.predict(X_test)

        print(f"\nRound {round_no}/{ROUNDS}")
        print(f"  Accuracy:  {accuracy_score(y_test, predictions):.4f}")
        print(
            f"  Precision: "
            f"{precision_score(y_test, predictions, zero_division=0):.4f}"
        )
        print(
            f"  Recall:    "
            f"{recall_score(y_test, predictions, zero_division=0):.4f}"
        )
        print(
            f"  F1-score:  "
            f"{f1_score(y_test, predictions, zero_division=0):.4f}"
        )

    # Final independent evaluation.
    final_predictions = global_model.predict(X_test)

    print("\nFinal independent test results")
    print(f"Accuracy:  {accuracy_score(y_test, final_predictions):.4f}")
    print(
        f"Precision: "
        f"{precision_score(y_test, final_predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:    "
        f"{recall_score(y_test, final_predictions, zero_division=0):.4f}"
    )
    print(
        f"F1-score:  "
        f"{f1_score(y_test, final_predictions, zero_division=0):.4f}"
    )

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, final_predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            final_predictions,
            labels=[0, 1],
            target_names=["Normal", "DDoS"],
            zero_division=0,
        )
    )

    print("Federated learning completed.")
    print("Training source: CLIENT_DATA")
    print("Evaluation source: data/independent_test.csv")
    print("Clients:", NUMBER_OF_CLIENTS)
    print("Communication rounds:", ROUNDS)
    print("Local epochs:", LOCAL_EPOCHS)


if __name__ == "__main__":
    main()
