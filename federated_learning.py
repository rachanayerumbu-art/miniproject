
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def average_client_models(client_models, client_sizes):
    """Average model parameters, weighted by client training-set size."""
    total = sum(client_sizes)

    if total == 0:
        raise ValueError("Total client training samples must be positive.")

    coefficients = sum(
        model.coef_ * size
        for model, size in zip(client_models, client_sizes)
    ) / total

    intercepts = sum(
        model.intercept_ * size
        for model, size in zip(client_models, client_sizes)
    ) / total

    return coefficients, intercepts


def main():
    # Temporary synthetic data. Replace with Rachana's prepared data later.
    X, y = make_classification(
        n_samples=2000,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit preprocessing on training data only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Split training examples among five simulated clients.
    client_data = np.array_split(
        np.random.default_rng(42).permutation(len(X_train)), 5
    )

    # Initialize a binary logistic model.
    global_model = SGDClassifier(
        loss="log_loss", random_state=42
    )
    global_model.partial_fit(
        X_train[client_data[0]],
        y_train[client_data[0]],
        classes=np.array([0, 1]),
    )

    rounds = 5
    local_epochs = 3

    for round_no in range(1, rounds + 1):
        client_models = []
        client_sizes = []

        # Each client starts from the current global parameters.
        for indices in client_data:
            local_model = SGDClassifier(
                loss="log_loss", random_state=42
            )
            local_model.coef_ = global_model.coef_.copy()
            local_model.intercept_ = global_model.intercept_.copy()
            local_model.classes_ = global_model.classes_.copy()
            local_model.n_features_in_ = X_train.shape[1]

            for _ in range(local_epochs):
                local_model.partial_fit(X_train[indices], y_train[indices])

            client_models.append(local_model)
            client_sizes.append(len(indices))

        # FedAvg: aggregate the client model parameters.
        global_model.coef_, global_model.intercept_ = (
            average_client_models(client_models, client_sizes)
        )

        predictions = global_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(
            f"Round {round_no}/{rounds} | "
            f"Test accuracy: {accuracy:.4f}"
        )

    print("\nSynthetic-data FL test completed.")
    print("These results are NOT CICIoT2023 project results.")


if __name__ == "__main__":
    main()
