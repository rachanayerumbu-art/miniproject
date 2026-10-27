
from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


def main():
    # Update this path after Rachana confirms the prepared CSV filename.
    data_path = Path("data/prepared_dataset.csv")

    if not data_path.exists():
        print(f"Dataset not found: {data_path.resolve()}")
        print("Ask Rachana for the prepared CSV filename and location.")
        return

    df = pd.read_csv(data_path)

    # Change 'label' if Rachana's label column has a different name.
    label_column = "label"

    if label_column not in df.columns:
        print("Label column not found.")
        print("Available columns:", df.columns.tolist())
        return

    df = df.dropna(subset=[label_column])
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # This starter assumes all feature columns are numeric.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([float("inf"), float("-inf")], float("nan"))

    # Remove rows with invalid feature values.
    valid_rows = X.notna().all(axis=1)
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if y.nunique() < 2:
        print("The dataset must contain at least two classes.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Baseline model results")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
