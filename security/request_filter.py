from functools import lru_cache
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from .baseline import LABEL_COLUMN, RANDOM_SEED, TRAIN_FILE, load_data
except ImportError:
    from baseline import LABEL_COLUMN, RANDOM_SEED, TRAIN_FILE, load_data


@lru_cache(maxsize=1)
def load_request_model():
    """Load and train the centralized request model once."""
    X_train, y_train = load_data(TRAIN_FILE)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    feature_names = tuple(str(name) for name in X_train.columns)
    if len(feature_names) != 37:
        raise ValueError(
            f"Expected 37 traffic features, but found {len(feature_names)}"
        )

    return model, feature_names


def _record_to_values(record, feature_names):
    if isinstance(record, Mapping):
        record_keys = set(record.keys())
        expected_keys = set(feature_names)
        missing = [
            name for name in feature_names
            if name not in record_keys
        ]
        extra = sorted(
            str(name) for name in record_keys - expected_keys
        )

        if missing:
            return None, "Missing feature values: " + ", ".join(missing)
        if extra:
            return None, "Unexpected features: " + ", ".join(extra)

        values = [record[name] for name in feature_names]

    elif isinstance(record, pd.Series):
        if set(record.index) != set(feature_names):
            missing = [
                name for name in feature_names
                if name not in record.index
            ]
            extra = [
                str(name) for name in record.index
                if name not in feature_names
            ]
            details = []
            if missing:
                details.append(
                    "Missing feature values: " + ", ".join(missing)
                )
            if extra:
                details.append(
                    "Unexpected features: " + ", ".join(extra)
                )
            return None, "; ".join(details)

        values = [record[name] for name in feature_names]

    elif isinstance(record, (Sequence, np.ndarray)) and not isinstance(
        record, (str, bytes)
    ):
        array = np.asarray(record)
        if array.ndim != 1:
            return None, "Record must contain exactly one traffic record"
        if array.size != len(feature_names):
            return None, (
                f"Record must contain exactly {len(feature_names)} "
                f"features; received {array.size}"
            )
        values = array.tolist()

    else:
        return None, (
            "Record must be a feature mapping, pandas Series, "
            "or one-dimensional numeric sequence"
        )

    try:
        numeric_values = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None, "Feature values must be numeric"

    if numeric_values.size != len(feature_names):
        return None, (
            f"Record must contain exactly {len(feature_names)} "
            f"features; received {numeric_values.size}"
        )

    if not np.all(np.isfinite(numeric_values)):
        return None, "Feature values cannot be NaN or infinite"

    return pd.DataFrame(
        [numeric_values],
        columns=list(feature_names),
    ), None


def predict_request(record):
    """Classify one IoT traffic record and return an allow/block decision."""
    model, feature_names = load_request_model()
    values, validation_error = _record_to_values(record, feature_names)

    if validation_error is not None:
        return {
            "prediction": None,
            "status": "invalid",
            "reason": validation_error,
        }

    prediction = int(model.predict(values)[0])

    if prediction == 1:
        return {
            "prediction": 1,
            "status": "blocked",
            "reason": "Predicted DDoS traffic",
        }

    if prediction == 0:
        return {
            "prediction": 0,
            "status": "allowed",
            "reason": "Predicted normal traffic",
        }

    return {
        "prediction": prediction,
        "status": "invalid",
        "reason": f"Unexpected model prediction: {prediction}",
    }
