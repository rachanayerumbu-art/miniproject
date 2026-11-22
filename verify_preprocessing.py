from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
CLIENT_DIR = PROJECT_DIR / "CLIENT_DATA"
TEST_FILE = PROJECT_DIR / "data" / "independent_test.csv"
REPORT_FILE = PROJECT_DIR / "preprocessing_verification_report.csv"
LABEL_COLUMN = "Label"
RANDOM_SEED = 42


def load_clean_features(path):
    frame = pd.read_csv(path)
    if LABEL_COLUMN not in frame.columns:
        raise ValueError(f"Missing Label column in {path.name}")

    features = frame.drop(columns=[LABEL_COLUMN]).apply(
        pd.to_numeric, errors="coerce"
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    labels = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce")
    valid = features.notna().all(axis=1) & labels.isin([0, 1])
    return frame, features, labels, valid


def main():
    np.random.seed(RANDOM_SEED)
    client_frames = []
    reference_columns = None
    rows = []

    for client_id in range(1, 6):
        path = CLIENT_DIR / f"iot_client_{client_id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing client file: {path}")

        original, features, labels, valid = load_clean_features(path)
        if reference_columns is None:
            reference_columns = list(features.columns)
        elif list(features.columns) != reference_columns:
            raise ValueError(f"Feature columns differ in {path.name}")

        valid_features = features.loc[valid].reset_index(drop=True)
        valid_labels = labels.loc[valid].reset_index(drop=True)
        client_frames.append(valid_features)

        rows.append({
            "Client": f"Client {client_id}",
            "Original_Rows": len(original),
            "Valid_Rows": len(valid_features),
            "Invalid_Rows_Removed": int((~valid).sum()),
            "Has_NonIID_Labels": set(valid_labels.unique()) == {0, 1},
        })

    training_features = pd.concat(client_frames, ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(training_features)

    all_finite_after_scaling = True
    for client_id, features in enumerate(client_frames, start=1):
        transformed = scaler.transform(features)
        if not np.isfinite(transformed).all():
            all_finite_after_scaling = False
            raise ValueError(f"Scaled Client {client_id} contains invalid values")

    test_result = "Not checked"
    test_overlap_training = 0

    if TEST_FILE.exists():
        test = pd.read_csv(TEST_FILE)
        test_features = test.drop(columns=[LABEL_COLUMN]).apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        test_features = test_features.loc[test_features.notna().all(axis=1)]

        if list(test_features.columns) != reference_columns:
            raise ValueError("Test feature columns/order differ from client features")

        test_scaled = scaler.transform(test_features)
        test_result = "Passed"

        if not np.isfinite(test_scaled).all():
            raise ValueError("Independent test contains invalid values after scaling")

    report = pd.DataFrame(rows)
    report["Training_Feature_Count"] = len(reference_columns)
    report["Scaler_Fitted_On"] = "Five client training CSVs only"
    report["Test_Used_For_Fitting"] = False
    report["Independent_Test_Transform"] = test_result
    report["All_Scaled_Values_Finite"] = all_finite_after_scaling
    report.to_csv(REPORT_FILE, index=False)

    print(report.to_string(index=False))
    print("\nScaler mean and scale are fitted using client training data only.")
    print("Independent test was transformed, not fitted.")
    print("Saved:", REPORT_FILE)


if __name__ == "__main__":
    main()
