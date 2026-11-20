from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
CLIENT_DIR = PROJECT_DIR / "CLIENT_DATA"
TRAIN_FILE = PROJECT_DIR / "data" / "training_pool.csv"
TEST_FILE = PROJECT_DIR / "data" / "independent_test.csv"
REPORT_FILE = PROJECT_DIR / "client_validation_report.csv"
DISTRIBUTION_FILE = PROJECT_DIR / "client_distribution.csv"
RANDOM_SEED = 42
LABEL_COLUMN = "Label"


def find_label_column(columns):
    matches = [column for column in columns if str(column).strip().lower() == "label"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Label column, found {matches}")
    return matches[0]


def normalize_labels(series):
    labels = series.astype("string").str.strip().str.lower()
    mapping = {"0": 0, "benign": 0, "normal": 0, "1": 1, "ddos": 1}
    mapped = labels.map(mapping)
    return mapped


def main():
    np.random.seed(RANDOM_SEED)
    client_files = [
        CLIENT_DIR / f"iot_client_{client_id}.csv"
        for client_id in range(1, 6)
    ]

    missing_files = [str(path) for path in client_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing client CSV files:\n" + "\n".join(missing_files)
        )

    frames = []
    for path in client_files:
        frame = pd.read_csv(path)
        source_label = find_label_column(frame.columns)
        if source_label != LABEL_COLUMN:
            frame = frame.rename(columns={source_label: LABEL_COLUMN})
        frame.columns = [str(column).strip() for column in frame.columns]
        frames.append((path, frame))

    reference_columns = list(frames[0][1].columns)
    reference_types = frames[0][1].dtypes.astype(str).to_dict()

    report_rows = []
    distribution_rows = []

    for client_id, (path, frame) in enumerate(frames, start=1):
        feature_frame = frame.drop(columns=[LABEL_COLUMN])
        numeric_features = feature_frame.apply(pd.to_numeric, errors="coerce")
        missing_count = int(frame.isna().sum().sum())
        inf_count = int(
            np.isinf(numeric_features.to_numpy(dtype=float, na_value=np.nan)).sum()
        )
        duplicate_count = int(frame.duplicated().sum())
        labels = normalize_labels(frame[LABEL_COLUMN])

        if labels.isna().any():
            unknown = sorted(
                frame.loc[labels.isna(), LABEL_COLUMN]
                .astype("string")
                .dropna()
                .unique()
                .tolist()
            )
            raise ValueError(f"{path.name} contains unknown labels: {unknown}")

        benign_count = int((labels == 0).sum())
        ddos_count = int((labels == 1).sum())
        total = len(frame)

        report_rows.append({
            "Client": f"Client {client_id}",
            "File": path.name,
            "Exists": True,
            "Rows": total,
            "Columns": len(frame.columns),
            "Columns_Consistent": list(frame.columns) == reference_columns,
            "Dtypes_Consistent": frame.dtypes.astype(str).to_dict() == reference_types,
            "Missing_Values": missing_count,
            "Infinite_Values": inf_count,
            "Duplicate_Rows": duplicate_count,
            "BENIGN_Count": benign_count,
            "DDoS_Count": ddos_count,
            "BENIGN_Percentage": round(100 * benign_count / total, 4),
            "DDoS_Percentage": round(100 * ddos_count / total, 4),
            "Both_Labels_Present": benign_count > 0 and ddos_count > 0,
        })

        distribution_rows.append({
            "Client": f"Client {client_id}",
            "BENIGN": benign_count,
            "DDoS": ddos_count,
            "BENIGN_Percentage": 100 * benign_count / total,
            "DDoS_Percentage": 100 * ddos_count / total,
        })

    combined = pd.concat(
        [frame.assign(_client=path.name) for path, frame in frames],
        ignore_index=True,
    )
    cross_client_duplicates = int(combined.duplicated(subset=[column for column in combined.columns if column != "_client"]).sum())

    train_test_overlap = 0
    train_test_files_exist = TRAIN_FILE.exists() and TEST_FILE.exists()
    train_duplicate_count = 0
    test_duplicate_count = 0

    if train_test_files_exist:
        train = pd.read_csv(TRAIN_FILE)
        test = pd.read_csv(TEST_FILE)
        train_duplicate_count = int(train.duplicated().sum())
        test_duplicate_count = int(test.duplicated().sum())

        if list(train.columns) == list(test.columns):
            train_hash = pd.util.hash_pandas_object(train, index=False)
            test_hash = pd.util.hash_pandas_object(test, index=False)
            train_hashes = set(train_hash.tolist())
            train_test_overlap = int(test_hash.isin(train_hashes).sum())

    report = pd.DataFrame(report_rows)
    report["Cross_Client_Duplicate_Rows"] = cross_client_duplicates
    report["Training_Test_Overlap"] = train_test_overlap
    report["Training_Duplicate_Rows"] = train_duplicate_count
    report["Test_Duplicate_Rows"] = test_duplicate_count
    report["Train_Test_Files_Checked"] = train_test_files_exist
    report.to_csv(REPORT_FILE, index=False)

    pd.DataFrame(distribution_rows).to_csv(DISTRIBUTION_FILE, index=False)

    print(report.to_string(index=False))
    print("\nCross-client duplicate rows:", cross_client_duplicates)
    print("Training/testing exact overlap:", train_test_overlap)
    print("Training duplicates:", train_duplicate_count)
    print("Test duplicates:", test_duplicate_count)
    print("Saved:", REPORT_FILE)
    print("Saved:", DISTRIBUTION_FILE)


if __name__ == "__main__":
    main()
