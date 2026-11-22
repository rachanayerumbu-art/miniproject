
from pathlib import Path
import pandas as pd

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "MERGED_CSV"
OUTPUT_DIR = PROJECT_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "prepared_dataset.csv"


def main():
    # Find the original client datasets
    files = sorted(INPUT_DIR.glob("client_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No client_*.csv files found in {INPUT_DIR}"
        )

    print(f"Found {len(files)} client CSV files.")

    dataframes = []
    expected_columns = None
    total_rows_before = 0

    for file in files:
        print(f"Reading: {file.name}")
        df = pd.read_csv(file)

        # Clean accidental whitespace in column names
        df.columns = [str(col).strip() for col in df.columns]

        # Find the label column regardless of capitalization
        label_columns = [
            col for col in df.columns
            if col.lower() == "label"
        ]

        if len(label_columns) != 1:
            raise ValueError(
                f"{file.name}: expected exactly one Label column; "
                f"found {label_columns}"
            )

        # Standardize the target column name
        df = df.rename(columns={label_columns[0]: "Label"})

        if df.columns.duplicated().any():
            raise ValueError(
                f"{file.name}: duplicate column names detected."
            )

        # Ensure every client has compatible features
        if expected_columns is None:
            expected_columns = list(df.columns)
        elif set(df.columns) != set(expected_columns):
            missing = sorted(set(expected_columns) - set(df.columns))
            extra = sorted(set(df.columns) - set(expected_columns))
            raise ValueError(
                f"{file.name} has a different schema.\n"
                f"Missing columns: {missing}\n"
                f"Extra columns: {extra}"
            )

        # Reorder columns consistently
        df = df[expected_columns]

        # Clean label text
        df["Label"] = df["Label"].astype("string").str.strip()
        df["Label"] = df["Label"].replace("", pd.NA)

        total_rows_before += len(df)
        dataframes.append(df)

    # Combine for the centralized baseline
    combined = pd.concat(dataframes, ignore_index=True)

    # Make all input features numeric
    feature_columns = [
        col for col in combined.columns if col != "Label"
    ]

    for col in feature_columns:
        combined[col] = pd.to_numeric(
            combined[col], errors="coerce"
        )

    # Remove rows with missing labels or invalid feature values
    invalid_rows = (
        combined["Label"].isna()
        | combined[feature_columns].isna().any(axis=1)
    )

    removed_rows = int(invalid_rows.sum())
    combined = combined.loc[~invalid_rows].copy()

    # Reject infinite feature values
    finite_rows = combined[feature_columns].apply(
        lambda column: column.map(
            lambda value: pd.notna(value) and abs(value) != float("inf")
        )
    ).all(axis=1)

    infinite_rows_removed = int((~finite_rows).sum())
    combined = combined.loc[finite_rows].copy()

    if combined.empty:
        raise ValueError("No valid rows remain after cleaning.")

    # Save the prepared dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print("\n--- Dataset preparation complete ---")
    print(f"Input files: {len(files)}")
    print(f"Rows before cleaning: {total_rows_before}")
    print(f"Rows removed for missing/invalid values: {removed_rows}")
    print(f"Rows removed for infinite values: {infinite_rows_removed}")
    print(f"Rows saved: {len(combined)}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Label distribution:\n{combined['Label'].value_counts()}")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
