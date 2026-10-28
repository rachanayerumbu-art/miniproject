
from pathlib import Path
import csv

# Dataset folder is inside the current project folder
DATASET_FOLDER = Path(__file__).parent / "MERGED_CSV"

# Number of sample rows to inspect per CSV
SAMPLE_ROWS = 3


def inspect_csv(file_path):
    """Print a CSV's columns and a few sample values."""
    size_mb = file_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 75)
    print(f"File: {file_path.name}")
    print(f"Size: {size_mb:.2f} MB")

    try:
        with file_path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline=""
        ) as file:
            reader = csv.reader(file)

            header = next(reader, None)

            if not header:
                print("This file is empty or has no header.")
                return

            # Remove accidental spaces around column names
            header = [column.strip() for column in header]

            print(f"\nTotal columns: {len(header)}")
            print("\nColumn names:")

            for column in header:
                print(f"  - {column}")

            # Find columns that might contain class labels
            possible_labels = [
                column
                for column in header
                if any(
                    word in column.lower()
                    for word in ("label", "attack", "class", "category")
                )
            ]

            print("\nPossible label columns:")
            if possible_labels:
                for column in possible_labels:
                    print(f"  - {column}")
            else:
                print("  No obvious label column found.")

            # Print a few sample values without reading the whole file
            print(f"\nFirst {SAMPLE_ROWS} data rows:")

            for row_number, row in enumerate(reader, start=1):
                if row_number > SAMPLE_ROWS:
                    break

                print(f"\nRow {row_number}:")

                for index, column in enumerate(header):
                    value = row[index] if index < len(row) else ""
                    print(f"  {column}: {value}")

            # Re-open the file to count label values in a small sample
            if possible_labels:
                print("\nSample label values:")

                with file_path.open(
                    "r",
                    encoding="utf-8-sig",
                    errors="replace",
                    newline=""
                ) as file:
                    reader = csv.DictReader(file)

                    for column in possible_labels:
                        values = []

                        for row_number, row in enumerate(reader):
                            if row_number >= 20:
                                break

                            value = (row.get(column) or "").strip()
                            if value and value not in values:
                                values.append(value)

                        print(f"  {column}: {values[:10]}")

    except Exception as error:
        print(f"Could not inspect this file: {error}")


def main():
    if not DATASET_FOLDER.is_dir():
        print(f"ERROR: Dataset folder not found:\n{DATASET_FOLDER}")
        print("Make sure MERGED_CSV is inside your miniproject-1 folder.")
        return

    csv_files = sorted(DATASET_FOLDER.glob("*.csv"))

    print("=" * 75)
    print("CICIoT2023 DATASET INSPECTION")
    print(f"Dataset folder: {DATASET_FOLDER}")
    print(f"CSV files found: {len(csv_files)}")
    print("=" * 75)

    if not csv_files:
        print("No CSV files found in MERGED_CSV.")
        return

    for file_path in csv_files:
        inspect_csv(file_path)

    print("\n" + "=" * 75)
    print("INSPECTION COMPLETE")
    print("Send the output to ChatGPT so we can prepare the five-device dataset.")
    print("=" * 75)


if __name__ == "__main__":
    main()
