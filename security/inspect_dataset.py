
from pathlib import Path
import csv

DATASET_FOLDER = Path(__file__).parent.parent / "MERGED_CSV"


def inspect_csv(file_path):
    print("\n" + "=" * 60)
    print(f"File: {file_path.name}")
    print(f"Size: {file_path.stat().st_size / (1024 * 1024):.2f} MB")

    try:
        with file_path.open(
            "r", encoding="utf-8-sig", errors="replace", newline=""
        ) as f:
            reader = csv.reader(f)
            header = next(reader, None)

            if not header:
                print("Empty file")
                return

            header = [column.strip() for column in header]
            print(f"\nColumns ({len(header)}):")
            print(header)

            label_columns = [
                column for column in header
                if any(word in column.lower()
                       for word in ("label", "attack", "class", "category"))
            ]

            print("\nPossible label columns:", label_columns or "None found")

            for row_number, row in enumerate(reader):
                if row_number >= 3:
                    break
                print(f"\nSample row {row_number + 1}:")
                print(dict(zip(header, row)))

            for column in label_columns:
                values = set()

                with file_path.open(
                    "r", encoding="utf-8-sig",
                    errors="replace", newline=""
                ) as f:
                    reader = csv.DictReader(f)

                    for row in reader:
                        value = (row.get(column) or "").strip()
                        if value:
                            values.add(value)
                        if len(values) >= 20:
                            break

                print(f"\n{column} sample values:")
                print(sorted(values))

    except Exception as error:
        print("Error:", error)


def main():
    if not DATASET_FOLDER.is_dir():
        print(f"Dataset folder not found: {DATASET_FOLDER}")
        return

    csv_files = sorted(DATASET_FOLDER.glob("*.csv"))
    print(f"CSV files found: {len(csv_files)}")

    if not csv_files:
        print("No CSV files found.")
        return

    for file_path in csv_files:
        inspect_csv(file_path)


if __name__ == "__main__":
    main()
