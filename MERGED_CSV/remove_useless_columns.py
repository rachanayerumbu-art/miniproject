"""Safely remove only objectively useless columns from 20 client CSV files."""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd


CLIENT_COUNT = 20
ROWS_PER_FILE = 5_000
CHUNK_SIZE = 25_000
ALLOWED_LABELS = {"Benign", "DDoS"}
CONFIRMATION_PHRASE = "BACK UP AND MODIFY THE 20 CLIENT FILES"
INDEX_COLUMN = re.compile(r"^Unnamed(?::\s*\d+)?$", re.IGNORECASE)


def expected_client_paths(folder: Path) -> list[Path]:
    paths = [folder / f"client_{number:02d}.csv" for number in range(1, CLIENT_COUNT + 1)]
    missing = [path.name for path in paths if not path.is_file()]
    csv_files = [path for path in folder.glob("*.csv") if path.is_file()]
    unexpected = [path.name for path in csv_files if path not in paths]
    if missing or unexpected or len(csv_files) != CLIENT_COUNT:
        details = []
        if missing:
            details.append(f"missing client files: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected root-level CSV files: {', '.join(unexpected)}")
        raise ValueError("Expected exactly client_01.csv through client_20.csv; " + "; ".join(details))
    return paths


def read_chunked(path: Path):
    return pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )


def inspect_files(
    paths: list[Path],
) -> tuple[list[str], list[dict[str, object]], set[str], set[str], set[str]]:
    expected_columns: list[str] | None = None
    stats: list[dict[str, object]] = []
    labels: set[str] = set()
    empty_columns: set[str] | None = None
    constant_columns: set[str] | None = None
    first_values: dict[str, str] = {}

    for path in paths:
        rows = 0
        file_labels: set[str] = set()
        try:
            chunks = read_chunked(path)
            for chunk in chunks:
                columns = chunk.columns.tolist()
                if expected_columns is None:
                    expected_columns = columns
                    if columns.count("Label") != 1:
                        raise ValueError(f"{path.name} must contain exactly one Label column")
                    empty_columns = set(columns)
                    constant_columns = set(columns) - {"Label"}
                elif columns != expected_columns:
                    raise ValueError(
                        f"{path.name} has different column names or order from the first file"
                    )

                if not first_values and len(chunk):
                    first_values = {
                        column: str(chunk[column].iloc[0]) for column in columns
                    }

                for column in columns:
                    values = chunk[column]
                    if values.ne("").any():
                        empty_columns.discard(column)
                    if column != "Label" and column in constant_columns:
                        if len(values) and values.ne(first_values[column]).any():
                            constant_columns.discard(column)

                labels.update(chunk["Label"].unique().tolist())
                file_labels.update(chunk["Label"].unique().tolist())
                rows += len(chunk)
        except pd.errors.EmptyDataError as error:
            raise ValueError(f"{path.name} is empty or has no CSV header") from error

        if rows != ROWS_PER_FILE:
            raise ValueError(f"{path.name} has {rows:,} rows; expected {ROWS_PER_FILE:,}")
        stats.append({"file": path.name, "rows": rows, "labels": file_labels})

    if expected_columns is None or empty_columns is None or constant_columns is None:
        raise ValueError("No input CSV files were inspected")
    if not labels.issubset(ALLOWED_LABELS):
        raise ValueError(f"Unexpected Label value(s): {sorted(labels - ALLOWED_LABELS)}")
    return expected_columns, stats, labels, empty_columns, constant_columns


def find_duplicate_columns(paths: list[Path], columns: list[str]) -> list[tuple[str, str]]:
    candidate_groups: dict[str, list[str]] = {}
    digests = {column: hashlib.sha256() for column in columns}

    for path in paths:
        for chunk in read_chunked(path):
            for column in columns:
                hashes = pd.util.hash_pandas_object(chunk[column], index=False)
                digests[column].update(hashes.to_numpy().tobytes())

    for column, digest in digests.items():
        candidate_groups.setdefault(digest.hexdigest(), []).append(column)

    duplicate_pairs: list[tuple[str, str]] = []
    for candidates in candidate_groups.values():
        if len(candidates) < 2:
            continue
        retained: list[str] = []
        for column in candidates:
            duplicate_of = next(
                (
                    prior
                    for prior in retained
                    if columns_equal_across_files(paths, prior, column)
                ),
                None,
            )
            if duplicate_of is None:
                retained.append(column)
            else:
                duplicate_pairs.append((column, duplicate_of))
    return duplicate_pairs


def columns_equal_across_files(paths: list[Path], first: str, second: str) -> bool:
    for path in paths:
        for chunk in read_chunked(path):
            if not chunk[first].equals(chunk[second]):
                return False
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_staged_files(
    paths: list[Path],
    staging_dir: Path,
    columns: list[str],
    removed: set[str],
) -> list[str]:
    retained = [column for column in columns if column not in removed]
    for source_path in paths:
        destination = staging_dir / source_path.name
        with source_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            header = next(reader, None)
            if header != columns:
                raise ValueError(f"{source_path.name} header changed during staging")
            indexes = [index for index, name in enumerate(header) if name in retained]
            with destination.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(retained)
                for row in reader:
                    if len(row) != len(header):
                        raise ValueError(
                            f"Malformed row in {source_path.name}: "
                            f"expected {len(header)} fields, found {len(row)}"
                        )
                    writer.writerow([row[index] for index in indexes])
    return retained


def validate_staged(
    source_paths: list[Path],
    staged_dir: Path,
    original_columns: list[str],
    retained_columns: list[str],
    expected_rows: int,
) -> None:
    expected_set = set(retained_columns)
    total = 0
    shared_header: list[str] | None = None

    for original in source_paths:
        staged = staged_dir / original.name
        with staged.open("r", encoding="utf-8-sig", newline="") as new_file:
            new_reader = csv.reader(new_file, strict=True)
            new_header = next(new_reader, None)
            if new_header != retained_columns or (shared_header is not None and new_header != shared_header):
                raise ValueError(f"Column names/order validation failed for {staged.name}")
            shared_header = new_header

            with original.open("r", encoding="utf-8-sig", newline="") as old_file:
                old_reader = csv.reader(old_file, strict=True)
                old_header = next(old_reader, None)
                if old_header != original_columns:
                    raise ValueError(f"Original header changed during validation: {original.name}")
                old_indexes = [i for i, name in enumerate(old_header) if name in expected_set]
                row_count = 0
                for old_row, new_row in zip(old_reader, new_reader):
                    if len(old_row) != len(old_header) or len(new_row) != len(retained_columns):
                        raise ValueError(f"Malformed row found while validating {original.name}")
                    if [old_row[i] for i in old_indexes] != new_row:
                        raise ValueError(
                            f"Retained values differ from original in {original.name}, "
                            f"data row {row_count + 1}"
                        )
                    row_count += 1
                if next(old_reader, None) is not None:
                    raise ValueError(f"Original file has extra rows: {original.name}")
                if next(new_reader, None) is not None:
                    raise ValueError(f"Staged file has extra rows: {staged.name}")
                if row_count != ROWS_PER_FILE:
                    raise ValueError(
                        f"Staged file {staged.name} has {row_count:,} rows; expected "
                        f"{ROWS_PER_FILE:,}"
                    )
                total += row_count

        label_values: set[str] = set()
        for chunk in read_chunked(staged):
            if chunk.columns.tolist() != retained_columns:
                raise ValueError(f"Column validation failed for {staged.name}")
            label_values.update(chunk["Label"].unique().tolist())
        if not label_values.issubset(ALLOWED_LABELS):
            raise ValueError(f"Unexpected Label value(s) in {staged.name}")

    if len(source_paths) != CLIENT_COUNT or total != expected_rows:
        raise ValueError(
            f"Staged dataset totals are incorrect: {len(source_paths)} files, "
            f"{total:,} rows"
        )


def create_backup(paths: list[Path], backup_dir: Path) -> None:
    backup_dir.mkdir()
    for path in paths:
        backup_path = backup_dir / path.name
        shutil.copy2(path, backup_path)
        if sha256_file(path) != sha256_file(backup_path):
            raise IOError(f"Backup verification failed for {path.name}")


def replace_originals(paths: list[Path], staged_dir: Path, backup_dir: Path) -> None:
    replaced: list[Path] = []
    try:
        for path in paths:
            replaced.append(path)
            shutil.copy2(staged_dir / path.name, path)
        for path in paths:
            if sha256_file(staged_dir / path.name) != sha256_file(path):
                raise IOError(f"Post-replacement verification failed for {path.name}")
    except Exception as error:
        rollback_errors: list[str] = []
        for path in reversed(replaced):
            try:
                shutil.copy2(backup_dir / path.name, path)
            except OSError as rollback_error:
                rollback_errors.append(f"{path.name}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError(
                f"Replacement failed ({error}); rollback failed for "
                + "; ".join(rollback_errors)
            ) from error
        raise


def print_inspection(
    paths: list[Path],
    stats: list[dict[str, object]],
    labels: set[str],
    columns: list[str],
    reasons: dict[str, str],
) -> None:
    print("\nInput row counts:")
    for entry in stats:
        print(f"  {entry['file']}: {entry['rows']:,}")
    print(f"Total rows: {sum(int(entry['rows']) for entry in stats):,}")
    print(f"Unique Label values: {', '.join(sorted(labels))}")
    print("\nColumns to remove:")
    if reasons:
        for column, reason in reasons.items():
            print(f"  {column}: {reason}")
    else:
        print("  (none)")
    print("\nColumns to retain:")
    for column in columns:
        if column not in reasons:
            print(f"  {column}")
    print(f"\nInput files inspected: {len(paths)}")


def main() -> int:
    folder = Path(__file__).resolve().parent
    paths = expected_client_paths(folder)
    columns, stats, labels, empty_columns, constant_columns = inspect_files(paths)
    total_rows = sum(int(entry["rows"]) for entry in stats)
    if total_rows != CLIENT_COUNT * ROWS_PER_FILE:
        raise ValueError(f"Expected 100,000 rows total; found {total_rows:,}")

    reasons: dict[str, str] = {}
    for column in columns:
        if column != "Label" and INDEX_COLUMN.fullmatch(column):
            reasons[column] = "accidental index column name"
    for column in empty_columns:
        if column != "Label":
            reasons.setdefault(column, "completely empty across all rows")
    for column in constant_columns:
        if column != "Label":
            reasons.setdefault(column, "one identical value across all rows")

    duplicates = find_duplicate_columns(paths, columns)
    for duplicate, original in duplicates:
        if duplicate != "Label" and duplicate not in reasons:
            reasons[duplicate] = f"exact duplicate of column {original!r} across all rows"

    print_inspection(paths, stats, labels, columns, reasons)
    staging_dir = Path(tempfile.mkdtemp(prefix=".useless_columns_staging_", dir=folder))
    backup_dir = folder / f"client_csv_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        retained_columns = create_staged_files(
            paths, staging_dir, columns, set(reasons)
        )
        validate_staged(
            paths,
            staging_dir,
            columns,
            retained_columns,
            CLIENT_COUNT * ROWS_PER_FILE,
        )

        print(
            "\nInspection and staged-output validation passed. "
            "No original CSV has been modified."
        )
        response = input(
            f'Type "{CONFIRMATION_PHRASE}" to create an exact backup and replace '
            "the originals; anything else cancels: "
        )
        if response != CONFIRMATION_PHRASE:
            print("Cancelled. Original CSV files were not changed.")
            return 1

        create_backup(paths, backup_dir)
        replace_originals(paths, staging_dir, backup_dir)
        print("\nReplacement complete.")
        print(f"Backup folder (unchanged copies): {backup_dir}")
        print(f"Files processed: {len(paths)}; total rows: {total_rows:,}")
        print(f"Columns removed: {len(reasons)}")
        return 0
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


if __name__ == "__main__":
    raise SystemExit(main())
