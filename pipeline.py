"""Small, reproducible survey data quality ETL pipeline.

Uses only Python's standard library. All sample data is synthetic.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "sample_survey.csv"
OUTPUT_DIR = ROOT / "outputs"

EXPECTED_COLUMNS = [
    "record_id", "interview_date", "region", "respondent_age", "group"
]
VALID_REGIONS = {"Hhohho", "Manzini", "Lubombo", "Shiselweni"}
VALID_GROUPS = {"treatment", "control"}


def validate_row(row: dict[str, str], seen_ids: set[str]) -> list[str]:
    """Return issue codes for a row; an empty list means it passed."""
    issues: list[str] = []
    record_id = (row.get("record_id") or "").strip()
    interview_date = (row.get("interview_date") or "").strip()
    region = (row.get("region") or "").strip()
    age_text = (row.get("respondent_age") or "").strip()
    group = (row.get("group") or "").strip().lower()

    if not record_id:
        issues.append("MISSING_RECORD_ID")
    elif record_id in seen_ids:
        issues.append("DUPLICATE_RECORD_ID")

    if not interview_date:
        issues.append("MISSING_INTERVIEW_DATE")
    else:
        try:
            date.fromisoformat(interview_date)
        except ValueError:
            issues.append("INVALID_INTERVIEW_DATE")

    if not region:
        issues.append("MISSING_REGION")
    elif region not in VALID_REGIONS:
        issues.append("INVALID_REGION")

    if not age_text:
        issues.append("MISSING_AGE")
    else:
        try:
            age = int(age_text)
            if not 18 <= age <= 100:
                issues.append("INVALID_AGE")
        except ValueError:
            issues.append("INVALID_AGE")

    if not group:
        issues.append("MISSING_GROUP")
    elif group not in VALID_GROUPS:
        issues.append("INVALID_GROUP")

    return issues


def read_records(path: Path = INPUT_PATH) -> list[dict[str, str]]:
    """Read CSV and fail clearly if expected columns are absent."""
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = set(EXPECTED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input is missing required columns: {sorted(missing)}")
        return [{column: (row.get(column) or "").strip()
                 for column in EXPECTED_COLUMNS} for row in reader]


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline(input_path: Path = INPUT_PATH, output_dir: Path = OUTPUT_DIR) -> dict:
    """Validate records, write outputs, and return the run report."""
    records = read_records(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    issue_counts: Counter[str] = Counter()

    for row_number, row in enumerate(records, start=2):  # header is line 1
        issues = validate_row(row, seen_ids)
        record_id = row["record_id"]
        # Track first occurrence only; a duplicate is rejected but cannot
        # replace the original valid ID in the seen set.
        if record_id:
            seen_ids.add(record_id)

        if issues:
            issue_counts.update(issues)
            rejected.append({
                **row,
                "source_row": str(row_number),
                "issue_codes": "|".join(issues),
            })
        else:
            accepted.append(row)

    rejected_columns = EXPECTED_COLUMNS + ["source_row", "issue_codes"]
    write_csv(output_dir / "accepted_records.csv", accepted, EXPECTED_COLUMNS)
    write_csv(output_dir / "rejected_records.csv", rejected, rejected_columns)

    total = len(records)
    report = {
        "pipeline": "eswatini-data-quality-pipeline",
        "input_file": input_path.name,
        "total_records": total,
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
        "acceptance_rate_percent": round((len(accepted) / total * 100), 2) if total else 0.0,
        "issue_counts": dict(sorted(issue_counts.items())),
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    summary_rows = [
        {"metric": "total_records", "value": str(total)},
        {"metric": "accepted_records", "value": str(len(accepted))},
        {"metric": "rejected_records", "value": str(len(rejected))},
        {"metric": "acceptance_rate_percent", "value": str(report["acceptance_rate_percent"])},
    ] + [
        {"metric": f"issue_{code.lower()}", "value": str(count)}
        for code, count in sorted(issue_counts.items())
    ]
    write_csv(output_dir / "quality_summary.csv", summary_rows, ["metric", "value"])
    return report


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
