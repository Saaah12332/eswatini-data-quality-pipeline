# Eswatini Data Quality Pipeline

A portfolio project demonstrating how to ingest, validate, clean, and report on survey data before it is used for analysis.

**Built with:** Python (standard library), CSV, JSON, SQL, `unittest`

> All included records are synthetic and created for demonstration. Never upload confidential respondent data, real GPS coordinates, names, phone numbers, or restricted research data to a public repository.

## Why this project matters

Research teams need data that is complete, consistent, and fit for analysis. This pipeline applies explicit validation rules to incoming survey records, preserves the original input, separates accepted and rejected rows, and produces a machine-readable quality report.

## Pipeline

1. **Extract** records from `data/sample_survey.csv`.
2. **Transform / validate** required fields, ID uniqueness, dates, age ranges, regions, and survey groups.
3. **Load** accepted records and rejected records into separate CSV outputs.
4. **Report** record counts, issue counts, and the acceptance rate.

```text
CSV source
   |
   v
Python validation pipeline
   |----------------------|
   v                      v
accepted_records.csv   rejected_records.csv
   |
   v
quality_report.json + quality_summary.csv
```

## Quick start

Requires Python 3.10+.

```bash
python src/pipeline.py
python -m unittest discover -s tests -v
```

The pipeline uses only Python's standard library; no paid tools or cloud account are required.

## Outputs

Generated in `outputs/`:

- `accepted_records.csv` — rows that passed all validation rules.
- `rejected_records.csv` — rows with one or more validation issues, including issue codes.
- `quality_report.json` — run summary and issue counts.
- `quality_summary.csv` — compact summary suitable for loading into a dashboard or warehouse.

The input CSV is not modified.

## Validation rules

| Rule | Example issue code | Behaviour |
|---|---|---|
| Record ID must be present | `MISSING_RECORD_ID` | Reject row |
| Record ID must be unique | `DUPLICATE_RECORD_ID` | Reject duplicate row |
| Interview date must be YYYY-MM-DD | `INVALID_INTERVIEW_DATE` | Reject row |
| Age must be an integer from 18 to 100 | `INVALID_AGE` / `MISSING_AGE` | Reject row |
| Region must be a known Eswatini region | `INVALID_REGION` / `MISSING_REGION` | Reject row |
| Group must be `treatment` or `control` | `INVALID_GROUP` / `MISSING_GROUP` | Reject row |

The rules are intentionally strict for this demonstration. Real projects should define validation rules with the data owner and study protocol.

## Example results

Run the pipeline to generate results from the supplied synthetic sample. The sample deliberately contains missing values, duplicate IDs, an invalid date, and an invalid group so that the checks can be demonstrated.

## Repository structure

```text
eswatini-data-quality-pipeline/
├── data/
│   └── sample_survey.csv
├── docs/
│   └── architecture.md
├── outputs/
│   └── .gitkeep
├── sql/
│   └── quality_analysis.sql
├── src/
│   └── pipeline.py
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── LICENSE
└── README.md
```

## Next improvements

- Add configurable rules and a larger synthetic dataset.
- Store run metadata and quality metrics in SQLite or PostgreSQL.
- Add continuous integration to run tests on each GitHub push.
- Add a cloud storage landing zone (AWS S3 or Azure Data Lake Storage).
- Orchestrate scheduled runs and publish monitoring metrics.
- Build a dashboard for quality trends across multiple runs.

## Skills demonstrated

Data ingestion, ETL design, validation rules, exception handling, reproducible outputs, automated tests, SQL analysis, documentation, and a cloud-ready architecture.

## Author

Portfolio project by a Computer Science graduate building practical skills in data engineering. Connect via GitHub or LinkedIn.
