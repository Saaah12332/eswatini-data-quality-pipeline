# Architecture and design notes

## Current version

The first version is a local batch ETL pipeline.

- **Source:** CSV file containing synthetic survey records.
- **Processing:** Python reads each record and applies explicit validation rules.
- **Accepted output:** rows that pass every rule.
- **Rejected output:** rows retained for review with source row number and issue codes.
- **Monitoring:** JSON report and CSV summary for record counts and issue frequencies.

## Reliability choices

1. The original source file is never overwritten.
2. Rejected records are retained rather than silently discarded.
3. Validation issues use stable codes that can be counted and monitored.
4. The pipeline checks that expected columns exist before processing.
5. Automated tests cover normal and invalid cases.
6. Synthetic data is used to avoid exposing confidential research information.

## Proposed cloud evolution

```text
Cloud object storage (S3 or Azure Data Lake Storage)
                 |
                 v
        Scheduled ETL job
                 |
        +--------+--------+
        v                 v
 Curated data       Rejected/quarantine data
        |
        v
 SQL warehouse / analytics layer
        |
        v
 Dashboard + quality alerts
```

A future implementation could use Amazon S3 with AWS Glue, or Azure Data Lake Storage with Azure Data Factory/Fabric. These are design options, not services deployed by this repository yet.

## Production considerations

Before production use, add schema versioning, configurable rules, structured logging, retry handling, access controls, encryption, secrets management, lineage, monitoring, and a documented data-retention policy.
