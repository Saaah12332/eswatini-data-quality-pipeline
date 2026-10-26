-- Example SQL for a future SQLite/PostgreSQL table named accepted_records.
-- Adapt column types and table creation to your chosen database.

-- 1. Records per region
SELECT region, COUNT(*) AS record_count
FROM accepted_records
GROUP BY region
ORDER BY record_count DESC;

-- 2. Treatment/control split
SELECT "group", COUNT(*) AS record_count
FROM accepted_records
GROUP BY "group"
ORDER BY "group";

-- 3. Average age by group
SELECT "group", ROUND(AVG(CAST(respondent_age AS REAL)), 1) AS average_age
FROM accepted_records
GROUP BY "group";

-- 4. A basic monitoring metric for each pipeline run
-- Populate pipeline_runs from quality_report.json when database loading is added.
SELECT run_date, total_records, accepted_records, rejected_records,
       acceptance_rate_percent
FROM pipeline_runs
ORDER BY run_date DESC;
