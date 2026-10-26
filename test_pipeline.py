import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline import run_pipeline, validate_row


class ValidationTests(unittest.TestCase):
    def test_valid_row_has_no_issues(self):
        row = {
            "record_id": "S100",
            "interview_date": "2026-10-20",
            "region": "Manzini",
            "respondent_age": "29",
            "group": "treatment",
        }
        self.assertEqual(validate_row(row, set()), [])

    def test_missing_id_is_flagged(self):
        row = {
            "record_id": "",
            "interview_date": "2026-10-20",
            "region": "Manzini",
            "respondent_age": "29",
            "group": "control",
        }
        self.assertIn("MISSING_RECORD_ID", validate_row(row, set()))

    def test_duplicate_id_is_flagged(self):
        row = {
            "record_id": "S001",
            "interview_date": "2026-10-20",
            "region": "Manzini",
            "respondent_age": "29",
            "group": "control",
        }
        self.assertIn("DUPLICATE_RECORD_ID", validate_row(row, {"S001"}))

    def test_invalid_age_and_group_are_flagged(self):
        row = {
            "record_id": "S101",
            "interview_date": "2026-10-20",
            "region": "Manzini",
            "respondent_age": "17",
            "group": "unknown",
        }
        issues = validate_row(row, set())
        self.assertIn("INVALID_AGE", issues)
        self.assertIn("INVALID_GROUP", issues)

    def test_pipeline_writes_separate_outputs(self):
        source = Path(__file__).resolve().parents[1] / "data" / "sample_survey.csv"
        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_pipeline(source, Path(temp_dir))
            self.assertEqual(report["total_records"], 10)
            self.assertGreater(report["accepted_records"], 0)
            self.assertGreater(report["rejected_records"], 0)
            self.assertTrue((Path(temp_dir) / "accepted_records.csv").exists())
            self.assertTrue((Path(temp_dir) / "rejected_records.csv").exists())
            saved = json.loads((Path(temp_dir) / "quality_report.json").read_text())
            self.assertEqual(saved["total_records"], 10)


if __name__ == "__main__":
    unittest.main()
