"""Offline contracts for the paired piano/acoustic-guitar holdout."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_producer_constraint_instrument_pair_holdout import (
    InstrumentPairHoldoutManifest,
    build_manifest,
)
from scripts.evaluate_producer_constraint_instrument_pair_holdout import run_holdout


class ProducerConstraintInstrumentPairHoldoutTests(unittest.TestCase):
    def test_manifest_has_independent_seven_case_mix(self) -> None:
        manifest = InstrumentPairHoldoutManifest.model_validate(build_manifest())
        self.assertEqual(len(manifest.cases), 7)
        self.assertEqual(
            [case.case_id for case in manifest.cases],
            [
                "ip01-unique-cn-replacement",
                "ip02-unique-en-replacement",
                "ip03-append-cn-no-piano",
                "ip04-compliant-pass-through",
                "ip05-multiple-piano-fail-closed",
                "ip06-ambiguous-piano-fail-closed",
                "ip07-only-forbidden-no-pair-fail-closed",
            ],
        )

    def test_holdout_runs_without_model_calls_and_matches_all_actions(self) -> None:
        manifest = InstrumentPairHoldoutManifest.model_validate(build_manifest())
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            output_path = Path(directory) / "summary.json"
            manifest_path.write_text(
                manifest.model_dump_json(indent=2), encoding="utf-8"
            )
            summary = run_holdout(
                manifest, manifest_path=manifest_path, output=output_path
            )
        self.assertEqual(
            summary["action_counts"],
            {"pass_through": 1, "deterministic_repair": 3, "fail_closed": 3},
        )
        self.assertEqual(summary["accepted_repairs"], 3)
        self.assertEqual(summary["model_calls"], 0)
        self.assertTrue(summary["all_repaired_cases_hard_clear"])

    def test_holdout_never_changes_unregistered_fields(self) -> None:
        manifest = InstrumentPairHoldoutManifest.model_validate(build_manifest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            summary = run_holdout(
                manifest,
                manifest_path=path,
                output=Path(directory) / "summary.json",
            )
        for row in summary["all_cases"]:
            if row["actual_action"] == "deterministic_repair":
                self.assertTrue(row["validation"]["accepted"])
                self.assertEqual(row["validation"]["new_advisory"], [])
                self.assertEqual(row["validation"]["new_registered_concepts"], [])
                self.assertEqual(row["validation"]["new_bpm_values"], [])


if __name__ == "__main__":
    unittest.main()
