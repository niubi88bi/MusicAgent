"""Contracts for the independent minimal-patch field-position holdout."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.build_producer_constraint_minimal_patch_holdout import build_manifest
from scripts.evaluate_producer_constraint_fault_injection import (
    FaultManifest,
    validate_fault_contracts,
)


HOLDOUT_PATH = Path(
    "data/miaoxiang/producer_constraint_minimal_patch_field_position_holdout.v1.json"
)


class ProducerConstraintMinimalPatchHoldoutTests(unittest.TestCase):
    def test_frozen_holdout_has_four_unique_families(self) -> None:
        manifest = FaultManifest.model_validate_json(
            HOLDOUT_PATH.read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest.cases), 4)
        self.assertEqual(
            {case.fault_family for case in manifest.cases},
            {
                "wrong_vocal_gender",
                "forbidden_rap_mc_slogans",
                "vocal_material_in_instrumental",
                "hard_fixed_bpm",
            },
        )

    def test_holdout_changes_expected_field_only(self) -> None:
        manifest = FaultManifest.model_validate_json(
            HOLDOUT_PATH.read_text(encoding="utf-8")
        )
        contracts = validate_fault_contracts(manifest)
        self.assertEqual(len(contracts), 4)
        self.assertTrue(
            all(
                row["base_hard_violation_codes"] == []
                and row["injected_hard_violation_codes"]
                for row in contracts
            )
        )
        self.assertEqual(
            {case.target_field for case in manifest.cases},
            {"style_prompt", "arrangement_note"},
        )

    def test_builder_is_reproducible(self) -> None:
        manifest = FaultManifest.model_validate(build_manifest())
        self.assertEqual(
            [case.case_id for case in manifest.cases],
            [
                "ho01-vina-style-gender",
                "ho02-dnb-style-rap",
                "ho03-techno-arrangement-vocal",
                "ho04-deep-house-arrangement-bpm",
            ],
        )


if __name__ == "__main__":
    unittest.main()
