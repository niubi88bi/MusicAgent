"""Offline contracts for strictly localized fault injection v2."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.evaluate_producer_constraint_fault_injection import (
    build_blind_card,
    load_manifest,
    resolve_cases,
    validate_fault_contracts,
)


MANIFEST_PATH = Path(
    "data/miaoxiang/producer_constraint_local_fault_injection.v2.json"
)


@unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/output/evaluations").is_dir(), "private evaluation artifacts are not published")
class ProducerConstraintLocalFaultInjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest(MANIFEST_PATH)
        cls.contracts = validate_fault_contracts(cls.manifest)

    def test_every_case_changes_exactly_one_registered_field(self) -> None:
        self.assertEqual(len(self.contracts), 4)
        for row in self.contracts:
            self.assertEqual(
                row["base_to_injected_changed_fields"], [row["target_field"]]
            )

    def test_every_case_has_only_expected_hard_faults(self) -> None:
        for case, contract in zip(self.manifest.cases, self.contracts, strict=True):
            self.assertEqual(
                contract["injected_hard_violation_codes"],
                case.expected_injected_hard_codes,
            )
            self.assertEqual(contract["injected_advisory_codes"], [])
            self.assertEqual(contract["base_hard_violation_codes"], [])

    def test_manifest_declares_single_fragment_scope(self) -> None:
        self.assertEqual(
            self.manifest.injection_scope, "single_field_single_fragment"
        )
        self.assertEqual(
            {case.target_field for case in self.manifest.cases},
            {"style_prompt", "vocal_direction", "arrangement_note"},
        )

    def test_visible_case_ids_do_not_reveal_fault_family(self) -> None:
        for source_case, fault in resolve_cases(self.manifest):
            card, _ = build_blind_card(source_case, fault, fault.base_plan)
            visible_id = card["case_id"].lower()
            for leaked in ("fault", "gender", "rap", "vocal", "bpm", "故障"):
                self.assertNotIn(leaked, visible_id)


if __name__ == "__main__":
    unittest.main()
