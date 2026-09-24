"""Offline contracts for the natural hard-constraint audit manifest."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.evaluate_producer_constraint_natural_audit import (
    load_manifest,
    validate_manifest_contracts,
)


MANIFEST_PATH = Path("data/miaoxiang/producer_constraint_natural_audit.v1.json")


class ProducerConstraintNaturalAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_manifest(MANIFEST_PATH)

    def test_manifest_has_frozen_twelve_case_mix(self) -> None:
        self.assertEqual(len(self.manifest.cases), 12)
        counts: dict[str, int] = {}
        for case in self.manifest.cases:
            counts[case.risk_focus] = counts.get(case.risk_focus, 0) + 1
        self.assertEqual(
            counts,
            {
                "explicit_vocal_gender": 3,
                "explicit_no_rap_mc_slogans": 3,
                "explicit_instrumental_only": 2,
                "explicit_instrument_substitution": 2,
                "explicit_flexible_bpm": 2,
            },
        )

    def test_explicit_constraint_contracts_are_detectable(self) -> None:
        contracts = validate_manifest_contracts(self.manifest)
        self.assertEqual(len(contracts), 12)
        self.assertTrue(all(row["expected_hard_concepts"] for row in contracts))

    def test_audit_manifest_has_no_repair_or_rag_fields(self) -> None:
        self.assertTrue(all(case.requirement.target_rhyme_group is None for case in self.manifest.cases))
        self.assertTrue(all(case.lyric_draft.hook in case.lyric_draft.lyrics for case in self.manifest.cases))


if __name__ == "__main__":
    unittest.main()
