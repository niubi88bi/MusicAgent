"""Offline contracts for the natural paired-instrument shadow audit."""

from __future__ import annotations

import unittest

from scripts.build_producer_constraint_instrument_pair_natural_manifest import (
    PairNaturalManifest,
    build_manifest,
)


class ProducerConstraintInstrumentPairNaturalTests(unittest.TestCase):
    def test_manifest_has_eight_new_cases(self) -> None:
        manifest = PairNaturalManifest.model_validate(build_manifest())
        self.assertEqual(len(manifest.cases), 8)
        self.assertEqual(
            {case.risk_focus for case in manifest.cases},
            {
                "paired_unique_substitution",
                "paired_missing_required",
                "paired_compliant",
                "forbidden_only",
                "paired_ambiguous_context",
            },
        )

    def test_manifest_does_not_reuse_previous_case_ids(self) -> None:
        manifest = PairNaturalManifest.model_validate(build_manifest())
        self.assertFalse(any(case.case_id in {"na09-sad-slap-no-piano", "na10-vina-guitar-substitution"} for case in manifest.cases))

    def test_manifest_requests_keep_pair_semantics_visible(self) -> None:
        manifest = PairNaturalManifest.model_validate(build_manifest())
        pair_cases = [
            case
            for case in manifest.cases
            if case.risk_focus.startswith("paired_")
        ]
        self.assertGreaterEqual(len(pair_cases), 6)
        self.assertTrue(
            all(
                ("钢琴" in case.user_request_text or "piano" in case.user_request_text.lower())
                and (
                    "木吉他" in case.user_request_text
                    or "原声吉他" in case.user_request_text
                    or "guitar" in case.user_request_text.lower()
                )
                for case in pair_cases
            )
        )


if __name__ == "__main__":
    unittest.main()
