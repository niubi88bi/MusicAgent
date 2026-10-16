"""Tests for deterministic decoding of the gated Producer blind review."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.evaluate_producer_rag_gated_ab import load_cases
from scripts.summarize_producer_rag_ab_review import (
    CandidateHumanReview,
    CaseHumanReview,
    HumanBlindReview,
)
from scripts.summarize_producer_rag_gated_ab_review import decode_review


def _score(value: int) -> CandidateHumanReview:
    return CandidateHumanReview(
        style_fit=value,
        production_executability=value,
        internal_consistency=value,
        over_constraint=False,
        unsupported_expansion=False,
    )


@unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/output/evaluations").is_dir(), "private evaluation artifacts are not published")
class ProducerRAGGatedReviewTests(unittest.TestCase):
    def test_decode_separates_injected_and_no_rag_pairs(self) -> None:
        manifest = load_cases()
        keys = [
            {"case_id": case.case_id, "X": "gated", "Y": "baseline"}
            for case in manifest.cases
        ]
        cases = []
        for case in manifest.cases:
            cases.append(
                CaseHumanReview(
                    case_id=case.case_id,
                    X=_score(5),
                    Y=_score(4),
                    preference="X",
                    notes="blind note",
                )
            )
        review = HumanBlindReview(
            schema_version="1.0",
            reviewer_count=1,
            reviewer_profile="test reviewer",
            review_modality="text only",
            blind_at_rating=True,
            recorded_at="2026-09-17",
            limitations=["test fixture"],
            cases=cases,
        )

        result = decode_review(review, manifest, keys)

        self.assertEqual(result["aggregate_all_pairs"]["case_count"], 4)
        self.assertEqual(
            result["aggregate_injection_authorized_subset"]["case_count"], 3
        )
        self.assertEqual(result["aggregate_no_rag_subset"]["case_count"], 1)
        self.assertEqual(
            result["aggregate_injection_authorized_subset"]["gated"]["wins"], 3
        )

    def test_decode_rejects_review_that_was_not_blind(self) -> None:
        manifest = load_cases()
        review = HumanBlindReview(
            schema_version="1.0",
            reviewer_count=1,
            reviewer_profile="test reviewer",
            review_modality="text only",
            blind_at_rating=False,
            recorded_at="2026-09-17",
            limitations=["test fixture"],
            cases=[
                CaseHumanReview(
                    case_id=case.case_id,
                    X=_score(5),
                    Y=_score(4),
                    preference="X",
                    notes="not blind",
                )
                for case in manifest.cases
            ],
        )
        keys = [
            {"case_id": case.case_id, "X": "gated", "Y": "baseline"}
            for case in manifest.cases
        ]
        with self.assertRaisesRegex(ValueError, "review_was_not_blind_at_rating"):
            decode_review(review, manifest, keys)


if __name__ == "__main__":
    unittest.main()
