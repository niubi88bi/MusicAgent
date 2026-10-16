"""Tests for deterministic decoding of the Producer RAG blind review."""

from __future__ import annotations

import unittest

from scripts.evaluate_producer_rag_ab import CASES_PATH, load_cases
from scripts.summarize_producer_rag_ab_review import (
    CaseHumanReview,
    CandidateHumanReview,
    HumanBlindReview,
    decode_review,
)


def candidate(score: int, *, unsupported: bool = False) -> CandidateHumanReview:
    return CandidateHumanReview(
        style_fit=score,
        production_executability=score,
        internal_consistency=score,
        over_constraint=False,
        unsupported_expansion=unsupported,
    )


@unittest.skipUnless(CASES_PATH.is_file(), "private Producer RAG cases are not published")
class ProducerRAGReviewTests(unittest.TestCase):
    def test_decode_uses_key_and_preserves_single_reviewer_boundary(self) -> None:
        manifest = load_cases()
        reviews = []
        keys = []
        for index, case in enumerate(manifest.cases):
            reviews.append(
                CaseHumanReview(
                    case_id=case.case_id,
                    X=candidate(5),
                    Y=candidate(4, unsupported=index == 0),
                    preference="X" if index < 7 else "tie",
                    notes="blind note",
                )
            )
            keys.append(
                {"case_id": case.case_id, "X": "rag_on", "Y": "rag_off"}
            )
        review = HumanBlindReview(
            schema_version="1.0",
            reviewer_count=1,
            reviewer_profile="non-professional",
            review_modality="text",
            blind_at_rating=True,
            recorded_at="2026-09-17",
            limitations=["single reviewer"],
            cases=reviews,
        )

        result = decode_review(review, manifest, keys)

        self.assertEqual(result["aggregate"]["rag_on"]["wins"], 7)
        self.assertEqual(result["aggregate"]["ties"], 1)
        self.assertEqual(
            result["aggregate"]["rag_off"]["unsupported_expansion"], 1
        )
        self.assertEqual(result["reviewer_count"], 1)
        self.assertIsNone(result["method_decision"])

    def test_decode_rejects_non_blind_review(self) -> None:
        manifest = load_cases()
        reviews = [
            CaseHumanReview(
                case_id=case.case_id,
                X=candidate(5),
                Y=candidate(4),
                preference="X",
                notes="not blind",
            )
            for case in manifest.cases
        ]
        review = HumanBlindReview(
            schema_version="1.0",
            reviewer_count=1,
            reviewer_profile="reviewer",
            review_modality="text",
            blind_at_rating=False,
            recorded_at="2026-09-17",
            limitations=["single reviewer"],
            cases=reviews,
        )
        keys = [
            {"case_id": case.case_id, "X": "rag_on", "Y": "rag_off"}
            for case in manifest.cases
        ]

        with self.assertRaisesRegex(ValueError, "review_was_not_blind_at_rating"):
            decode_review(review, manifest, keys)


if __name__ == "__main__":
    unittest.main()
