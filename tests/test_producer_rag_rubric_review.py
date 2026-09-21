"""Tests for decoding the three-model Producer rubric-v2 blind review."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.evaluate_producer_rag_rubric import CASES_PATH, load_manifest
from scripts.summarize_producer_rag_rubric_review import (
    MultiModelBlindReview,
    decode_reviews,
)


REVIEW_PATH = "data/miaoxiang/producer_rag_rubric_multi_model_review.v1.json"


class ProducerRAGRubricReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.review = MultiModelBlindReview.model_validate(
            json.loads(Path(REVIEW_PATH).read_text(encoding="utf-8"))
        )
        cls.manifest = load_manifest()
        cls.key_rows = [
            {
                "case_id": case.case_id,
                "X": "baseline" if index in {0, 1, 5} else "gated",
                "Y": "gated" if index in {0, 1, 5} else "baseline",
            }
            for index, case in enumerate(cls.manifest.cases)
        ]

    def test_decode_preserves_three_reviewers_and_eighteen_decisions(self) -> None:
        result = decode_reviews(
            self.review,
            manifest_path=CASES_PATH,
            key_rows=self.key_rows,
        )
        self.assertEqual(result["reviewer_count"], 3)
        self.assertEqual(result["aggregate_all_decisions"]["decision_count"], 18)
        self.assertEqual(result["aggregate_all_decisions"]["gated"]["wins"], 11)
        self.assertEqual(result["aggregate_all_decisions"]["baseline"]["wins"], 3)
        self.assertEqual(result["aggregate_all_decisions"]["ties"], 4)

    def test_case_consensus_keeps_failure_tie(self) -> None:
        result = decode_reviews(
            self.review,
            manifest_path=CASES_PATH,
            key_rows=self.key_rows,
        )
        by_id = {row["case_id"]: row for row in result["case_consensus"]}
        self.assertEqual(
            by_id["r01-deep-house-mix-voice-flex-tempo"]["majority_preference"],
            "baseline",
        )
        self.assertEqual(
            by_id["r02-dnb-rain-courier-sung-no-mc"]["majority_preference"],
            "gated",
        )
        self.assertEqual(
            by_id["r06-sad-slap-house-no-piano"]["majority_preference"],
            "tie",
        )

    def test_decode_rejects_missing_case(self) -> None:
        bad_review = self.review.model_copy(deep=True)
        bad_review.reviewers[0].cases.pop()
        with self.assertRaisesRegex(ValueError, "rubric_review_case_set_mismatch"):
            decode_reviews(
                bad_review,
                manifest_path=CASES_PATH,
                key_rows=self.key_rows,
            )


if __name__ == "__main__":
    unittest.main()
