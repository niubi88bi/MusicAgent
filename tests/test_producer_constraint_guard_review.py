"""Tests for the corrected three-model constraint-guard blind review."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.evaluate_producer_constraint_guard import CASES_PATH
from scripts.summarize_producer_constraint_guard_review import (
    MultiModelGuardReview,
    decode_reviews,
)


REVIEW_PATH = Path(
    "data/miaoxiang/producer_constraint_guard_multi_model_review.v1.json"
)


@unittest.skipUnless(
    REVIEW_PATH.is_file() and CASES_PATH.is_file(),
    "private constraint-review dataset is not published",
)
class ProducerConstraintGuardReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.review = MultiModelGuardReview.model_validate(
            json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        )
        cls.key_rows = [
            {"case_id": "r01-deep-house-mix-voice-flex-tempo", "X": "gated", "Y": "guarded"},
            {"case_id": "r05-vina-house-female-vocal-override", "X": "guarded", "Y": "gated"},
            {"case_id": "r06-sad-slap-house-no-piano", "X": "gated", "Y": "guarded"},
        ]

    def test_decode_reports_expected_votes_and_hard_failures(self) -> None:
        result = decode_reviews(
            self.review,
            manifest_path=CASES_PATH,
            key_rows=self.key_rows,
        )
        aggregate = result["aggregate_all_decisions"]
        self.assertEqual(aggregate["decision_count"], 9)
        self.assertEqual(aggregate["gated"]["wins"], 3)
        self.assertEqual(aggregate["guarded"]["wins"], 4)
        self.assertEqual(aggregate["ties"], 2)
        self.assertEqual(aggregate["gated"]["hard_requirement_violation"], 3)
        self.assertEqual(aggregate["guarded"]["hard_requirement_violation"], 0)

    def test_case_consensus_rejects_global_upfront_guard(self) -> None:
        result = decode_reviews(
            self.review,
            manifest_path=CASES_PATH,
            key_rows=self.key_rows,
        )
        by_id = {row["case_id"]: row for row in result["case_consensus"]}
        self.assertEqual(
            by_id["r01-deep-house-mix-voice-flex-tempo"]["majority_preference"],
            "tie",
        )
        self.assertEqual(
            by_id["r05-vina-house-female-vocal-override"]["majority_preference"],
            "gated",
        )
        self.assertEqual(
            by_id["r06-sad-slap-house-no-piano"]["majority_preference"],
            "guarded",
        )
        self.assertEqual(result["method_decision"], "reject_upfront_guard_for_all_cases")

    def test_decode_rejects_non_blind_review(self) -> None:
        review = self.review.model_copy(update={"blind_at_rating": False})
        with self.assertRaisesRegex(ValueError, "reviews_were_not_blind"):
            decode_reviews(review, manifest_path=CASES_PATH, key_rows=self.key_rows)


if __name__ == "__main__":
    unittest.main()
