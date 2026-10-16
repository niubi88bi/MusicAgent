"""Contracts for decoding the three-model post-check repair review."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.summarize_producer_constraint_postcheck_review import (
    MultiModelReview,
    decode_reviews,
)


REVIEW_PATH = Path(
    "data/miaoxiang/producer_constraint_postcheck_multi_model_review.v1.json"
)
MANIFEST_PATH = Path(
    "data/miaoxiang/producer_constraint_postcheck_validation.v1.json"
)


@unittest.skipUnless(
    REVIEW_PATH.is_file() and MANIFEST_PATH.is_file(),
    "private constraint-review dataset is not published",
)
class ProducerConstraintPostcheckReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.review = MultiModelReview.model_validate_json(
            REVIEW_PATH.read_text(encoding="utf-8")
        )
        self.key = [
            {
                "case_id": "r06-sad-slap-house-no-piano",
                "X": "postcheck",
                "Y": "initial",
            }
        ]

    def test_decode_matches_unanimous_postcheck_preference(self) -> None:
        decoded = decode_reviews(
            self.review,
            manifest_path=MANIFEST_PATH,
            key_rows=self.key,
        )
        aggregate = decoded["aggregate_all_decisions"]
        self.assertEqual(aggregate["decision_count"], 3)
        self.assertEqual(aggregate["postcheck"]["wins"], 3)
        self.assertEqual(aggregate["initial"]["wins"], 0)
        self.assertEqual(aggregate["postcheck"]["hard_requirement_violation"], 0)
        self.assertEqual(aggregate["initial"]["hard_requirement_violation"], 3)
        self.assertEqual(aggregate["postcheck"]["unsupported_expansion"], 1)
        self.assertEqual(
            decoded["method_decision"],
            "continue_postcheck_repair_shadow_validation",
        )

    def test_case_consensus_is_unanimous(self) -> None:
        decoded = decode_reviews(
            self.review,
            manifest_path=MANIFEST_PATH,
            key_rows=self.key,
        )
        consensus = decoded["case_consensus"][0]
        self.assertEqual(consensus["majority_preference"], "postcheck")
        self.assertTrue(consensus["unanimous_preference"])
        self.assertEqual(consensus["preference_votes"]["postcheck"], 3)

    def test_non_blind_review_is_rejected(self) -> None:
        payload = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        payload["blind_at_rating"] = False
        review = MultiModelReview.model_validate(payload)
        with self.assertRaisesRegex(ValueError, "reviews_were_not_blind"):
            decode_reviews(review, manifest_path=MANIFEST_PATH, key_rows=self.key)


if __name__ == "__main__":
    unittest.main()
