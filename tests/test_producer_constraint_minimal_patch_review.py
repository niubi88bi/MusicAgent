"""Contracts for the minimal-fragment-patch multi-model review."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.summarize_producer_constraint_minimal_patch_review import (
    MultiModelReview,
    decode_reviews,
)


REVIEW_PATH = Path(
    "data/miaoxiang/producer_constraint_minimal_patch_multi_model_review.v1.json"
)
KEY_PATH = Path(
    "data/output/evaluations/"
    "producer-constraint-minimal-patch-v1-e61f598feb6f41ee980628dc0367205e/"
    "blind_review_key.json"
)


class ProducerConstraintMinimalPatchReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.review = MultiModelReview.model_validate_json(
            REVIEW_PATH.read_text(encoding="utf-8")
        )
        self.key = json.loads(KEY_PATH.read_text(encoding="utf-8"))

    def test_decode_records_unanimous_base_preference(self) -> None:
        decoded = decode_reviews(self.review, key_rows=self.key)
        aggregate = decoded["aggregate_all_decisions"]
        self.assertEqual(aggregate["decision_count"], 3)
        self.assertEqual(aggregate["base"]["wins"], 3)
        self.assertEqual(aggregate["minimal"]["wins"], 0)
        self.assertEqual(aggregate["ties"], 0)
        self.assertEqual(
            decoded["case_consensus"][0]["majority_preference"], "base"
        )
        self.assertTrue(decoded["case_consensus"][0]["unanimous_preference"])

    def test_method_rejects_free_fragment_and_fails_closed(self) -> None:
        decoded = decode_reviews(self.review, key_rows=self.key)
        self.assertEqual(
            decoded["method_decision"], "reject_free_llm_fragment_replacement"
        )
        self.assertEqual(decoded["retained_branch"], "deterministic_patch_shadow_only")
        self.assertEqual(decoded["non_deterministic_behavior"], "fail_closed")
        self.assertEqual(decoded["runtime_decision"], "unchanged_disabled")

    def test_no_candidate_is_marked_as_hard_violation(self) -> None:
        decoded = decode_reviews(self.review, key_rows=self.key)
        aggregate = decoded["aggregate_all_decisions"]
        self.assertEqual(aggregate["base"]["hard_requirement_violation"], 0)
        self.assertEqual(aggregate["minimal"]["hard_requirement_violation"], 0)

    def test_non_blind_review_is_rejected(self) -> None:
        payload = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        payload["blind_at_rating"] = False
        review = MultiModelReview.model_validate(payload)
        with self.assertRaisesRegex(ValueError, "reviews_were_not_blind"):
            decode_reviews(review, key_rows=self.key)


if __name__ == "__main__":
    unittest.main()
