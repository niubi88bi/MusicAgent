"""Contracts for decoding localized-fault multi-model reviews."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.evaluate_producer_constraint_fault_injection import (
    blind_mapping,
    load_manifest,
)
from scripts.summarize_producer_constraint_fault_review import (
    MultiModelReview,
    decode_reviews,
)


REVIEW_PATH = Path(
    "data/miaoxiang/producer_constraint_local_fault_multi_model_review.v2.json"
)
MANIFEST_PATH = Path(
    "data/miaoxiang/producer_constraint_local_fault_injection.v2.json"
)


@unittest.skipUnless(
    REVIEW_PATH.is_file() and MANIFEST_PATH.is_file(),
    "private constraint-review dataset is not published",
)
class ProducerConstraintLocalFaultReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.review = MultiModelReview.model_validate_json(
            REVIEW_PATH.read_text(encoding="utf-8")
        )
        self.key = [
            {
                "case_id": case.review_case_id or case.case_id,
                **blind_mapping(case.case_id),
            }
            for case in load_manifest(MANIFEST_PATH).cases
        ]

    def test_decode_matches_expected_overall_votes(self) -> None:
        decoded = decode_reviews(
            self.review,
            manifest_path=MANIFEST_PATH,
            key_rows=self.key,
        )
        aggregate = decoded["aggregate_all_decisions"]
        self.assertEqual(aggregate["decision_count"], 12)
        self.assertEqual(aggregate["base"]["wins"], 8)
        self.assertEqual(aggregate["repaired"]["wins"], 0)
        self.assertEqual(aggregate["ties"], 4)
        self.assertEqual(aggregate["base"]["hard_requirement_violation"], 0)
        self.assertEqual(aggregate["repaired"]["hard_requirement_violation"], 0)

    def test_case_consensus_preserves_ties_and_majorities(self) -> None:
        decoded = decode_reviews(
            self.review,
            manifest_path=MANIFEST_PATH,
            key_rows=self.key,
        )
        by_case = {row["case_id"]: row for row in decoded["case_consensus"]}
        self.assertEqual(by_case["l01-library-closing"]["majority_preference"], "base")
        self.assertEqual(
            by_case["l02-rain-night-bus"]["preference_votes"],
            {"base": 2, "repaired": 0, "tie": 1, "both_fail": 0},
        )
        self.assertEqual(by_case["l03-greenhouse-startup"]["majority_preference"], "base")
        self.assertEqual(by_case["l04-morning-ferry"]["majority_preference"], "tie")

    def test_no_reviewer_marks_a_hard_violation(self) -> None:
        decoded = decode_reviews(
            self.review,
            manifest_path=MANIFEST_PATH,
            key_rows=self.key,
        )
        aggregate = decoded["aggregate_all_decisions"]
        self.assertEqual(aggregate["base"]["hard_requirement_violation"], 0)
        self.assertEqual(aggregate["repaired"]["hard_requirement_violation"], 0)

    def test_non_blind_review_is_rejected(self) -> None:
        payload = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        payload["blind_at_rating"] = False
        review = MultiModelReview.model_validate(payload)
        with self.assertRaisesRegex(ValueError, "reviews_were_not_blind"):
            decode_reviews(review, manifest_path=MANIFEST_PATH, key_rows=self.key)


if __name__ == "__main__":
    unittest.main()
