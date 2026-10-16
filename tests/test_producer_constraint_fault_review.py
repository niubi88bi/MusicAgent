"""Contracts for decoding controlled-fault multi-model reviews."""

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
    "data/miaoxiang/producer_constraint_fault_multi_model_review.v1.json"
)
MANIFEST_PATH = Path("data/miaoxiang/producer_constraint_fault_injection.v1.json")


@unittest.skipUnless(
    REVIEW_PATH.is_file() and MANIFEST_PATH.is_file(),
    "private constraint-review dataset is not published",
)
class ProducerConstraintFaultReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.review = MultiModelReview.model_validate_json(
            REVIEW_PATH.read_text(encoding="utf-8")
        )
        self.key = [
            {"case_id": case.case_id, **blind_mapping(case.case_id)}
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
        self.assertEqual(aggregate["base"]["wins"], 7)
        self.assertEqual(aggregate["repaired"]["wins"], 2)
        self.assertEqual(aggregate["ties"], 3)
        self.assertEqual(aggregate["base"]["hard_requirement_violation"], 0)
        self.assertEqual(aggregate["repaired"]["hard_requirement_violation"], 0)
        self.assertEqual(
            decoded["method_decision"],
            "do_not_admit_current_free_rewrite_repair",
        )

    def test_case_consensus_preserves_disagreement(self) -> None:
        decoded = decode_reviews(
            self.review,
            manifest_path=MANIFEST_PATH,
            key_rows=self.key,
        )
        by_case = {row["case_id"]: row for row in decoded["case_consensus"]}
        self.assertEqual(by_case["f01-wrong-vocal-gender"]["majority_preference"], "tie")
        self.assertTrue(by_case["f01-wrong-vocal-gender"]["unanimous_preference"])
        self.assertEqual(
            by_case["f02-forbidden-rap-mc-slogans"]["preference_votes"],
            {"base": 2, "repaired": 1, "tie": 0, "both_fail": 0},
        )
        self.assertEqual(
            by_case["f03-vocal-material-in-instrumental"]["preference_votes"],
            {"base": 2, "repaired": 1, "tie": 0, "both_fail": 0},
        )
        self.assertEqual(
            by_case["f04-hard-fixed-bpm"]["preference_votes"],
            {"base": 3, "repaired": 0, "tie": 0, "both_fail": 0},
        )

    def test_non_blind_review_is_rejected(self) -> None:
        payload = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        payload["blind_at_rating"] = False
        review = MultiModelReview.model_validate(payload)
        with self.assertRaisesRegex(ValueError, "reviews_were_not_blind"):
            decode_reviews(review, manifest_path=MANIFEST_PATH, key_rows=self.key)

    def test_incomplete_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "key_case_set_mismatch"):
            decode_reviews(
                self.review,
                manifest_path=MANIFEST_PATH,
                key_rows=self.key[:-1],
            )


if __name__ == "__main__":
    unittest.main()
