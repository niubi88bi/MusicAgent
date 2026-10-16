"""Tests for decoding repeated boundary blind reviews."""

from __future__ import annotations

import unittest

from scripts.evaluate_producer_rag_boundary import CASES_PATH, load_manifest
from scripts.summarize_producer_rag_ab_review import (
    CandidateHumanReview,
    CaseHumanReview,
    HumanBlindReview,
)
from scripts.summarize_producer_rag_boundary_review import decode_review


def _score(value: int) -> CandidateHumanReview:
    return CandidateHumanReview(
        style_fit=value,
        production_executability=value,
        internal_consistency=value,
        over_constraint=False,
        unsupported_expansion=False,
    )


@unittest.skipUnless(CASES_PATH.is_file(), "private Producer RAG cases are not published")
class ProducerRAGBoundaryReviewTests(unittest.TestCase):
    def test_decode_reports_family_stability(self) -> None:
        manifest = load_manifest()
        key_rows = []
        review_cases = []
        for family in manifest.generation_families:
            for replicate in range(1, family.repeat_count + 1):
                case_id = f"{family.family_id}-r{replicate}"
                key_rows.append(
                    {"case_id": case_id, "family_id": family.family_id, "X": "gated", "Y": "baseline"}
                )
                review_cases.append(
                    CaseHumanReview(
                        case_id=case_id,
                        X=_score(5),
                        Y=_score(4),
                        preference="X",
                        notes="test note",
                    )
                )
        review = HumanBlindReview(
            schema_version="1.0",
            reviewer_count=1,
            reviewer_profile="test reviewer",
            review_modality="text only",
            blind_at_rating=True,
            recorded_at="2026-09-18",
            limitations=["test fixture"],
            cases=review_cases,
        )
        result = decode_review(
            review,
            manifest_path=__import__(
                "scripts.evaluate_producer_rag_boundary", fromlist=["CASES_PATH"]
            ).CASES_PATH,
            key_rows=key_rows,
        )
        self.assertEqual(result["aggregate_all_pairs"]["gated"]["wins"], 4)
        self.assertTrue(
            all(row["stable_gated_win"] for row in result["by_family"].values())
        )

    def test_decode_rejects_non_blind_review(self) -> None:
        manifest = load_manifest()
        key_rows = []
        review_cases = []
        for family in manifest.generation_families:
            for replicate in range(1, family.repeat_count + 1):
                case_id = f"{family.family_id}-r{replicate}"
                key_rows.append({"case_id": case_id, "X": "gated", "Y": "baseline"})
                review_cases.append(
                    CaseHumanReview(
                        case_id=case_id,
                        X=_score(5),
                        Y=_score(4),
                        preference="X",
                        notes="not blind",
                    )
                )
        review = HumanBlindReview(
            schema_version="1.0",
            reviewer_count=1,
            reviewer_profile="test reviewer",
            review_modality="text only",
            blind_at_rating=False,
            recorded_at="2026-09-18",
            limitations=["test fixture"],
            cases=review_cases,
        )
        with self.assertRaisesRegex(ValueError, "review_was_not_blind_at_rating"):
            decode_review(review, manifest_path=__import__(
                "scripts.evaluate_producer_rag_boundary", fromlist=["CASES_PATH"]
            ).CASES_PATH, key_rows=key_rows)


if __name__ == "__main__":
    unittest.main()
