"""Validate, decode, and summarize the three-model rubric-v2 blind review."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scripts.evaluate_producer_rag_rubric import load_manifest
from scripts.summarize_producer_rag_ab_review import CaseHumanReview


VARIANTS = ("baseline", "gated")
DIMENSIONS = ("style_fit", "production_executability", "internal_consistency")
ISSUES = ("over_constraint", "unsupported_expansion")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ModelReviewer(_StrictModel):
    reviewer_id: str = Field(min_length=1)
    reported_family: str = Field(min_length=1)
    cases: list[CaseHumanReview] = Field(min_length=1)


class MultiModelBlindReview(_StrictModel):
    schema_version: str = Field(min_length=1)
    recorded_at: str = Field(min_length=1)
    review_modality: str = Field(min_length=1)
    blind_at_rating: bool
    model_versions: str = Field(min_length=1)
    rubric: str = Field(min_length=1)
    limitations: list[str] = Field(min_length=1)
    reviewers: list[ModelReviewer] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_reviewer_ids(self) -> "MultiModelBlindReview":
        ids = [reviewer.reviewer_id for reviewer in self.reviewers]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate_reviewer_id")
        return self


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins: Counter[str] = Counter(row["preferred_variant"] for row in rows)
    result: dict[str, Any] = {"decision_count": len(rows)}
    for variant in VARIANTS:
        variant_scores = [row["scores"][variant] for row in rows]
        result[variant] = {
            "wins": wins[variant],
            "mean_scores": {
                dimension: round(
                    sum(float(score[dimension]) for score in variant_scores)
                    / len(variant_scores),
                    3,
                )
                for dimension in DIMENSIONS
            },
            **{
                issue: sum(int(bool(score[issue])) for score in variant_scores)
                for issue in ISSUES
            },
        }
    result["ties"] = wins["tie"]
    return result


def decode_reviews(
    review: MultiModelBlindReview,
    *,
    manifest_path: Path,
    key_rows: list[dict[str, str]],
) -> dict[str, Any]:
    if not review.blind_at_rating:
        raise ValueError("reviews_were_not_blind_at_rating")
    manifest = load_manifest(manifest_path)
    expected_ids = {case.case_id for case in manifest.cases}
    key_by_case = {row.get("case_id", ""): row for row in key_rows}
    if set(key_by_case) != expected_ids or len(key_rows) != len(expected_ids):
        raise ValueError("rubric_key_case_set_mismatch")

    decoded_reviewers: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    rows_by_case: dict[str, list[dict[str, Any]]] = {
        case_id: [] for case_id in expected_ids
    }
    for reviewer in review.reviewers:
        reviewed_ids = [case.case_id for case in reviewer.cases]
        if set(reviewed_ids) != expected_ids or len(reviewed_ids) != len(expected_ids):
            raise ValueError(f"rubric_review_case_set_mismatch:{reviewer.reviewer_id}")
        reviewer_rows: list[dict[str, Any]] = []
        for case in reviewer.cases:
            key = key_by_case[case.case_id]
            if {key.get("X"), key.get("Y")} != set(VARIANTS):
                raise ValueError(f"invalid_blind_mapping:{case.case_id}")
            candidates = {"X": case.X, "Y": case.Y}
            scores = {
                key[label]: candidate.model_dump(mode="json")
                for label, candidate in candidates.items()
            }
            preferred = "tie" if case.preference == "tie" else key[case.preference]
            row = {
                "case_id": case.case_id,
                "preferred_variant": preferred,
                "scores": scores,
                "notes": case.notes,
            }
            reviewer_rows.append(row)
            all_rows.append(row)
            rows_by_case[case.case_id].append(row)
        decoded_reviewers.append(
            {
                "reviewer_id": reviewer.reviewer_id,
                "reported_family": reviewer.reported_family,
                "aggregate": _aggregate_rows(reviewer_rows),
                "cases": reviewer_rows,
            }
        )

    consensus: list[dict[str, Any]] = []
    for case in manifest.cases:
        rows = rows_by_case[case.case_id]
        votes = Counter(row["preferred_variant"] for row in rows)
        majority, majority_count = votes.most_common(1)[0]
        consensus.append(
            {
                "case_id": case.case_id,
                "risk_focus": case.risk_focus,
                "preference_votes": {
                    "baseline": votes["baseline"],
                    "gated": votes["gated"],
                    "tie": votes["tie"],
                },
                "majority_preference": majority if majority_count >= 2 else None,
                "unanimous_preference": len(votes) == 1,
                "aggregate": _aggregate_rows(rows),
            }
        )

    return {
        "schema_version": "1.0",
        "reviewer_count": len(review.reviewers),
        "review_modality": review.review_modality,
        "blind_at_rating": review.blind_at_rating,
        "model_versions": review.model_versions,
        "rubric": review.rubric,
        "limitations": review.limitations,
        "aggregate_all_decisions": _aggregate_rows(all_rows),
        "reviewers": decoded_reviewers,
        "case_consensus": consensus,
        "method_decision": None,
        "runtime_decision": "unchanged_disabled",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="解盲并汇总Producer风格RAG评审口径v2三模型复核。"
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    review = MultiModelBlindReview.model_validate(_read_json(args.review))
    key_rows = _read_json(args.key)
    if not isinstance(key_rows, list):
        raise ValueError("blind_key_root_must_be_list")
    result = decode_reviews(
        review,
        manifest_path=args.cases,
        key_rows=key_rows,
    )
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "aggregate_all_decisions": result["aggregate_all_decisions"],
                "case_consensus": result["case_consensus"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
