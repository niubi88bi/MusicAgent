"""Validate and decode the three-model controlled-fault review."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scripts.evaluate_producer_constraint_fault_injection import load_manifest


VARIANTS = ("base", "repaired")
DIMENSIONS = ("style_fit", "production_executability", "internal_consistency")
ISSUES = (
    "over_constraint",
    "unsupported_expansion",
    "hard_requirement_violation",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CandidateReview(_StrictModel):
    style_fit: int = Field(ge=1, le=5)
    production_executability: int = Field(ge=1, le=5)
    internal_consistency: int = Field(ge=1, le=5)
    over_constraint: bool
    unsupported_expansion: bool
    hard_requirement_violation: bool


class CaseReview(_StrictModel):
    case_id: str = Field(min_length=1)
    X: CandidateReview
    Y: CandidateReview
    preference: Literal["X", "Y", "tie", "both_fail"]
    notes: str = Field(min_length=1)


class Reviewer(_StrictModel):
    reviewer_id: str = Field(min_length=1)
    reported_family: str = Field(min_length=1)
    cases: list[CaseReview] = Field(min_length=4, max_length=4)


class MultiModelReview(_StrictModel):
    schema_version: str = Field(min_length=1)
    recorded_at: str = Field(min_length=1)
    review_modality: str = Field(min_length=1)
    blind_at_rating: bool
    model_versions: str = Field(min_length=1)
    rubric: str = Field(min_length=1)
    limitations: list[str] = Field(min_length=1)
    reviewers: list[Reviewer] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_reviewer_ids(self) -> "MultiModelReview":
        reviewer_ids = [reviewer.reviewer_id for reviewer in self.reviewers]
        if len(reviewer_ids) != len(set(reviewer_ids)):
            raise ValueError("duplicate_reviewer_id")
        return self


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    votes = Counter(row["preferred_variant"] for row in rows)
    result: dict[str, Any] = {"decision_count": len(rows)}
    for variant in VARIANTS:
        scores = [row["scores"][variant] for row in rows]
        result[variant] = {
            "wins": votes[variant],
            "mean_scores": {
                dimension: round(
                    sum(float(score[dimension]) for score in scores) / len(scores),
                    3,
                )
                for dimension in DIMENSIONS
            },
            **{
                issue: sum(int(bool(score[issue])) for score in scores)
                for issue in ISSUES
            },
        }
    result["ties"] = votes["tie"]
    result["both_fail"] = votes["both_fail"]
    return result


def decode_reviews(
    review: MultiModelReview,
    *,
    manifest_path: Path,
    key_rows: list[dict[str, str]],
) -> dict[str, Any]:
    if not review.blind_at_rating:
        raise ValueError("reviews_were_not_blind_at_rating")
    manifest_ids = {
        case.review_case_id or case.case_id
        for case in load_manifest(manifest_path).cases
    }
    key_by_case = {row.get("case_id", ""): row for row in key_rows}
    if set(key_by_case) != manifest_ids:
        raise ValueError("fault_review_key_case_set_mismatch")

    all_rows: list[dict[str, Any]] = []
    decoded_reviewers: list[dict[str, Any]] = []
    rows_by_case: dict[str, list[dict[str, Any]]] = {
        case_id: [] for case_id in key_by_case
    }
    for reviewer in review.reviewers:
        reviewed_ids = [case.case_id for case in reviewer.cases]
        if set(reviewed_ids) != manifest_ids or len(reviewed_ids) != len(manifest_ids):
            raise ValueError(f"fault_review_case_set_mismatch:{reviewer.reviewer_id}")
        rows: list[dict[str, Any]] = []
        for case in reviewer.cases:
            key = key_by_case[case.case_id]
            if {key.get("X"), key.get("Y")} != set(VARIANTS):
                raise ValueError(f"invalid_fault_blind_mapping:{case.case_id}")
            candidates = {"X": case.X, "Y": case.Y}
            scores = {
                key[label]: candidate.model_dump(mode="json")
                for label, candidate in candidates.items()
            }
            preferred = (
                case.preference
                if case.preference in {"tie", "both_fail"}
                else key[case.preference]
            )
            row = {
                "case_id": case.case_id,
                "preferred_variant": preferred,
                "scores": scores,
                "notes": case.notes,
            }
            rows.append(row)
            all_rows.append(row)
            rows_by_case[case.case_id].append(row)
        decoded_reviewers.append(
            {
                "reviewer_id": reviewer.reviewer_id,
                "reported_family": reviewer.reported_family,
                "aggregate": _aggregate(rows),
                "cases": rows,
            }
        )

    consensus: list[dict[str, Any]] = []
    for case_id in sorted(manifest_ids):
        rows = rows_by_case[case_id]
        votes = Counter(row["preferred_variant"] for row in rows)
        majority, count = votes.most_common(1)[0]
        consensus.append(
            {
                "case_id": case_id,
                "preference_votes": {
                    "base": votes["base"],
                    "repaired": votes["repaired"],
                    "tie": votes["tie"],
                    "both_fail": votes["both_fail"],
                },
                "majority_preference": majority if count >= 2 else None,
                "unanimous_preference": len(votes) == 1,
                "aggregate": _aggregate(rows),
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
        "aggregate_all_decisions": _aggregate(all_rows),
        "reviewers": decoded_reviewers,
        "case_consensus": consensus,
        "evidence_scope": "conditional_repair_capability_not_natural_failure_rate",
        "method_decision": "do_not_admit_current_free_rewrite_repair",
        "next_candidate": "discuss_minimal_diff_field_level_repair",
        "runtime_decision": "unchanged_disabled",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="解盲并汇总Producer受控故障注入三模型复核。"
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    review = MultiModelReview.model_validate(_read_json(args.review))
    key_rows = _read_json(args.key)
    if not isinstance(key_rows, list):
        raise ValueError("blind_key_root_must_be_list")
    result = decode_reviews(review, manifest_path=args.cases, key_rows=key_rows)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "aggregate_all_decisions": result["aggregate_all_decisions"],
                "case_consensus": result["case_consensus"],
                "method_decision": result["method_decision"],
                "next_candidate": result["next_candidate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
