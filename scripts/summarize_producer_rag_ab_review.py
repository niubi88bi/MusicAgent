"""Validate and decode one blind human review for the Producer RAG A/B pilot."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scripts.evaluate_producer_rag_ab import ProducerRAGManifest, load_cases


Variant = Literal["rag_off", "rag_on"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CandidateHumanReview(_StrictModel):
    style_fit: int = Field(ge=1, le=5)
    production_executability: int = Field(ge=1, le=5)
    internal_consistency: int = Field(ge=1, le=5)
    over_constraint: bool
    unsupported_expansion: bool


class CaseHumanReview(_StrictModel):
    case_id: str = Field(min_length=1)
    X: CandidateHumanReview
    Y: CandidateHumanReview
    preference: Literal["X", "Y", "tie"]
    notes: str = Field(min_length=1)


class HumanBlindReview(_StrictModel):
    schema_version: str = Field(min_length=1)
    reviewer_count: int = Field(ge=1)
    reviewer_profile: str = Field(min_length=1)
    review_modality: str = Field(min_length=1)
    blind_at_rating: bool
    recorded_at: str = Field(min_length=1)
    limitations: list[str] = Field(min_length=1)
    cases: list[CaseHumanReview] = Field(min_length=1)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_exact_case_set(
    review: HumanBlindReview,
    manifest: ProducerRAGManifest,
    key_rows: list[dict[str, str]],
) -> None:
    expected = {case.case_id for case in manifest.cases}
    reviewed = [case.case_id for case in review.cases]
    keyed = [row.get("case_id", "") for row in key_rows]
    if len(reviewed) != len(set(reviewed)):
        raise ValueError("duplicate_review_case_id")
    if len(keyed) != len(set(keyed)):
        raise ValueError("duplicate_key_case_id")
    if set(reviewed) != expected:
        raise ValueError("review_case_set_mismatch")
    if set(keyed) != expected:
        raise ValueError("blind_key_case_set_mismatch")


def decode_review(
    review: HumanBlindReview,
    manifest: ProducerRAGManifest,
    key_rows: list[dict[str, str]],
) -> dict[str, Any]:
    _validate_exact_case_set(review, manifest, key_rows)
    if not review.blind_at_rating:
        raise ValueError("review_was_not_blind_at_rating")

    key_by_case = {row["case_id"]: row for row in key_rows}
    case_type_by_id = {case.case_id: case.case_type for case in manifest.cases}
    dimensions = (
        "style_fit",
        "production_executability",
        "internal_consistency",
    )
    sums: dict[str, dict[str, float]] = {
        variant: {dimension: 0.0 for dimension in dimensions}
        for variant in ("rag_off", "rag_on")
    }
    issue_counts = {
        variant: {"over_constraint": 0, "unsupported_expansion": 0}
        for variant in ("rag_off", "rag_on")
    }
    wins: Counter[str] = Counter()
    type_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    decoded_cases: list[dict[str, Any]] = []

    for case in review.cases:
        key = key_by_case[case.case_id]
        if {key.get("X"), key.get("Y")} != {"rag_off", "rag_on"}:
            raise ValueError(f"invalid_blind_mapping:{case.case_id}")
        candidates = {"X": case.X, "Y": case.Y}
        decoded_scores: dict[str, dict[str, Any]] = {}
        for label, candidate in candidates.items():
            variant = key[label]
            payload = candidate.model_dump(mode="json")
            decoded_scores[variant] = payload
            for dimension in dimensions:
                sums[variant][dimension] += float(payload[dimension])
            for issue in ("over_constraint", "unsupported_expansion"):
                issue_counts[variant][issue] += int(bool(payload[issue]))

        preferred_variant = (
            "tie" if case.preference == "tie" else key[case.preference]
        )
        wins[preferred_variant] += 1
        type_outcomes[case_type_by_id[case.case_id]][preferred_variant] += 1
        decoded_cases.append(
            {
                "case_id": case.case_id,
                "case_type": case_type_by_id[case.case_id],
                "preferred_variant": preferred_variant,
                "scores": decoded_scores,
                "notes": case.notes,
            }
        )

    denominator = len(review.cases)
    aggregate: dict[str, Any] = {}
    for variant in ("rag_off", "rag_on"):
        aggregate[variant] = {
            "wins": wins[variant],
            "mean_scores": {
                dimension: round(sums[variant][dimension] / denominator, 3)
                for dimension in dimensions
            },
            **issue_counts[variant],
        }
    aggregate["ties"] = wins["tie"]

    return {
        "schema_version": "1.0",
        "reviewer_count": review.reviewer_count,
        "reviewer_profile": review.reviewer_profile,
        "review_modality": review.review_modality,
        "blind_at_rating": review.blind_at_rating,
        "limitations": review.limitations,
        "aggregate": aggregate,
        "outcomes_by_case_type": {
            case_type: {
                "rag_off": outcomes["rag_off"],
                "rag_on": outcomes["rag_on"],
                "tie": outcomes["tie"],
            }
            for case_type, outcomes in sorted(type_outcomes.items())
        },
        "cases": decoded_cases,
        "method_decision": None,
        "decision_note": (
            "Descriptive single-reviewer evidence only; a method decision must "
            "consider retrieval failures and sampling variance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="解盲并汇总 Producer RAG 人工复核。")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    review = HumanBlindReview.model_validate(_read_json(args.review))
    manifest = load_cases(args.cases) if args.cases else load_cases()
    key_rows = _read_json(args.key)
    if not isinstance(key_rows, list):
        raise ValueError("blind_key_root_must_be_list")
    result = decode_review(review, manifest, key_rows)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
