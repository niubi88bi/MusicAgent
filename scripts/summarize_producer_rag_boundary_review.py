"""Validate and decode the repeated high-risk boundary blind review."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.evaluate_producer_rag_boundary import load_manifest
from scripts.summarize_producer_rag_ab_review import HumanBlindReview


VARIANTS = ("baseline", "gated")
DIMENSIONS = (
    "style_fit",
    "production_executability",
    "internal_consistency",
)
ISSUES = ("over_constraint", "unsupported_expansion")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins: Counter[str] = Counter()
    sums = {
        variant: {dimension: 0.0 for dimension in DIMENSIONS}
        for variant in VARIANTS
    }
    issues = {
        variant: {issue: 0 for issue in ISSUES}
        for variant in VARIANTS
    }
    for row in rows:
        wins[row["preferred_variant"]] += 1
        for variant in VARIANTS:
            score = row["scores"][variant]
            for dimension in DIMENSIONS:
                sums[variant][dimension] += float(score[dimension])
            for issue in ISSUES:
                issues[variant][issue] += int(bool(score[issue]))
    denominator = len(rows)
    result: dict[str, Any] = {"case_count": denominator}
    for variant in VARIANTS:
        result[variant] = {
            "wins": wins[variant],
            "mean_scores": {
                dimension: (
                    round(sums[variant][dimension] / denominator, 3)
                    if denominator
                    else None
                )
                for dimension in DIMENSIONS
            },
            **issues[variant],
        }
    result["ties"] = wins["tie"]
    return result


def decode_review(
    review: HumanBlindReview,
    *,
    manifest_path: Path,
    key_rows: list[dict[str, str]],
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    expected_ids = {
        f"{family.family_id}-r{replicate}"
        for family in manifest.generation_families
        for replicate in range(1, family.repeat_count + 1)
    }
    reviewed_ids = {row.case_id for row in review.cases}
    keyed_ids = {row.get("case_id", "") for row in key_rows}
    if reviewed_ids != expected_ids:
        raise ValueError("boundary_review_case_set_mismatch")
    if keyed_ids != expected_ids:
        raise ValueError("boundary_key_case_set_mismatch")
    if len(review.cases) != len(expected_ids):
        raise ValueError("boundary_review_case_count_mismatch")
    if not review.blind_at_rating:
        raise ValueError("review_was_not_blind_at_rating")

    key_by_case = {row["case_id"]: row for row in key_rows}
    family_by_sample = {
        f"{family.family_id}-r{replicate}": family.family_id
        for family in manifest.generation_families
        for replicate in range(1, family.repeat_count + 1)
    }
    decoded: list[dict[str, Any]] = []
    for case_review in review.cases:
        key = key_by_case[case_review.case_id]
        if {key.get("X"), key.get("Y")} != set(VARIANTS):
            raise ValueError(f"invalid_blind_mapping:{case_review.case_id}")
        candidates = {"X": case_review.X, "Y": case_review.Y}
        scores = {
            key[label]: candidate.model_dump(mode="json")
            for label, candidate in candidates.items()
        }
        preferred = (
            "tie" if case_review.preference == "tie" else key[case_review.preference]
        )
        decoded.append(
            {
                "case_id": case_review.case_id,
                "family_id": family_by_sample[case_review.case_id],
                "preferred_variant": preferred,
                "scores": scores,
                "notes": case_review.notes,
            }
        )

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decoded:
        by_family[row["family_id"]].append(row)
    family_results: dict[str, Any] = {}
    for family_id, rows in sorted(by_family.items()):
        aggregate = _aggregate(rows)
        aggregate["stable_gated_win"] = (
            aggregate["gated"]["wins"] == aggregate["case_count"]
            and aggregate["case_count"] >= 2
        )
        family_results[family_id] = aggregate

    return {
        "schema_version": "1.0",
        "reviewer_count": review.reviewer_count,
        "reviewer_profile": review.reviewer_profile,
        "review_modality": review.review_modality,
        "blind_at_rating": review.blind_at_rating,
        "limitations": review.limitations,
        "aggregate_all_pairs": _aggregate(decoded),
        "by_family": family_results,
        "offline_contract_cases": [
            {
                "case_id": case.case_id,
                "expected_action": case.expected_action,
                "expected_reason": case.expected_reason,
                "deepseek_calls": 0,
            }
            for case in manifest.offline_cases
        ],
        "cases": decoded,
        "method_decision": None,
        "decision_note": (
            "Descriptive single-reviewer evidence only. Stable family wins are "
            "screening evidence, not production admission or audio-quality proof."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="解盲并汇总Producer风格RAG高风险边界复核。"
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    review = HumanBlindReview.model_validate(_read_json(args.review))
    key_rows = _read_json(args.key)
    if not isinstance(key_rows, list):
        raise ValueError("blind_key_root_must_be_list")
    result = decode_review(review, manifest_path=args.cases, key_rows=key_rows)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "all_pairs": result["aggregate_all_pairs"],
                "by_family": result["by_family"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
