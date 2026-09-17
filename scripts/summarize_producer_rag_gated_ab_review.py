"""Validate and decode the conservative-gate Producer blind review."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.evaluate_producer_rag_gated_ab import GatedProducerManifest, load_cases
from scripts.summarize_producer_rag_ab_review import HumanBlindReview
from tools.rag.style_retrieval_policy import decide_style_retrieval


VARIANTS = ("baseline", "gated")
DIMENSIONS = (
    "style_fit",
    "production_executability",
    "internal_consistency",
)
ISSUES = ("over_constraint", "unsupported_expansion")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_exact_case_set(
    review: HumanBlindReview,
    manifest: GatedProducerManifest,
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


def _aggregate(decoded_cases: list[dict[str, Any]]) -> dict[str, Any]:
    wins: Counter[str] = Counter()
    sums = {
        variant: {dimension: 0.0 for dimension in DIMENSIONS}
        for variant in VARIANTS
    }
    issues = {
        variant: {issue: 0 for issue in ISSUES}
        for variant in VARIANTS
    }
    for case in decoded_cases:
        wins[case["preferred_variant"]] += 1
        for variant in VARIANTS:
            scores = case["scores"][variant]
            for dimension in DIMENSIONS:
                sums[variant][dimension] += float(scores[dimension])
            for issue in ISSUES:
                issues[variant][issue] += int(bool(scores[issue]))
    denominator = len(decoded_cases)
    payload: dict[str, Any] = {"case_count": denominator}
    for variant in VARIANTS:
        payload[variant] = {
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
    payload["ties"] = wins["tie"]
    return payload


def decode_review(
    review: HumanBlindReview,
    manifest: GatedProducerManifest,
    key_rows: list[dict[str, str]],
) -> dict[str, Any]:
    _validate_exact_case_set(review, manifest, key_rows)
    if not review.blind_at_rating:
        raise ValueError("review_was_not_blind_at_rating")

    key_by_case = {row["case_id"]: row for row in key_rows}
    manifest_by_id = {case.case_id: case for case in manifest.cases}
    decoded_cases: list[dict[str, Any]] = []
    for case_review in review.cases:
        key = key_by_case[case_review.case_id]
        if {key.get("X"), key.get("Y")} != set(VARIANTS):
            raise ValueError(f"invalid_blind_mapping:{case_review.case_id}")
        case = manifest_by_id[case_review.case_id]
        decision = decide_style_retrieval(case.user_request_text)
        candidates = {"X": case_review.X, "Y": case_review.Y}
        decoded_scores: dict[str, dict[str, Any]] = {}
        for label, candidate in candidates.items():
            decoded_scores[key[label]] = candidate.model_dump(mode="json")
        preferred_variant = (
            "tie"
            if case_review.preference == "tie"
            else key[case_review.preference]
        )
        decoded_cases.append(
            {
                "case_id": case.case_id,
                "case_type": case.case_type,
                "gate_action": decision.action,
                "experimental_context_injected": decision.action != "no_rag",
                "preferred_variant": preferred_variant,
                "scores": decoded_scores,
                "notes": case_review.notes,
            }
        )

    injected = [
        case for case in decoded_cases if case["experimental_context_injected"]
    ]
    no_rag = [
        case for case in decoded_cases if not case["experimental_context_injected"]
    ]
    return {
        "schema_version": "1.0",
        "reviewer_count": review.reviewer_count,
        "reviewer_profile": review.reviewer_profile,
        "review_modality": review.review_modality,
        "blind_at_rating": review.blind_at_rating,
        "limitations": review.limitations,
        "aggregate_all_pairs": _aggregate(decoded_cases),
        "aggregate_injection_authorized_subset": _aggregate(injected),
        "aggregate_no_rag_subset": _aggregate(no_rag),
        "cases": decoded_cases,
        "method_decision": None,
        "decision_note": (
            "Descriptive single-reviewer evidence only. The no_rag pair uses "
            "identical prompts and must not be counted as a RAG-effect sample."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="解盲并汇总Producer保守门控第二轮人工复核。"
    )
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
    print(
        json.dumps(
            {
                "all_pairs": result["aggregate_all_pairs"],
                "injection_authorized_subset": result[
                    "aggregate_injection_authorized_subset"
                ],
                "no_rag_subset": result["aggregate_no_rag_subset"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
