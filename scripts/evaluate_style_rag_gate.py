"""Offline acceptance evaluation for the deterministic style-RAG shadow gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scripts.evaluate_producer_rag_ab import load_cases
from tools.rag.style_retrieval_policy import decide_style_retrieval


ACCEPTANCE_PATH = Path("data/miaoxiang/style_rag_gate_acceptance.v1.json")
HOLDOUT_PATH = Path("data/miaoxiang/style_rag_gate_holdout.v1.json")
DEFAULT_OUTPUT = Path(
    "data/output/evaluations/style-rag-gate-shadow-v1/result.json"
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GateExpectation(_StrictModel):
    query_id: str
    user_text: str
    expected_action: Literal["no_rag", "single_style", "multi_candidate"]
    expected_origin: str
    expected_reason: str
    expected_candidate_ids: list[str]
    expected_template_id: str | None = None


class GateManifest(_StrictModel):
    schema_version: str
    kind: Literal["style_rag_gate_acceptance", "style_rag_gate_holdout"]
    frozen_at: str
    notes: list[str] = Field(min_length=1)
    queries: list[GateExpectation] = Field(min_length=1)


DEV_EXPECTATIONS = {
    "p01-deep-house-night-shift": (
        "single_style",
        "user_explicit",
        "explicit_supported_style",
        ["style-edm-deep-house-v2"],
        None,
    ),
    "p02-tropical-house-seaside-reunion": (
        "single_style",
        "user_explicit",
        "explicit_supported_style",
        ["style-edm-tropical-house-v2"],
        None,
    ),
    "p03-zen-rap-insomnia": (
        "single_style",
        "user_explicit",
        "explicit_supported_style",
        ["style-rap-zen-fusion-v2"],
        None,
    ),
    "p04-melodic-electronic-boundary": (
        "multi_candidate",
        "user_multiple",
        "explicit_multiple_styles",
        ["style-edm-future-bass-v2", "style-edm-trance-v2"],
        None,
    ),
    "p05-guofeng-cyber-boundary": (
        "multi_candidate",
        "user_multiple",
        "explicit_multiple_styles",
        ["style-guofeng-electronic-fusion-v2", "style-rap-cyber-zen-v2"],
        None,
    ),
    "p06-sad-folk-electronic-composite": (
        "multi_candidate",
        "user_composite",
        "named_composite",
        ["style-dynamic-sad-slap-house-v2", "style-dynamic-vina-house-v2"],
        "style-dynamic-sad-folk-pop-remix-v2",
    ),
    "p07-emotion-only-unspecified-style": (
        "no_rag",
        "none",
        "no_explicit_supported_style",
        [],
        None,
    ),
    "p08-out-of-scope-bossa-jazz": (
        "no_rag",
        "unsupported_explicit",
        "unsupported_explicit_style",
        [],
        None,
    ),
}


def load_manifest(path: Path) -> GateManifest:
    return GateManifest.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_acceptance(path: Path = ACCEPTANCE_PATH) -> GateManifest:
    return load_manifest(path)


def _evaluate(expectation: GateExpectation, *, dataset: str) -> dict[str, object]:
    decision = decide_style_retrieval(expectation.user_text)
    checks = {
        "action": decision.action == expectation.expected_action,
        "origin": decision.style_intent_origin == expectation.expected_origin,
        "reason": decision.reason == expectation.expected_reason,
        "candidate_ids": decision.candidate_ids == expectation.expected_candidate_ids,
        "template_id": decision.composite_template_id == expectation.expected_template_id,
        "runtime_injection_disabled": decision.runtime_injection_allowed is False,
        "candidate_limit": len(decision.candidate_ids) <= 2,
    }
    return {
        "dataset": dataset,
        "query_id": expectation.query_id,
        "user_text": expectation.user_text,
        "expected": expectation.model_dump(mode="json"),
        "actual": decision.model_dump(mode="json"),
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_report(
    acceptance: GateManifest,
    holdout: GateManifest | None = None,
) -> dict[str, object]:
    acceptance_rows = [
        _evaluate(expectation, dataset="acceptance")
        for expectation in acceptance.queries
    ]
    producer_cases = load_cases().cases
    dev_rows: list[dict[str, object]] = []
    for case in producer_cases:
        action, origin, reason, candidates, template_id = DEV_EXPECTATIONS[case.case_id]
        dev_rows.append(
            _evaluate(
                GateExpectation(
                    query_id=case.case_id,
                    user_text=case.retrieval_query,
                    expected_action=action,
                    expected_origin=origin,
                    expected_reason=reason,
                    expected_candidate_ids=candidates,
                    expected_template_id=template_id,
                ),
                dataset="producer_ab_development",
            )
        )
    holdout_rows = (
        [_evaluate(expectation, dataset="holdout") for expectation in holdout.queries]
        if holdout is not None
        else []
    )
    rows = dev_rows + acceptance_rows + holdout_rows
    reasons = Counter(str(row["actual"]["reason"]) for row in rows)
    actions = Counter(str(row["actual"]["action"]) for row in rows)
    return {
        "schema_version": "1.0",
        "method": "deterministic_raw_user_text_shadow_gate",
        "runtime_agent_connected": False,
        "deepseek_calls": 0,
        "audio_generated": False,
        "single_similarity_threshold_used": False,
        "development": {
            "count": len(dev_rows),
            "passed": sum(bool(row["passed"]) for row in dev_rows),
        },
        "acceptance": {
            "count": len(acceptance_rows),
            "passed": sum(bool(row["passed"]) for row in acceptance_rows),
            "independent_holdout": False,
            "boundary": "Used during parser correction; contract acceptance only.",
        },
        "holdout": {
            "count": len(holdout_rows),
            "passed": sum(bool(row["passed"]) for row in holdout_rows),
            "frozen_after_implementation": holdout is not None,
            "boundary": "Project-authored contract holdout; not external validation.",
        },
        "aggregate": {
            "total": len(rows),
            "passed": sum(bool(row["passed"]) for row in rows),
            "actions": dict(sorted(actions.items())),
            "reasons": dict(sorted(reasons.items())),
            "runtime_injection_enabled_count": sum(
                bool(row["actual"]["runtime_injection_allowed"]) for row in rows
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="离线评估风格RAG影子门控。")
    parser.add_argument("--acceptance", type=Path, default=ACCEPTANCE_PATH)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(load_manifest(args.acceptance), load_manifest(args.holdout))
    report["input_files"] = {
        "acceptance": {
            "path": str(args.acceptance),
            "sha256": file_sha256(args.acceptance),
        },
        "holdout": {
            "path": str(args.holdout),
            "sha256": file_sha256(args.holdout),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    if report["aggregate"]["passed"] != report["aggregate"]["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
