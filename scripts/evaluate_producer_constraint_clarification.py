"""Evaluate the deterministic Producer clarification shadow contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.build_producer_constraint_clarification_holdout import (
    DEFAULT_OUTPUT as DEFAULT_CASES,
    ClarificationHoldoutManifest,
)
from tools.rag.producer_constraint_clarification import (
    analyze_producer_constraint_clarification,
)


DEFAULT_REPORT = Path(
    "data/output/evaluations/producer-constraint-clarification-shadow-v1/summary.json"
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: Path = DEFAULT_CASES) -> ClarificationHoldoutManifest:
    return ClarificationHoldoutManifest.model_validate(_read(path))


def run_evaluation(
    manifest: ClarificationHoldoutManifest,
    *,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in manifest.cases:
        actual = analyze_producer_constraint_clarification(case.user_text)
        checks = {
            "status": actual.status == case.expected_status,
            "replacement_mode": (
                actual.replacement_mode == case.expected_replacement_mode
            ),
            "explicit_replacement": (
                actual.explicit_replacement == case.expected_explicit_replacement
            ),
            "question_count": len(actual.questions) == case.expected_question_count,
        }
        rows.append(
            {
                "case_id": case.case_id,
                "user_text": case.user_text,
                "expected": {
                    "status": case.expected_status,
                    "replacement_mode": case.expected_replacement_mode,
                    "explicit_replacement": case.expected_explicit_replacement,
                    "question_count": case.expected_question_count,
                },
                "actual": actual.model_dump(mode="json"),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    summary = {
        "schema_version": "1.0",
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "case_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "failed_count": sum(not row["passed"] for row in rows),
        "clarification_required_count": sum(
            row["actual"]["status"] == "clarification_required" for row in rows
        ),
        "advisory_only_count": sum(
            row["actual"]["status"] == "advisory_only" for row in rows
        ),
        "no_clarification_count": sum(
            row["actual"]["status"] == "no_clarification" for row in rows
        ),
        "rows": rows,
        "llm_calls": 0,
        "producer_calls": 0,
        "rag_injected": False,
        "audio_generated": False,
        "runtime_agent_connected": False,
    }
    _write(output_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="评估Producer制作约束澄清影子规则。")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    summary = run_evaluation(
        load_manifest(args.cases), manifest_path=args.cases, output_path=args.out
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
