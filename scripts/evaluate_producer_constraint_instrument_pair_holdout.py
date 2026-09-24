"""Run the deterministic paired piano/acoustic-guitar shadow holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.build_producer_constraint_instrument_pair_holdout import (
    DEFAULT_OUTPUT,
    InstrumentPairHoldoutManifest,
)
from scripts.evaluate_producer_constraint_minimal_patch import plan_deterministic_stage
from tools.rag.producer_constraint_guard import (
    check_plan_constraints,
    extract_explicit_constraints,
    split_constraint_violations,
)
from tools.rag.producer_constraint_patch import (
    apply_fragment_patch,
    changed_fields,
    validate_patched_plan,
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = DEFAULT_OUTPUT) -> InstrumentPairHoldoutManifest:
    return InstrumentPairHoldoutManifest.model_validate(_read(path))


def run_holdout(manifest: InstrumentPairHoldoutManifest, *, manifest_path: Path, output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in manifest.cases:
        constraints = extract_explicit_constraints(case.request)
        initial_violations = check_plan_constraints(
            case.injected_plan,
            user_text=case.request,
            constraints=constraints,
        )
        initial_hard, initial_advisory = split_constraint_violations(initial_violations)
        final, patches, unresolved = plan_deterministic_stage(
            case.injected_plan,
            user_text=case.request,
        )
        if not initial_hard:
            action = "pass_through"
            validation = None
        elif not unresolved:
            action = "deterministic_repair"
            validation_model = validate_patched_plan(
                case.injected_plan,
                final,
                patches=patches,
                user_text=case.request,
                constraints=constraints,
            )
            validation = validation_model.model_dump(mode="json")
            if not validation_model.accepted:
                raise AssertionError(f"holdout_repair_rejected:{case.case_id}:{validation_model.reason}")
        else:
            action = "fail_closed"
            validation = None
        if action != case.expected_action:
            raise AssertionError(
                f"holdout_action_mismatch:{case.case_id}:{action}!={case.expected_action}"
            )
        final_violations = check_plan_constraints(
            final,
            user_text=case.request,
            constraints=constraints,
        )
        final_hard, final_advisory = split_constraint_violations(final_violations)
        if action == "deterministic_repair" and final_hard:
            raise AssertionError(f"holdout_repair_hard_remaining:{case.case_id}")
        rows.append(
            {
                "case_id": case.case_id,
                "expected_action": case.expected_action,
                "actual_action": action,
                "initial_hard": [row.model_dump(mode="json") for row in initial_hard],
                "initial_advisory": [row.model_dump(mode="json") for row in initial_advisory],
                "patches": [row.model_dump(mode="json") for row in patches],
                "remaining_hard": [row.model_dump(mode="json") for row in unresolved],
                "final_hard": [row.model_dump(mode="json") for row in final_hard],
                "final_advisory": [row.model_dump(mode="json") for row in final_advisory],
                "changed_fields": changed_fields(case.injected_plan, final),
                "final_equals_base": final == case.base_plan,
                "expected_final_equals_base": case.expected_final_equals_base,
                "validation": validation,
            }
        )
    summary = {
        "schema_version": "1.0",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "case_count": len(rows),
        "action_counts": {
            action: sum(row["actual_action"] == action for row in rows)
            for action in ("pass_through", "deterministic_repair", "fail_closed")
        },
        "accepted_repairs": sum(
            row["actual_action"] == "deterministic_repair"
            and bool(row["validation"] and row["validation"]["accepted"])
            for row in rows
        ),
        "all_expected_actions_match": True,
        "all_repaired_cases_hard_clear": all(
            not row["final_hard"]
            for row in rows
            if row["actual_action"] == "deterministic_repair"
        ),
        "all_cases": rows,
        "model_calls": 0,
        "rag_injected": False,
        "audio_generated": False,
        "runtime_agent_connected": False,
    }
    _write(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="运行成对乐器约束独立holdout。")
    parser.add_argument("--cases", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = run_holdout(
        load_manifest(args.cases),
        manifest_path=args.cases,
        output=args.out,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
