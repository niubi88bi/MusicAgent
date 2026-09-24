"""Shadow-audit natural Producer outputs with the paired instrument patch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from agents.schemas import ProductionPlan, pydantic_validation_feedback
from scripts.build_producer_constraint_instrument_pair_natural_manifest import (
    DEFAULT_OUTPUT as DEFAULT_CASES,
    PairNaturalManifest,
)
from scripts.evaluate_producer_constraint_minimal_patch import plan_deterministic_stage
from scripts.evaluate_producer_rag_ab import build_producer_messages
from services.config import get_settings
from services.llm import DeepSeekLLM
from tools.rag.producer_constraint_guard import (
    check_plan_constraints,
    extract_explicit_constraints,
    split_constraint_violations,
)
from tools.rag.producer_constraint_patch import (
    changed_fields,
    validate_patched_plan,
)


DEFAULT_OUTPUT_ROOT = Path("data/output/evaluations")
TEMPERATURE = 0.55
MAX_TOKENS = 1000


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = DEFAULT_CASES) -> PairNaturalManifest:
    return PairNaturalManifest.model_validate(_read(path))


def _call_initial_plan(llm: DeepSeekLLM, case: Any) -> ProductionPlan:
    payload = llm.chat_json(
        build_producer_messages(case.requirement, case.lyric_draft),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        operation="producer_constraint_instrument_pair_natural_initial",
        payload_validator=lambda value: pydantic_validation_feedback(
            ProductionPlan, value
        ),
    )
    return ProductionPlan.model_validate(payload)


def run_audit(
    manifest: PairNaturalManifest,
    *,
    manifest_path: Path,
    output_root: Path,
) -> Path:
    output_dir = output_root / (
        f"producer-constraint-instrument-pair-natural-v1-{uuid4().hex}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = get_settings()
    _write(
        output_dir / "experiment_config.json",
        {
            "schema_version": "1.0",
            "case_manifest": str(manifest_path),
            "case_manifest_sha256": _sha256(manifest_path),
            "case_count": len(manifest.cases),
            "initial_calls_planned": len(manifest.cases),
            "repair_calls_planned": 0,
            "paired_patch_enabled": True,
            "rag_injected": False,
            "audio_generated": False,
            "runtime_agent_connected": False,
            "llm": {
                "provider": "DeepSeek",
                "model": settings.deepseek_model,
                "json_thinking_mode": (
                    "disabled" if settings.llm_json_disable_thinking else "default"
                ),
            },
        },
    )

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    llm = DeepSeekLLM(settings)
    try:
        for case in manifest.cases:
            case_dir = output_dir / "cases" / case.case_id
            case_dir.mkdir(parents=True, exist_ok=False)
            print(f"运行成对乐器自然审计：{case.case_id}", flush=True)
            try:
                constraints = extract_explicit_constraints(case.user_request_text)
                before_calls = len(llm.telemetry_snapshot())
                initial = _call_initial_plan(llm, case)
                initial_violations = check_plan_constraints(
                    initial,
                    user_text=case.user_request_text,
                    constraints=constraints,
                )
                initial_hard, initial_advisory = split_constraint_violations(
                    initial_violations
                )
                candidate, patches, unresolved = plan_deterministic_stage(
                    initial,
                    user_text=case.user_request_text,
                )
                shadow_final = initial
                validation = None
                action = "pass_through" if not initial_hard else "fail_closed"
                if initial_hard and not unresolved:
                    validation_model = validate_patched_plan(
                        initial,
                        candidate,
                        patches=patches,
                        user_text=case.user_request_text,
                        constraints=constraints,
                    )
                    validation = validation_model.model_dump(mode="json")
                    if validation_model.accepted:
                        shadow_final = candidate
                        action = "deterministic_shadow_repair"
                    else:
                        patches = []
                records = llm.telemetry_snapshot()[before_calls:]
                final_violations = check_plan_constraints(
                    shadow_final,
                    user_text=case.user_request_text,
                    constraints=constraints,
                )
                final_hard, final_advisory = split_constraint_violations(
                    final_violations
                )
                report = {
                    "case_id": case.case_id,
                    "risk_focus": case.risk_focus,
                    "user_request_text": case.user_request_text,
                    "initial_plan": initial.model_dump(mode="json"),
                    "initial_hard_violations": [
                        row.model_dump(mode="json") for row in initial_hard
                    ],
                    "initial_advisories": [
                        row.model_dump(mode="json") for row in initial_advisory
                    ],
                    "deterministic_patches": [
                        row.model_dump(mode="json") for row in patches
                    ],
                    "remaining_hard_after_shadow_preview": [
                        row.model_dump(mode="json") for row in unresolved
                    ],
                    "shadow_final_plan": shadow_final.model_dump(mode="json"),
                    "final_hard": [row.model_dump(mode="json") for row in final_hard],
                    "final_advisory": [
                        row.model_dump(mode="json") for row in final_advisory
                    ],
                    "natural_action": action,
                    "shadow_changed_fields": changed_fields(initial, shadow_final),
                    "validation": validation,
                    "repair_performed": action == "deterministic_shadow_repair",
                    "telemetry": DeepSeekLLM.summarize_call_records(records),
                }
                _write(case_dir / "result.json", report)
                completed.append(report)
            except Exception as error:
                failure = {
                    "case_id": case.case_id,
                    "status": "audit_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                _write(case_dir / "failure.json", failure)
                failures.append(failure)
    finally:
        telemetry = DeepSeekLLM.summarize_call_records(llm.telemetry_snapshot())
        llm.close()

    _write(
        output_dir / "summary.json",
        {
            "completed_case_count": len(completed),
            "failed_case_count": len(failures),
            "pass_through_count": sum(
                row["natural_action"] == "pass_through" for row in completed
            ),
            "deterministic_shadow_repair_count": sum(
                row["natural_action"] == "deterministic_shadow_repair"
                for row in completed
            ),
            "fail_closed_count": sum(
                row["natural_action"] == "fail_closed" for row in completed
            ),
            "initial_hard_violation_count": sum(
                bool(row["initial_hard_violations"]) for row in completed
            ),
            "shadow_final_hard_violation_count": sum(
                bool(row["final_hard"]) for row in completed
            ),
            "repair_calls": 0,
            "completed": completed,
            "failures": failures,
            "llm_telemetry": telemetry,
            "paired_patch_enabled": True,
            "rag_injected": False,
            "audio_generated": False,
            "runtime_agent_connected": False,
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="运行成对乐器自然影子审计。")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.cases)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "case_count": len(manifest.cases),
                    "initial_calls_planned": len(manifest.cases),
                    "repair_calls_planned": 0,
                    "rag_injected": False,
                    "audio_generated": False,
                    "runtime_agent_connected": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    output = run_audit(manifest, manifest_path=args.cases, output_root=args.out)
    print(f"实验输出：{output}", flush=True)


if __name__ == "__main__":
    main()
