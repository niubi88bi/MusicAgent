"""Measure natural Producer hard-constraint violations without repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.schemas import LyricDraft, ProductionPlan, SongRequirement, pydantic_validation_feedback
from scripts.evaluate_producer_rag_ab import build_producer_messages
from scripts.evaluate_producer_constraint_minimal_patch import plan_deterministic_stage
from services.config import get_settings
from services.llm import DeepSeekLLM
from tools.rag.producer_constraint_guard import (
    check_plan_constraints,
    extract_explicit_constraints,
    split_constraint_violations,
)


CASES_PATH = Path("data/miaoxiang/producer_constraint_natural_audit.v1.json")
DEFAULT_OUTPUT_ROOT = Path("data/output/evaluations")
TEMPERATURE = 0.55
MAX_TOKENS = 1000


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class NaturalAuditCase(_StrictModel):
    case_id: str = Field(min_length=1)
    risk_focus: Literal[
        "explicit_vocal_gender",
        "explicit_no_rap_mc_slogans",
        "explicit_instrumental_only",
        "explicit_instrument_substitution",
        "explicit_flexible_bpm",
    ]
    user_request_text: str = Field(min_length=10)
    expected_hard_concepts: list[str] = Field(min_length=1)
    requirement: SongRequirement
    lyric_draft: LyricDraft


class NaturalAuditManifest(_StrictModel):
    schema_version: str = Field(min_length=1)
    kind: Literal["producer_constraint_natural_audit"]
    frozen_at: str = Field(min_length=1)
    rubric: str = Field(min_length=1)
    notes: list[str] = Field(min_length=1)
    cases: list[NaturalAuditCase] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def validate_cases(self) -> "NaturalAuditManifest":
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("natural_audit_case_ids_must_be_unique")
        focus_counts = {focus: 0 for focus in {
            "explicit_vocal_gender",
            "explicit_no_rap_mc_slogans",
            "explicit_instrumental_only",
            "explicit_instrument_substitution",
            "explicit_flexible_bpm",
        }}
        for case in self.cases:
            focus_counts[case.risk_focus] += 1
        if focus_counts != {
            "explicit_vocal_gender": 3,
            "explicit_no_rap_mc_slogans": 3,
            "explicit_instrumental_only": 2,
            "explicit_instrument_substitution": 2,
            "explicit_flexible_bpm": 2,
        }:
            raise ValueError(f"natural_audit_case_mix_mismatch:{focus_counts}")
        return self


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = CASES_PATH) -> NaturalAuditManifest:
    return NaturalAuditManifest.model_validate(_read_json(path))


def validate_manifest_contracts(manifest: NaturalAuditManifest) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for case in manifest.cases:
        constraints = extract_explicit_constraints(case.user_request_text)
        actual = {
            "must_include": [row.concept for row in constraints.must_include],
            "must_avoid": [row.concept for row in constraints.must_avoid],
            "do_not_infer": constraints.do_not_infer,
        }
        actual_hard = set(actual["must_include"] + actual["must_avoid"])
        expected = set(case.expected_hard_concepts)
        if not expected.issubset(actual_hard) and not (
            "exact_bpm_as_hard_requirement" in constraints.do_not_infer
            and "bpm" in expected
        ):
            raise ValueError(
                f"natural_audit_constraint_mismatch:{case.case_id}:{actual}:{sorted(expected)}"
            )
        contracts.append(
            {
                "case_id": case.case_id,
                "risk_focus": case.risk_focus,
                "constraints": constraints.model_dump(mode="json"),
                "expected_hard_concepts": case.expected_hard_concepts,
            }
        )
    return contracts


def _call_initial_plan(llm: DeepSeekLLM, case: NaturalAuditCase) -> ProductionPlan:
    payload = llm.chat_json(
        build_producer_messages(case.requirement, case.lyric_draft),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        operation="producer_constraint_natural_audit_initial",
        payload_validator=lambda value: pydantic_validation_feedback(
            ProductionPlan, value
        ),
    )
    return ProductionPlan.model_validate(payload)


def run_audit(
    manifest: NaturalAuditManifest,
    *,
    manifest_path: Path,
    output_root: Path,
) -> Path:
    contracts = validate_manifest_contracts(manifest)
    output_dir = output_root / f"producer-constraint-natural-audit-v1-{uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = get_settings()
    _write_json(output_dir / "audit_contracts.json", contracts)
    _write_json(
        output_dir / "experiment_config.json",
        {
            "schema_version": "1.0",
            "case_manifest": str(manifest_path),
            "case_manifest_sha256": _file_hash(manifest_path),
            "case_count": len(manifest.cases),
            "initial_calls_planned": len(manifest.cases),
            "repair_calls_planned": 0,
            "rag_injected": False,
            "repair_enabled": False,
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
            print(f"运行自然违规审计：{case.case_id}", flush=True)
            try:
                constraints = extract_explicit_constraints(case.user_request_text)
                before_calls = len(llm.telemetry_snapshot())
                initial = _call_initial_plan(llm, case)
                violations = check_plan_constraints(
                    initial,
                    user_text=case.user_request_text,
                    constraints=constraints,
                )
                hard, advisory = split_constraint_violations(violations)
                intermediate, patches, unresolved = plan_deterministic_stage(
                    initial,
                    user_text=case.user_request_text,
                )
                records = llm.telemetry_snapshot()[before_calls:]
                report = {
                    "case_id": case.case_id,
                    "risk_focus": case.risk_focus,
                    "user_request_text": case.user_request_text,
                    "initial_plan": initial.model_dump(mode="json"),
                    "initial_hard_violations": [row.model_dump(mode="json") for row in hard],
                    "initial_advisories": [row.model_dump(mode="json") for row in advisory],
                    "deterministic_patches_preview": [
                        row.model_dump(mode="json") for row in patches
                    ],
                    "remaining_hard_after_deterministic_preview": [
                        row.model_dump(mode="json") for row in unresolved
                    ],
                    "natural_action": (
                        "pass_through"
                        if not hard
                        else "deterministic_candidate"
                        if not unresolved
                        else "fail_closed"
                    ),
                    "repair_performed": False,
                    "telemetry": DeepSeekLLM.summarize_call_records(records),
                }
                _write_json(case_dir / "result.json", report)
                completed.append(report)
            except Exception as error:
                failure = {
                    "case_id": case.case_id,
                    "status": "audit_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                _write_json(case_dir / "failure.json", failure)
                failures.append(failure)
    finally:
        telemetry = DeepSeekLLM.summarize_call_records(llm.telemetry_snapshot())
        llm.close()

    _write_json(
        output_dir / "summary.json",
        {
            "completed_case_count": len(completed),
            "failed_case_count": len(failures),
            "pass_through_count": sum(row["natural_action"] == "pass_through" for row in completed),
            "hard_violation_count": sum(bool(row["initial_hard_violations"]) for row in completed),
            "deterministic_candidate_count": sum(
                row["natural_action"] == "deterministic_candidate" for row in completed
            ),
            "fail_closed_count": sum(row["natural_action"] == "fail_closed" for row in completed),
            "repair_calls": 0,
            "completed": completed,
            "failures": failures,
            "llm_telemetry": telemetry,
            "rag_injected": False,
            "repair_enabled": False,
            "audio_generated": False,
            "runtime_agent_connected": False,
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="运行Producer自然hard约束违规率审计。")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true", help="调用一次首稿Producer；不修复、不生成音频")
    args = parser.parse_args()
    manifest = load_manifest(args.cases)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "case_count": len(manifest.cases),
                    "repair_calls": 0,
                    "rag_injected": False,
                    "audio_generated": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    output = run_audit(manifest, manifest_path=args.cases, output_root=args.out)
    print(f"实验输出：{output}")


if __name__ == "__main__":
    main()
