"""Evaluate conditional Producer repair with controlled hard-constraint faults."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.schemas import ProductionPlan, pydantic_validation_feedback
from scripts.evaluate_producer_constraint_severity_holdout import (
    SeverityHoldoutCase,
    build_initial_messages,
    build_repair_messages,
    load_manifest as load_source_manifest,
)
from services.config import get_settings
from services.llm import DeepSeekLLM
from tools.rag.producer_constraint_guard import (
    check_plan_constraints,
    extract_explicit_constraints,
    split_constraint_violations,
)


CASES_PATH = Path("data/miaoxiang/producer_constraint_fault_injection.v1.json")
DEFAULT_OUTPUT_ROOT = Path("data/output/evaluations")
TEMPERATURE = 0.55
MAX_TOKENS = 1000


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FaultCase(_StrictModel):
    case_id: str = Field(min_length=1)
    source_case_id: str = Field(min_length=1)
    fault_family: Literal[
        "wrong_vocal_gender",
        "forbidden_rap_mc_slogans",
        "vocal_material_in_instrumental",
        "hard_fixed_bpm",
    ]
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_injected_hard_codes: list[str] = Field(min_length=1)
    base_plan: ProductionPlan
    injected_plan: ProductionPlan


class FaultManifest(_StrictModel):
    schema_version: str = Field(min_length=1)
    kind: Literal["producer_constraint_fault_injection"]
    frozen_at: str = Field(min_length=1)
    source_manifest: str = Field(min_length=1)
    source_experiment: str = Field(min_length=1)
    rubric: str = Field(min_length=1)
    max_constraint_repairs: Literal[1]
    notes: list[str] = Field(min_length=1)
    cases: list[FaultCase] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_fixed_set(self) -> "FaultManifest":
        case_ids = [case.case_id for case in self.cases]
        source_ids = [case.source_case_id for case in self.cases]
        families = [case.fault_family for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("fault_case_ids_must_be_unique")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("fault_source_case_ids_must_be_unique")
        if len(families) != len(set(families)):
            raise ValueError("fault_families_must_be_unique")
        return self


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = CASES_PATH) -> FaultManifest:
    return FaultManifest.model_validate(_read_json(path))


def resolve_cases(
    manifest: FaultManifest,
) -> list[tuple[SeverityHoldoutCase, FaultCase]]:
    source = load_source_manifest(Path(manifest.source_manifest))
    by_id = {case.case_id: case for case in source.cases}
    missing = [case.source_case_id for case in manifest.cases if case.source_case_id not in by_id]
    if missing:
        raise ValueError(f"fault_source_cases_missing:{missing}")
    return [(by_id[row.source_case_id], row) for row in manifest.cases]


def validate_fault_contracts(
    manifest: FaultManifest,
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for source_case, fault in resolve_cases(manifest):
        constraints = extract_explicit_constraints(source_case.user_request_text)
        base_all = check_plan_constraints(
            fault.base_plan,
            user_text=source_case.user_request_text,
            constraints=constraints,
        )
        base_hard, base_advisory = split_constraint_violations(base_all)
        if base_hard:
            raise ValueError(f"fault_base_has_hard_violation:{fault.case_id}")
        injected_all = check_plan_constraints(
            fault.injected_plan,
            user_text=source_case.user_request_text,
            constraints=constraints,
        )
        injected_hard, injected_advisory = split_constraint_violations(injected_all)
        codes = [row.code for row in injected_hard]
        if codes != fault.expected_injected_hard_codes:
            raise ValueError(
                f"fault_code_mismatch:{fault.case_id}:{codes}:"
                f"{fault.expected_injected_hard_codes}"
            )
        contracts.append(
            {
                "case_id": fault.case_id,
                "source_case_id": source_case.case_id,
                "fault_family": fault.fault_family,
                "base_hard_violation_codes": [row.code for row in base_hard],
                "base_advisory_codes": [row.code for row in base_advisory],
                "injected_hard_violation_codes": codes,
                "injected_advisory_codes": [row.code for row in injected_advisory],
                "planned_repair_calls": 1,
            }
        )
    return contracts


def _call_repair(
    llm: DeepSeekLLM,
    messages: list[dict[str, str]],
) -> ProductionPlan:
    payload = llm.chat_json(
        messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        operation="producer_constraint_fault_injection_repair",
        payload_validator=lambda value: pydantic_validation_feedback(
            ProductionPlan, value
        ),
    )
    return ProductionPlan.model_validate(payload)


def blind_mapping(case_id: str) -> dict[str, str]:
    digest = hashlib.sha256(f"fault:{case_id}".encode("utf-8")).digest()[0]
    if digest % 2:
        return {"X": "repaired", "Y": "base"}
    return {"X": "base", "Y": "repaired"}


def build_blind_card(
    source_case: SeverityHoldoutCase,
    fault: FaultCase,
    repaired_plan: ProductionPlan,
) -> tuple[dict[str, Any], dict[str, str]]:
    mapping = blind_mapping(fault.case_id)
    plans = {"base": fault.base_plan, "repaired": repaired_plan}
    return (
        {
            "case_id": fault.case_id,
            "song_request": source_case.user_request_text,
            "song_requirement": source_case.requirement.model_dump(mode="json"),
            "lyrics": source_case.lyric_draft.model_dump(mode="json"),
            "X": plans[mapping["X"]].model_dump(mode="json"),
            "Y": plans[mapping["Y"]].model_dump(mode="json"),
        },
        mapping,
    )


def render_blind_review(cards: list[dict[str, Any]]) -> str:
    lines = [
        "# Producer 制作方案匿名复核",
        "",
        "只评价制作方案文本，不评价音频，不猜测X/Y身份。",
        "明确硬要求违反是独立字段；方案违反用户写出的必须、禁止或替代要求时标记为是。",
        "即使两份方案都满足硬要求，也要比较是否出现不必要改动、信息丢失、过度约束或无依据扩写。",
        "",
    ]
    for index, card in enumerate(cards, start=1):
        requirement = card["song_requirement"]
        lyric = card["lyrics"]
        lines.extend(
            [
                f"## {index}. {card['case_id']}",
                "",
                f"- 用户原始请求：{card['song_request']}",
                f"- 主题：{requirement['theme']}",
                f"- 曲风：{requirement['genre']}",
                f"- 人声：{requirement['vocal_type']}",
                f"- 标题：{lyric['title']}",
                "",
            ]
        )
        for label in ("X", "Y"):
            plan = card[label]
            lines.extend(
                [
                    f"### {label}",
                    "",
                    f"- style_prompt：{plan['style_prompt']}",
                    f"- tags：{'、'.join(plan['tags'])}",
                    f"- vocal_direction：{plan['vocal_direction']}",
                    f"- arrangement_note：{plan['arrangement_note']}",
                    "",
                ]
            )
        lines.extend(
            [
                "评审记录：",
                "",
                "- X：契合度__/5；可执行性__/5；一致性__/5；过度约束 是/否；无依据扩写 是/否；明确硬要求违反 是/否",
                "- Y：契合度__/5；可执行性__/5；一致性__/5；过度约束 是/否；无依据扩写 是/否；明确硬要求违反 是/否",
                "- 偏好：X / Y / 平局 / 两者均不合格",
                "- 备注：",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def run_experiment(
    manifest: FaultManifest,
    *,
    manifest_path: Path,
    output_root: Path,
) -> Path:
    contracts = validate_fault_contracts(manifest)
    pairs = resolve_cases(manifest)
    experiment_dir = output_root / f"producer-constraint-fault-v1-{uuid4().hex}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    settings = get_settings()
    _write_json(experiment_dir / "fault_contracts.json", contracts)
    _write_json(
        experiment_dir / "experiment_config.json",
        {
            "schema_version": "1.0",
            "case_manifest": str(manifest_path),
            "case_manifest_sha256": _file_hash(manifest_path),
            "source_experiment": manifest.source_experiment,
            "case_count": len(pairs),
            "initial_generation_calls": 0,
            "planned_repair_calls": len(pairs),
            "max_repair_calls_per_case": manifest.max_constraint_repairs,
            "evidence_scope": "conditional_repair_capability_not_natural_failure_rate",
            "audio_generated": False,
            "runtime_agent_connected": False,
            "runtime_injection_changed": False,
            "llm": {
                "provider": "DeepSeek",
                "model": settings.deepseek_model,
                "max_retries": settings.llm_max_retries,
                "json_thinking_mode": (
                    "disabled" if settings.llm_json_disable_thinking else "default"
                ),
            },
        },
    )

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    keys: list[dict[str, str]] = []
    llm = DeepSeekLLM(settings)
    try:
        for source_case, fault in pairs:
            print(f"运行受控故障：{fault.case_id}", flush=True)
            case_dir = experiment_dir / "cases" / fault.case_id
            case_dir.mkdir(parents=True, exist_ok=False)
            try:
                constraints = extract_explicit_constraints(
                    source_case.user_request_text
                )
                injected_all = check_plan_constraints(
                    fault.injected_plan,
                    user_text=source_case.user_request_text,
                    constraints=constraints,
                )
                injected_hard, injected_advisory = split_constraint_violations(
                    injected_all
                )
                before = len(llm.telemetry_snapshot())
                repaired = _call_repair(
                    llm,
                    build_repair_messages(
                        build_initial_messages(source_case),
                        fault.injected_plan,
                        injected_hard,
                    ),
                )
                final_all = check_plan_constraints(
                    repaired,
                    user_text=source_case.user_request_text,
                    constraints=constraints,
                )
                final_hard, final_advisory = split_constraint_violations(final_all)
                records = llm.telemetry_snapshot()[before:]
                report = {
                    "case_id": fault.case_id,
                    "source_case_id": source_case.case_id,
                    "fault_family": fault.fault_family,
                    "source_artifact_sha256": fault.source_artifact_sha256,
                    "base_plan": fault.base_plan.model_dump(mode="json"),
                    "injected_plan": fault.injected_plan.model_dump(mode="json"),
                    "injected_hard_violations": [
                        row.model_dump(mode="json") for row in injected_hard
                    ],
                    "injected_advisories": [
                        row.model_dump(mode="json") for row in injected_advisory
                    ],
                    "repaired_plan": repaired.model_dump(mode="json"),
                    "final_hard_violations": [
                        row.model_dump(mode="json") for row in final_hard
                    ],
                    "final_advisories": [
                        row.model_dump(mode="json") for row in final_advisory
                    ],
                    "repair_succeeded": not final_hard,
                    "repaired_equals_base": (
                        repaired.model_dump(mode="json")
                        == fault.base_plan.model_dump(mode="json")
                    ),
                    "telemetry": DeepSeekLLM.summarize_call_records(records),
                }
                _write_json(case_dir / "result.json", report)
                completed.append(report)
                card, mapping = build_blind_card(source_case, fault, repaired)
                cards.append(card)
                keys.append({"case_id": fault.case_id, **mapping})
            except Exception as error:
                failure = {
                    "case_id": fault.case_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                _write_json(case_dir / "failure.json", failure)
                print(f"受控故障失败：{fault.case_id}", flush=True)
    finally:
        telemetry = DeepSeekLLM.summarize_call_records(llm.telemetry_snapshot())
        llm.close()

    _write_json(experiment_dir / "blind_review_cards.json", cards)
    _write_json(experiment_dir / "blind_review_key.json", keys)
    (experiment_dir / "blind_review.md").write_text(
        render_blind_review(cards), encoding="utf-8"
    )
    _write_json(
        experiment_dir / "summary.json",
        {
            "completed_case_count": len(completed),
            "failed_case_count": len(failures),
            "repair_succeeded_count": sum(
                row["repair_succeeded"] for row in completed
            ),
            "remaining_hard_violation_case_count": sum(
                bool(row["final_hard_violations"]) for row in completed
            ),
            "repaired_equals_base_count": sum(
                row["repaired_equals_base"] for row in completed
            ),
            "completed": completed,
            "failures": failures,
            "llm_telemetry": telemetry,
            "blind_review_status": "pending",
            "evidence_scope": "conditional_repair_capability_not_natural_failure_rate",
            "runtime_agent_connected": False,
            "runtime_injection_changed": False,
        },
    )
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行Producer显式制作约束受控故障修复实验。"
    )
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="确认仅调用4次受控故障修复；不生成首稿或音频",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.cases)
    contracts = validate_fault_contracts(manifest)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "case_count": len(manifest.cases),
                    "initial_generation_calls": 0,
                    "planned_repair_calls": len(manifest.cases),
                    "audio_generated": False,
                    "runtime_agent_connected": False,
                    "evidence_scope": "conditional_repair_capability_not_natural_failure_rate",
                    "contracts": contracts,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    output = run_experiment(
        manifest,
        manifest_path=args.cases,
        output_root=args.out,
    )
    print(f"实验输出：{output}", flush=True)


if __name__ == "__main__":
    main()
