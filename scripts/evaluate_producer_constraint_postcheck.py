"""Evaluate post-generation constraint checking with repair only on violation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.schemas import ProductionPlan, pydantic_validation_feedback
from scripts.evaluate_producer_rag_gated_ab import (
    build_gate_context,
    build_variant_messages,
)
from scripts.evaluate_producer_rag_rubric import (
    RubricValidationCase,
    load_manifest as load_source_manifest,
)
from services.config import get_settings
from services.llm import DeepSeekLLM
from tools.rag.producer_constraint_guard import (
    ConstraintViolation,
    check_plan_constraints,
    extract_explicit_constraints,
    split_constraint_violations,
)
from tools.rag.style_retrieval_policy import decide_style_retrieval


CASES_PATH = Path("data/miaoxiang/producer_constraint_postcheck_validation.v1.json")
DEFAULT_OUTPUT_ROOT = Path("data/output/evaluations")
TEMPERATURE = 0.55
MAX_TOKENS = 1000


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PostcheckCase(_StrictModel):
    case_id: str = Field(min_length=1)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_initial_violation_codes: list[str]
    initial_plan: ProductionPlan


class PostcheckManifest(_StrictModel):
    schema_version: str = Field(min_length=1)
    kind: Literal["producer_constraint_postcheck_validation"]
    frozen_at: str = Field(min_length=1)
    source_manifest: str = Field(min_length=1)
    source_experiment: str = Field(min_length=1)
    rubric: str = Field(min_length=1)
    max_constraint_repairs: Literal[1]
    notes: list[str] = Field(min_length=1)
    cases: list[PostcheckCase] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_fixed_cases(self) -> "PostcheckManifest":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("postcheck_case_ids_must_be_unique")
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


def load_manifest(path: Path = CASES_PATH) -> PostcheckManifest:
    return PostcheckManifest.model_validate(_read_json(path))


def resolve_cases(
    manifest: PostcheckManifest,
) -> list[tuple[RubricValidationCase, PostcheckCase]]:
    source = load_source_manifest(Path(manifest.source_manifest))
    by_id = {case.case_id: case for case in source.cases}
    missing = [case.case_id for case in manifest.cases if case.case_id not in by_id]
    if missing:
        raise ValueError(f"postcheck_source_cases_missing:{missing}")
    return [(by_id[case.case_id], case) for case in manifest.cases]


def validate_postcheck_contracts(
    manifest: PostcheckManifest,
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for source_case, frozen in resolve_cases(manifest):
        constraints = extract_explicit_constraints(source_case.user_request_text)
        violations = check_plan_constraints(
            frozen.initial_plan,
            user_text=source_case.user_request_text,
            constraints=constraints,
        )
        hard, advisory = split_constraint_violations(violations)
        codes = [row.code for row in violations]
        if codes != frozen.expected_initial_violation_codes:
            raise ValueError(
                f"postcheck_violation_mismatch:{source_case.case_id}:"
                f"{codes}:{frozen.expected_initial_violation_codes}"
            )
        contracts.append(
            {
                "case_id": source_case.case_id,
                "initial_violation_codes": codes,
                "initial_hard_violation_codes": [row.code for row in hard],
                "initial_advisory_codes": [row.code for row in advisory],
                "action": "repair_once" if hard else "pass_through",
                "planned_llm_calls": int(bool(hard)),
            }
        )
    return contracts


def build_repair_messages(
    case: RubricValidationCase,
    initial_plan: ProductionPlan,
    violations: list[ConstraintViolation],
) -> list[dict[str, str]]:
    decision = decide_style_retrieval(case.user_request_text)
    context = build_gate_context(decision)
    messages = build_variant_messages(
        case,
        variant="gated",
        decision=decision,
        context=context,
    )
    return [
        *messages,
        {
            "role": "assistant",
            "content": json.dumps(
                initial_plan.model_dump(mode="json"), ensure_ascii=False
            ),
        },
        {
            "role": "user",
            "content": (
                "确定性后检发现上一个制作方案违反了用户的显式制作约束：\n"
                + json.dumps(
                    [row.model_dump(mode="json") for row in violations],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n只修正列出的冲突，不新增曲风、乐器、人声性别或叙事方向；"
                "其余内容尽量保持不变。重新输出完整JSON对象。"
            ),
        },
    ]


def _call_plan(
    llm: DeepSeekLLM,
    messages: list[dict[str, str]],
) -> ProductionPlan:
    payload = llm.chat_json(
        messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        operation="producer_constraint_postcheck_repair",
        payload_validator=lambda value: pydantic_validation_feedback(
            ProductionPlan, value
        ),
    )
    return ProductionPlan.model_validate(payload)


def blind_mapping(case_id: str) -> dict[str, str]:
    first = hashlib.sha256(f"postcheck:{case_id}".encode("utf-8")).digest()[0]
    if first % 2:
        return {"X": "postcheck", "Y": "initial"}
    return {"X": "initial", "Y": "postcheck"}


def build_blind_card(
    case: RubricValidationCase,
    *,
    initial_plan: ProductionPlan,
    final_plan: ProductionPlan,
) -> tuple[dict[str, Any], dict[str, str]]:
    mapping = blind_mapping(case.case_id)
    plans = {"initial": initial_plan, "postcheck": final_plan}
    return (
        {
            "case_id": case.case_id,
            "song_request": case.user_request_text,
            "song_requirement": case.requirement.model_dump(mode="json"),
            "lyrics": case.lyric_draft.model_dump(mode="json"),
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
        "如果一份方案修正硬冲突但明显伤害主题、风格或自然度，也应如实扣分。",
        "",
    ]
    for index, card in enumerate(cards, start=1):
        requirement = card["song_requirement"]
        lyrics = card["lyrics"]
        lines.extend(
            [
                f"## {index}. {card['case_id']}",
                "",
                f"- 用户原始请求：{card['song_request']}",
                f"- 主题：{requirement['theme']}",
                f"- 曲风：{requirement['genre']}",
                f"- 人声：{requirement['vocal_type']}",
                f"- 标题：{lyrics['title']}",
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
    manifest: PostcheckManifest,
    *,
    manifest_path: Path,
    output_root: Path,
) -> Path:
    contracts = validate_postcheck_contracts(manifest)
    cases = resolve_cases(manifest)
    experiment_dir = output_root / f"producer-constraint-postcheck-v1-{uuid4().hex}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    settings = get_settings()
    _write_json(experiment_dir / "postcheck_contracts.json", contracts)
    _write_json(
        experiment_dir / "experiment_config.json",
        {
            "schema_version": "1.0",
            "case_manifest": str(manifest_path),
            "case_manifest_sha256": _file_hash(manifest_path),
            "source_experiment": manifest.source_experiment,
            "case_count": len(cases),
            "initial_generation_calls": 0,
            "planned_repair_calls": sum(row["planned_llm_calls"] for row in contracts),
            "max_repair_calls_per_case": manifest.max_constraint_repairs,
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

    llm: DeepSeekLLM | None = None
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    keys: list[dict[str, str]] = []
    try:
        for source_case, frozen in cases:
            case_dir = experiment_dir / "cases" / source_case.case_id
            case_dir.mkdir(parents=True, exist_ok=False)
            try:
                constraints = extract_explicit_constraints(source_case.user_request_text)
                initial_violations = check_plan_constraints(
                    frozen.initial_plan,
                    user_text=source_case.user_request_text,
                    constraints=constraints,
                )
                initial_hard, initial_advisory = split_constraint_violations(
                    initial_violations
                )
                final_plan = frozen.initial_plan
                repair_attempted = bool(initial_hard)
                records: list[Any] = []
                if repair_attempted:
                    if llm is None:
                        llm = DeepSeekLLM(settings)
                    before = len(llm.telemetry_snapshot())
                    final_plan = _call_plan(
                        llm,
                        build_repair_messages(
                            source_case,
                            frozen.initial_plan,
                            initial_hard,
                        ),
                    )
                    records = llm.telemetry_snapshot()[before:]
                final_violations = check_plan_constraints(
                    final_plan,
                    user_text=source_case.user_request_text,
                    constraints=constraints,
                )
                final_hard, final_advisory = split_constraint_violations(
                    final_violations
                )
                unchanged = (
                    final_plan.model_dump(mode="json")
                    == frozen.initial_plan.model_dump(mode="json")
                )
                if not repair_attempted and not unchanged:
                    raise AssertionError("pass_through_plan_changed")
                report = {
                    "case_id": source_case.case_id,
                    "source_artifact_sha256": frozen.source_artifact_sha256,
                    "initial_plan": frozen.initial_plan.model_dump(mode="json"),
                    "initial_violations": [
                        row.model_dump(mode="json") for row in initial_violations
                    ],
                    "initial_hard_violations": [
                        row.model_dump(mode="json") for row in initial_hard
                    ],
                    "initial_advisories": [
                        row.model_dump(mode="json") for row in initial_advisory
                    ],
                    "action": "repair_once" if repair_attempted else "pass_through",
                    "repair_attempted": repair_attempted,
                    "final_plan": final_plan.model_dump(mode="json"),
                    "final_violations": [
                        row.model_dump(mode="json") for row in final_violations
                    ],
                    "final_hard_violations": [
                        row.model_dump(mode="json") for row in final_hard
                    ],
                    "final_advisories": [
                        row.model_dump(mode="json") for row in final_advisory
                    ],
                    "unchanged": unchanged,
                    "telemetry": DeepSeekLLM.summarize_call_records(records),
                }
                _write_json(case_dir / "result.json", report)
                completed.append(report)
                if repair_attempted:
                    card, mapping = build_blind_card(
                        source_case,
                        initial_plan=frozen.initial_plan,
                        final_plan=final_plan,
                    )
                    cards.append(card)
                    keys.append({"case_id": source_case.case_id, **mapping})
            except Exception as error:
                failure = {
                    "case_id": source_case.case_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                _write_json(case_dir / "failure.json", failure)
    finally:
        if llm is None:
            telemetry = DeepSeekLLM.summarize_call_records([])
        else:
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
            "pass_through_count": sum(
                row["action"] == "pass_through" for row in completed
            ),
            "repair_attempted_count": sum(row["repair_attempted"] for row in completed),
            "repair_succeeded_count": sum(
                row["repair_attempted"] and not row["final_hard_violations"]
                for row in completed
            ),
            "remaining_hard_violation_case_count": sum(
                bool(row["final_hard_violations"]) for row in completed
            ),
            "completed": completed,
            "failures": failures,
            "llm_telemetry": telemetry,
            "blind_review_status": "pending" if cards else "not_required",
            "runtime_agent_connected": False,
            "runtime_injection_changed": False,
        },
    )
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行Producer显式制作约束按需后检修复影子实验。"
    )
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="确认仅对后检违规案例调用一次DeepSeek修复；不生成首稿或音频",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.cases)
    contracts = validate_postcheck_contracts(manifest)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "case_count": len(manifest.cases),
                    "initial_generation_calls": 0,
                    "planned_repair_calls": sum(
                        row["planned_llm_calls"] for row in contracts
                    ),
                    "audio_generated": False,
                    "runtime_agent_connected": False,
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
