"""Run an isolated hard-vs-advisory Producer constraint holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.schemas import LyricDraft, ProductionPlan, SongRequirement, pydantic_validation_feedback
from scripts.evaluate_producer_rag_gated_ab import build_gate_context, build_variant_messages
from services.config import get_settings
from services.llm import DeepSeekLLM
from tools.rag.producer_constraint_guard import (
    ConstraintViolation,
    ExplicitProductionConstraints,
    check_plan_constraints,
    extract_explicit_constraints,
    split_constraint_violations,
)
from tools.rag.style_retrieval_policy import decide_style_retrieval


CASES_PATH = Path("data/miaoxiang/producer_constraint_severity_holdout.v1.json")
DEFAULT_OUTPUT_ROOT = Path("data/output/evaluations")
TEMPERATURE = 0.55
MAX_TOKENS = 1000


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SeverityHoldoutCase(_StrictModel):
    case_id: str = Field(min_length=1)
    risk_focus: Literal[
        "explicit_male_vocal",
        "explicit_no_rap_mc_slogans",
        "explicit_instrumental_only",
        "explicit_instrument_substitution",
        "explicit_flexible_bpm",
        "explicit_female_vocal",
        "advisory_unspecified_gender",
        "advisory_secondary_genre",
    ]
    user_request_text: str = Field(min_length=5)
    expected_action: Literal["single_style"]
    expected_candidate_ids: list[str] = Field(min_length=1, max_length=1)
    expected_must_include: list[str]
    expected_must_avoid: list[str]
    expected_do_not_infer: list[str]
    requirement: SongRequirement
    lyric_draft: LyricDraft


class SeverityHoldoutManifest(_StrictModel):
    schema_version: str = Field(min_length=1)
    kind: Literal["producer_constraint_severity_holdout"]
    frozen_at: str = Field(min_length=1)
    rubric: str = Field(min_length=1)
    max_constraint_repairs: Literal[1]
    notes: list[str] = Field(min_length=1)
    cases: list[SeverityHoldoutCase] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def validate_fixed_set(self) -> "SeverityHoldoutManifest":
        case_ids = [case.case_id for case in self.cases]
        risk_focuses = [case.risk_focus for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("severity_holdout_case_ids_must_be_unique")
        if len(risk_focuses) != len(set(risk_focuses)):
            raise ValueError("severity_holdout_risk_focuses_must_be_unique")
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


def _concepts(rules: list[Any]) -> list[str]:
    return [rule.concept for rule in rules]


def load_manifest(path: Path = CASES_PATH) -> SeverityHoldoutManifest:
    return SeverityHoldoutManifest.model_validate(_read_json(path))


def has_explicit_hard_contract(
    constraints: ExplicitProductionConstraints,
) -> bool:
    return bool(
        constraints.must_include
        or constraints.must_avoid
        or "exact_bpm_as_hard_requirement" in constraints.do_not_infer
    )


def should_repair(violations: list[ConstraintViolation]) -> bool:
    hard, _ = split_constraint_violations(violations)
    return bool(hard)


def validate_holdout_contracts(
    manifest: SeverityHoldoutManifest,
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for case in manifest.cases:
        decision = decide_style_retrieval(case.user_request_text)
        if decision.action != case.expected_action:
            raise ValueError(
                f"severity_gate_action_mismatch:{case.case_id}:{decision.action}"
            )
        if decision.candidate_ids != case.expected_candidate_ids:
            raise ValueError(
                f"severity_gate_candidates_mismatch:{case.case_id}:"
                f"{decision.candidate_ids}"
            )
        constraints = extract_explicit_constraints(case.user_request_text)
        actual = {
            "must_include": _concepts(constraints.must_include),
            "must_avoid": _concepts(constraints.must_avoid),
            "do_not_infer": constraints.do_not_infer,
        }
        expected = {
            "must_include": case.expected_must_include,
            "must_avoid": case.expected_must_avoid,
            "do_not_infer": case.expected_do_not_infer,
        }
        if actual != expected:
            raise ValueError(
                f"severity_constraint_mismatch:{case.case_id}:{actual}:{expected}"
            )
        contracts.append(
            {
                "case_id": case.case_id,
                "risk_focus": case.risk_focus,
                "candidate_ids": decision.candidate_ids,
                "has_explicit_hard_contract": has_explicit_hard_contract(
                    constraints
                ),
                **actual,
            }
        )
    return contracts


def _call_plan(
    llm: DeepSeekLLM,
    messages: list[dict[str, str]],
    *,
    operation: str,
) -> ProductionPlan:
    payload = llm.chat_json(
        messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        operation=operation,
        payload_validator=lambda value: pydantic_validation_feedback(
            ProductionPlan, value
        ),
    )
    return ProductionPlan.model_validate(payload)


def build_initial_messages(
    case: SeverityHoldoutCase,
) -> list[dict[str, str]]:
    decision = decide_style_retrieval(case.user_request_text)
    return build_variant_messages(
        case,
        variant="gated",
        decision=decision,
        context=build_gate_context(decision),
    )


def build_repair_messages(
    initial_messages: list[dict[str, str]],
    initial_plan: ProductionPlan,
    hard_violations: list[ConstraintViolation],
) -> list[dict[str, str]]:
    if not hard_violations or any(
        row.severity != "hard" for row in hard_violations
    ):
        raise ValueError("repair_requires_hard_violations_only")
    return [
        *initial_messages,
        {
            "role": "assistant",
            "content": json.dumps(
                initial_plan.model_dump(mode="json"), ensure_ascii=False
            ),
        },
        {
            "role": "user",
            "content": (
                "确定性后检发现上一个制作方案违反了用户明确写出的硬约束：\n"
                + json.dumps(
                    [row.model_dump(mode="json") for row in hard_violations],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n只修正列出的硬冲突；不要处理advisory，不新增曲风、乐器、人声性别或叙事方向；"
                "其余字段尽量保持不变。重新输出完整JSON对象。"
            ),
        },
    ]


def blind_mapping(case_id: str) -> dict[str, str]:
    digest = hashlib.sha256(f"severity:{case_id}".encode("utf-8")).digest()[0]
    if digest % 2:
        return {"X": "repaired", "Y": "initial"}
    return {"X": "initial", "Y": "repaired"}


def build_blind_card(
    case: SeverityHoldoutCase,
    *,
    initial_plan: ProductionPlan,
    repaired_plan: ProductionPlan,
) -> tuple[dict[str, Any], dict[str, str]]:
    mapping = blind_mapping(case.case_id)
    plans = {"initial": initial_plan, "repaired": repaired_plan}
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
        "明确硬要求违反是独立字段；仅因用户没有指定而产生的性别或邻近曲风选择不自动算硬违反。",
        "即使一份方案满足硬要求，但明显伤害主题、风格或自然度，也应如实扣分。",
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
    manifest: SeverityHoldoutManifest,
    *,
    manifest_path: Path,
    output_root: Path,
) -> Path:
    contracts = validate_holdout_contracts(manifest)
    experiment_dir = output_root / f"producer-constraint-severity-v1-{uuid4().hex}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    settings = get_settings()
    _write_json(experiment_dir / "severity_contracts.json", contracts)
    _write_json(
        experiment_dir / "experiment_config.json",
        {
            "schema_version": "1.0",
            "case_manifest": str(manifest_path),
            "case_manifest_sha256": _file_hash(manifest_path),
            "case_count": len(manifest.cases),
            "planned_initial_calls": len(manifest.cases),
            "maximum_repair_calls": sum(
                int(row["has_explicit_hard_contract"]) for row in contracts
            ),
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

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    keys: list[dict[str, str]] = []
    llm = DeepSeekLLM(settings)
    try:
        for case in manifest.cases:
            print(f"运行严重度holdout：{case.case_id}", flush=True)
            case_dir = experiment_dir / "cases" / case.case_id
            case_dir.mkdir(parents=True, exist_ok=False)
            try:
                decision = decide_style_retrieval(case.user_request_text)
                constraints = extract_explicit_constraints(case.user_request_text)
                initial_messages = build_initial_messages(case)
                before = len(llm.telemetry_snapshot())
                initial = _call_plan(
                    llm,
                    initial_messages,
                    operation="producer_constraint_severity_initial",
                )
                initial_violations = check_plan_constraints(
                    initial,
                    user_text=case.user_request_text,
                    constraints=constraints,
                )
                initial_hard, initial_advisory = split_constraint_violations(
                    initial_violations
                )
                final = initial
                repair_attempted = bool(initial_hard)
                if repair_attempted:
                    final = _call_plan(
                        llm,
                        build_repair_messages(
                            initial_messages, initial, initial_hard
                        ),
                        operation="producer_constraint_severity_repair",
                    )
                final_violations = check_plan_constraints(
                    final,
                    user_text=case.user_request_text,
                    constraints=constraints,
                )
                final_hard, final_advisory = split_constraint_violations(
                    final_violations
                )
                records = llm.telemetry_snapshot()[before:]
                unchanged = (
                    initial.model_dump(mode="json")
                    == final.model_dump(mode="json")
                )
                if not repair_attempted and not unchanged:
                    raise AssertionError("non_repaired_case_changed")
                report = {
                    "case_id": case.case_id,
                    "risk_focus": case.risk_focus,
                    "candidate_ids": decision.candidate_ids,
                    "constraints": constraints.model_dump(mode="json"),
                    "initial_plan": initial.model_dump(mode="json"),
                    "initial_hard_violations": [
                        row.model_dump(mode="json") for row in initial_hard
                    ],
                    "initial_advisories": [
                        row.model_dump(mode="json") for row in initial_advisory
                    ],
                    "action": "repair_once" if repair_attempted else "pass_through",
                    "repair_attempted": repair_attempted,
                    "final_plan": final.model_dump(mode="json"),
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
                        case,
                        initial_plan=initial,
                        repaired_plan=final,
                    )
                    cards.append(card)
                    keys.append({"case_id": case.case_id, **mapping})
            except Exception as error:
                failure = {
                    "case_id": case.case_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                _write_json(case_dir / "failure.json", failure)
                print(f"严重度holdout失败：{case.case_id}", flush=True)
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
            "advisory_only_repair_count": sum(
                row["repair_attempted"]
                and not row["initial_hard_violations"]
                and bool(row["initial_advisories"])
                for row in completed
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
        description="运行Producer硬约束与advisory分级holdout影子实验。"
    )
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="确认调用8次首稿和最多6次硬约束修复；不生成音频",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.cases)
    contracts = validate_holdout_contracts(manifest)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "case_count": len(manifest.cases),
                    "planned_initial_calls": len(manifest.cases),
                    "maximum_repair_calls": sum(
                        int(row["has_explicit_hard_contract"])
                        for row in contracts
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
