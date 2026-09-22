"""Run an isolated gated-vs-guarded Producer constraint experiment."""

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
    ExplicitProductionConstraints,
    check_plan_constraints,
    extract_explicit_constraints,
)
from tools.rag.style_retrieval_policy import decide_style_retrieval


CASES_PATH = Path("data/miaoxiang/producer_constraint_guard_validation.v1.json")
DEFAULT_OUTPUT_ROOT = Path("data/output/evaluations")
VARIANTS = ("gated", "guarded")
TEMPERATURE = 0.55
MAX_TOKENS = 1000


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GuardCaseReference(_StrictModel):
    case_id: str = Field(min_length=1)
    expected_must_include: list[str]
    expected_must_avoid: list[str]
    expected_do_not_infer: list[str]


class GuardManifest(_StrictModel):
    schema_version: str = Field(min_length=1)
    kind: Literal["producer_constraint_guard_validation"]
    frozen_at: str = Field(min_length=1)
    source_manifest: str = Field(min_length=1)
    rubric: str = Field(min_length=1)
    variants: list[Literal["gated", "guarded"]] = Field(min_length=2, max_length=2)
    max_constraint_repairs: Literal[1]
    notes: list[str] = Field(min_length=1)
    cases: list[GuardCaseReference] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_fixed_set(self) -> "GuardManifest":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("guard_case_ids_must_be_unique")
        if self.variants != list(VARIANTS):
            raise ValueError("guard_variants_must_be_gated_then_guarded")
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


def load_manifest(path: Path = CASES_PATH) -> GuardManifest:
    return GuardManifest.model_validate(_read_json(path))


def resolve_cases(
    manifest: GuardManifest,
    *,
    manifest_path: Path = CASES_PATH,
) -> list[tuple[RubricValidationCase, GuardCaseReference]]:
    source_path = Path(manifest.source_manifest)
    if not source_path.is_absolute():
        source_path = source_path.resolve()
    source = load_source_manifest(source_path)
    by_id = {case.case_id: case for case in source.cases}
    missing = [row.case_id for row in manifest.cases if row.case_id not in by_id]
    if missing:
        raise ValueError(f"guard_source_cases_missing:{missing}")
    return [(by_id[row.case_id], row) for row in manifest.cases]


def _concepts(rules: list[Any]) -> list[str]:
    return [rule.concept for rule in rules]


def validate_guard_contracts(
    manifest: GuardManifest,
    *,
    manifest_path: Path = CASES_PATH,
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for case, expected in resolve_cases(manifest, manifest_path=manifest_path):
        decision = decide_style_retrieval(case.user_request_text)
        if decision.action != "single_style":
            raise ValueError(f"guard_case_not_single_style:{case.case_id}")
        constraints = extract_explicit_constraints(case.user_request_text)
        actual = {
            "must_include": _concepts(constraints.must_include),
            "must_avoid": _concepts(constraints.must_avoid),
            "do_not_infer": constraints.do_not_infer,
        }
        expected_payload = {
            "must_include": expected.expected_must_include,
            "must_avoid": expected.expected_must_avoid,
            "do_not_infer": expected.expected_do_not_infer,
        }
        if actual != expected_payload:
            raise ValueError(
                f"guard_constraint_mismatch:{case.case_id}:{actual}:{expected_payload}"
            )
        context = build_gate_context(decision)
        gated = build_variant_messages(
            case, variant="gated", decision=decision, context=context
        )
        guarded = build_guarded_messages(
            case,
            decision=decision,
            context=context,
            constraints=constraints,
        )
        if gated == guarded:
            raise ValueError(f"guarded_prompt_unexpectedly_equal:{case.case_id}")
        contracts.append(
            {
                "case_id": case.case_id,
                "candidate_ids": decision.candidate_ids,
                **actual,
                "prompts_identical": False,
            }
        )
    return contracts


def build_guarded_messages(
    case: RubricValidationCase,
    *,
    decision: Any,
    context: list[dict[str, Any]],
    constraints: ExplicitProductionConstraints,
) -> list[dict[str, str]]:
    messages = build_variant_messages(
        case,
        variant="gated",
        decision=decision,
        context=context,
    )
    payload = constraints.model_dump(mode="json")
    messages[1]["content"] += (
        "\n\n显式制作约束（优先级高于风格参考卡，不得被卡片默认值覆盖）：\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n必须满足 must_include；不得出现 must_avoid；do_not_infer 表示用户未授权的推断。"
        "如果风格参考与这些约束冲突，必须忽略冲突部分。"
    )
    return messages


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


def _repair_messages(
    messages: list[dict[str, str]],
    plan: ProductionPlan,
    violations: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        *messages,
        {
            "role": "assistant",
            "content": json.dumps(plan.model_dump(mode="json"), ensure_ascii=False),
        },
        {
            "role": "user",
            "content": (
                "上一个制作方案违反了显式约束：\n"
                + json.dumps(violations, ensure_ascii=False, indent=2)
                + "\n请只修正这些冲突，保留原主题和主风格，并重新输出完整JSON对象。"
            ),
        },
    ]


def variant_order(case_id: str) -> tuple[str, str]:
    if hashlib.sha256(case_id.encode("utf-8")).digest()[0] % 2:
        return "guarded", "gated"
    return "gated", "guarded"


def blind_mapping(case_id: str) -> dict[str, str]:
    order = variant_order(f"blind:{case_id}")
    return {"X": order[0], "Y": order[1]}


def build_blind_card(
    case: RubricValidationCase,
    plans: dict[str, ProductionPlan],
) -> tuple[dict[str, Any], dict[str, str]]:
    mapping = blind_mapping(case.case_id)
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
        "# Producer 显式制作约束匿名复核",
        "",
        "只评价制作方案文本，不评价音频。X/Y身份已隐藏。",
        "风格原生细节不自动算扩写；非必要细节被写成不可替换的强制实现才算过度约束。",
        "明确硬要求违反是独立字段：方案直接违反用户写出的必须、禁止、唯一或替代要求时标记为是。",
        "如果两份方案都违反同一核心要求，可以判两者均不合格；如仍排序，必须说明胜者也不可直接使用。",
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


def reanalyze_existing_run(experiment_dir: Path) -> Path:
    """Recompute violations and rebuild a blind packet without new LLM calls.

    The corrected packet uses each variant's initial plan. This is appropriate
    when a previous checker falsely triggered a repair on a negated forbidden
    term; the original files remain untouched for auditability.
    """

    summary = _read_json(experiment_dir / "summary.json")
    cards: list[dict[str, Any]] = []
    keys: list[dict[str, str]] = []
    cases: list[dict[str, Any]] = []
    for completed in summary["completed"]:
        case_id = completed["case_id"]
        case_dir = experiment_dir / "cases" / case_id
        case = RubricValidationCase.model_validate(_read_json(case_dir / "input.json"))
        constraints = extract_explicit_constraints(case.user_request_text)
        plans: dict[str, ProductionPlan] = {}
        variants: dict[str, Any] = {}
        for variant in VARIANTS:
            stored = _read_json(case_dir / f"{variant}.json")
            initial = ProductionPlan.model_validate(stored["initial_plan"])
            violations = check_plan_constraints(
                initial,
                user_text=case.user_request_text,
                constraints=constraints,
            )
            plans[variant] = initial
            variants[variant] = {
                "selected_plan_source": "initial_plan",
                "violations": [row.model_dump(mode="json") for row in violations],
                "original_repair_attempted": bool(stored["repair_attempted"]),
                "original_final_plan_excluded": bool(stored["repair_attempted"]),
            }
        card, mapping = build_blind_card(case, plans)
        cards.append(card)
        keys.append({"case_id": case_id, **mapping})
        cases.append({"case_id": case_id, "variants": variants})

    corrected_cards = experiment_dir / "blind_review_cards_corrected.json"
    corrected_key = experiment_dir / "blind_review_key_corrected.json"
    corrected_review = experiment_dir / "blind_review_corrected.md"
    corrected_summary = experiment_dir / "reanalysis_corrected.json"
    _write_json(corrected_cards, cards)
    _write_json(corrected_key, keys)
    corrected_review.write_text(render_blind_review(cards), encoding="utf-8")
    _write_json(
        corrected_summary,
        {
            "schema_version": "1.0",
            "source_experiment": str(experiment_dir),
            "reason": "negated_forbidden_terms_were_false_positives",
            "new_llm_calls": 0,
            "evaluation_plan_source": "initial_plan_for_both_variants",
            "actual_source_run_calls": summary["llm_telemetry"]["successful_calls"],
            "evaluation_main_calls": len(cases) * 2,
            "excluded_repair_calls": sum(
                int(row["variants"]["guarded"]["original_final_plan_excluded"])
                for row in cases
            ),
            "cases": cases,
            "blind_review_status": "pending",
            "method_decision": None,
        },
    )
    return corrected_review


def run_experiment(
    manifest: GuardManifest,
    *,
    manifest_path: Path,
    output_root: Path,
) -> Path:
    contracts = validate_guard_contracts(manifest, manifest_path=manifest_path)
    cases = resolve_cases(manifest, manifest_path=manifest_path)
    experiment_dir = output_root / f"producer-constraint-guard-v1-{uuid4().hex}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    settings = get_settings()
    _write_json(experiment_dir / "guard_contracts.json", contracts)
    _write_json(
        experiment_dir / "experiment_config.json",
        {
            "schema_version": "1.0",
            "case_manifest": str(manifest_path),
            "case_manifest_sha256": _file_hash(manifest_path),
            "source_manifest": manifest.source_manifest,
            "rubric": manifest.rubric,
            "variants": list(VARIANTS),
            "case_count": len(cases),
            "planned_main_calls": len(cases) * 2,
            "max_repair_calls": len(cases) * manifest.max_constraint_repairs,
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

    blind_cards: list[dict[str, Any]] = []
    blind_keys: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    llm = DeepSeekLLM(settings)
    try:
        for case, _ in cases:
            print(f"运行显式约束案例：{case.case_id}", flush=True)
            case_dir = experiment_dir / "cases" / case.case_id
            case_dir.mkdir(parents=True, exist_ok=False)
            try:
                decision = decide_style_retrieval(case.user_request_text)
                context = build_gate_context(decision)
                constraints = extract_explicit_constraints(case.user_request_text)
                _write_json(case_dir / "input.json", case.model_dump(mode="json"))
                _write_json(
                    case_dir / "gate_decision.json", decision.model_dump(mode="json")
                )
                _write_json(
                    case_dir / "constraints.json", constraints.model_dump(mode="json")
                )

                plans: dict[str, ProductionPlan] = {}
                variant_reports: dict[str, Any] = {}
                for variant in variant_order(case.case_id):
                    before = len(llm.telemetry_snapshot())
                    if variant == "gated":
                        messages = build_variant_messages(
                            case,
                            variant="gated",
                            decision=decision,
                            context=context,
                        )
                    else:
                        messages = build_guarded_messages(
                            case,
                            decision=decision,
                            context=context,
                            constraints=constraints,
                        )
                    initial = _call_plan(
                        llm,
                        messages,
                        operation=f"producer_constraint_guard_{variant}_initial",
                    )
                    initial_violations = check_plan_constraints(
                        initial,
                        user_text=case.user_request_text,
                        constraints=constraints,
                    )
                    final = initial
                    repair_attempted = False
                    if variant == "guarded" and initial_violations:
                        repair_attempted = True
                        final = _call_plan(
                            llm,
                            _repair_messages(
                                messages,
                                initial,
                                [
                                    violation.model_dump(mode="json")
                                    for violation in initial_violations
                                ],
                            ),
                            operation="producer_constraint_guard_repair",
                        )
                    final_violations = check_plan_constraints(
                        final,
                        user_text=case.user_request_text,
                        constraints=constraints,
                    )
                    records = llm.telemetry_snapshot()[before:]
                    plans[variant] = final
                    variant_reports[variant] = {
                        "initial_plan": initial.model_dump(mode="json"),
                        "initial_violations": [
                            row.model_dump(mode="json") for row in initial_violations
                        ],
                        "repair_attempted": repair_attempted,
                        "final_plan": final.model_dump(mode="json"),
                        "final_violations": [
                            row.model_dump(mode="json") for row in final_violations
                        ],
                        "telemetry": DeepSeekLLM.summarize_call_records(records),
                    }
                    _write_json(case_dir / f"{variant}.json", variant_reports[variant])

                card, mapping = build_blind_card(case, plans)
                blind_cards.append(card)
                blind_keys.append({"case_id": case.case_id, **mapping})
                completed.append(
                    {
                        "case_id": case.case_id,
                        "candidate_ids": decision.candidate_ids,
                        "call_order": list(variant_order(case.case_id)),
                        "variants": variant_reports,
                    }
                )
            except Exception as error:
                failure = {
                    "case_id": case.case_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                _write_json(case_dir / "failure.json", failure)
                print(f"显式约束案例失败：{case.case_id}", flush=True)
    finally:
        telemetry = DeepSeekLLM.summarize_call_records(llm.telemetry_snapshot())
        llm.close()

    _write_json(experiment_dir / "blind_review_cards.json", blind_cards)
    _write_json(experiment_dir / "blind_review_key.json", blind_keys)
    (experiment_dir / "blind_review.md").write_text(
        render_blind_review(blind_cards), encoding="utf-8"
    )
    _write_json(
        experiment_dir / "summary.json",
        {
            "completed_case_count": len(completed),
            "failed_case_count": len(failures),
            "completed": completed,
            "failures": failures,
            "llm_telemetry": telemetry,
            "multi_model_review_status": "pending",
            "method_decision": None,
        },
    )
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行Producer显式制作约束保护的三案例影子实验。"
    )
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="确认调用6次主生成和最多3次约束修复；不生成音频",
    )
    parser.add_argument(
        "--reanalyze-existing",
        type=Path,
        help="使用修正后的检查器重算既有运行；不调用DeepSeek",
    )
    args = parser.parse_args()
    if args.reanalyze_existing is not None:
        if args.execute:
            raise ValueError("execute_and_reanalyze_are_mutually_exclusive")
        output = reanalyze_existing_run(args.reanalyze_existing)
        print(f"纠正后盲评文件：{output}", flush=True)
        return
    manifest = load_manifest(args.cases)
    contracts = validate_guard_contracts(manifest, manifest_path=args.cases)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "case_count": len(manifest.cases),
                    "planned_main_calls": len(manifest.cases) * 2,
                    "max_repair_calls": len(manifest.cases),
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
