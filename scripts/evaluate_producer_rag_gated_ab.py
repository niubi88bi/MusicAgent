"""Run a four-case blind Producer comparison with the conservative style gate.

This is an isolated experiment.  It manually applies the shadow decision to
one variant, never changes the runtime Producer/LangGraph route, and never
invokes an audio backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.schemas import (
    LyricDraft,
    ProductionPlan,
    SongRequirement,
    pydantic_validation_feedback,
)
from scripts.evaluate_producer_rag_ab import _compact_card, build_producer_messages
from services.config import get_settings
from services.llm import DeepSeekLLM
from tools.rag.style_index import load_index_inputs, select_local_demo_cards
from tools.rag.style_retrieval_policy import (
    Action,
    StyleRetrievalDecision,
    decide_style_retrieval,
)


CASES_PATH = Path("data/miaoxiang/producer_rag_gated_ab_cases.v1.json")
DEFAULT_OUTPUT_ROOT = Path("data/output/evaluations")
TEMPERATURE = 0.55
MAX_TOKENS = 1000
VARIANTS = ("baseline", "gated")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GatedProducerCase(_StrictModel):
    case_id: str = Field(min_length=1)
    case_type: Literal[
        "clear_single_style",
        "named_composite",
        "emotion_only",
        "hard_vocal_constraint",
    ]
    user_request_text: str = Field(min_length=5)
    expected_action: Action
    expected_candidate_ids: list[str] = Field(max_length=2)
    requirement: SongRequirement
    lyric_draft: LyricDraft


class GatedProducerManifest(_StrictModel):
    schema_version: str = Field(min_length=1)
    kind: Literal["producer_rag_gated_ab_fixed_cases"]
    frozen_at: str = Field(min_length=1)
    notes: list[str] = Field(min_length=1)
    cases: list[GatedProducerCase] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_fixed_set(self) -> "GatedProducerManifest":
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("gated_producer_case_ids_must_be_unique")
        required_types = {
            "clear_single_style",
            "named_composite",
            "emotion_only",
            "hard_vocal_constraint",
        }
        actual_types = {case.case_type for case in self.cases}
        if actual_types != required_types:
            raise ValueError(f"gated_producer_case_mix_mismatch:{actual_types}")
        return self


def load_cases(path: Path = CASES_PATH) -> GatedProducerManifest:
    return GatedProducerManifest.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_gate_context(
    decision: StyleRetrievalDecision,
) -> list[dict[str, Any]]:
    if decision.action == "no_rag":
        return []
    cards, sources, policy = load_index_inputs()
    direct, _, blocked = select_local_demo_cards(cards, sources, policy)
    if blocked:
        raise RuntimeError(f"blocked_cards_in_current_admission_state:{blocked}")
    direct_by_id = {card.id: card for card in direct}
    context: list[dict[str, Any]] = []
    for card_id in decision.candidate_ids:
        card = direct_by_id.get(card_id)
        if card is None:
            raise RuntimeError(f"gate_returned_non_direct_card:{card_id}")
        payload = _compact_card(card, score=1.0)
        payload.pop("retrieval_score", None)
        payload["selection_basis"] = "explicit_user_style_gate"
        context.append(payload)
    return context


def build_variant_messages(
    case: GatedProducerCase,
    *,
    variant: str,
    decision: StyleRetrievalDecision,
    context: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown_variant:{variant}")
    if variant == "baseline" or decision.action == "no_rag":
        return build_producer_messages(case.requirement, case.lyric_draft)
    messages = build_producer_messages(
        case.requirement,
        case.lyric_draft,
        rag_context=context,
    )
    if decision.composite_guidance is not None:
        messages[1]["content"] += (
            "\n\n组合风格边界（只用于防止堆叠，不是额外生成要求）：\n"
            + json.dumps(
                decision.composite_guidance.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )
    return messages


def call_producer(
    llm: DeepSeekLLM,
    case: GatedProducerCase,
    *,
    variant: str,
    decision: StyleRetrievalDecision,
    context: list[dict[str, Any]],
) -> ProductionPlan:
    payload = llm.chat_json(
        build_variant_messages(
            case,
            variant=variant,
            decision=decision,
            context=context,
        ),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        operation=f"producer_rag_gated_ab_{variant}",
        payload_validator=lambda value: pydantic_validation_feedback(
            ProductionPlan, value
        ),
    )
    return ProductionPlan.model_validate(payload)


def variant_order(case_id: str) -> tuple[str, str]:
    if hashlib.sha256(case_id.encode("utf-8")).digest()[0] % 2:
        return "gated", "baseline"
    return "baseline", "gated"


def blind_mapping(case_id: str) -> dict[str, str]:
    variants = variant_order(f"blind:{case_id}")
    return {"X": variants[0], "Y": variants[1]}


def build_blind_card(
    case: GatedProducerCase,
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
        "# Producer 制作方案第二轮匿名复核",
        "",
        "只评价歌曲制作方案文本，不评价音频。X/Y 身份已隐藏。",
        "未指定曲风的案例中，两份方案可能来自同一提示条件；差异只能视为采样波动。",
        "",
        "每组分别评价：风格/需求契合度、制作可执行性、内部一致性（均为1-5），",
        "以及是否过度约束、是否有无依据扩写，最后选择 X / Y / 平局。",
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
                f"- 情绪：{requirement['emotion']}",
                f"- 曲风：{requirement['genre']}",
                f"- 人声：{requirement['vocal_type']}",
                f"- 标题：{lyrics['title']}",
                "",
                "歌词/器乐结构：",
                "",
                lyrics["lyrics"],
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
                "- X：契合度__/5；可执行性__/5；一致性__/5；过度约束 是/否；无依据扩写 是/否",
                "- Y：契合度__/5；可执行性__/5；一致性__/5；过度约束 是/否；无依据扩写 是/否",
                "- 偏好：X / Y / 平局",
                "- 备注：",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def run_experiment(
    manifest: GatedProducerManifest,
    *,
    output_root: Path,
) -> Path:
    experiment_dir = output_root / f"producer-rag-gated-ab-v1-{uuid4().hex}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    settings = get_settings()
    _write_json(
        experiment_dir / "experiment_config.json",
        {
            "schema_version": "1.0",
            "case_manifest": str(CASES_PATH),
            "case_manifest_sha256": _manifest_hash(CASES_PATH),
            "variants": list(VARIANTS),
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "planned_deepseek_calls": len(manifest.cases) * 2,
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
            "limitations": [
                "Four single-sample pairs are a development screen, not statistical evidence.",
                "A no_rag pair uses identical prompts, so differences are sampling variation.",
                "Text plans do not prove audio quality or backend control.",
            ],
        },
    )

    blind_cards: list[dict[str, Any]] = []
    blind_keys: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    llm = DeepSeekLLM(settings)
    try:
        for case in manifest.cases:
            print(f"运行案例：{case.case_id}", flush=True)
            case_dir = experiment_dir / "cases" / case.case_id
            case_dir.mkdir(parents=True, exist_ok=False)
            try:
                decision = decide_style_retrieval(case.user_request_text)
                if decision.action != case.expected_action:
                    raise RuntimeError(
                        f"gate_action_mismatch:{decision.action}:{case.expected_action}"
                    )
                if decision.candidate_ids != case.expected_candidate_ids:
                    raise RuntimeError(
                        "gate_candidate_mismatch:"
                        f"{decision.candidate_ids}:{case.expected_candidate_ids}"
                    )
                context = build_gate_context(decision)
                baseline_messages = build_variant_messages(
                    case,
                    variant="baseline",
                    decision=decision,
                    context=context,
                )
                gated_messages = build_variant_messages(
                    case,
                    variant="gated",
                    decision=decision,
                    context=context,
                )
                prompts_identical = baseline_messages == gated_messages
                if (decision.action == "no_rag") != prompts_identical:
                    raise RuntimeError("no_rag_prompt_identity_contract_failed")
                _write_json(case_dir / "input.json", case.model_dump(mode="json"))
                _write_json(case_dir / "gate_decision.json", decision.model_dump(mode="json"))
                _write_json(case_dir / "experimental_context.json", context)

                plans: dict[str, ProductionPlan] = {}
                call_telemetry: dict[str, Any] = {}
                for variant in variant_order(case.case_id):
                    before = len(llm.telemetry_snapshot())
                    plans[variant] = call_producer(
                        llm,
                        case,
                        variant=variant,
                        decision=decision,
                        context=context,
                    )
                    records = llm.telemetry_snapshot()[before:]
                    call_telemetry[variant] = DeepSeekLLM.summarize_call_records(records)
                    _write_json(
                        case_dir / f"{variant}.json",
                        plans[variant].model_dump(mode="json"),
                    )
                card, mapping = build_blind_card(case, plans)
                blind_cards.append(card)
                blind_keys.append({"case_id": case.case_id, **mapping})
                completed.append(
                    {
                        "case_id": case.case_id,
                        "gate_action": decision.action,
                        "candidate_ids": decision.candidate_ids,
                        "call_order": list(variant_order(case.case_id)),
                        "prompts_identical": prompts_identical,
                        "contracts_valid": True,
                        "telemetry": call_telemetry,
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
                print(f"案例失败：{case.case_id}", flush=True)
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
            "expected_logical_calls": len(manifest.cases) * 2,
            "completed": completed,
            "failures": failures,
            "llm_telemetry": telemetry,
            "human_review_status": "pending",
            "method_decision": None,
        },
    )
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行Producer保守风格门控的四案例匿名文本对照。"
    )
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--execute", action="store_true", help="确认调用8次真实DeepSeek；不生成音频"
    )
    args = parser.parse_args()
    manifest = load_cases(args.cases)
    if not args.execute:
        decisions = [
            {
                "case_id": case.case_id,
                "decision": decide_style_retrieval(case.user_request_text).model_dump(
                    mode="json"
                ),
            }
            for case in manifest.cases
        ]
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "case_count": len(manifest.cases),
                    "planned_deepseek_calls": len(manifest.cases) * 2,
                    "audio_generated": False,
                    "runtime_agent_connected": False,
                    "decisions": decisions,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    output = run_experiment(manifest, output_root=args.out)
    print(f"实验输出：{output}", flush=True)


if __name__ == "__main__":
    main()
