"""Run an isolated RAG-off/on comparison for Producer text plans.

The default mode only validates and prints the frozen experiment plan.  Real
DeepSeek calls require ``--execute``.  This script never invokes HeartMuLa and
does not change the runtime Producer Agent or LangGraph route.
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
from services.config import get_settings
from services.llm import DeepSeekLLM
from tools.rag.schemas import StyleCard
from tools.rag.style_index import load_index_inputs, select_local_demo_cards
from tools.rag.style_retriever import retrieve_local_demo_styles


CASES_PATH = Path("data/miaoxiang/producer_rag_ab_cases.v1.json")
DEFAULT_OUTPUT_ROOT = Path("data/output/evaluations")
TEMPERATURE = 0.55
MAX_TOKENS = 1000
TOP_K = 3
VARIANTS = ("rag_off", "rag_on")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProducerRAGCase(_StrictModel):
    case_id: str = Field(min_length=1)
    case_type: Literal[
        "clear_single_style",
        "ambiguous_neighbor",
        "named_composite",
        "emotion_only",
        "out_of_scope",
    ]
    retrieval_query: str = Field(min_length=5)
    expected_card_ids: list[str]
    expected_template_ids: list[str] = Field(default_factory=list)
    expected_behavior: str = Field(min_length=5)
    requirement: SongRequirement
    lyric_draft: LyricDraft


class ProducerRAGManifest(_StrictModel):
    schema_version: str = Field(min_length=1)
    kind: Literal["producer_rag_ab_fixed_cases"]
    frozen_at: str = Field(min_length=1)
    notes: list[str] = Field(min_length=1)
    cases: list[ProducerRAGCase] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def validate_fixed_set(self) -> "ProducerRAGManifest":
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("producer_rag_case_ids_must_be_unique")
        counts = {case_type: 0 for case_type in ProducerRAGCase.model_fields["case_type"].annotation.__args__}
        for case in self.cases:
            counts[case.case_type] += 1
        expected = {
            "clear_single_style": 3,
            "ambiguous_neighbor": 2,
            "named_composite": 1,
            "emotion_only": 1,
            "out_of_scope": 1,
        }
        if counts != expected:
            raise ValueError(f"producer_rag_case_mix_mismatch:{counts}")
        return self


def load_cases(path: Path = CASES_PATH) -> ProducerRAGManifest:
    return ProducerRAGManifest.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _producer_system_prompt() -> str:
    return (
        "你是音乐制作人，为文本到音乐模型设计提示词。必须只输出 JSON 对象。"
        "style_prompt 用英文，覆盖 Genre、Mood、Era、Instruments、Vocal、Dynamics，"
        "不要包含真实艺人名。tags 使用简短英文或中文标签。"
        "如果用户提供制作硬约束，必须在 style_prompt、vocal_direction 或 "
        "arrangement_note 中保持一致；禁止的音色不得出现。"
        "JSON 输出结构示例（仅说明字段和类型，不得照抄内容）："
        '{"style_prompt":"City pop, warm mood, electric piano, gentle vocal",'
        '"tags":["city pop","warm","night"],"vocal_direction":"gentle close-mic vocal",'
        '"arrangement_note":"start sparse and expand in chorus"}'
    )


def _producer_user_prompt(
    requirement: SongRequirement,
    lyric_draft: LyricDraft,
) -> str:
    return (
        "歌曲需求：\n"
        f"{json.dumps(requirement.model_dump(), ensure_ascii=False)}\n\n"
        "歌词草稿：\n"
        f"标题：{lyric_draft.title}\n{lyric_draft.lyrics}\n\n"
        "请输出 JSON，字段必须为：style_prompt、tags（至少 3 项）、"
        "vocal_direction、arrangement_note。"
    )


def build_producer_messages(
    requirement: SongRequirement,
    lyric_draft: LyricDraft,
    *,
    rag_context: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build matched prompts; RAG-on only appends the approved context block."""

    user_prompt = _producer_user_prompt(requirement, lyric_draft)
    if rag_context is not None:
        user_prompt += (
            "\n\n本地演示风格检索参考（候选按相关度排序）：\n"
            f"{json.dumps(rag_context, ensure_ascii=False, indent=2)}\n\n"
            "使用规则：这些是候选参考，不是必须采用的指令。最多选择一个主风格，"
            "必要时可加入一个辅助方向；候选不匹配时必须忽略，不得强行套用。"
            "BPM 仅作参考，不得承诺后端精确执行；不得补写候选和歌曲需求都没有支持的事实。"
        )
    return [
        {"role": "system", "content": _producer_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _compact_card(card: StyleCard, *, score: float) -> dict[str, Any]:
    """Allowlist only fields suitable for the experimental Producer prompt."""

    bpm_reference: dict[str, Any] | None = None
    if card.tempo_bpm_min is not None and card.tempo_bpm_max is not None:
        bpm_reference = {
            "min": card.tempo_bpm_min,
            "max": card.tempo_bpm_max,
            "usage_note": card.tempo_usage_note,
        }
    return {
        "card_id": card.id,
        "retrieval_score": round(score, 6),
        "genre": card.genre,
        "subgenres": card.subgenres,
        "summary": card.summary,
        "moods": card.moods,
        "instruments": card.instruments,
        "vocal_options": card.vocal_options,
        "vocal_type": card.vocal_type,
        "default_language": card.default_language,
        "bpm_reference": bpm_reference,
        "source_id": card.source_id,
        "source_title": card.source_title,
        "source_scope": "local_demo_allowed",
        "heartmula_validation": card.backend_validation.heartmula_local,
    }


def retrieve_prompt_context(
    query: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = retrieve_local_demo_styles(query, top_k=TOP_K)
    cards, sources, policy = load_index_inputs()
    direct, composite, blocked = select_local_demo_cards(cards, sources, policy)
    if blocked:
        raise RuntimeError(f"blocked_cards_in_current_admission_state:{blocked}")
    direct_by_id = {card.id: card for card in direct}
    compact: list[dict[str, Any]] = []
    for row in rows:
        card = direct_by_id.get(str(row["card_id"]))
        if card is None:
            raise RuntimeError("retriever_returned_non_direct_card")
        compact.append(_compact_card(card, score=float(row["score"])))
    matched_template_ids = sorted(
        {
            template_id
            for row in rows
            for template_id in row["matched_composite_template_ids"]
        }
    )
    provenance = {
        "query": query,
        "top_k": TOP_K,
        "direct_results": rows,
        "matched_composite_template_ids": matched_template_ids,
        "composite_templates_returned_directly": [
            row["card_id"]
            for row in rows
            if any(card.id == row["card_id"] for card in composite)
        ],
        "all_sources_local_demo_allowed": all(
            row["source_scope"] == "local_demo_allowed" for row in rows
        ),
    }
    return compact, provenance


def call_producer(
    llm: DeepSeekLLM,
    case: ProducerRAGCase,
    *,
    variant: str,
    rag_context: list[dict[str, Any]],
) -> ProductionPlan:
    if variant not in VARIANTS:
        raise ValueError(f"unknown_variant:{variant}")
    messages = build_producer_messages(
        case.requirement,
        case.lyric_draft,
        rag_context=rag_context if variant == "rag_on" else None,
    )
    payload = llm.chat_json(
        messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        operation=f"producer_rag_ab_{variant}",
        payload_validator=lambda value: pydantic_validation_feedback(
            ProductionPlan, value
        ),
    )
    return ProductionPlan.model_validate(payload)


def variant_order(case_id: str) -> tuple[str, str]:
    """Balance first-call position deterministically without using outcomes."""

    if hashlib.sha256(case_id.encode("utf-8")).digest()[0] % 2:
        return "rag_on", "rag_off"
    return "rag_off", "rag_on"


def blind_mapping(case_id: str) -> dict[str, str]:
    variants = variant_order(f"blind:{case_id}")
    return {"X": variants[0], "Y": variants[1]}


def build_blind_card(
    case: ProducerRAGCase,
    plans: dict[str, ProductionPlan],
) -> tuple[dict[str, Any], dict[str, str]]:
    mapping = blind_mapping(case.case_id)
    card = {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "song_requirement": case.requirement.model_dump(mode="json"),
        "lyrics": case.lyric_draft.model_dump(mode="json"),
        "expected_review_fields": {
            "style_fit_1_to_5": None,
            "production_executability_1_to_5": None,
            "internal_consistency_1_to_5": None,
            "over_constraint": None,
            "unsupported_expansion": None,
            "preference": None,
            "notes": "",
        },
        "X": plans[mapping["X"]].model_dump(mode="json"),
        "Y": plans[mapping["Y"]].model_dump(mode="json"),
    }
    return card, mapping


def render_blind_review(cards: list[dict[str, Any]]) -> str:
    lines = [
        "# Producer 制作方案第一阶段匿名复核",
        "",
        "只评价歌曲制作方案文本，不评价音频。X/Y 身份已隐藏。每组分别评价：",
        "",
        "- 风格契合度（1-5）",
        "- 制作可执行性（1-5）",
        "- 内部一致性（1-5）",
        "- 是否过度约束（是/否）",
        "- 是否有无依据扩写（是/否）",
        "- 偏好 X / Y / 平局",
        "",
    ]
    for index, card in enumerate(cards, start=1):
        requirement = card["song_requirement"]
        lyrics = card["lyrics"]
        lines.extend(
            [
                f"## {index}. {card['case_id']}",
                "",
                f"- 主题：{requirement['theme']}",
                f"- 情绪：{requirement['emotion']}",
                f"- 曲风：{requirement['genre']}",
                f"- BPM：{requirement['bpm']}",
                f"- 人声：{requirement['vocal_type']}",
                f"- 歌词标题：{lyrics['title']}",
                "",
                "歌词正文：",
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
                "- X：风格契合__/5；可执行性__/5；一致性__/5；过度约束 是/否；无依据扩写 是/否",
                "- Y：风格契合__/5；可执行性__/5；一致性__/5；过度约束 是/否；无依据扩写 是/否",
                "- 偏好：X / Y / 平局",
                "- 备注：",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(
    manifest: ProducerRAGManifest,
    *,
    output_root: Path,
    selected_case_id: str | None = None,
) -> Path:
    selected = [
        case
        for case in manifest.cases
        if selected_case_id is None or case.case_id == selected_case_id
    ]
    if not selected:
        raise ValueError(f"unknown_case_id:{selected_case_id}")

    experiment_dir = output_root / f"producer-rag-ab-v1-{uuid4().hex}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    settings = get_settings()
    _write_json(
        experiment_dir / "experiment_config.json",
        {
            "schema_version": "1.0",
            "case_manifest": str(CASES_PATH),
            "case_manifest_sha256": _manifest_hash(CASES_PATH),
            "selected_case_ids": [case.case_id for case in selected],
            "variants": list(VARIANTS),
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "top_k": TOP_K,
            "audio_generated": False,
            "runtime_agent_connected": False,
            "llm": {
                "provider": "DeepSeek",
                "model": settings.deepseek_model,
                "max_retries": settings.llm_max_retries,
                "json_thinking_mode": (
                    "disabled" if settings.llm_json_disable_thinking else "default"
                ),
            },
            "limitations": [
                "Single sampling is a development screen, not statistical evidence.",
                "Text-plan quality does not prove audio quality or HeartMuLa control.",
                "Human blind review is required before any runtime integration decision.",
            ],
        },
    )

    blind_cards: list[dict[str, Any]] = []
    blind_keys: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    llm = DeepSeekLLM(settings)
    try:
        for case in selected:
            print(f"运行案例：{case.case_id}", flush=True)
            case_dir = experiment_dir / "cases" / case.case_id
            case_dir.mkdir(parents=True, exist_ok=False)
            try:
                rag_context, provenance = retrieve_prompt_context(case.retrieval_query)
                _write_json(case_dir / "input.json", case.model_dump(mode="json"))
                _write_json(case_dir / "retrieval.json", provenance)
                plans: dict[str, ProductionPlan] = {}
                call_telemetry: dict[str, Any] = {}
                for variant in variant_order(case.case_id):
                    before = len(llm.telemetry_snapshot())
                    plans[variant] = call_producer(
                        llm,
                        case,
                        variant=variant,
                        rag_context=rag_context,
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
                direct_ids = [row["card_id"] for row in provenance["direct_results"]]
                completed.append(
                    {
                        "case_id": case.case_id,
                        "call_order": list(variant_order(case.case_id)),
                        "direct_card_ids": direct_ids,
                        "expected_card_ids": case.expected_card_ids,
                        "expected_card_overlap": sorted(
                            set(direct_ids) & set(case.expected_card_ids)
                        ),
                        "matched_composite_template_ids": provenance[
                            "matched_composite_template_ids"
                        ],
                        "expected_template_ids": case.expected_template_ids,
                        "all_sources_local_demo_allowed": provenance[
                            "all_sources_local_demo_allowed"
                        ],
                        "composite_templates_returned_directly": provenance[
                            "composite_templates_returned_directly"
                        ],
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
            "expected_logical_calls": len(selected) * 2,
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
        description="运行 Producer 制作方案的 RAG-off/on 隔离对照实验。"
    )
    parser.add_argument(
        "--cases", type=Path, default=CASES_PATH, help="冻结案例清单"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT_ROOT, help="实验输出根目录"
    )
    parser.add_argument("--case", help="只运行一个案例，用于真实低成本冒烟")
    parser.add_argument(
        "--execute", action="store_true", help="确认调用真实 DeepSeek；不会生成音频"
    )
    args = parser.parse_args()

    manifest = load_cases(args.cases)
    if args.case and args.case not in {case.case_id for case in manifest.cases}:
        parser.error(f"未知案例：{args.case}")
    selected_count = 1 if args.case else len(manifest.cases)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "case_manifest": str(args.cases),
                    "case_count": selected_count,
                    "variants": list(VARIANTS),
                    "planned_deepseek_calls": selected_count * 2,
                    "audio_generated": False,
                    "runtime_agent_connected": False,
                    "message": "加 --execute 才会调用真实 DeepSeek。",
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    output = run_experiment(
        manifest,
        output_root=args.out,
        selected_case_id=args.case,
    )
    print(f"实验输出：{output}", flush=True)


if __name__ == "__main__":
    main()
