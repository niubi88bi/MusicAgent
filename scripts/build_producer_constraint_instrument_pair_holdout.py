"""Build an independent holdout for the narrow piano/acoustic-guitar pair rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents.schemas import ProductionPlan
from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_OUTPUT = Path(
    "data/miaoxiang/producer_constraint_instrument_pair_holdout.v1.json"
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class InstrumentPairHoldoutCase(_StrictModel):
    case_id: str = Field(min_length=1)
    request: str = Field(min_length=10)
    expected_action: str = Field(min_length=1)
    expected_hard_concepts: list[str]
    expected_final_equals_base: bool
    base_plan: ProductionPlan
    injected_plan: ProductionPlan


class InstrumentPairHoldoutManifest(_StrictModel):
    schema_version: str = Field(min_length=1)
    kind: str = "producer_constraint_instrument_pair_holdout"
    frozen_at: str = Field(min_length=1)
    notes: list[str] = Field(min_length=1)
    cases: list[InstrumentPairHoldoutCase] = Field(min_length=7, max_length=7)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "InstrumentPairHoldoutManifest":
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("instrument_pair_holdout_case_ids_must_be_unique")
        return self


def _plan(
    style_prompt: str,
    tags: list[str],
    vocal_direction: str,
    arrangement_note: str,
) -> dict[str, Any]:
    return ProductionPlan(
        style_prompt=style_prompt,
        tags=tags,
        vocal_direction=vocal_direction,
        arrangement_note=arrangement_note,
    ).model_dump(mode="json")


def build_manifest() -> dict[str, Any]:
    cases = [
        {
            "case_id": "ip01-unique-cn-replacement",
            "request": "Sad Slap House中文歌，不要钢琴，必须用木吉他做伤感动机，电子推动克制",
            "expected_action": "deterministic_repair",
            "expected_hard_concepts": [],
            "expected_final_equals_base": True,
            "base_plan": _plan(
                "Sad Slap House，克制忧郁与中速电子律动",
                ["Sad Slap House", "克制", "夜晚"],
                "中文独唱，近距离克制表达",
                "以悲伤木吉他开场，随后加入低频脉冲并保持动态克制",
            ),
            "injected_plan": _plan(
                "Sad Slap House，克制忧郁与中速电子律动",
                ["Sad Slap House", "克制", "夜晚"],
                "中文独唱，近距离克制表达",
                "以悲伤钢琴开场，随后加入低频脉冲并保持动态克制",
            ),
        },
        {
            "case_id": "ip02-unique-en-replacement",
            "request": "Produce a restrained Sad Slap House song. No piano; acoustic guitar must carry the wistful motif.",
            "expected_action": "deterministic_repair",
            "expected_hard_concepts": [],
            "expected_final_equals_base": True,
            "base_plan": _plan(
                "Sad Slap House, restrained pulse and intimate solo vocal",
                ["Sad Slap House", "restrained", "night"],
                "Chinese solo vocal, close-mic and controlled",
                "Open with acoustic guitar, then add a soft bass pulse without an explosive drop",
            ),
            "injected_plan": _plan(
                "Sad Slap House, restrained pulse and intimate solo vocal",
                ["Sad Slap House", "restrained", "night"],
                "Chinese solo vocal, close-mic and controlled",
                "Open with piano, then add a soft bass pulse without an explosive drop",
            ),
        },
        {
            "case_id": "ip03-append-cn-no-piano",
            "request": "Vina House中文歌，不要钢琴，必须用木吉他承担轻微伤感动机，整体明亮轻盈",
            "expected_action": "deterministic_repair",
            "expected_hard_concepts": [],
            "expected_final_equals_base": False,
            "base_plan": _plan(
                "Vina House，明亮轻盈的中文舞曲",
                ["Vina House", "明亮", "轻盈"],
                "中文清亮独唱，保持自然",
                "木吉他承担主要伤感动机；主歌以弹跳低音和稀疏合成器推进，副歌保持明亮而克制；不使用钢琴",
            ),
            "injected_plan": _plan(
                "Vina House，明亮轻盈的中文舞曲",
                ["Vina House", "明亮", "轻盈"],
                "中文清亮独唱，保持自然",
                "主歌以弹跳低音和稀疏合成器推进，副歌保持明亮而克制",
            ),
        },
        {
            "case_id": "ip04-compliant-pass-through",
            "request": "做一首中文Sad Slap House，不要钢琴，用木吉他做伤感动机，电子推动克制",
            "expected_action": "pass_through",
            "expected_hard_concepts": [],
            "expected_final_equals_base": True,
            "base_plan": _plan(
                "Sad Slap House，克制忧郁的中文独唱",
                ["Sad Slap House", "克制", "中文歌"],
                "中文独唱，近距离表达",
                "木吉他承担伤感动机，低频脉冲克制推进，不使用钢琴",
            ),
            "injected_plan": _plan(
                "Sad Slap House，克制忧郁的中文独唱",
                ["Sad Slap House", "克制", "中文歌"],
                "中文独唱，近距离表达",
                "木吉他承担伤感动机，低频脉冲克制推进，不使用钢琴",
            ),
        },
        {
            "case_id": "ip05-multiple-piano-fail-closed",
            "request": "Sad Slap House中文歌，不要钢琴，必须用木吉他做伤感动机",
            "expected_action": "fail_closed",
            "expected_hard_concepts": ["piano", "acoustic_guitar"],
            "expected_final_equals_base": False,
            "base_plan": _plan(
                "Sad Slap House，克制电子律动",
                ["Sad Slap House", "克制", "夜晚"],
                "中文独唱，近距离表达",
                "木吉他承担伤感动机，低频脉冲克制推进，不使用钢琴",
            ),
            "injected_plan": _plan(
                "Sad Slap House，悲伤钢琴与克制电子律动",
                ["Sad Slap House", "克制", "夜晚"],
                "中文独唱，近距离表达",
                "主歌使用简约钢琴，桥段再次回到钢琴并降低鼓组密度",
            ),
        },
        {
            "case_id": "ip06-ambiguous-piano-fail-closed",
            "request": "不要钢琴，必须用木吉他做伤感动机，钢琴只能作为反例录音噪声",
            "expected_action": "fail_closed",
            "expected_hard_concepts": ["piano"],
            "expected_final_equals_base": False,
            "base_plan": _plan(
                "Sad Slap House，木吉他与克制低频脉冲",
                ["Sad Slap House", "克制", "实验"],
                "中文独唱，近距离表达",
                "木吉他承担伤感动机，电子鼓保持克制，不使用钢琴",
            ),
            "injected_plan": _plan(
                "Sad Slap House，保留旧钢琴录音的噪声作为反例，不使用钢琴演奏",
                ["Sad Slap House", "克制", "实验"],
                "中文独唱，近距离表达",
                "低频脉冲保持克制，木吉他尚未进入",
            ),
        },
        {
            "case_id": "ip07-only-forbidden-no-pair-fail-closed",
            "request": "做一首中文Deep House，不要钢琴，保持温暖克制，不指定替代乐器",
            "expected_action": "fail_closed",
            "expected_hard_concepts": ["piano"],
            "expected_final_equals_base": False,
            "base_plan": _plan(
                "Deep House，温暖克制的中文独唱",
                ["Deep House", "温暖", "克制"],
                "中文独唱，近距离表达",
                "以Rhodes和柔和铺底推进，不使用钢琴",
            ),
            "injected_plan": _plan(
                "Deep House，温暖克制的中文独唱",
                ["Deep House", "温暖", "克制"],
                "中文独唱，近距离表达",
                "以悲伤钢琴和柔和铺底推进",
            ),
        },
    ]
    payload = {
        "schema_version": "1.0",
        "kind": "producer_constraint_instrument_pair_holdout",
        "frozen_at": "2026-09-24",
        "notes": [
            "7个未复用na09/na10的独立成对乐器约束案例。",
            "覆盖唯一精确替换、固定约束句追加、已合规直通、多处钢琴、语义歧义和仅禁止钢琴。",
            "只验证确定性补丁与失败关闭，不调用模型、RAG或音频。",
        ],
        "cases": cases,
    }
    InstrumentPairHoldoutManifest.model_validate(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="生成成对乐器约束独立holdout。")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.out), "case_count": len(payload["cases"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
