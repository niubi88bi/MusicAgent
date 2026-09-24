"""Build a frozen holdout for deterministic Producer clarification decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_OUTPUT = Path(
    "data/miaoxiang/producer_constraint_clarification_holdout.v1.json"
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ClarificationHoldoutCase(_StrictModel):
    case_id: str = Field(min_length=1)
    user_text: str = Field(min_length=2)
    expected_status: Literal[
        "clarification_required", "advisory_only", "no_clarification"
    ]
    expected_replacement_mode: Literal[
        "unspecified",
        "explicit",
        "auto_style_compatible",
        "no_replacement",
        "not_applicable",
    ]
    expected_explicit_replacement: str | None = None
    expected_question_count: int = Field(ge=0, le=1)


class ClarificationHoldoutManifest(_StrictModel):
    schema_version: str = Field(min_length=1)
    kind: str = "producer_constraint_clarification_holdout"
    frozen_at: str = Field(min_length=1)
    notes: list[str] = Field(min_length=1)
    cases: list[ClarificationHoldoutCase] = Field(min_length=16, max_length=16)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "ClarificationHoldoutManifest":
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("clarification_holdout_case_ids_must_be_unique")
        return self


def build_manifest() -> dict:
    raw_cases = [
        ("cq01-hard-no-replacement-cn", "做一首Deep House，不要钢琴，保持温暖克制", "clarification_required", "unspecified", None, 1),
        ("cq02-hard-forbid-cn", "禁止使用钢琴，做一首夜归主题中文歌", "clarification_required", "unspecified", None, 1),
        ("cq03-hard-no-piano-en", "Deep House with no piano and a restrained mood", "clarification_required", "unspecified", None, 1),
        ("cq04-hard-without-piano-en", "A warm electronic song without any piano", "clarification_required", "unspecified", None, 1),
        ("cq05-explicit-guitar-cn", "不要钢琴，必须用木吉他承担伤感动机", "no_clarification", "explicit", "acoustic_guitar", 0),
        ("cq06-explicit-acoustic-cn", "钢琴全部拿掉，换成原声吉他", "no_clarification", "explicit", "acoustic_guitar", 0),
        ("cq07-explicit-guitar-en", "No piano; use acoustic guitar for the wistful motif", "no_clarification", "explicit", "acoustic_guitar", 0),
        ("cq08-explicit-synth-cn", "不要钢琴，改用合成器Pad承担和声", "no_clarification", "explicit", "synth_pad_or_pluck", 0),
        ("cq09-explicit-strings-cn", "不使用钢琴，用弦乐做情绪铺垫", "no_clarification", "explicit", "strings", 0),
        ("cq10-auto-style-cn", "不要钢琴，替代配器由系统按曲风选择", "no_clarification", "auto_style_compatible", None, 0),
        ("cq11-auto-user-cn", "禁止钢琴，其他配器你来决定", "no_clarification", "auto_style_compatible", None, 0),
        ("cq12-no-replacement-cn", "不要钢琴，也不需要替代乐器", "no_clarification", "no_replacement", None, 0),
        ("cq13-soft-prefer-cn", "最好少用钢琴，整体保持轻盈", "advisory_only", "not_applicable", None, 0),
        ("cq14-soft-avoid-cn", "尽量避免钢琴，但不需要阻断生成", "advisory_only", "not_applicable", None, 0),
        ("cq15-positive-piano-cn", "使用钢琴和弦写一首温暖的中文歌", "no_clarification", "not_applicable", None, 0),
        ("cq16-unrelated-cn", "做一首关于清晨渡船的Deep House中文歌", "no_clarification", "not_applicable", None, 0),
    ]
    cases = [
        {
            "case_id": case_id,
            "user_text": user_text,
            "expected_status": status,
            "expected_replacement_mode": mode,
            "expected_explicit_replacement": replacement,
            "expected_question_count": question_count,
        }
        for case_id, user_text, status, mode, replacement, question_count in raw_cases
    ]
    payload = {
        "schema_version": "1.0",
        "kind": "producer_constraint_clarification_holdout",
        "frozen_at": "2026-09-24",
        "notes": [
            "16个固定正反例只验证追问触发语义，不调用LLM、Producer、RAG或音频。",
            "覆盖硬禁止无替代、显式替代、系统决定、无需替代、软偏好和无关请求。",
            "澄清器保持影子模式，不改变正式Requirement Agent、LangGraph或Web行为。",
        ],
        "cases": cases,
    }
    ClarificationHoldoutManifest.model_validate(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="生成Producer制作约束澄清holdout。")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.out), "case_count": len(payload["cases"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
