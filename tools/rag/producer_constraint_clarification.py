"""Deterministic shadow clarification for incomplete Producer constraints.

This module only analyzes raw user text. It does not connect itself to the
Requirement Agent, LangGraph, Web routes, Producer generation, or RAG.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ClarificationOption(_StrictModel):
    code: Literal[
        "acoustic_guitar",
        "synth_pad_or_pluck",
        "strings",
        "no_replacement",
        "auto_style_compatible",
    ]
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)


class ClarificationQuestion(_StrictModel):
    code: Literal["forbidden_instrument_without_replacement"]
    question: str = Field(min_length=1)
    options: list[ClarificationOption] = Field(min_length=5, max_length=5)
    recommended_option: Literal["auto_style_compatible"]


class ClarificationAnswer(_StrictModel):
    """用户对一次澄清问题的结构化选择。"""

    code: Literal["forbidden_instrument_without_replacement"]
    option_code: Literal[
        "acoustic_guitar",
        "synth_pad_or_pluck",
        "strings",
        "no_replacement",
        "auto_style_compatible",
    ]


class ProducerConstraintClarification(_StrictModel):
    status: Literal[
        "clarification_required",
        "advisory_only",
        "no_clarification",
    ]
    forbidden_instruments: list[str] = Field(default_factory=list)
    forbidden_scope: list[str] = Field(default_factory=list)
    replacement_mode: Literal[
        "unspecified",
        "explicit",
        "auto_style_compatible",
        "no_replacement",
        "not_applicable",
    ]
    explicit_replacement: str | None = None
    advisories: list[str] = Field(default_factory=list)
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


PIANO_SCOPE = ["acoustic_piano", "electric_piano", "rhodes"]

QUESTION = ClarificationQuestion(
    code="forbidden_instrument_without_replacement",
    question="已记录不使用钢琴。你希望由什么承担主要和声或情绪动机？",
    options=[
        ClarificationOption(
            code="acoustic_guitar",
            label="木吉他",
            description="用木吉他或原声吉他承担主要和声与情绪动机。",
        ),
        ClarificationOption(
            code="synth_pad_or_pluck",
            label="合成器",
            description="使用合成器 Pad 或 Pluck，避免引入钢琴类音色。",
        ),
        ClarificationOption(
            code="strings",
            label="弦乐",
            description="用弦乐承担和声与情绪铺垫。",
        ),
        ClarificationOption(
            code="no_replacement",
            label="不需要替代",
            description="减少和声乐器，不强制加入新的替代乐器。",
        ),
        ClarificationOption(
            code="auto_style_compatible",
            label="系统按曲风选择",
            description="保留禁用钢琴，由系统选择与曲风兼容的其他配器。",
        ),
    ],
    recommended_option="auto_style_compatible",
)

REPLACEMENT_INSTRUMENTS = {
    "acoustic_guitar": "acoustic_guitar",
    "synth_pad_or_pluck": "synth_pad_or_pluck",
    "strings": "strings",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def _first_match(text: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


SOFT_PIANO_PATTERNS = (
    r"最好[^，。；,.]{0,8}(?:少用|不用)钢琴",
    r"尽量[^，。；,.]{0,8}(?:避免|少用|不用)钢琴",
    r"钢琴[^，。；,.]{0,6}(?:少一点|弱一点)",
    r"prefer(?:ably)?\s+(?:less|no)\s+piano",
)

HARD_PIANO_PATTERNS = (
    r"(?:(?:不要|禁止|禁用)(?:使用)?|不能用|不使用|不用)(?:任何)?(?:原声|电)?钢琴",
    r"钢琴(?:全部)?(?:拿掉|移除|去掉)",
    r"(?:no|without)\s+(?:any\s+)?(?:acoustic\s+|electric\s+)?piano",
)

EXPLICIT_REPLACEMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "acoustic_guitar",
        (
            r"(?:必须用|改用|换成|使用|用)(?:木吉他|原声吉他)",
            r"(?:must\s+(?:use|include)|use|replace[^.;，。；]{0,12}with)\s+(?:an?\s+)?acoustic\s+guitar",
            r"acoustic\s+guitar\s+must\s+(?:carry|serve|provide|be)",
        ),
    ),
    (
        "synth_pad_or_pluck",
        (
            r"(?:改用|换成|使用|用)(?:合成器|pad|pluck)",
            r"(?:use|replace[^.;，。；]{0,12}with)\s+(?:synth|pad|pluck)",
        ),
    ),
    (
        "strings",
        (
            r"(?:改用|换成|使用|用)(?:弦乐|弦乐组)",
            r"(?:use|replace[^.;，。；]{0,12}with)\s+strings",
        ),
    ),
)

AUTO_PATTERNS = (
    r"(?:其他|替代)配器(?:由你|你来|系统|自动)(?:决定|选择)",
    r"(?:由系统|系统自动|你来|自行)(?:按曲风)?(?:决定|选择)",
    r"(?:按|根据)(?:当前)?曲风(?:自动)?选择",
    r"(?:you|system)\s+(?:may\s+)?choose",
    r"choose\s+(?:an?\s+)?alternative\s+by\s+style",
)

NO_REPLACEMENT_PATTERNS = (
    r"不需要(?:任何)?替代(?:乐器)?",
    r"无需(?:任何)?替代(?:乐器)?",
    r"不要替代(?:乐器)?",
    r"no\s+replacement(?:\s+instrument)?",
)


def analyze_producer_constraint_clarification(
    user_text: str,
) -> ProducerConstraintClarification:
    """Return a narrow, deterministic clarification decision for piano bans."""

    text = _normalize(user_text)
    soft = _first_match(text, SOFT_PIANO_PATTERNS)
    hard_text = text.replace(soft, " ", 1) if soft else text
    hard = _first_match(hard_text, HARD_PIANO_PATTERNS)
    if soft and not hard:
        return ProducerConstraintClarification(
            status="advisory_only",
            replacement_mode="not_applicable",
            advisories=["用户表达的是减少或尽量避免钢琴，不作为阻断性硬约束。"],
            evidence=[soft],
        )
    if not hard:
        return ProducerConstraintClarification(
            status="no_clarification",
            replacement_mode="not_applicable",
        )

    base = {
        "forbidden_instruments": ["piano"],
        "forbidden_scope": PIANO_SCOPE,
        "evidence": [hard],
    }
    for replacement, patterns in EXPLICIT_REPLACEMENTS:
        evidence = _first_match(text, patterns)
        if evidence:
            return ProducerConstraintClarification(
                status="no_clarification",
                replacement_mode="explicit",
                explicit_replacement=replacement,
                **{**base, "evidence": [hard, evidence]},
            )
    auto = _first_match(text, AUTO_PATTERNS)
    if auto:
        return ProducerConstraintClarification(
            status="no_clarification",
            replacement_mode="auto_style_compatible",
            **{**base, "evidence": [hard, auto]},
        )
    no_replacement = _first_match(text, NO_REPLACEMENT_PATTERNS)
    if no_replacement:
        return ProducerConstraintClarification(
            status="no_clarification",
            replacement_mode="no_replacement",
            **{**base, "evidence": [hard, no_replacement]},
        )
    return ProducerConstraintClarification(
        status="clarification_required",
        replacement_mode="unspecified",
        questions=[QUESTION],
        **base,
    )


def resolve_clarification_answer(
    decision: ProducerConstraintClarification,
    answer: ClarificationAnswer,
) -> dict[str, object]:
    """将一个合法选项转换为后续制作阶段可消费的硬约束。"""

    if decision.status != "clarification_required":
        raise ValueError("当前需求不需要澄清答案")
    if answer.code != QUESTION.code:
        raise ValueError("澄清问题代码不匹配")
    allowed = {option.code for option in QUESTION.options}
    if answer.option_code not in allowed:
        raise ValueError("澄清选项不在允许范围内")
    result: dict[str, object] = {
        "forbidden_instruments": list(decision.forbidden_instruments),
        "forbidden_scope": list(decision.forbidden_scope),
        "replacement_mode": answer.option_code,
    }
    replacement = REPLACEMENT_INSTRUMENTS.get(answer.option_code)
    if replacement is not None:
        result["replacement_instrument"] = replacement
    return result
