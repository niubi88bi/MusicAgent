"""Deterministic explicit-constraint guard for isolated Producer experiments.

The guard only extracts a narrow allowlist of explicit user constraints. It
does not infer taste, rewrite requirements, or connect itself to runtime code.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.schemas import ProductionPlan
from tools.rag.style_retrieval_policy import decide_style_retrieval, parse_constraints


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ConstraintTerm(_StrictModel):
    concept: str = Field(min_length=1)
    accepted_terms: list[str] = Field(min_length=1)
    evidence: str = Field(min_length=1)


class ExplicitProductionConstraints(_StrictModel):
    must_include: list[ConstraintTerm] = Field(default_factory=list)
    must_avoid: list[ConstraintTerm] = Field(default_factory=list)
    do_not_infer: list[
        Literal[
            "vocal_gender",
            "secondary_genre",
            "exact_bpm_as_hard_requirement",
        ]
    ] = Field(default_factory=list)


class ConstraintViolation(_StrictModel):
    code: Literal[
        "required_term_missing",
        "forbidden_term_present",
        "unrequested_vocal_gender",
        "unrequested_secondary_genre",
        "exact_bpm_written_as_hard_requirement",
    ]
    severity: Literal["hard", "advisory"]
    concept: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


TERM_GROUPS: dict[str, tuple[str, ...]] = {
    "female_vocal": (
        "女声",
        "female vocal",
        "female voice",
        "female lead",
        "female solo",
    ),
    "male_vocal": (
        "男声",
        "male vocal",
        "male voice",
        "male lead",
        "male solo",
    ),
    "rap_or_mc": (
        "说唱",
        "rap",
        "mc",
        "spoken bars",
        "口号",
        "spoken slogan",
        "spoken slogans",
        "sampled slogan",
        "sampled slogans",
    ),
    "vocal_material": (
        "人声",
        "vocal",
        "spoken word",
        "口白",
        "voice sample",
        "vocal sample",
    ),
    "piano": ("钢琴", "piano"),
    "acoustic_guitar": ("木吉他", "acoustic guitar"),
}

SECONDARY_GENRES: tuple[str, ...] = (
    "city pop",
    "city-pop",
    "jazz",
    "bossa nova",
    "rock",
    "techno",
    "trance",
    "future bass",
    "drum & bass",
    "drum and bass",
    "deep house",
    "tropical house",
    "vina house",
    "slap house",
)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def _contains(text: str, term: str) -> bool:
    normalized_term = _normalize(term)
    if re.fullmatch(r"[a-z0-9 &+_-]+", normalized_term):
        escaped = re.escape(normalized_term).replace(r"\ ", r"\s+")
        return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None
    return normalized_term in text


def _positive_occurrence(text: str, term: str) -> str | None:
    """Return a non-negated occurrence of ``term`` if one exists."""

    normalized_term = _normalize(term)
    if re.fullmatch(r"[a-z0-9 &+_-]+", normalized_term):
        escaped = re.escape(normalized_term).replace(r"\ ", r"\s+")
        pattern = re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")
    else:
        pattern = re.compile(re.escape(normalized_term))
    negation = re.compile(
        r"(?:no|without|avoid|avoiding|exclude|excluding|禁止|不要|避免|不用|无|不使用)"
        r"[^,.;，。；]{0,16}$"
    )
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 24) : match.start()]
        if not negation.search(prefix):
            return match.group(0)
    return None


def _term(concept: str, evidence: str) -> ConstraintTerm:
    return ConstraintTerm(
        concept=concept,
        accepted_terms=list(TERM_GROUPS[concept]),
        evidence=evidence,
    )


def extract_explicit_constraints(user_text: str) -> ExplicitProductionConstraints:
    """Extract only registered constraints explicitly present in raw user text."""

    text = _normalize(user_text)
    parsed = parse_constraints(user_text)
    must_include: list[ConstraintTerm] = []
    must_avoid: list[ConstraintTerm] = []
    do_not_infer: list[str] = []

    female_forbidden = bool(re.search(r"(?:不要|禁用|禁止)(?:[^，。；]{0,8})女声", text))
    male_forbidden = bool(re.search(r"(?:不要|禁用|禁止)(?:[^，。；]{0,8})男声", text))
    female_required = "女声" in text and not female_forbidden
    male_required = "男声" in text and not male_forbidden
    if female_required and not male_required:
        must_include.append(_term("female_vocal", "用户明确指定女声"))
        must_avoid.append(_term("male_vocal", "用户明确排除或覆盖男声默认值"))
    elif male_required and not female_required:
        must_include.append(_term("male_vocal", "用户明确指定男声"))
        must_avoid.append(_term("female_vocal", "用户明确排除或覆盖女声默认值"))
    elif not female_required and not male_required:
        do_not_infer.append("vocal_gender")

    if parsed.rap == "forbidden":
        must_avoid.append(_term("rap_or_mc", "用户明确禁止说唱或MC"))
    if parsed.instrumental == "required":
        must_avoid.append(_term("vocal_material", "用户明确要求纯音乐或无人声"))

    instrument_patterns = (
        (
            "piano",
            r"(?:不要|禁用|禁止|避免)(?:使用)?钢琴|"
            r"(?:no|without|avoid|excluding)\s+(?:the\s+)?piano",
        ),
        (
            "acoustic_guitar",
            r"(?:必须用|改用|使用|用)木吉他|"
            r"(?:must\s+(?:use|include)|use)\s+(?:an?\s+)?acoustic\s+guitar|"
            r"acoustic\s+guitar\s+must\s+(?:carry|serve|provide|be)",
        ),
    )
    for concept, pattern in instrument_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        rule = _term(concept, match.group(0))
        if concept == "piano":
            must_avoid.append(rule)
        else:
            must_include.append(rule)

    decision = decide_style_retrieval(user_text)
    if decision.action == "single_style":
        do_not_infer.append("secondary_genre")
    if any(
        phrase in text
        for phrase in ("bpm只作参考", "bpm 只作参考", "不要固定数值", "速度可随演唱微调")
    ):
        do_not_infer.append("exact_bpm_as_hard_requirement")

    return ExplicitProductionConstraints(
        must_include=must_include,
        must_avoid=must_avoid,
        do_not_infer=list(dict.fromkeys(do_not_infer)),
    )


def _plan_text(plan: ProductionPlan) -> str:
    return _normalize(
        "\n".join(
            [
                plan.style_prompt,
                " ".join(plan.tags),
                plan.vocal_direction,
                plan.arrangement_note,
            ]
        )
    )


def check_plan_constraints(
    plan: ProductionPlan,
    *,
    user_text: str,
    constraints: ExplicitProductionConstraints,
) -> list[ConstraintViolation]:
    """Return deterministic violations without modifying the plan."""

    text = _plan_text(plan)
    user = _normalize(user_text)
    violations: list[ConstraintViolation] = []
    for rule in constraints.must_include:
        if not any(_positive_occurrence(text, term) for term in rule.accepted_terms):
            violations.append(
                ConstraintViolation(
                    code="required_term_missing",
                    severity="hard",
                    concept=rule.concept,
                    evidence=rule.evidence,
                )
            )
    for rule in constraints.must_avoid:
        present = next(
            (
                occurrence
                for term in rule.accepted_terms
                if (occurrence := _positive_occurrence(text, term))
            ),
            None,
        )
        if present:
            violations.append(
                ConstraintViolation(
                    code="forbidden_term_present",
                    severity="hard",
                    concept=rule.concept,
                    evidence=present,
                )
            )

    if "vocal_gender" in constraints.do_not_infer:
        for concept in ("female_vocal", "male_vocal"):
            present = next(
                (
                    occurrence
                    for term in TERM_GROUPS[concept]
                    if (occurrence := _positive_occurrence(text, term))
                ),
                None,
            )
            if present:
                violations.append(
                    ConstraintViolation(
                        code="unrequested_vocal_gender",
                        severity="advisory",
                        concept=concept,
                        evidence=present,
                    )
                )

    if "secondary_genre" in constraints.do_not_infer:
        for genre in SECONDARY_GENRES:
            occurrence = _positive_occurrence(text, genre)
            if occurrence and not _contains(user, genre):
                violations.append(
                    ConstraintViolation(
                        code="unrequested_secondary_genre",
                        severity="advisory",
                        concept="secondary_genre",
                        evidence=occurrence,
                    )
                )

    if "exact_bpm_as_hard_requirement" in constraints.do_not_infer:
        hard_bpm = re.search(
            r"(?:must|strictly|exactly|固定|必须)[^\n,.，。]{0,20}\b\d{2,3}\s*bpm\b",
            text,
        )
        if hard_bpm:
            violations.append(
            ConstraintViolation(
                code="exact_bpm_written_as_hard_requirement",
                severity="hard",
                concept="bpm",
                evidence=hard_bpm.group(0),
                )
            )

    unique: dict[tuple[str, str, str], ConstraintViolation] = {}
    for violation in violations:
        key = (violation.code, violation.concept, violation.evidence)
        unique[key] = violation
    return list(unique.values())


def split_constraint_violations(
    violations: list[ConstraintViolation],
) -> tuple[list[ConstraintViolation], list[ConstraintViolation]]:
    """Separate auto-repairable hard conflicts from report-only advisories."""

    hard = [row for row in violations if row.severity == "hard"]
    advisory = [row for row in violations if row.severity == "advisory"]
    return hard, advisory
