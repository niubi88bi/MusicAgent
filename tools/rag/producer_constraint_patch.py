"""Minimal-diff patching for isolated Producer constraint experiments.

This module never connects itself to runtime routing. It creates and applies
exact fragment replacements, then rejects any result that changes unapproved
fields or introduces new registered production concepts.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.schemas import ProductionPlan
from tools.rag.producer_constraint_guard import (
    SECONDARY_GENRES,
    TERM_GROUPS,
    ConstraintViolation,
    ExplicitProductionConstraints,
    check_plan_constraints,
    split_constraint_violations,
)


TEXT_FIELDS = ("style_prompt", "vocal_direction", "arrangement_note")
PlanField = Literal["style_prompt", "tags", "vocal_direction", "arrangement_note"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FragmentPatch(_StrictModel):
    field: PlanField
    original_fragment: str = Field(min_length=1)
    replacement_fragment: str = Field(min_length=1)
    violation_code: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    strategy: Literal["deterministic", "llm_fragment"]
    operation: Literal["replace", "append"] = "replace"


class LocatedViolation(_StrictModel):
    field: PlanField
    original_fragment: str = Field(min_length=1)
    violation_code: str = Field(min_length=1)
    concept: str = Field(min_length=1)


class PatchValidation(_StrictModel):
    accepted: bool
    reason: str = Field(min_length=1)
    changed_fields: list[str] = Field(default_factory=list)
    final_hard: list[ConstraintViolation] = Field(default_factory=list)
    final_advisory: list[ConstraintViolation] = Field(default_factory=list)
    new_advisory: list[ConstraintViolation] = Field(default_factory=list)
    new_registered_concepts: list[str] = Field(default_factory=list)
    new_bpm_values: list[str] = Field(default_factory=list)


def changed_fields(left: ProductionPlan, right: ProductionPlan) -> list[str]:
    before = left.model_dump(mode="json")
    after = right.model_dump(mode="json")
    return [field for field in before if before[field] != after[field]]


def _is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 28) : start].lower()
    return bool(
        re.search(
            r"(?:no|without|avoid|avoiding|exclude|excluding|禁止|不要|避免|"
            r"不用|无|不使用)[^,.;，。；]{0,18}$",
            prefix,
        )
    )


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    if re.fullmatch(r"[a-z0-9 &+_-]+", term.lower()):
        return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def _positive_spans(text: str, concept: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for term in sorted(TERM_GROUPS[concept], key=len, reverse=True):
        for match in _term_pattern(term).finditer(text):
            if _is_negated(text, match.start()):
                continue
            span = (match.start(), match.end())
            if any(not (span[1] <= old[0] or span[0] >= old[1]) for old in spans):
                continue
            spans.append(span)
    return sorted(spans)


def _clause_fragment(text: str, start: int, end: int) -> str:
    left_matches = list(re.finditer(r"[,，;；。]", text[:start]))
    left = left_matches[-1].end() if left_matches else 0
    right_match = re.search(r"[,，;；。]", text[end:])
    right = end + right_match.start() if right_match else len(text)
    return text[left:right].strip()


def locate_forbidden_violation(
    plan: ProductionPlan,
    violation: ConstraintViolation,
) -> list[LocatedViolation]:
    if violation.code != "forbidden_term_present":
        return []
    locations: dict[tuple[str, str], LocatedViolation] = {}
    payload = plan.model_dump(mode="json")
    for field in TEXT_FIELDS:
        value = str(payload[field])
        spans = _positive_spans(value, violation.concept)
        for start, end in spans:
            fragment = _clause_fragment(value, start, end)
            locations[(field, fragment)] = LocatedViolation(
                field=field,
                original_fragment=fragment,
                violation_code=violation.code,
                concept=violation.concept,
            )
    for tag in plan.tags:
        if _positive_spans(tag, violation.concept):
            locations[("tags", tag)] = LocatedViolation(
                field="tags",
                original_fragment=tag,
                violation_code=violation.code,
                concept=violation.concept,
            )
    return list(locations.values())


def _required_concepts(constraints: ExplicitProductionConstraints) -> set[str]:
    return {rule.concept for rule in constraints.must_include}


def _explicit_piano_guitar_pair(
    constraints: ExplicitProductionConstraints,
) -> bool:
    """Return whether the user explicitly supplied the paired substitution relation."""

    required = _required_concepts(constraints)
    forbidden = {rule.concept for rule in constraints.must_avoid}
    return "acoustic_guitar" in required and "piano" in forbidden


def _positive_concept_spans(plan: ProductionPlan, concept: str) -> list[tuple[str, tuple[int, int]]]:
    payload = plan.model_dump(mode="json")
    spans: list[tuple[str, tuple[int, int]]] = []
    for field in TEXT_FIELDS:
        value = str(payload[field])
        spans.extend((field, span) for span in _positive_spans(value, concept))
    for index, tag in enumerate(plan.tags):
        spans.extend((f"tags[{index}]", span) for span in _positive_spans(tag, concept))
    return spans


def _ambiguous_instrument_context(fragment: str) -> bool:
    lowered = fragment.lower()
    return bool(
        re.search(
            r"反例|模拟|录音|噪声|采样|reference|example|simulat|recording|noise|sample",
            lowered,
        )
    )


def _append_pair_constraint_patch(
    plan: ProductionPlan,
    *,
    violation: ConstraintViolation,
    constraints: ExplicitProductionConstraints,
) -> FragmentPatch | None:
    if (
        violation.code != "required_term_missing"
        or violation.concept != "acoustic_guitar"
        or not _explicit_piano_guitar_pair(constraints)
    ):
        return None
    # If a positive piano occurrence exists, the paired replacement path must
    # resolve it first. Appending a sentence would leave the hard conflict in
    # place and would hide an unsafe replacement decision.
    if _positive_concept_spans(plan, "piano"):
        return None
    if _positive_concept_spans(plan, "acoustic_guitar"):
        return None
    original = plan.arrangement_note
    if not original.strip():
        return None
    chinese = bool(re.search(r"[\u3400-\u9fff]", original))
    guitar_label = "木吉他"
    for rule in constraints.must_include:
        if rule.concept == "acoustic_guitar" and "原声吉他" in rule.evidence:
            guitar_label = "原声吉他"
            break
        if rule.concept == "acoustic_guitar" and "acoustic guitar" in rule.evidence.lower():
            guitar_label = "acoustic guitar"
            break
    suffix = (
        f"；{guitar_label}承担主要伤感动机；不使用钢琴"
        if chinese
        else "; acoustic guitar carries the main wistful motif; no piano"
    )
    if suffix in original:
        return None
    return FragmentPatch(
        field="arrangement_note",
        original_fragment=original,
        replacement_fragment=original + suffix,
        violation_code=violation.code,
        concept=violation.concept,
        strategy="deterministic",
        operation="append",
    )


def deterministic_patch(
    plan: ProductionPlan,
    *,
    violation: ConstraintViolation,
    constraints: ExplicitProductionConstraints,
) -> FragmentPatch | None:
    appended = _append_pair_constraint_patch(
        plan,
        violation=violation,
        constraints=constraints,
    )
    if appended is not None:
        return appended

    if violation.code == "exact_bpm_written_as_hard_requirement":
        matches: list[tuple[str, re.Match[str]]] = []
        pattern = re.compile(
            r"(?:must\s+(?:stay\s+)?exactly\s+at|must\s+be|strictly|"
            r"必须(?:保持|固定)?|固定)\s*(\d{2,3})\s*BPM",
            re.IGNORECASE,
        )
        for field in TEXT_FIELDS:
            value = str(getattr(plan, field))
            matches.extend((field, match) for match in pattern.finditer(value))
        if len(matches) != 1:
            return None
        field, match = matches[0]
        return FragmentPatch(
            field=field,
            original_fragment=match.group(0),
            replacement_fragment=f"around {match.group(1)} BPM",
            violation_code=violation.code,
            concept=violation.concept,
            strategy="deterministic",
        )

    locations = locate_forbidden_violation(plan, violation)
    if len(locations) != 1:
        return None
    location = locations[0]
    required = _required_concepts(constraints)
    replacements = {
        ("female_vocal", "male_vocal"): ("女声", "男声", "female", "male"),
        ("male_vocal", "female_vocal"): ("男声", "女声", "male", "female"),
        ("piano", "acoustic_guitar"): ("钢琴", "木吉他", "piano", "acoustic guitar"),
    }
    for (forbidden, wanted), (zh_old, zh_new, en_old, en_new) in replacements.items():
        if violation.concept == forbidden and wanted in required:
            if (
                forbidden == "piano"
                and wanted == "acoustic_guitar"
                and (
                    len(_positive_concept_spans(plan, "piano")) != 1
                    or _ambiguous_instrument_context(location.original_fragment)
                )
            ):
                return None
            original = location.original_fragment
            guitar_label = zh_new
            for rule in constraints.must_include:
                if rule.concept == "acoustic_guitar" and "原声吉他" in rule.evidence:
                    guitar_label = "原声吉他"
                    break
            if zh_old in original:
                replacement = original.replace(zh_old, guitar_label, 1)
            elif re.search(en_old, original, flags=re.IGNORECASE):
                replacement = re.sub(
                    en_old, en_new, original, count=1, flags=re.IGNORECASE
                )
            else:
                return None
            return FragmentPatch(
                field=location.field,
                original_fragment=original,
                replacement_fragment=replacement,
                violation_code=violation.code,
                concept=violation.concept,
                strategy="deterministic",
            )

    if violation.concept == "vocal_material" and location.field in {
        "style_prompt",
        "vocal_direction",
    }:
        english = bool(re.search(r"[a-z]", location.original_fragment, re.IGNORECASE))
        replacement = (
            "no vocals, no speech, no vocal samples"
            if english
            else "无人声、无口白、无人声采样"
        )
        return FragmentPatch(
            field=location.field,
            original_fragment=location.original_fragment,
            replacement_fragment=replacement,
            violation_code=violation.code,
            concept=violation.concept,
            strategy="deterministic",
        )
    return None


def unresolved_location(
    plan: ProductionPlan,
    violation: ConstraintViolation,
) -> LocatedViolation | None:
    locations = locate_forbidden_violation(plan, violation)
    return locations[0] if len(locations) == 1 else None


def apply_fragment_patch(plan: ProductionPlan, patch: FragmentPatch) -> ProductionPlan:
    payload = plan.model_dump(mode="json")
    value = payload[patch.field]
    if patch.field == "tags":
        if not isinstance(value, list) or value.count(patch.original_fragment) != 1:
            raise ValueError("patch_original_fragment_must_match_once")
        payload[patch.field] = [
            patch.replacement_fragment if item == patch.original_fragment else item
            for item in value
        ]
    else:
        if not isinstance(value, str) or value.count(patch.original_fragment) != 1:
            raise ValueError("patch_original_fragment_must_match_once")
        if patch.operation == "append":
            if value == patch.replacement_fragment:
                return plan
            if not patch.replacement_fragment.startswith(patch.original_fragment):
                raise ValueError("append_patch_must_preserve_original_prefix")
        payload[patch.field] = value.replace(
            patch.original_fragment, patch.replacement_fragment, 1
        )
    return ProductionPlan.model_validate(payload)


def _registered_inventory(plan: ProductionPlan) -> tuple[set[str], set[str]]:
    text = "\n".join(
        [plan.style_prompt, " ".join(plan.tags), plan.vocal_direction, plan.arrangement_note]
    )
    concepts = {
        concept
        for concept in TERM_GROUPS
        if _positive_spans(text, concept)
    }
    lowered = text.lower()
    concepts.update(
        f"genre:{genre}"
        for genre in SECONDARY_GENRES
        if re.search(rf"(?<![a-z0-9]){re.escape(genre)}(?![a-z0-9])", lowered)
    )
    bpm_values = set(re.findall(r"\b\d{2,3}\s*bpm\b", lowered))
    return concepts, bpm_values


def validate_patched_plan(
    original: ProductionPlan,
    patched: ProductionPlan,
    *,
    patches: list[FragmentPatch],
    user_text: str,
    constraints: ExplicitProductionConstraints,
) -> PatchValidation:
    reconstructed = original
    try:
        for patch in patches:
            reconstructed = apply_fragment_patch(reconstructed, patch)
    except ValueError as error:
        return PatchValidation(
            accepted=False,
            reason=f"patch_reconstruction_failed:{error}",
            changed_fields=changed_fields(original, patched),
        )
    if reconstructed != patched:
        return PatchValidation(
            accepted=False,
            reason="patch_reconstruction_mismatch",
            changed_fields=changed_fields(original, patched),
        )
    allowed_fields = {patch.field for patch in patches}
    changed = changed_fields(original, patched)
    unauthorized = [field for field in changed if field not in allowed_fields]
    if unauthorized:
        return PatchValidation(
            accepted=False,
            reason=f"unauthorized_fields_changed:{unauthorized}",
            changed_fields=changed,
        )
    original_violations = check_plan_constraints(
        original, user_text=user_text, constraints=constraints
    )
    _, original_advisory = split_constraint_violations(original_violations)
    violations = check_plan_constraints(
        patched, user_text=user_text, constraints=constraints
    )
    hard, advisory = split_constraint_violations(violations)
    original_advisory_keys = {
        (row.code, row.concept, row.evidence) for row in original_advisory
    }
    new_advisory = [
        row
        for row in advisory
        if (row.code, row.concept, row.evidence) not in original_advisory_keys
    ]
    before_concepts, before_bpm = _registered_inventory(original)
    after_concepts, after_bpm = _registered_inventory(patched)
    allowed_new = _required_concepts(constraints)
    new_concepts = sorted((after_concepts - before_concepts) - allowed_new)
    new_bpm = sorted(after_bpm - before_bpm)
    if hard:
        reason = "hard_violation_remains"
    elif new_advisory:
        reason = "new_advisory"
    elif new_concepts:
        reason = "new_registered_concepts"
    elif new_bpm:
        reason = "new_bpm_values"
    else:
        reason = "accepted"
    return PatchValidation(
        accepted=reason == "accepted",
        reason=reason,
        changed_fields=changed,
        final_hard=hard,
        final_advisory=advisory,
        new_advisory=new_advisory,
        new_registered_concepts=new_concepts,
        new_bpm_values=new_bpm,
    )
