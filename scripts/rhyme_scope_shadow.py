"""实验专用的分段押韵影子分析与均衡反馈。

本模块不被生产 LangGraph、正式评分器或 API 导入。它用于比较不同
``rhyme_scope``，不会输出正式通过/失败结论，也不会修改现有评分口径。
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Literal

from pypinyin import Style, lazy_pinyin

from tools.lyrics_quality.thirteen_rhymes import (
    RHYME_GROUPS,
    normalize_final,
    normalize_rhyme_group,
    rhyme_group_for_final,
)


RhymeScope = Literal["global_target", "chorus_target", "section_consistency"]
TargetOrigin = Literal[
    "user_explicit",
    "system_suggested",
    "none",
    "legacy_unknown",
]
RHYME_SCOPES: tuple[RhymeScope, ...] = (
    "global_target",
    "chorus_target",
    "section_consistency",
)
TARGET_ORIGINS: tuple[TargetOrigin, ...] = (
    "user_explicit",
    "system_suggested",
    "none",
    "legacy_unknown",
)
THRESHOLD_LEVELS = (0.5, 0.75, 1.0)
RHYME_EXAMPLES = {"中东": ("东", "风", "梦", "星")}

_SECTION_PATTERN = re.compile(
    r"^\[(Intro|Verse|Pre-?Chorus|Chorus|Hook|Bridge|Outro|Interlude|Instrumental)"
    r"(?:\s+\d+)?\]$",
    re.I,
)
_CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class ShadowLine:
    section: str
    stanza_index: int
    line_no: int
    text: str
    ending_char: str | None
    final: str | None
    rhyme_group: str | None


def _canonical_section(raw: str) -> str:
    compact = raw.lower().replace("-", "")
    names = {
        "intro": "Intro",
        "verse": "Verse",
        "prechorus": "Pre-Chorus",
        "chorus": "Chorus",
        "hook": "Hook",
        "bridge": "Bridge",
        "outro": "Outro",
        "interlude": "Interlude",
        "instrumental": "Instrumental",
    }
    return names[compact]


def _analyse_ending(text: str) -> tuple[str | None, str | None, str | None]:
    characters = "".join(_CHINESE_PATTERN.findall(text))
    if not characters:
        return None, None, None
    finals = lazy_pinyin(
        characters,
        style=Style.FINALS,
        strict=True,
        v_to_u=True,
        errors="ignore",
    )
    final = normalize_final(finals[-1]) if finals else None
    group = rhyme_group_for_final(final) if final else None
    return characters[-1], final, group


def parse_shadow_lines(lyrics: str) -> list[ShadowLine]:
    """识别扩展段落标签，并为每个同类 stanza 单独编号。"""

    current_section: str | None = None
    current_stanza_index = 0
    stanza_counts: Counter[str] = Counter()
    lines: list[ShadowLine] = []
    for line_no, raw_line in enumerate(lyrics.splitlines(), start=1):
        text = raw_line.strip()
        match = _SECTION_PATTERN.fullmatch(text)
        if match:
            current_section = _canonical_section(match.group(1))
            stanza_counts[current_section] += 1
            current_stanza_index = stanza_counts[current_section]
            continue
        if not current_section or not text:
            continue
        ending_char, final, rhyme_group = _analyse_ending(text)
        lines.append(
            ShadowLine(
                section=current_section,
                stanza_index=current_stanza_index,
                line_no=line_no,
                text=text,
                ending_char=ending_char,
                final=final,
                rhyme_group=rhyme_group,
            )
        )
    return lines


def _normalise_template_line(text: str) -> str:
    return " ".join(text.split())


def _template_id(lines: list[ShadowLine]) -> str:
    canonical = "\n".join(_normalise_template_line(line.text) for line in lines)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _dominant_pattern(groups: list[str]) -> tuple[str | None, float, str]:
    if not groups:
        return None, 0.0, "none"
    counts = Counter(groups)
    dominant_group, dominant_count = counts.most_common(1)[0]
    dominant_rate = dominant_count / len(groups)
    if len(groups) == 2:
        return dominant_group, (1.0 if groups[0] == groups[1] else 0.0), "AA"
    if len(groups) == 4:
        aabb_rate = (int(groups[0] == groups[1]) + int(groups[2] == groups[3])) / 2
        abab_rate = (int(groups[0] == groups[2]) + int(groups[1] == groups[3])) / 2
        best_rate, pattern = max(
            ((dominant_rate, "dominant"), (aabb_rate, "AABB"), (abab_rate, "ABAB")),
            key=lambda item: item[0],
        )
        return dominant_group, best_rate, pattern
    return dominant_group, dominant_rate, "dominant"


def _target_applies(section: str, scope: RhymeScope) -> bool:
    if scope == "global_target":
        return section in {"Verse", "Pre-Chorus", "Chorus", "Hook", "Bridge"}
    if scope == "chorus_target":
        return section in {"Chorus", "Hook"}
    return False


def _normalise_target_origin(
    target: str | None,
    target_origin: TargetOrigin | None,
) -> TargetOrigin:
    """保留旧调用的未知来源，不把历史目标伪装成用户选择。"""

    origin: TargetOrigin = target_origin or (
        "legacy_unknown" if target else "none"
    )
    if origin not in TARGET_ORIGINS:
        raise ValueError(f"未知 target_origin：{origin}")
    if origin == "none" and target is not None:
        raise ValueError("target_origin=none 时不能同时提供目标辙")
    if origin in {"user_explicit", "system_suggested"} and target is None:
        raise ValueError(f"target_origin={origin} 时必须提供目标辙")
    return origin


def _target_policy(scope: RhymeScope, origin: TargetOrigin) -> str:
    if origin == "none":
        return "no_target"
    if origin == "legacy_unknown":
        return "provenance_unknown_not_eligible_for_method_decision"
    if scope == "section_consistency" and origin == "system_suggested":
        return "advisory_metadata_only_preserve_natural_section_rhyme"
    if scope == "section_consistency" and origin == "user_explicit":
        return "report_explicit_target_separately_without_automatic_rewrite"
    if origin == "user_explicit":
        return "explicit_target_evaluation"
    return "system_target_experiment_only"


def analyze_rhyme_scope(
    lyrics: str,
    hook: str,
    target_rhyme_group: str | None,
    scope: RhymeScope,
    target_origin: TargetOrigin | None = None,
) -> dict[str, object]:
    """返回可序列化影子指标；刻意不包含正式 ``passed`` 字段。"""

    if scope not in RHYME_SCOPES:
        raise ValueError(f"未知 rhyme_scope：{scope}")
    target = normalize_rhyme_group(target_rhyme_group) if target_rhyme_group else None
    if target_rhyme_group and target is None:
        raise ValueError(f"未知目标辙：{target_rhyme_group}")
    origin = _normalise_target_origin(target, target_origin)
    lines = parse_shadow_lines(lyrics)
    by_stanza: dict[tuple[str, int], list[ShadowLine]] = defaultdict(list)
    for line in lines:
        by_stanza[(line.section, line.stanza_index)].append(line)

    chorus_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    stanza_template_ids: dict[tuple[str, int], str] = {}
    for key, stanza_lines in by_stanza.items():
        section, stanza_index = key
        if section != "Chorus":
            continue
        template_id = _template_id(stanza_lines)
        stanza_template_ids[key] = template_id
        chorus_occurrences[template_id].append(
            {
                "stanza_index": stanza_index,
                "start_line_no": min(line.line_no for line in stanza_lines),
                "end_line_no": max(line.line_no for line in stanza_lines),
            }
        )

    stanzas: list[dict[str, object]] = []
    for (section, stanza_index), stanza_lines in by_stanza.items():
        analyzable = [line for line in stanza_lines if line.rhyme_group]
        groups = [str(line.rhyme_group) for line in analyzable]
        dominant_group, local_rate, local_pattern = _dominant_pattern(groups)
        target_match_count = (
            sum(line.rhyme_group == target for line in analyzable) if target else 0
        )
        target_rate = (
            target_match_count / len(analyzable) if target and analyzable else None
        )
        template_id = stanza_template_ids.get((section, stanza_index))
        occurrence_count = (
            len(chorus_occurrences[template_id]) if template_id is not None else 1
        )
        stanzas.append(
            {
                "section": section,
                "stanza_index": stanza_index,
                "template_id": template_id,
                "occurrence_count": occurrence_count,
                "start_line_no": min(line.line_no for line in stanza_lines),
                "end_line_no": max(line.line_no for line in stanza_lines),
                "line_count": len(stanza_lines),
                "analyzable_line_count": len(analyzable),
                "target_applies": bool(target and _target_applies(section, scope)),
                "target_match_count": target_match_count if target else None,
                "target_match_rate": round(target_rate, 3)
                if target_rate is not None
                else None,
                "target_thresholds": {
                    f"gte_{str(level).replace('.', '_')}": bool(
                        target_rate is not None and target_rate >= level
                    )
                    for level in THRESHOLD_LEVELS
                }
                if target_rate is not None
                else None,
                "dominant_rhyme_group": dominant_group,
                "dominant_group_rate": round(
                    groups.count(dominant_group) / len(groups), 3
                )
                if groups and dominant_group
                else None,
                "local_pattern": local_pattern,
                "local_consistency_rate": round(local_rate, 3),
                "lines": [asdict(line) for line in stanza_lines],
            }
        )

    hook_ending, hook_final, hook_group = _analyse_ending(hook)
    chorus_templates = [
        {
            "template_id": template_id,
            "occurrence_count": len(occurrences),
            "occurrences": occurrences,
        }
        for template_id, occurrences in chorus_occurrences.items()
    ]
    threshold_distribution: dict[str, dict[str, int]] = {}
    for section in ("Verse", "Chorus", "Bridge", "Outro"):
        section_stanzas = [item for item in stanzas if item["section"] == section]
        if not section_stanzas:
            continue
        threshold_distribution[section] = {
            f"gte_{str(level).replace('.', '_')}": sum(
                bool(item["target_thresholds"][f"gte_{str(level).replace('.', '_')}"])
                for item in section_stanzas
                if item["target_thresholds"] is not None
            )
            for level in THRESHOLD_LEVELS
        }
        threshold_distribution[section]["stanza_count"] = len(section_stanzas)
    return {
        "shadow_only": True,
        "rhyme_scope": scope,
        "target_rhyme_group": target,
        "target_origin": origin,
        "target_policy": _target_policy(scope, origin),
        "stanzas": stanzas,
        "chorus_templates": chorus_templates,
        "hook": {
            "text": hook,
            "ending_char": hook_ending,
            "final": hook_final,
            "rhyme_group": hook_group,
            "matches_target": hook_group == target if target else None,
            "target_is_user_explicit": (
                True
                if origin == "user_explicit"
                else False
                if origin in {"system_suggested", "none"}
                else None
            ),
        },
        "target_threshold_distribution": threshold_distribution,
        "limitations": [
            "Exact normalized text only; near-duplicate choruses are not merged.",
            "Threshold slices are descriptive and do not define a pass gate.",
            "Bridge and Outro naturalness are not inferred from rhyme metrics.",
            "Target origin changes interpretation, not production routing.",
        ],
    }


def _stanza_label(stanza: dict[str, object]) -> str:
    return f"{stanza['section']} {stanza['stanza_index']}"


def _mapping_text(target: str | None) -> str:
    if not target:
        return ""
    finals = "、".join(sorted(RHYME_GROUPS[target]))
    examples = RHYME_EXAMPLES.get(target)
    example_text = f"例字：{'、'.join(examples)}；" if examples else ""
    return (
        f"目标辙“{target}”对应普通话韵母 {finals}；{example_text}"
        "例字只用于理解分类，不要求直接写入歌词。"
    )


def _mismatch_candidates(
    stanza: dict[str, object],
    expected_group: str,
    limit: int,
    instruction: str,
) -> list[str]:
    candidates: list[str] = []
    seen_text: set[str] = set()
    for line in stanza["lines"]:
        text = str(line["text"])
        if line["rhyme_group"] == expected_group or text in seen_text:
            continue
        seen_text.add(text)
        candidates.append(
            f"第 {line['line_no']} 行“{text}”末字“{line['ending_char']}”属于"
            f"{line['rhyme_group'] or '未分类'}，{instruction}"
        )
        if len(candidates) >= limit:
            break
    return candidates


def build_balanced_feedback(
    lyrics: str,
    hook: str,
    target_rhyme_group: str | None,
    scope: RhymeScope,
    issue_budget: int,
    target_origin: TargetOrigin | None = None,
) -> dict[str, object]:
    """按段落/模板轮转选择问题，避免歌词前部占满反馈。"""

    if issue_budget < 1:
        raise ValueError("issue_budget 必须大于 0")
    report = analyze_rhyme_scope(
        lyrics,
        hook,
        target_rhyme_group,
        scope,
        target_origin=target_origin,
    )
    target = report["target_rhyme_group"]
    stanzas = list(report["stanzas"])
    summaries = ["段落押韵影子摘要："]
    seen_chorus_templates: set[str] = set()
    for stanza in stanzas:
        template_id = stanza["template_id"]
        if stanza["section"] == "Chorus" and template_id in seen_chorus_templates:
            continue
        if stanza["section"] == "Chorus" and template_id:
            seen_chorus_templates.add(str(template_id))
        label = _stanza_label(stanza)
        repeat = (
            f"，相同模板共 {stanza['occurrence_count']} 次"
            if stanza["section"] == "Chorus" and int(stanza["occurrence_count"]) > 1
            else ""
        )
        if stanza["target_applies"]:
            summaries.append(
                f"- {label}：目标{target}命中 {stanza['target_match_count']}/"
                f"{stanza['analyzable_line_count']}{repeat}。"
            )
        else:
            summaries.append(
                f"- {label}：主导辙{stanza['dominant_rhyme_group'] or '无'}，"
                f"局部一致性 {float(stanza['local_consistency_rate']):.0%}{repeat}；"
                "不据此判断自然度。"
            )

    grouped_candidates: list[tuple[str, int, list[str]]] = []
    hook_report = report["hook"]
    if target and hook_report["matches_target"] is False and scope != "section_consistency":
        if scope == "global_target":
            hook_instruction = (
                "允许调整 Hook 句尾以匹配目标辙，但保留核心含义、副歌重复位置和记忆点。"
            )
        else:
            hook_instruction = (
                "这是副歌目标辙的修改建议，不是硬失败；仅在不损害核心含义和记忆点时调整句尾。"
            )
        grouped_candidates.append(("Hook", 0, [f"Hook“{hook}”：{hook_instruction}"]))

    processed_templates: set[str] = set()
    for stanza in stanzas:
        section = str(stanza["section"])
        template_id = stanza["template_id"]
        if section == "Chorus" and template_id in processed_templates:
            continue
        if section == "Chorus" and template_id:
            processed_templates.add(str(template_id))
        label = _stanza_label(stanza)
        candidates: list[str] = []
        priority = 4
        if stanza["target_applies"] and target:
            limit = 3 if section == "Chorus" else 2 if section == "Verse" else 1
            priority = 1 if section == "Chorus" else 2 if section == "Verse" else 3
            candidates = _mismatch_candidates(
                stanza,
                str(target),
                limit,
                f"建议保持原意后改为{target}辙。",
            )
        elif scope == "chorus_target" and section == "Verse":
            dominant = stanza["dominant_rhyme_group"]
            if dominant and float(stanza["local_consistency_rate"]) < 1.0:
                priority = 2
                candidates = _mismatch_candidates(
                    stanza,
                    str(dominant),
                    2,
                    f"只需参考本段主导的{dominant}辙改善局部一致性，不必改成副歌目标辙。",
                )
        elif scope == "section_consistency" and section in {
            "Verse",
            "Pre-Chorus",
            "Chorus",
        }:
            dominant = stanza["dominant_rhyme_group"]
            if dominant and float(stanza["local_consistency_rate"]) < 1.0:
                limit = 3 if section == "Chorus" else 2
                priority = 1 if section == "Chorus" else 2
                candidates = _mismatch_candidates(
                    stanza,
                    str(dominant),
                    limit,
                    f"仅参考本段自然形成的{dominant}辙改善局部一致性，不追随系统目标辙。",
                )
        if candidates:
            if section == "Chorus" and int(stanza["occurrence_count"]) > 1:
                candidates.append(
                    f"{label} 的相同模板出现 {stanza['occurrence_count']} 次，"
                    "修改后必须同步全部位置，并重新检查各次内容一致。"
                )
            grouped_candidates.append((label, priority, candidates))

    grouped_candidates.sort(key=lambda item: (item[1], item[0]))
    selected: list[str] = []
    selected_units: set[str] = set()
    max_rounds = max((len(items) for _, _, items in grouped_candidates), default=0)
    for round_index in range(max_rounds):
        for unit, _, candidates in grouped_candidates:
            if round_index >= len(candidates) or len(selected) >= issue_budget:
                continue
            selected.append(candidates[round_index])
            selected_units.add(unit)

    total_candidates = sum(len(items) for _, _, items in grouped_candidates)
    problem_units = [unit for unit, _, _ in grouped_candidates]
    uncovered_units = [unit for unit in problem_units if unit not in selected_units]
    show_mapping = bool(
        target
        and not (
            scope == "section_consistency"
            and report["target_origin"] == "system_suggested"
        )
    )
    instructions = [
        "修改边界：",
        f"- {_mapping_text(str(target))}"
        if show_mapping
        else "- 系统建议目标的详细映射在本轮隐藏。"
        if target
        else "- 未指定目标辙。",
    ]
    if scope == "global_target":
        instructions.append("- Verse、Pre-Chorus、Chorus、Hook、Bridge 使用目标辙；Outro 只报告。")
    elif scope == "chorus_target":
        instructions.append(
            "- 目标辙只用于 Chorus；Hook 是否匹配只作为建议，Verse 保持局部押韵，Bridge 与 Outro 只报告。"
        )
    else:
        if report["target_origin"] == "system_suggested" and target:
            instructions.append(
                f"- 系统曾建议目标辙“{target}”，但这里只作为记录；"
                "不得为命中它而破坏已经自然成立的段内押韵。"
            )
        elif report["target_origin"] == "user_explicit" and target:
            instructions.append(
                f"- 用户明确目标辙“{target}”另行报告；本轮只检查段内一致性，"
                "不自动强制改写。"
            )
        instructions.append(
            "- 不指定新的目标辙，只改善各 Verse/Chorus 的局部押韵一致性；"
            "局部一致性已经完整时保持原文。"
        )
    feedback_lines = [
        "请以上一版歌词为基础做最小必要修改，不要为了押韵重写整首。",
        *instructions,
        *summaries,
        "代表性问题：",
        *(f"- {item}" for item in selected),
    ]
    return {
        "rhyme_scope": scope,
        "target_origin": report["target_origin"],
        "target_policy": report["target_policy"],
        "issue_budget": issue_budget,
        "selected_issue_count": len(selected),
        "omitted_issue_count": max(total_candidates - len(selected), 0),
        "problem_units": problem_units,
        "covered_units": sorted(selected_units),
        "uncovered_units": uncovered_units,
        "estimated_characters": len("\n".join(feedback_lines)),
        "text": "\n".join(feedback_lines),
        "shadow_report": report,
    }


def build_section_consistency_triage(
    lyrics: str,
    hook: str,
    system_target_rhyme_group: str | None,
) -> dict[str, object]:
    """对系统建议目标做离线分流；不自动把描述性比例升级成失败门槛。"""

    report = analyze_rhyme_scope(
        lyrics,
        hook,
        system_target_rhyme_group,
        "section_consistency",
        target_origin="system_suggested"
        if system_target_rhyme_group
        else "none",
    )
    unique_choruses: list[dict[str, object]] = []
    seen_templates: set[str] = set()
    for stanza in report["stanzas"]:
        if stanza["section"] != "Chorus":
            continue
        template_id = str(stanza["template_id"])
        if template_id in seen_templates:
            continue
        seen_templates.add(template_id)
        unique_choruses.append(
            {
                "template_id": template_id,
                "occurrence_count": stanza["occurrence_count"],
                "line_count": stanza["analyzable_line_count"],
                "local_pattern": stanza["local_pattern"],
                "local_consistency_rate": stanza["local_consistency_rate"],
                "dominant_rhyme_group": stanza["dominant_rhyme_group"],
                "system_target_match_rate_for_reference_only": stanza[
                    "target_match_rate"
                ],
            }
        )
    rates = [float(item["local_consistency_rate"]) for item in unique_choruses]
    all_complete = bool(rates) and all(rate == 1.0 for rate in rates)
    return {
        "shadow_only": True,
        "target_origin": report["target_origin"],
        "target_policy": report["target_policy"],
        "system_target_rhyme_group_for_reference_only": report[
            "target_rhyme_group"
        ],
        "unique_chorus_template_count": len(unique_choruses),
        "chorus_templates": unique_choruses,
        "minimum_chorus_local_consistency": min(rates) if rates else None,
        "observed_threshold_slices": {
            f"gte_{str(level).replace('.', '_')}": sum(rate >= level for rate in rates)
            for level in THRESHOLD_LEVELS
        },
        "triage_action": (
            "preserve_no_rewrite"
            if all_complete
            else "human_review_before_any_rewrite"
        ),
        "automatic_rewrite_authorized": False,
        "formal_pass_decision": None,
        "limitations": [
            "A 1.0 local pattern is preserved in this preflight, not admitted as a production threshold.",
            "Values below 1.0 require human review before any rewrite call.",
            "Lyrics text metrics do not judge naturalness or singability.",
        ],
    }


def choose_feedback_budget(
    cases: list[tuple[str, str, str | None]],
    budgets: tuple[int, ...] = (10, 12, 16),
) -> dict[str, object]:
    """选择能覆盖全部问题单元的最小候选预算，并保留各预算测量。"""

    measurements: list[dict[str, object]] = []
    for budget in budgets:
        packages = [
            build_balanced_feedback(lyrics, hook, target, scope, budget)
            for lyrics, hook, target in cases
            for scope in ("global_target", "chorus_target")
        ]
        uncovered = sum(len(package["uncovered_units"]) for package in packages)
        measurements.append(
            {
                "issue_budget": budget,
                "packages": len(packages),
                "uncovered_problem_units": uncovered,
                "max_estimated_characters": max(
                    int(package["estimated_characters"]) for package in packages
                ),
                "mean_estimated_characters": round(
                    sum(int(package["estimated_characters"]) for package in packages)
                    / len(packages),
                    1,
                ),
            }
        )
    selected = next(
        (
            int(item["issue_budget"])
            for item in measurements
            if item["uncovered_problem_units"] == 0
        ),
        max(budgets),
    )
    return {
        "candidate_budgets": list(budgets),
        "selected_issue_budget": selected,
        "selection_rule": "smallest budget covering every problem unit; otherwise maximum candidate",
        "measurements": measurements,
    }


def section_change_metrics(
    source_lyrics: str,
    output_lyrics: str,
    target_sections: set[str],
) -> dict[str, object]:
    """按源稿行对齐估计各段改动；插入行单列，不伪造段落归属。"""

    source = parse_shadow_lines(source_lyrics)
    output = parse_shadow_lines(output_lyrics)
    matcher = SequenceMatcher(
        a=[line.text for line in source],
        b=[line.text for line in output],
        autojunk=False,
    )
    changed_source_indexes: set[int] = set()
    inserted_line_count = 0
    deleted_line_count = 0
    for tag, source_start, source_end, output_start, output_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_source_indexes.update(range(source_start, source_end))
        if tag in {"insert", "replace"}:
            inserted_line_count += max((output_end - output_start) - (source_end - source_start), 0)
        if tag in {"delete", "replace"}:
            deleted_line_count += max((source_end - source_start) - (output_end - output_start), 0)
    changed_by_section = Counter(source[index].section for index in changed_source_indexes)
    non_target_changed = sum(
        count for section, count in changed_by_section.items() if section not in target_sections
    )
    changed_source_count = len(changed_source_indexes)
    return {
        "source_line_count": len(source),
        "output_line_count": len(output),
        "changed_source_line_count": changed_source_count,
        "changed_source_lines_by_section": dict(sorted(changed_by_section.items())),
        "non_target_changed_source_line_count": non_target_changed,
        "non_target_changed_source_line_rate": round(
            non_target_changed / changed_source_count, 3
        )
        if changed_source_count
        else 0.0,
        "inserted_unattributed_line_count": inserted_line_count,
        "deleted_line_count": deleted_line_count,
        "line_sequence_similarity": round(matcher.ratio(), 3),
        "limitations": "Section attribution follows aligned source lines; inserted lines are reported separately.",
    }
