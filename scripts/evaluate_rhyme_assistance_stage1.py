"""第一阶段韵脚辅助对照：现有反馈 vs 固定辙部映射。

默认只打印计划；显式传入 ``--execute`` 才调用真实 DeepSeek。实验固定使用
D9.4 源池中两个“双候选均未过押韵门槛”的主题，共 4 份原始草稿、8 次逻辑
改写。它不读取 390 词候选、不加载 RAG 或音频模型，也不修改生产提示词。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

from agents.lyricist_agent import LyricistAgent
from agents.schemas import LyricDraft, SongRequirement
from services.config import get_settings
from services.llm import DeepSeekLLM
from scripts.rhyme_scope_shadow import (
    analyze_rhyme_scope,
    build_balanced_feedback,
    build_section_consistency_triage,
    choose_feedback_budget,
    section_change_metrics,
)
from tools.lyrics_quality.parser import parse_lyric_lines
from tools.lyrics_quality.scorer import format_revision_feedback, score_lyrics
from tools.lyrics_quality.thirteen_rhymes import RHYME_GROUPS


SOURCE_ROOT = Path("data/output/evaluations/d9-4-source-pool")
OUTPUT_ROOT = Path("data/output/evaluations")
SOURCE_FAMILIES = ("comeback-rock", "summer-storm-dancepop")
PERSONAS = ("structured", "imagery")
VARIANTS = ("current_feedback", "fixed_mapping")
B2_VARIANT = "fixed_mapping_b2"
B3_VARIANT = "balanced_global_b3"
B4_VARIANT = "balanced_chorus_b4"
B3_B4_VARIANTS = (B3_VARIANT, B4_VARIANT)
SECTION_CONSISTENCY_VARIANT = "section_consistency_b5"
SECTION_CONSISTENCY_FAMILIES = (
    "campus-graduation-rap",
    "late-cafe-jazz",
    "morning-commute-pop",
    "night-running-electronic",
)
RHYME_EXAMPLES = {"中东": ("东", "风", "梦", "星")}
BLIND_SEED = 20260913
DEFAULT_STAGE1_BASELINE = Path(
    "data/output/evaluations/"
    "rhyme-assistance-stage1-4be77077fd4b48bf82cab57f1e8c4285"
)


@dataclass(frozen=True)
class SourceCase:
    source_id: str
    family_id: str
    persona_id: str
    requirement: SongRequirement
    draft: LyricDraft


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_cases(
    family_ids: tuple[str, ...],
    source_root: Path,
) -> tuple[SourceCase, ...]:
    cases: list[SourceCase] = []
    for family_id in family_ids:
        family_dir = source_root / family_id
        requirement = SongRequirement.model_validate_json(
            (family_dir / "requirement.json").read_text(encoding="utf-8")
        )
        candidates = json.loads(
            (family_dir / "lyric_candidates.json").read_text(encoding="utf-8")
        )
        by_persona = {item["persona_id"]: item for item in candidates}
        for persona_id in PERSONAS:
            if persona_id not in by_persona:
                raise ValueError(f"{family_id} 缺少 {persona_id} 候选")
            cases.append(
                SourceCase(
                    source_id=f"{family_id}--{persona_id}",
                    family_id=family_id,
                    persona_id=persona_id,
                    requirement=requirement,
                    draft=LyricDraft.model_validate(by_persona[persona_id]["draft"]),
                )
            )
    return tuple(cases)


def load_source_cases(source_root: Path = SOURCE_ROOT) -> tuple[SourceCase, ...]:
    """只读取预先锁定的两个失败主题，不扫描或挑选新结果。"""

    return _load_cases(SOURCE_FAMILIES, source_root)


def load_section_consistency_cases(
    source_root: Path = SOURCE_ROOT,
) -> tuple[SourceCase, ...]:
    """读取对押韵方法留出的四个 D9.4 验证家族，共八份原始稿。"""

    overlap = set(SECTION_CONSISTENCY_FAMILIES) & set(SOURCE_FAMILIES)
    if overlap:
        raise ValueError(f"section consistency 验证集与 B3/B4 开发集重叠：{overlap}")
    return _load_cases(SECTION_CONSISTENCY_FAMILIES, source_root)


def build_mapping_note(target_rhyme_group: str | None) -> str:
    """由项目确定性映射生成说明；不读取候选词库。"""

    if not target_rhyme_group or target_rhyme_group not in RHYME_GROUPS:
        raise ValueError("第一阶段要求已有标准目标辙名")
    finals = "、".join(sorted(RHYME_GROUPS[target_rhyme_group]))
    examples = RHYME_EXAMPLES.get(target_rhyme_group)
    example_text = f"正确例字：{'、'.join(examples)}。" if examples else ""
    return (
        f"补充确定性辙部知识：目标辙“{target_rhyme_group}”对应普通话韵母"
        f" {finals}。{example_text}其他韵母不属于该目标辙。"
        "这些例字只用于理解分类，不要求直接写入歌词。"
        "只修改确实不匹配的句子，保留原主题、Hook、段落和原本正确的行。"
    )


def build_b2_mapping_note(target_rhyme_group: str | None) -> str:
    """修正 B 组的 Hook 冲突，同时不增加候选词或 RAG 信息。"""

    if not target_rhyme_group or target_rhyme_group not in RHYME_GROUPS:
        raise ValueError("B2 要求已有标准目标辙名")
    finals = "、".join(sorted(RHYME_GROUPS[target_rhyme_group]))
    examples = RHYME_EXAMPLES.get(target_rhyme_group)
    example_text = f"正确例字：{'、'.join(examples)}。" if examples else ""
    return (
        f"补充确定性辙部知识：目标辙“{target_rhyme_group}”对应普通话韵母"
        f" {finals}。{example_text}其他韵母不属于该目标辙。"
        "这些例字只用于理解分类，不要求直接写入歌词。"
        "如果原 Hook 的句尾不属于目标辙，允许调整 Hook 句尾；"
        "必须保留 Hook 的核心含义、副歌中的重复位置和记忆点。"
        "如果原 Hook 已属于目标辙，则原样保留。"
        "其余只修改确实不匹配的句子，保留原主题、段落和原本正确的行。"
    )


def build_feedback(case: SourceCase, variant: str) -> str:
    report = score_lyrics(
        case.draft.lyrics,
        case.draft.hook,
        case.requirement.duration_seconds,
        case.requirement.target_rhyme_group,
        case.requirement.theme,
        case.requirement.keywords,
    )
    feedback = format_revision_feedback(report)
    if variant == "current_feedback":
        return feedback
    if variant == "fixed_mapping":
        return f"{feedback}\n\n{build_mapping_note(case.requirement.target_rhyme_group)}"
    if variant == B2_VARIANT:
        return f"{feedback}\n\n{build_b2_mapping_note(case.requirement.target_rhyme_group)}"
    raise ValueError(f"未知实验变体：{variant}")


def quality_metrics(draft: LyricDraft, requirement: SongRequirement) -> dict[str, object]:
    report = score_lyrics(
        draft.lyrics,
        draft.hook,
        requirement.duration_seconds,
        requirement.target_rhyme_group,
        requirement.theme,
        requirement.keywords,
    )
    target = requirement.target_rhyme_group
    analyzable = [line for line in report.rhyme.lines if line.rhyme_group]
    matched = sum(line.rhyme_group == target for line in analyzable)
    return {
        "passed": report.passed,
        "total_score": report.total_score,
        "rhyme_score": report.rhyme.score,
        "structure_score": report.structure.score,
        "target_rhyme_group": target,
        "target_match_count": matched,
        "analyzable_line_count": len(analyzable),
        "target_match_rate": round(matched / len(analyzable), 3)
        if analyzable
        else None,
        "issue_codes": [issue.code for issue in report.issues],
        "quality_report": report.to_dict(),
    }


def line_change_metrics(source: LyricDraft, output: LyricDraft) -> dict[str, object]:
    """记录改写幅度；它只衡量文本保留，不评价改写是否自然。"""

    source_lines = [line.text for line in parse_lyric_lines(source.lyrics)]
    output_lines = [line.text for line in parse_lyric_lines(output.lyrics)]
    matcher = SequenceMatcher(a=source_lines, b=output_lines, autojunk=False)
    unchanged = sum(block.size for block in matcher.get_matching_blocks())
    changed = max(len(source_lines), len(output_lines)) - unchanged
    return {
        "source_line_count": len(source_lines),
        "output_line_count": len(output_lines),
        "unchanged_line_count": unchanged,
        "changed_line_count": changed,
        "line_sequence_similarity": round(matcher.ratio(), 3),
        "original_hook_preserved_exactly": source.hook in output.lyrics,
        "hook_changed": source.hook != output.hook,
    }


def build_call_plan(cases: tuple[SourceCase, ...]) -> list[tuple[SourceCase, str]]:
    """交替 A/B 与 B/A，降低固定调用顺序对单次筛查的影响。"""

    plan: list[tuple[SourceCase, str]] = []
    for index, case in enumerate(cases):
        order = VARIANTS if index % 2 == 0 else tuple(reversed(VARIANTS))
        plan.extend((case, variant) for variant in order)
    return plan


def build_b3_b4_call_plan(
    cases: tuple[SourceCase, ...],
) -> list[tuple[SourceCase, str]]:
    """同一源稿内交替 B3/B4 顺序，减少固定调用顺序偏置。"""

    plan: list[tuple[SourceCase, str]] = []
    for index, case in enumerate(cases):
        order = B3_B4_VARIANTS if index % 2 == 0 else tuple(reversed(B3_B4_VARIANTS))
        plan.extend((case, variant) for variant in order)
    return plan


def _scope_for_variant(variant: str) -> str:
    if variant == B3_VARIANT:
        return "global_target"
    if variant == B4_VARIANT:
        return "chorus_target"
    raise ValueError(f"变体 {variant} 没有影子 scope")


def build_b3_b4_plan_summary(cases: tuple[SourceCase, ...]) -> dict[str, object]:
    """离线测量候选反馈预算；不调用 LLM。"""

    budget = choose_feedback_budget(
        [
            (case.draft.lyrics, case.draft.hook, case.requirement.target_rhyme_group)
            for case in cases
        ]
    )
    selected = int(budget["selected_issue_budget"])
    coverage: list[dict[str, object]] = []
    for case in cases:
        for variant in B3_B4_VARIANTS:
            scope = _scope_for_variant(variant)
            package = build_balanced_feedback(
                case.draft.lyrics,
                case.draft.hook,
                case.requirement.target_rhyme_group,
                scope,
                selected,
            )
            coverage.append(
                {
                    "source_id": case.source_id,
                    "variant": variant,
                    "rhyme_scope": scope,
                    "selected_issue_count": package["selected_issue_count"],
                    "omitted_issue_count": package["omitted_issue_count"],
                    "problem_units": package["problem_units"],
                    "uncovered_units": package["uncovered_units"],
                    "estimated_characters": package["estimated_characters"],
                }
            )
    return {
        "logical_calls": len(cases) * 2,
        "variants": list(B3_B4_VARIANTS),
        "production_behavior_changed": False,
        "shadow_only": True,
        "feedback_budget_selection": budget,
        "feedback_coverage": coverage,
        "call_plan": [
            {"source_id": case.source_id, "variant": variant}
            for case, variant in build_b3_b4_call_plan(cases)
        ],
    }


def build_section_consistency_plan_summary(
    cases: tuple[SourceCase, ...],
) -> dict[str, object]:
    """生成 B5 离线分流；不把 75% 或 100% 写成生产通过门槛。"""

    source_rows: list[dict[str, object]] = []
    for case in cases:
        triage = build_section_consistency_triage(
            case.draft.lyrics,
            case.draft.hook,
            case.requirement.target_rhyme_group,
        )
        source_rows.append(
            {
                "source_id": case.source_id,
                "family_id": case.family_id,
                "persona_id": case.persona_id,
                "target_origin": "system_suggested",
                "system_target_rhyme_group_for_reference_only": (
                    case.requirement.target_rhyme_group
                ),
                "triage_action": triage["triage_action"],
                "automatic_rewrite_authorized": False,
                "minimum_chorus_local_consistency": triage[
                    "minimum_chorus_local_consistency"
                ],
                "chorus_templates": triage["chorus_templates"],
            }
        )
    preserve = [
        row for row in source_rows if row["triage_action"] == "preserve_no_rewrite"
    ]
    human_review = [
        row
        for row in source_rows
        if row["triage_action"] == "human_review_before_any_rewrite"
    ]
    return {
        "stage": "section-consistency-heldout-preflight",
        "variant": SECTION_CONSISTENCY_VARIANT,
        "source_drafts": len(cases),
        "source_families": list(SECTION_CONSISTENCY_FAMILIES),
        "excluded_development_families": list(SOURCE_FAMILIES),
        "target_origin": "system_suggested",
        "target_origin_evidence": (
            "The source pool was generated by RequirementAgent from theme-only plans; "
            "users did not explicitly choose rhyme groups."
        ),
        "system_target_groups_for_reference_only": sorted(
            {
                str(case.requirement.target_rhyme_group)
                for case in cases
                if case.requirement.target_rhyme_group
            }
        ),
        "original_source_control_included": True,
        "production_behavior_changed": False,
        "shadow_only": True,
        "formal_threshold_selected": False,
        "planned_llm_calls_now": 0,
        "preserve_no_rewrite_count": len(preserve),
        "human_review_before_any_rewrite_count": len(human_review),
        "maximum_future_calls_after_human_confirmation": len(human_review),
        "source_rows": source_rows,
        "future_comparison_if_approved": {
            "Z": "original source draft; no API call",
            "X": "section-consistency minimal rewrite; only for human-confirmed defects",
            "Y": "explicit-target rewrite; omitted until a genuinely user-explicit sample exists",
        },
        "formal_method_decision": "not_allowed_from_offline_preflight",
        "limitations": [
            "Held out from B3/B4 rhyme-method design, but reused from the D9.4 source pool.",
            "A local-consistency value is descriptive and does not judge naturalness.",
            "Rows below 1.0 require human review; no rewrite is automatically authorized.",
            "No genuinely user-explicit rhyme-target sample exists in this set.",
        ],
    }


def build_section_consistency_review_markdown(
    cases: tuple[SourceCase, ...],
    plan: dict[str, object],
) -> str:
    """展示只需人工判断的原始副歌；此时还没有任何改写候选。"""

    case_by_id = {case.source_id: case for case in cases}
    rows = [
        row
        for row in plan["source_rows"]
        if row["triage_action"] == "human_review_before_any_rewrite"
    ]
    markdown = [
        "# Section consistency 人工预检",
        "",
        "这里只判断原稿副歌的 75% 局部一致性是否真的需要修改。",
        "系统建议目标仅供背景参考，不是本轮修改目标；当前没有调用 DeepSeek。",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        case = case_by_id[str(row["source_id"])]
        report = analyze_rhyme_scope(
            case.draft.lyrics,
            case.draft.hook,
            case.requirement.target_rhyme_group,
            "section_consistency",
            target_origin="system_suggested",
        )
        markdown.extend(
            [
                f"## {index}. {case.source_id}",
                "",
                f"- 主题：{case.requirement.theme}",
                f"- Hook：{case.draft.hook}",
                f"- 系统建议目标（仅记录）：{case.requirement.target_rhyme_group}",
                "- 原稿控制：Z，不调用 API",
                "",
            ]
        )
        seen_templates: set[str] = set()
        for stanza in report["stanzas"]:
            if stanza["section"] != "Chorus":
                continue
            template_id = str(stanza["template_id"])
            if template_id in seen_templates:
                continue
            seen_templates.add(template_id)
            markdown.extend(
                [
                    f"### Chorus 模板（重复 {stanza['occurrence_count']} 次）",
                    "",
                    f"局部模式：{stanza['local_pattern']}；"
                    f"一致性：{float(stanza['local_consistency_rate']):.0%}；"
                    f"主导辙：{stanza['dominant_rhyme_group']}。",
                    "",
                ]
            )
            for line in stanza["lines"]:
                markdown.append(
                    f"- {line['text']}　`{line['rhyme_group'] or '未分类'}`"
                )
            markdown.extend(
                [
                    "",
                    "人工判断：保持原稿 / 确需局部调整 / 无法仅凭文本判断",
                    "",
                    "备注：",
                    "",
                ]
            )
    markdown.extend(
        [
            "## 解释边界",
            "",
            "75% 只是筛查切片，不是失败阈值。只有人工确认存在听感或文本问题后，"
            "才规划 section-consistency 最小改写；否则保持 Z 原稿。",
            "",
        ]
    )
    return "\n".join(markdown)


def build_aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    completed = [item for item in results if item["status"] == "completed"]
    by_variant: dict[str, dict[str, object]] = {}
    for variant in VARIANTS:
        items = [item for item in completed if item["variant"] == variant]
        by_variant[variant] = {
            "completed": len(items),
            "rhyme_pass_count": sum(
                float(item["output_metrics"]["rhyme_score"]) >= 70 for item in items
            ),
            "full_pass_count": sum(bool(item["output_metrics"]["passed"]) for item in items),
            "mean_rhyme_score": round(
                sum(float(item["output_metrics"]["rhyme_score"]) for item in items)
                / len(items),
                1,
            )
            if items
            else None,
            "mean_target_match_rate": round(
                sum(float(item["output_metrics"]["target_match_rate"]) for item in items)
                / len(items),
                3,
            )
            if items
            else None,
            "mean_changed_lines": round(
                sum(float(item["change_metrics"]["changed_line_count"]) for item in items)
                / len(items),
                1,
            )
            if items
            else None,
        }

    pairs: list[dict[str, object]] = []
    for source_id in dict.fromkeys(str(item["source_id"]) for item in results):
        pair = {
            str(item["variant"]): item
            for item in completed
            if item["source_id"] == source_id
        }
        if set(pair) != set(VARIANTS):
            continue
        current = pair["current_feedback"]["output_metrics"]
        mapping = pair["fixed_mapping"]["output_metrics"]
        pairs.append(
            {
                "source_id": source_id,
                "current_rhyme_score": current["rhyme_score"],
                "mapping_rhyme_score": mapping["rhyme_score"],
                "current_target_match_rate": current["target_match_rate"],
                "mapping_target_match_rate": mapping["target_match_rate"],
                "current_rhyme_passed": float(current["rhyme_score"]) >= 70,
                "mapping_rhyme_passed": float(mapping["rhyme_score"]) >= 70,
                "mapping_additional_rescue": (
                    float(mapping["rhyme_score"]) >= 70
                    and float(current["rhyme_score"]) < 70
                ),
            }
        )
    return {
        "logical_calls_planned": len(results),
        "logical_calls_completed": len(completed),
        "logical_calls_failed": len(results) - len(completed),
        "by_variant": by_variant,
        "paired_results": pairs,
        "mapping_additional_rescue_count": sum(
            bool(item["mapping_additional_rescue"]) for item in pairs
        ),
        "human_naturalness_review_status": "pending",
    }


def write_blind_review(
    experiment_dir: Path,
    cases: tuple[SourceCase, ...],
    results: list[dict[str, object]],
    variants: tuple[str, str] = VARIANTS,
    title: str = "第一阶段韵脚辅助匿名复核",
) -> None:
    """生成不暴露 A/B 身份的 X/Y 人工评审材料和单独密钥。"""

    rng = random.Random(BLIND_SEED)
    cards: list[dict[str, object]] = []
    keys: list[dict[str, str]] = []
    markdown = [
        f"# {title}",
        "",
        "只比较 X/Y 的自然度、原意保留和是否为押韵大改；不要参考规则分。",
        "每组建议选 X、Y 或平局，并各给 1～5 分自然度。",
    ]
    completed = [item for item in results if item["status"] == "completed"]
    for case in cases:
        pair = {
            str(item["variant"]): item
            for item in completed
            if item["source_id"] == case.source_id
        }
        if set(pair) != set(variants):
            continue
        labels = ["X", "Y"]
        rng.shuffle(labels)
        mapping = dict(zip(variants, labels))
        keys.extend(
            {"source_id": case.source_id, "label": label, "variant": variant}
            for variant, label in mapping.items()
        )
        markdown.extend(
            [
                "",
                f"## {case.source_id}",
                "",
                f"主题：{case.requirement.theme}",
                "",
                "### 原稿",
                "",
                case.draft.lyrics,
            ]
        )
        for label in ("X", "Y"):
            variant = next(name for name, value in mapping.items() if value == label)
            item = pair[variant]
            cards.append(
                {
                    "source_id": case.source_id,
                    "theme": case.requirement.theme,
                    "candidate_label": label,
                    "lyrics": item["output_draft"]["lyrics"],
                    "naturalness_1_to_5": None,
                    "meaning_preservation_1_to_5": None,
                    "forced_rhyme_problem": None,
                    "preferred_candidate": None,
                    "notes": "",
                }
            )
            markdown.extend(
                [
                    "",
                    f"### 候选 {label}",
                    "",
                    str(item["output_draft"]["lyrics"]),
                    "",
                    f"- {label} 自然度（1～5）：",
                    f"- {label} 原意保留（1～5）：",
                    f"- {label} 是否有为押韵硬改：",
                ]
            )
        markdown.extend(["", "- 本组偏好（X / Y / 平局）：", "- 备注："])
    _write_json(experiment_dir / "blind_review_cards.json", cards)
    _write_json(experiment_dir / "blind_review_key.json", keys)
    (experiment_dir / "blind_review.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )


def load_current_feedback_baseline(baseline_dir: Path) -> list[dict[str, object]]:
    """读取并验证上一轮 A 组，避免补跑 B2 时悄悄更换对照。"""

    results_path = baseline_dir / "results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    baseline = [
        item
        for item in payload
        if item.get("variant") == "current_feedback"
        and item.get("status") == "completed"
    ]
    expected_ids = {
        f"{family_id}--{persona_id}"
        for family_id in SOURCE_FAMILIES
        for persona_id in PERSONAS
    }
    actual_ids = {str(item.get("source_id")) for item in baseline}
    if len(baseline) != 4 or actual_ids != expected_ids:
        raise ValueError("B2 基线必须包含锁定 4 份源稿的完整 current_feedback 结果")
    return baseline


def build_b2_aggregate(
    baseline: list[dict[str, object]],
    b2_results: list[dict[str, object]],
) -> dict[str, object]:
    """比较复用的 A 基线与新 B2；失败调用不作为零分。"""

    completed = [item for item in b2_results if item["status"] == "completed"]

    def summarize(items: list[dict[str, object]]) -> dict[str, object]:
        return {
            "completed": len(items),
            "rhyme_pass_count": sum(
                float(item["output_metrics"]["rhyme_score"]) >= 70 for item in items
            ),
            "full_pass_count": sum(bool(item["output_metrics"]["passed"]) for item in items),
            "mean_rhyme_score": round(
                sum(float(item["output_metrics"]["rhyme_score"]) for item in items)
                / len(items),
                1,
            )
            if items
            else None,
            "mean_target_match_rate": round(
                sum(float(item["output_metrics"]["target_match_rate"]) for item in items)
                / len(items),
                3,
            )
            if items
            else None,
            "mean_changed_lines": round(
                sum(float(item["change_metrics"]["changed_line_count"]) for item in items)
                / len(items),
                1,
            )
            if items
            else None,
            "hook_changed_count": sum(
                bool(item["change_metrics"]["hook_changed"]) for item in items
            ),
        }

    pairs: list[dict[str, object]] = []
    baseline_by_id = {str(item["source_id"]): item for item in baseline}
    for item in completed:
        source_id = str(item["source_id"])
        current = baseline_by_id[source_id]["output_metrics"]
        b2 = item["output_metrics"]
        pairs.append(
            {
                "source_id": source_id,
                "current_rhyme_score": current["rhyme_score"],
                "b2_rhyme_score": b2["rhyme_score"],
                "current_target_match_rate": current["target_match_rate"],
                "b2_target_match_rate": b2["target_match_rate"],
                "current_rhyme_passed": float(current["rhyme_score"]) >= 70,
                "b2_rhyme_passed": float(b2["rhyme_score"]) >= 70,
                "b2_additional_rescue": (
                    float(b2["rhyme_score"]) >= 70
                    and float(current["rhyme_score"]) < 70
                ),
                "b2_hook_changed": item["change_metrics"]["hook_changed"],
            }
        )
    return {
        "logical_calls_planned": 4,
        "logical_calls_completed": len(completed),
        "logical_calls_failed": len(b2_results) - len(completed),
        "baseline_reused_not_recalled": True,
        "by_variant": {
            "current_feedback": summarize(baseline),
            B2_VARIANT: summarize(completed),
        },
        "paired_results": pairs,
        "b2_additional_rescue_count": sum(
            bool(item["b2_additional_rescue"]) for item in pairs
        ),
        "automatic_gate_passed": (
            len(completed) == 4
            and sum(bool(item["b2_additional_rescue"]) for item in pairs) >= 3
        ),
        "human_naturalness_review_status": "pending",
    }


def run_b2_experiment(
    source_root: Path,
    output_root: Path,
    baseline_dir: Path,
) -> Path:
    """复用 A 基线，只补跑 4 次去除 Hook 冲突的固定映射。"""

    cases = load_source_cases(source_root)
    baseline = load_current_feedback_baseline(baseline_dir)
    experiment_dir = output_root / f"rhyme-assistance-b2-{uuid4().hex}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    settings = get_settings()
    _write_json(
        experiment_dir / "experiment_config.json",
        {
            "stage": "1-b2-hook-conflict-correction",
            "baseline_dir": str(baseline_dir.resolve()),
            "baseline_variant": "current_feedback",
            "baseline_reused_not_recalled": True,
            "new_variant": B2_VARIANT,
            "logical_calls": 4,
            "model": settings.deepseek_model,
            "temperature": 0.85,
            "max_tokens": 2200,
            "rhyme_word_rag_used": False,
            "audio_generated": False,
            "limitations": [
                "B2 changes both rhyme-group knowledge and permission to repair a mismatched Hook ending.",
                "Each source has one stochastic B2 rewrite; this is a screening test, not statistical proof.",
                "Rule scores do not measure naturalness, singability, or audio quality.",
                "Only the 中东 rhyme group is covered.",
                "Human blind review is required before a method decision.",
            ],
        },
    )
    results: list[dict[str, object]] = []
    for call_index, case in enumerate(cases, start=1):
        print(f"[{call_index}/4] {case.source_id} / {B2_VARIANT}", flush=True)
        feedback = build_feedback(case, B2_VARIANT)
        llm = DeepSeekLLM(settings)
        started_at = time.perf_counter()
        result: dict[str, object] = {
            "call_index": call_index,
            "source_id": case.source_id,
            "family_id": case.family_id,
            "persona_id": case.persona_id,
            "variant": B2_VARIANT,
            "feedback": feedback,
            "source_metrics": quality_metrics(case.draft, case.requirement),
        }
        try:
            output = LyricistAgent(llm, persona=case.persona_id).run(
                case.requirement,
                revision_feedback=feedback,
                previous_draft=case.draft,
            )
            result.update(
                {
                    "status": "completed",
                    "wall_seconds": round(time.perf_counter() - started_at, 3),
                    "output_draft": output.model_dump(mode="json"),
                    "output_metrics": quality_metrics(output, case.requirement),
                    "change_metrics": line_change_metrics(case.draft, output),
                    "telemetry": DeepSeekLLM.summarize_call_records(
                        llm.telemetry_snapshot()
                    ),
                }
            )
        except Exception as error:
            result.update(
                {
                    "status": "failed",
                    "wall_seconds": round(time.perf_counter() - started_at, 3),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "telemetry": DeepSeekLLM.summarize_call_records(
                        llm.telemetry_snapshot()
                    ),
                }
            )
        finally:
            llm.close()
        results.append(result)
        _write_json(experiment_dir / f"call-{call_index:02d}.json", result)

    combined_for_review = [*baseline, *results]
    _write_json(experiment_dir / "baseline_current_feedback.json", baseline)
    _write_json(experiment_dir / "results.json", results)
    _write_json(experiment_dir / "summary.json", build_b2_aggregate(baseline, results))
    write_blind_review(
        experiment_dir,
        cases,
        combined_for_review,
        variants=("current_feedback", B2_VARIANT),
        title="B2 Hook 冲突修正匿名复核",
    )
    return experiment_dir


def _repeated_chorus_sync(
    source_report: dict[str, object], output_report: dict[str, object]
) -> dict[str, object]:
    """按源稿重复模板的 stanza 位置检查输出是否同步，近似模板不合并。"""

    output_by_index = {
        int(stanza["stanza_index"]): stanza["template_id"]
        for stanza in output_report["stanzas"]
        if stanza["section"] == "Chorus"
    }
    checks: list[dict[str, object]] = []
    for template in source_report["chorus_templates"]:
        if int(template["occurrence_count"]) < 2:
            continue
        indexes = [int(item["stanza_index"]) for item in template["occurrences"]]
        output_ids = [output_by_index.get(index) for index in indexes]
        checks.append(
            {
                "source_template_id": template["template_id"],
                "source_stanza_indexes": indexes,
                "output_template_ids": output_ids,
                "all_positions_present": all(item is not None for item in output_ids),
                "synchronized": (
                    all(item is not None for item in output_ids)
                    and len(set(output_ids)) == 1
                ),
            }
        )
    return {
        "repeated_template_groups_checked": len(checks),
        "repeated_template_groups_synchronized": sum(
            bool(item["synchronized"]) for item in checks
        ),
        "checks": checks,
    }


def _shadow_metric_mean(
    results: list[dict[str, object]], section: str, field: str
) -> float | None:
    values: list[float] = []
    for result in results:
        seen_templates: set[str] = set()
        for stanza in result["shadow_metrics"]["stanzas"]:
            if stanza["section"] != section:
                continue
            if section == "Chorus" and stanza["template_id"]:
                template_id = str(stanza["template_id"])
                if template_id in seen_templates:
                    continue
                seen_templates.add(template_id)
            value = stanza[field]
            if value is not None:
                values.append(float(value))
    return round(sum(values) / len(values), 3) if values else None


def build_b3_b4_aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    """只做描述性汇总，不输出生产准入结论。"""

    completed = [item for item in results if item["status"] == "completed"]
    by_variant: dict[str, dict[str, object]] = {}
    for variant in B3_B4_VARIANTS:
        items = [item for item in completed if item["variant"] == variant]
        by_variant[variant] = {
            "completed": len(items),
            "production_rhyme_score_mean_for_reference_only": round(
                sum(float(item["production_metrics"]["rhyme_score"]) for item in items)
                / len(items),
                1,
            )
            if items
            else None,
            "chorus_target_match_rate_mean": _shadow_metric_mean(
                items, "Chorus", "target_match_rate"
            ),
            "verse_local_consistency_rate_mean": _shadow_metric_mean(
                items, "Verse", "local_consistency_rate"
            ),
            "hook_matches_target_count": sum(
                item["shadow_metrics"]["hook"]["matches_target"] is True
                for item in items
            ),
            "mean_changed_source_lines": round(
                sum(
                    int(item["section_change_metrics"]["changed_source_line_count"])
                    for item in items
                )
                / len(items),
                1,
            )
            if items
            else None,
            "mean_non_target_changed_source_line_rate": round(
                sum(
                    float(
                        item["section_change_metrics"][
                            "non_target_changed_source_line_rate"
                        ]
                    )
                    for item in items
                )
                / len(items),
                3,
            )
            if items
            else None,
            "repeated_chorus_sync_count": sum(
                item["repeated_chorus_sync"][
                    "repeated_template_groups_synchronized"
                ]
                for item in items
            ),
            "repeated_chorus_groups_checked": sum(
                item["repeated_chorus_sync"]["repeated_template_groups_checked"]
                for item in items
            ),
        }
    return {
        "logical_calls_planned": 8,
        "logical_calls_completed": len(completed),
        "logical_calls_failed": len(results) - len(completed),
        "by_variant": by_variant,
        "formal_method_decision": "not_allowed_from_development_screening",
        "human_naturalness_review_status": "pending",
        "limitations": [
            "Four locked drafts, two themes, one rhyme group, one stochastic sample per condition.",
            "Production score is saved for historical reference and cannot judge chorus_target.",
            "B3 includes corrected shadow section attribution as part of the feedback pipeline.",
        ],
    }


def run_b3_b4_experiment(source_root: Path, output_root: Path) -> Path:
    """运行 B3/B4 开发筛查；不修改生产评分器或 LangGraph。"""

    cases = load_source_cases(source_root)
    plan = build_b3_b4_plan_summary(cases)
    issue_budget = int(plan["feedback_budget_selection"]["selected_issue_budget"])
    experiment_dir = output_root / f"rhyme-assistance-b3-b4-{uuid4().hex}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    settings = get_settings()
    _write_json(
        experiment_dir / "experiment_config.json",
        {
            "stage": "b3-b4-development-screening",
            "source_drafts": len(cases),
            "logical_calls": 8,
            "call_plan": plan["call_plan"],
            "feedback_budget_selection": plan["feedback_budget_selection"],
            "model": settings.deepseek_model,
            "temperature": 0.85,
            "max_tokens": 2200,
            "production_behavior_changed": False,
            "shadow_only": True,
            "rhyme_word_rag_used": False,
            "audio_generated": False,
            "scope_contract": {
                B3_VARIANT: "global_target",
                B4_VARIANT: "chorus_target",
            },
            "limitations": [
                "Development screening only; no production method decision is allowed.",
                "Each condition has one stochastic rewrite per source.",
                "Only two themes and the 中东 rhyme group are covered.",
                "Human blind review and held-out validation are required.",
            ],
        },
    )
    results: list[dict[str, object]] = []
    for call_index, (case, variant) in enumerate(
        build_b3_b4_call_plan(cases), start=1
    ):
        scope = _scope_for_variant(variant)
        feedback_package = build_balanced_feedback(
            case.draft.lyrics,
            case.draft.hook,
            case.requirement.target_rhyme_group,
            scope,
            issue_budget,
        )
        print(f"[{call_index}/8] {case.source_id} / {variant}", flush=True)
        llm = DeepSeekLLM(settings)
        started_at = time.perf_counter()
        result: dict[str, object] = {
            "call_index": call_index,
            "source_id": case.source_id,
            "family_id": case.family_id,
            "persona_id": case.persona_id,
            "variant": variant,
            "rhyme_scope": scope,
            "feedback_package": feedback_package,
            "source_shadow_metrics": analyze_rhyme_scope(
                case.draft.lyrics,
                case.draft.hook,
                case.requirement.target_rhyme_group,
                scope,
            ),
        }
        try:
            output = LyricistAgent(llm, persona=case.persona_id).run(
                case.requirement,
                revision_feedback=str(feedback_package["text"]),
                previous_draft=case.draft,
            )
            output_shadow = analyze_rhyme_scope(
                output.lyrics,
                output.hook,
                case.requirement.target_rhyme_group,
                scope,
            )
            target_sections = (
                {"Verse", "Pre-Chorus", "Chorus", "Hook", "Bridge"}
                if scope == "global_target"
                else {"Chorus", "Hook"}
            )
            result.update(
                {
                    "status": "completed",
                    "wall_seconds": round(time.perf_counter() - started_at, 3),
                    "output_draft": output.model_dump(mode="json"),
                    "production_metrics": quality_metrics(output, case.requirement),
                    "shadow_metrics": output_shadow,
                    "line_change_metrics": line_change_metrics(case.draft, output),
                    "section_change_metrics": section_change_metrics(
                        case.draft.lyrics, output.lyrics, target_sections
                    ),
                    "repeated_chorus_sync": _repeated_chorus_sync(
                        result["source_shadow_metrics"], output_shadow
                    ),
                    "telemetry": DeepSeekLLM.summarize_call_records(
                        llm.telemetry_snapshot()
                    ),
                }
            )
        except Exception as error:
            result.update(
                {
                    "status": "failed",
                    "wall_seconds": round(time.perf_counter() - started_at, 3),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "telemetry": DeepSeekLLM.summarize_call_records(
                        llm.telemetry_snapshot()
                    ),
                }
            )
        finally:
            llm.close()
        results.append(result)
        _write_json(experiment_dir / f"call-{call_index:02d}.json", result)

    _write_json(experiment_dir / "results.json", results)
    _write_json(experiment_dir / "summary.json", build_b3_b4_aggregate(results))
    write_blind_review(
        experiment_dir,
        cases,
        results,
        variants=B3_B4_VARIANTS,
        title="B3/B4 分段押韵匿名复核",
    )
    return experiment_dir


def run_experiment(source_root: Path, output_root: Path) -> Path:
    cases = load_source_cases(source_root)
    experiment_dir = output_root / f"rhyme-assistance-stage1-{uuid4().hex}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    settings = get_settings()
    config = {
        "stage": 1,
        "hypothesis": "fixed rhyme-group mapping may solve failures without rhyme-word RAG",
        "source_families": list(SOURCE_FAMILIES),
        "source_drafts": len(cases),
        "variants": list(VARIANTS),
        "logical_calls": len(cases) * len(VARIANTS),
        "model": settings.deepseek_model,
        "temperature": 0.85,
        "max_tokens": 2200,
        "rhyme_word_rag_used": False,
        "audio_generated": False,
        "limitations": [
            "Each condition has one stochastic rewrite per source; this is a screening test, not statistical proof.",
            "Rule scores do not measure naturalness, singability, or audio quality.",
            "Only the 中东 rhyme group is covered.",
            "Human blind review is required before a method decision.",
        ],
    }
    _write_json(experiment_dir / "experiment_config.json", config)
    results: list[dict[str, object]] = []
    for call_index, (case, variant) in enumerate(build_call_plan(cases), start=1):
        print(f"[{call_index}/8] {case.source_id} / {variant}", flush=True)
        feedback = build_feedback(case, variant)
        source_metrics = quality_metrics(case.draft, case.requirement)
        llm = DeepSeekLLM(settings)
        started_at = time.perf_counter()
        result: dict[str, object] = {
            "call_index": call_index,
            "source_id": case.source_id,
            "family_id": case.family_id,
            "persona_id": case.persona_id,
            "variant": variant,
            "feedback": feedback,
            "source_metrics": source_metrics,
        }
        try:
            output = LyricistAgent(llm, persona=case.persona_id).run(
                case.requirement,
                revision_feedback=feedback,
                previous_draft=case.draft,
            )
            result.update(
                {
                    "status": "completed",
                    "wall_seconds": round(time.perf_counter() - started_at, 3),
                    "output_draft": output.model_dump(mode="json"),
                    "output_metrics": quality_metrics(output, case.requirement),
                    "change_metrics": line_change_metrics(case.draft, output),
                    "telemetry": DeepSeekLLM.summarize_call_records(
                        llm.telemetry_snapshot()
                    ),
                }
            )
        except Exception as error:
            result.update(
                {
                    "status": "failed",
                    "wall_seconds": round(time.perf_counter() - started_at, 3),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "telemetry": DeepSeekLLM.summarize_call_records(
                        llm.telemetry_snapshot()
                    ),
                }
            )
        finally:
            llm.close()
        results.append(result)
        _write_json(experiment_dir / f"call-{call_index:02d}.json", result)

    aggregate = build_aggregate(results)
    _write_json(experiment_dir / "results.json", results)
    _write_json(experiment_dir / "summary.json", aggregate)
    write_blind_review(experiment_dir, cases, results)
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="运行第一阶段韵脚辅助对照实验。")
    parser.add_argument(
        "--mode",
        choices=("stage1", "b2", "b3_b4", "section_consistency"),
        default="stage1",
    )
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_STAGE1_BASELINE)
    parser.add_argument(
        "--execute", action="store_true", help="执行所选模式已获批准的 DeepSeek 改写"
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        help="可选：将离线计划写入指定 JSON；不会因此调用 DeepSeek。",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        help="可选：将 section consistency 人工预检稿写入指定 Markdown。",
    )
    args = parser.parse_args()
    if args.mode == "section_consistency":
        if args.execute:
            parser.error(
                "section_consistency 当前只获批离线预检；需人工确认候选后再实现真实调用。"
            )
        section_cases = load_section_consistency_cases(args.source_root)
        section_plan = build_section_consistency_plan_summary(section_cases)
        if args.plan_output:
            args.plan_output.parent.mkdir(parents=True, exist_ok=True)
            _write_json(args.plan_output, section_plan)
        if args.review_output:
            args.review_output.parent.mkdir(parents=True, exist_ok=True)
            args.review_output.write_text(
                build_section_consistency_review_markdown(
                    section_cases,
                    section_plan,
                ),
                encoding="utf-8",
            )
        print(json.dumps(section_plan, ensure_ascii=False, indent=2))
        return

    cases = load_source_cases(args.source_root)
    if not args.execute:
        b3_b4_plan = (
            build_b3_b4_plan_summary(cases) if args.mode == "b3_b4" else None
        )
        print(
            json.dumps(
                {
                    "source_drafts": len(cases),
                    "mode": args.mode,
                    "logical_calls": (
                        4
                        if args.mode == "b2"
                        else 8
                        if args.mode == "b3_b4"
                        else len(cases) * len(VARIANTS)
                    ),
                    "variants": (
                        [B2_VARIANT]
                        if args.mode == "b2"
                        else list(B3_B4_VARIANTS)
                        if args.mode == "b3_b4"
                        else list(VARIANTS)
                    ),
                    "source_ids": [case.source_id for case in cases],
                    "baseline_reused": str(args.baseline) if args.mode == "b2" else None,
                    "b3_b4_plan": b3_b4_plan,
                    "rhyme_word_rag_used": False,
                    "message": "加 --execute 才会调用真实 DeepSeek。",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.mode == "b2":
        experiment_dir = run_b2_experiment(
            args.source_root, args.out, args.baseline
        )
    elif args.mode == "b3_b4":
        experiment_dir = run_b3_b4_experiment(args.source_root, args.out)
    else:
        experiment_dir = run_experiment(args.source_root, args.out)
    print(f"实验完成：{experiment_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
