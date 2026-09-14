"""基于拼音韵母和十三辙的歌词行尾押韵检查器。"""

from __future__ import annotations

from collections import Counter, defaultdict

from pypinyin import Style, lazy_pinyin

from tools.lyrics_quality.models import QualityIssue, RhymeLineAnalysis, RhymeReport
from tools.lyrics_quality.parser import chinese_characters, parse_lyric_lines
from tools.lyrics_quality.thirteen_rhymes import (
    LOUD_RHYME_GROUPS,
    normalize_final,
    normalize_rhyme_group,
    rhyme_group_for_final,
)


def _analyse_line(section: str, line_no: int, text: str) -> RhymeLineAnalysis:
    characters = chinese_characters(text)
    if not characters:
        return RhymeLineAnalysis(section, line_no, None, None, None)

    # 整句输入让 pypinyin 的短语词典有机会处理多音字。
    finals = lazy_pinyin(
        characters,
        style=Style.FINALS,
        strict=True,
        v_to_u=True,
        errors="ignore",
    )
    final = normalize_final(finals[-1]) if finals else None
    return RhymeLineAnalysis(
        section=section,
        line_no=line_no,
        ending_char=characters[-1],
        final=final,
        rhyme_group=rhyme_group_for_final(final) if final else None,
    )


def _pattern_rate(groups: list[str]) -> tuple[float, str, str | None]:
    """为一段歌词选取最贴合的 v1 押韵模式。"""

    counts = Counter(groups)
    dominant_group, dominant_count = counts.most_common(1)[0]
    dominant_rate = dominant_count / len(groups)
    if len(groups) == 2:
        return (1.0 if groups[0] == groups[1] else 0.0), "AA", dominant_group
    if len(groups) == 4:
        aabb_rate = (int(groups[0] == groups[1]) + int(groups[2] == groups[3])) / 2
        abab_rate = (int(groups[0] == groups[2]) + int(groups[1] == groups[3])) / 2
        candidates = [
            (dominant_rate, "主导辙"),
            (aabb_rate, "AABB"),
            (abab_rate, "ABAB"),
        ]
        best_rate, pattern = max(candidates, key=lambda item: item[0])
        return best_rate, pattern, dominant_group
    return dominant_rate, "主导辙", dominant_group


def _mismatched_lines(
    items: list[RhymeLineAnalysis],
    pattern: str,
    expected_group: str | None,
    target_rhyme_group: str | None,
) -> list[tuple[RhymeLineAnalysis, str | None]]:
    """返回不符合当前段落模式的具体行，供回炉提示精确定位。"""

    if target_rhyme_group or pattern == "主导辙":
        return [
            (item, expected_group)
            for item in items
            if item.rhyme_group != expected_group
        ]
    if pattern == "AA" and len(items) == 2 and items[0].rhyme_group != items[1].rhyme_group:
        return [(items[1], items[0].rhyme_group)]
    if pattern == "AABB" and len(items) == 4:
        return [
            (items[right], items[left].rhyme_group)
            for left, right in ((0, 1), (2, 3))
            if items[left].rhyme_group != items[right].rhyme_group
        ]
    if pattern == "ABAB" and len(items) == 4:
        return [
            (items[right], items[left].rhyme_group)
            for left, right in ((0, 2), (1, 3))
            if items[left].rhyme_group != items[right].rhyme_group
        ]
    return []


def check_rhyme(
    lyrics: str,
    target_rhyme_group: str | None = None,
) -> RhymeReport:
    """分析歌词行尾并返回 0-100 押韵分。"""

    lyric_lines = parse_lyric_lines(lyrics)
    analyses = [_analyse_line(line.section, line.line_no, line.text) for line in lyric_lines]
    issues: list[QualityIssue] = []
    # 兼容旧产物或外部调用传入 ang/uang/iang 等韵母；内部统一使用
    # “江阳”等标准辙名，避免目标约束因表示法不同被静默忽略。
    valid_target = normalize_rhyme_group(target_rhyme_group) if target_rhyme_group else None
    if target_rhyme_group and not valid_target:
        issues.append(
            QualityIssue(
                code="unknown_target_rhyme_group",
                message=f"目标辙「{target_rhyme_group}」不在十三辙映射表中。",
                suggestion="删除该目标辙，或改为十三辙中的标准名称。",
            )
        )

    analysed = [item for item in analyses if item.rhyme_group]
    for item in analyses:
        if not item.rhyme_group:
            issues.append(
                QualityIssue(
                    code="unclassified_ending",
                    line_no=item.line_no,
                    message=f"第 {item.line_no} 行末字无法归入十三辙。",
                    suggestion="使用中文韵脚，或把该行末字替换为可识别的普通话常用字。",
                )
            )

    by_section: dict[str, list[RhymeLineAnalysis]] = defaultdict(list)
    for item in analysed:
        by_section[item.section].append(item)

    weighted_rate_sum = 0.0
    weighted_line_count = 0
    section_patterns: dict[str, str] = {}
    for section, items in by_section.items():
        groups = [item.rhyme_group for item in items if item.rhyme_group]
        if len(groups) < 2:
            section_patterns[section] = "单行，不参与一致性扣分"
            continue
        if valid_target:
            expected_group = valid_target
            rate = sum(group == valid_target for group in groups) / len(groups)
            pattern = f"目标辙：{valid_target}"
        else:
            rate, pattern, expected_group = _pattern_rate(groups)
        section_patterns[section] = pattern
        weighted_rate_sum += rate * len(groups)
        weighted_line_count += len(groups)
        if rate < 0.75:
            if valid_target or pattern == "主导辙":
                suggestion = (
                    f"优先把该段韵脚统一到「{expected_group}」辙，"
                    "保留语义后再检查句尾。"
                )
            else:
                suggestion = (
                    f"按 {pattern} 模式修正失配行的句尾，"
                    "保留其他行的语义和韵脚。"
                )
            issues.append(
                QualityIssue(
                    code="weak_section_rhyme",
                    message=f"{section} 段行尾押韵一致性仅为 {rate:.0%}（{pattern}）。",
                    suggestion=suggestion,
                )
            )
            for item, line_expected_group in _mismatched_lines(
                items, pattern, expected_group, valid_target
            ):
                issues.append(
                    QualityIssue(
                        code="rhyme_pattern_mismatch",
                        line_no=item.line_no,
                        message=(
                            f"第 {item.line_no} 行末字「{item.ending_char}」属于"
                            f"「{item.rhyme_group}」辙，与 {section} 段的"
                            f"{pattern} 要求不一致。"
                        ),
                        suggestion=(
                            f"将该行句尾改为「{line_expected_group}」辙的词，"
                            "并尽量保持该行语义。"
                        ),
                    )
                )

    coverage = len(analysed) / len(analyses) if analyses else 0.0
    consistency = weighted_rate_sum / weighted_line_count if weighted_line_count else 0.0
    loud_ratio = (
        sum(item.rhyme_group in LOUD_RHYME_GROUPS for item in analysed) / len(analysed)
        if analysed
        else 0.0
    )
    score = round(coverage * 20 + consistency * 70 + loud_ratio * 10, 1)
    return RhymeReport(
        score=score,
        total_lines=len(analyses),
        analyzable_lines=len(analysed),
        section_patterns=section_patterns,
        lines=analyses,
        issues=issues,
    )
