"""歌词主题关键词的确定性词面覆盖检查。"""

from __future__ import annotations

import re
from typing import Sequence

from tools.lyrics_quality.models import QualityIssue, ThemeSignalReport
from tools.lyrics_quality.parser import parse_lyric_lines


WORD_CHARACTER_PATTERN = re.compile(r"[\w\u4e00-\u9fff]", re.UNICODE)


def _normalize(text: str) -> str:
    return "".join(WORD_CHARACTER_PATTERN.findall(text.lower()))


def _clean_keywords(keywords: Sequence[str] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for keyword in keywords or ():
        cleaned = keyword.strip()
        normalized = _normalize(cleaned)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(cleaned)
    return result


def check_theme_signal(
    lyrics: str,
    hook: str,
    *,
    theme: str | None = None,
    keywords: Sequence[str] | None = None,
) -> ThemeSignalReport:
    """报告关键词直接命中；同义表达和主题方向仍交给 D8 评审。"""

    cleaned_keywords = _clean_keywords(keywords)
    normalized_keywords = {
        keyword: _normalize(keyword) for keyword in cleaned_keywords
    }
    normalized_lyrics = _normalize(lyrics)
    matched = [
        keyword
        for keyword, normalized in normalized_keywords.items()
        if normalized in normalized_lyrics
    ]
    missing = [keyword for keyword in cleaned_keywords if keyword not in matched]

    section_text: dict[str, str] = {}
    for line in parse_lyric_lines(lyrics):
        key = f"{line.section}#{line.stanza_index}"
        section_text[key] = section_text.get(key, "") + _normalize(line.text)
    section_matches = {
        section: [
            keyword
            for keyword, normalized in normalized_keywords.items()
            if normalized in text
        ]
        for section, text in section_text.items()
    }
    normalized_hook = _normalize(hook)
    hook_matches = [
        keyword
        for keyword, normalized in normalized_keywords.items()
        if normalized in normalized_hook
    ]
    coverage = (
        round(len(matched) / len(cleaned_keywords), 3)
        if cleaned_keywords
        else None
    )

    issues: list[QualityIssue] = []
    if cleaned_keywords and not matched:
        issues.append(
            QualityIssue(
                code="low_keyword_coverage",
                message="歌词没有直接命中任何需求关键词。",
                suggestion="核对主题场景是否仍被清楚表达；可以使用自然同义表达，不要机械堆词。",
            )
        )
    if cleaned_keywords and not hook_matches:
        issues.append(
            QualityIssue(
                code="hook_without_theme_anchor",
                message="Hook 没有直接包含需求关键词。",
                suggestion="检查 Hook 是否能独立承载主题；若已使用清晰同义表达，可保留原句。",
            )
        )

    return ThemeSignalReport(
        theme=theme,
        keywords=cleaned_keywords,
        matched_keywords=matched,
        missing_keywords=missing,
        coverage_ratio=coverage,
        section_matches=section_matches,
        hook_matches=hook_matches,
        advisory_only=True,
        issues=issues,
    )
