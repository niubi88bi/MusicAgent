"""不依赖旋律的歌词句长、停顿和平仄信号检查。"""

from __future__ import annotations

from collections import defaultdict
from statistics import median
import re

from pypinyin import Style, lazy_pinyin

from tools.lyrics_quality.models import (
    QualityIssue,
    RhythmLineAnalysis,
    RhythmReport,
)
from tools.lyrics_quality.parser import chinese_characters, parse_lyric_lines


PHRASE_SEPARATOR = re.compile(r"[，,。！？!?；;、：:\s]+")


def _tone_pattern(characters: str) -> tuple[str, int, int, int]:
    syllables = lazy_pinyin(
        characters,
        style=Style.TONE3,
        neutral_tone_with_five=True,
        errors="ignore",
    )
    pattern: list[str] = []
    ping = 0
    ze = 0
    neutral = 0
    for syllable in syllables:
        tone = int(syllable[-1]) if syllable and syllable[-1].isdigit() else None
        if tone in {1, 2}:
            ping += 1
            pattern.append("P")
        elif tone in {3, 4}:
            ze += 1
            pattern.append("Z")
        else:
            neutral += 1
            pattern.append("N")
    return "".join(pattern), ping, ze, neutral


def _phrase_count(text: str) -> int:
    phrases = [part for part in PHRASE_SEPARATOR.split(text.strip()) if part]
    return max(1, len(phrases))


def _stanza_pattern(lengths: list[int]) -> str:
    if len(lengths) < 2:
        return "样本不足，仅记录"
    if max(lengths) - min(lengths) <= 2:
        return "近似等长"
    if len(lengths) == 4:
        if abs(lengths[0] - lengths[2]) <= 2 and abs(lengths[1] - lengths[3]) <= 2:
            return "ABAB 长短句"
        if abs(lengths[0] - lengths[1]) <= 2 and abs(lengths[2] - lengths[3]) <= 2:
            return "AABB 长短句"
    return "自由句长"


def _expected_lengths(lengths: list[int], pattern: str) -> list[float]:
    if pattern == "ABAB 长短句":
        first = median([lengths[0], lengths[2]])
        second = median([lengths[1], lengths[3]])
        return [first, second, first, second]
    if pattern == "AABB 长短句":
        first = median(lengths[:2])
        second = median(lengths[2:])
        return [first, first, second, second]
    center = median(lengths)
    return [center] * len(lengths)


def check_rhythm(lyrics: str) -> RhythmReport:
    """输出可复现的节奏信号；第一版不改变歌词通过结论。"""

    lyric_lines = parse_lyric_lines(lyrics)
    analyses: list[RhythmLineAnalysis] = []
    grouped: dict[tuple[str, int], list[RhythmLineAnalysis]] = defaultdict(list)
    for line in lyric_lines:
        characters = chinese_characters(line.text)
        pattern, ping, ze, neutral = _tone_pattern(characters)
        analysis = RhythmLineAnalysis(
            section=line.section,
            stanza_index=line.stanza_index,
            line_no=line.line_no,
            chinese_char_count=len(characters),
            phrase_count=_phrase_count(line.text),
            ping_count=ping,
            ze_count=ze,
            neutral_count=neutral,
            tone_pattern=pattern,
        )
        analyses.append(analysis)
        grouped[(line.section, line.stanza_index)].append(analysis)

    issues: list[QualityIssue] = []
    stanza_patterns: dict[str, str] = {}
    scored_lines = 0
    penalty = 0.0
    for (section, stanza_index), items in grouped.items():
        key = f"{section}#{stanza_index}"
        lengths = [item.chinese_char_count for item in items]
        pattern = _stanza_pattern(lengths)
        stanza_patterns[key] = pattern
        if len(items) < 2:
            continue
        scored_lines += len(items)
        expected_lengths = _expected_lengths(lengths, pattern)
        for item, expected in zip(items, expected_lengths):
            allowed_deviation = max(4.0, expected * 0.4)
            if abs(item.chinese_char_count - expected) > allowed_deviation:
                penalty += 30.0
                issues.append(
                    QualityIssue(
                        code="line_length_outlier",
                        line_no=item.line_no,
                        message=(
                            f"第 {item.line_no} 行有 {item.chinese_char_count} 个汉字，"
                            f"明显偏离 {section} 第 {stanza_index} 段的 {pattern}。"
                        ),
                        suggestion="压缩或补足该行，并优先保持原意、韵脚和自然停顿。",
                    )
                )

        phrase_counts = [item.phrase_count for item in items]
        expected_phrases = median(phrase_counts)
        for item in items:
            if abs(item.phrase_count - expected_phrases) >= 2:
                penalty += 15.0
                issues.append(
                    QualityIssue(
                        code="phrase_count_mismatch",
                        line_no=item.line_no,
                        message=(
                            f"第 {item.line_no} 行包含 {item.phrase_count} 个停顿单元，"
                            f"与同段常见的 {expected_phrases:g} 个差异明显。"
                        ),
                        suggestion="调整标点或短语切分，使平行句的停顿层级更接近。",
                    )
                )

    score = 100.0 if not scored_lines else max(0.0, 100.0 - penalty / scored_lines)
    return RhythmReport(
        score=round(score, 1),
        advisory_only=True,
        stanza_patterns=stanza_patterns,
        lines=analyses,
        issues=issues,
    )
