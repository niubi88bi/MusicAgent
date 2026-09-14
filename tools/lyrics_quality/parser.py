"""歌词段落与行号解析，供多个规则检查器共用。"""

from __future__ import annotations

import re
from collections import Counter

from tools.lyrics_quality.models import LyricLine


SECTION_PATTERN = re.compile(r"^\[(Verse|Chorus|Bridge|Hook)(?:\s+\d+)?\]$", re.I)
CHINESE_CHARACTER_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def parse_lyric_lines(lyrics: str) -> list[LyricLine]:
    """只提取有明确 Verse/Chorus/Bridge/Hook 标签的非空歌词行。"""

    current_section: str | None = None
    current_stanza_index = 0
    stanza_counts: Counter[str] = Counter()
    lines: list[LyricLine] = []
    for line_no, raw_line in enumerate(lyrics.splitlines(), start=1):
        text = raw_line.strip()
        match = SECTION_PATTERN.fullmatch(text)
        if match:
            current_section = match.group(1).title()
            stanza_counts[current_section] += 1
            current_stanza_index = stanza_counts[current_section]
            continue
        if current_section and text:
            lines.append(
                LyricLine(
                    section=current_section,
                    line_no=line_no,
                    text=text,
                    stanza_index=current_stanza_index,
                )
            )
    return lines


def chinese_characters(text: str) -> str:
    """保留中文字符，便于在整句上下文里处理多音字。"""

    return "".join(CHINESE_CHARACTER_PATTERN.findall(text))
