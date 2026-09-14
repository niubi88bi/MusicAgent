"""歌词段落和 Hook 的规则结构检查器。"""

from __future__ import annotations

from collections import Counter

from tools.lyrics_quality.models import QualityIssue, StructureReport
from tools.lyrics_quality.parser import parse_lyric_lines


REQUIRED_SECTIONS = ("Verse", "Chorus", "Bridge")


def _minimum_line_counts(duration_seconds: int | None) -> dict[str, int]:
    """用时长决定 v1 最低行数，避免短试听被长歌标准误伤。"""

    if duration_seconds is None or duration_seconds <= 45:
        return {"Verse": 2, "Chorus": 2, "Bridge": 1}
    if duration_seconds <= 90:
        return {"Verse": 3, "Chorus": 3, "Bridge": 1}
    return {"Verse": 4, "Chorus": 4, "Bridge": 2}


def check_structure(
    lyrics: str,
    hook: str,
    duration_seconds: int | None = None,
) -> StructureReport:
    """检查必须段落、时长适配的最小行数和可定位的 Hook 候选。"""

    lines = parse_lyric_lines(lyrics)
    counts = Counter(line.section for line in lines)
    section_line_counts = {section: counts.get(section, 0) for section in REQUIRED_SECTIONS}
    issues: list[QualityIssue] = []
    existing_sections = sum(counts.get(section, 0) > 0 for section in REQUIRED_SECTIONS)
    label_score = existing_sections / len(REQUIRED_SECTIONS) * 40
    for section in REQUIRED_SECTIONS:
        if counts.get(section, 0) == 0:
            issues.append(
                QualityIssue(
                    code="missing_section",
                    message=f"缺少 [{section}] 段落标签或该段没有歌词。",
                    suggestion=f"补充至少一个 [{section}] 段落。",
                )
            )

    minimums = _minimum_line_counts(duration_seconds)
    sufficient_sections = 0
    for section, minimum in minimums.items():
        actual = counts.get(section, 0)
        if actual >= minimum:
            sufficient_sections += 1
        else:
            issues.append(
                QualityIssue(
                    code="short_section",
                    message=f"{section} 段只有 {actual} 行，当前时长至少需要 {minimum} 行。",
                    suggestion=f"为 [{section}] 补充 {minimum - actual} 行歌词，保持已有主题与韵脚。",
                )
            )
    length_score = sufficient_sections / len(REQUIRED_SECTIONS) * 40

    chorus_text = "\n".join(line.text for line in lines if line.section in {"Chorus", "Hook"})
    hook_present = bool(hook.strip()) and hook.strip() in chorus_text
    if not hook_present:
        issues.append(
            QualityIssue(
                code="missing_hook",
                message="Hook 未在 [Chorus] 或 [Hook] 段中原样出现。",
                suggestion="把 hook 字段中的短句原样放进副歌，形成可定位的 Hook 候选。",
            )
        )
    hook_score = 20 if hook_present else 0
    score = round(label_score + length_score + hook_score, 1)
    return StructureReport(
        score=score,
        section_line_counts=section_line_counts,
        hook_present=hook_present,
        issues=issues,
    )
