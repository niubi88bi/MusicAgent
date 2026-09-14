"""D6-D7 歌词质检 v1 的评分汇总与回炉反馈格式。"""

from __future__ import annotations

from tools.lyrics_quality.models import LyricQualityReport
from tools.lyrics_quality.rhyme_checker import check_rhyme
from tools.lyrics_quality.rhythm_checker import check_rhythm
from tools.lyrics_quality.structure_checker import check_structure
from tools.lyrics_quality.theme_checker import check_theme_signal


QUALITY_PASS_THRESHOLD = 75.0
MIN_RHYME_SCORE = 70.0


def score_lyrics(
    lyrics: str,
    hook: str,
    duration_seconds: int | None = None,
    target_rhyme_group: str | None = None,
    theme: str | None = None,
    theme_keywords: list[str] | None = None,
) -> LyricQualityReport:
    """保留 v1 总分，同时附加 D9 的节奏与主题词面建议报告。"""

    rhyme = check_rhyme(lyrics, target_rhyme_group=target_rhyme_group)
    structure = check_structure(lyrics, hook, duration_seconds=duration_seconds)
    rhythm = check_rhythm(lyrics)
    theme_signal = check_theme_signal(
        lyrics,
        hook,
        theme=theme,
        keywords=theme_keywords,
    )
    total_score = round(rhyme.score * 0.7 + structure.score * 0.3, 1)
    passed = (
        total_score >= QUALITY_PASS_THRESHOLD
        and rhyme.score >= MIN_RHYME_SCORE
        and structure.score >= QUALITY_PASS_THRESHOLD
        # Hook 是 D6 的结构验收项。仅靠分数放行会出现“报告明确缺 Hook，
        # 但总分仍通过”的矛盾，因此将其作为最小结构门槛。
        and structure.hook_present
    )
    return LyricQualityReport(
        total_score=total_score,
        passed=passed,
        rhyme=rhyme,
        structure=structure,
        rhythm=rhythm,
        theme_signal=theme_signal,
        issues=[
            *rhyme.issues,
            *structure.issues,
            *rhythm.issues,
            *theme_signal.issues,
        ],
    )


def format_revision_feedback(report: LyricQualityReport) -> str:
    """把规则报告压缩为作词 Agent 可消费的中文反馈。"""

    lines = [
        "请保留主题和段落标签，按以下规则反馈重写歌词：",
        (
            f"当前规则分 {report.total_score:.1f}/100 "
            f"（押韵 {report.rhyme.score:.1f}，结构 {report.structure.score:.1f}；"
            f"节奏参考 {report.rhythm.score:.1f}，主题词面信号仅作建议）。"
        ),
    ]
    for issue in report.issues[:8]:
        position = f"第 {issue.line_no} 行：" if issue.line_no else ""
        lines.append(f"- {position}{issue.message} 建议：{issue.suggestion}")
    if not report.issues:
        lines.append("- 请提高各段句尾同辙比例，并保持 Hook 位于副歌。")
    return "\n".join(lines)
