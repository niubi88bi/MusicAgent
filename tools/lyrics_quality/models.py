"""歌词质检模块内部使用的可序列化结果类型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LyricLine:
    """一行带段落与原始行号的歌词。"""

    section: str
    line_no: int
    text: str
    stanza_index: int = 1


@dataclass(frozen=True)
class QualityIssue:
    """可直接交给作词 Agent 的规则问题。"""

    code: str
    message: str
    suggestion: str
    line_no: int | None = None


@dataclass(frozen=True)
class RhymeLineAnalysis:
    """单行韵脚的可解释分析。"""

    section: str
    line_no: int
    ending_char: str | None
    final: str | None
    rhyme_group: str | None


@dataclass(frozen=True)
class RhymeReport:
    score: float
    total_lines: int
    analyzable_lines: int
    section_patterns: dict[str, str]
    lines: list[RhymeLineAnalysis]
    issues: list[QualityIssue]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StructureReport:
    score: float
    section_line_counts: dict[str, int]
    hook_present: bool
    issues: list[QualityIssue]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RhythmLineAnalysis:
    """不依赖旋律的单行节奏信号；不能等同于可唱性判断。"""

    section: str
    stanza_index: int
    line_no: int
    chinese_char_count: int
    phrase_count: int
    ping_count: int
    ze_count: int
    neutral_count: int
    tone_pattern: str


@dataclass(frozen=True)
class RhythmReport:
    """句长与停顿规律报告；D9 第一版仅作建议，不参与总分。"""

    score: float
    advisory_only: bool
    stanza_patterns: dict[str, str]
    lines: list[RhythmLineAnalysis]
    issues: list[QualityIssue]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ThemeSignalReport:
    """主题关键词的词面覆盖报告；不声称理解完整语义。"""

    theme: str | None
    keywords: list[str]
    matched_keywords: list[str]
    missing_keywords: list[str]
    coverage_ratio: float | None
    section_matches: dict[str, list[str]]
    hook_matches: list[str]
    advisory_only: bool
    issues: list[QualityIssue]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LyricQualityReport:
    """D6-D7 汇总报告；规则分不等同于听感评价。"""

    total_score: float
    passed: bool
    rhyme: RhymeReport
    structure: StructureReport
    rhythm: RhythmReport
    theme_signal: ThemeSignalReport
    issues: list[QualityIssue]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_score": self.total_score,
            "passed": self.passed,
            "rhyme": self.rhyme.to_dict(),
            "structure": self.structure.to_dict(),
            "rhythm": self.rhythm.to_dict(),
            "theme_signal": self.theme_signal.to_dict(),
            "issues": [asdict(issue) for issue in self.issues],
        }
