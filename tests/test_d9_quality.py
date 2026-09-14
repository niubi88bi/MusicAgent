"""D9 节奏与主题词面信号的离线回归测试。"""

import unittest

from tests.test_d6_quality import GOOD_LYRICS
from tools.lyrics_quality.parser import parse_lyric_lines
from tools.lyrics_quality.rhythm_checker import check_rhythm
from tools.lyrics_quality.scorer import score_lyrics
from tools.lyrics_quality.theme_checker import check_theme_signal


class D9QualityTests(unittest.TestCase):
    def test_repeated_sections_have_independent_stanza_indexes(self) -> None:
        lines = parse_lyric_lines(
            "[Chorus]\n第一次副歌\n\n[Verse 2]\n第二段主歌\n\n[Chorus 2]\n第二次副歌"
        )

        self.assertEqual(
            [(line.section, line.stanza_index) for line in lines],
            [("Chorus", 1), ("Verse", 1), ("Chorus", 2)],
        )

    def test_nearly_equal_lines_are_recognized_without_warning(self) -> None:
        lyrics = """[Verse]
晚风轻轻吹过街角
灯火慢慢照亮归途
脚步轻轻穿过小巷
月光静静落在窗前"""

        report = check_rhythm(lyrics)

        self.assertEqual(report.stanza_patterns["Verse#1"], "近似等长")
        self.assertFalse(report.issues)

    def test_abab_long_short_pattern_is_not_misclassified_as_outlier(self) -> None:
        lyrics = """[Verse]
灯灯灯灯灯灯灯灯灯灯灯灯灯灯
风风风风风风风风风风
雨雨雨雨雨雨雨雨雨雨雨雨雨雨
云云云云云云云云云云"""

        report = check_rhythm(lyrics)

        self.assertEqual(report.stanza_patterns["Verse#1"], "ABAB 长短句")
        self.assertNotIn("line_length_outlier", {issue.code for issue in report.issues})

    def test_obvious_line_length_outlier_is_reported(self) -> None:
        lyrics = """[Verse]
晚风轻轻吹过街角
灯火慢慢照亮归途
脚步轻轻穿过小巷
这是一个明显长得离谱并且完全打破整段节奏规律的歌词句子"""

        report = check_rhythm(lyrics)
        outliers = [issue for issue in report.issues if issue.code == "line_length_outlier"]

        self.assertEqual([issue.line_no for issue in outliers], [5])

    def test_phrase_count_mismatch_is_reported(self) -> None:
        lyrics = """[Verse]
晚风轻轻吹过街角
灯火慢慢照亮归途
脚步轻轻穿过小巷
晚风，穿过，街角，落进，灯光"""

        report = check_rhythm(lyrics)

        self.assertIn("phrase_count_mismatch", {issue.code for issue in report.issues})

    def test_all_oblique_tones_are_recorded_but_not_penalized(self) -> None:
        report = check_rhythm("[Verse]\n此刻只想快一点到\n晚风轻轻吹过街角")
        first_line = report.lines[0]

        self.assertEqual((first_line.ping_count, first_line.ze_count), (0, 8))
        self.assertEqual(first_line.tone_pattern, "ZZZZZZZZ")
        self.assertFalse(report.issues)

    def test_theme_signal_reports_hits_by_section_and_hook(self) -> None:
        lyrics = """[Verse]
清晨站台落着薄雾
父亲替我提起行囊

[Chorus]
汽笛响起 别回头望
故乡灯火留在身后"""

        report = check_theme_signal(
            lyrics,
            "汽笛响起 别回头望",
            theme="第一次离家",
            keywords=["站台", "汽笛", "父母", "远方"],
        )

        self.assertEqual(report.matched_keywords, ["站台", "汽笛"])
        self.assertEqual(report.missing_keywords, ["父母", "远方"])
        self.assertEqual(report.section_matches["Verse#1"], ["站台"])
        self.assertEqual(report.section_matches["Chorus#1"], ["汽笛"])
        self.assertEqual(report.hook_matches, ["汽笛"])
        self.assertEqual(report.coverage_ratio, 0.5)

    def test_zero_keyword_hits_warn_but_do_not_change_pass_result(self) -> None:
        baseline = score_lyrics(
            GOOD_LYRICS,
            "晚风在等我 回家的路上",
            30,
            "江阳",
        )
        with_theme_signal = score_lyrics(
            GOOD_LYRICS,
            "晚风在等我 回家的路上",
            30,
            "江阳",
            "海边度假",
            ["沙滩", "海浪"],
        )

        self.assertTrue(baseline.passed)
        self.assertTrue(with_theme_signal.passed)
        self.assertEqual(with_theme_signal.total_score, baseline.total_score)
        self.assertEqual(with_theme_signal.rhyme.score, baseline.rhyme.score)
        self.assertEqual(with_theme_signal.structure.score, baseline.structure.score)
        self.assertTrue(with_theme_signal.rhythm.advisory_only)
        self.assertTrue(with_theme_signal.theme_signal.advisory_only)
        self.assertIn(
            "low_keyword_coverage",
            {issue.code for issue in with_theme_signal.theme_signal.issues},
        )


if __name__ == "__main__":
    unittest.main()
