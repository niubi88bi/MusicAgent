"""D6 十三辙、押韵、结构和评分器的离线测试。"""

import unittest

from tools.lyrics_quality.rhyme_checker import check_rhyme
from tools.lyrics_quality.scorer import score_lyrics
from tools.lyrics_quality.thirteen_rhymes import (
    normalize_rhyme_group,
    rhyme_group_for_final,
)


GOOD_LYRICS = """[Verse]
路灯把影子拉得很长
晚风绕过熟悉的街巷
我把疲惫放进微亮的窗
回家的方向有温柔的光

[Chorus]
晚风在等我 回家的路上
晚风在等我 回家的路上
城市的心跳不再匆忙
今晚的月色替我原谅

[Bridge]
远处的车灯像星河流淌
我终于学会和自己相望"""

BAD_LYRICS = """[Verse]
路灯照亮寂静的窗
雨滴落在旧书的页
我听见远方的海
故事停在陌生的门

[Chorus]
不要说再见的光
不要说再见的光
明天会有新的风
我们走向不同的人

[Bridge]
这一刻想起你"""


class D6QualityTests(unittest.TestCase):
    def test_common_finals_map_to_expected_groups(self) -> None:
        cases = {
            "ia": "发花",
            "uo": "梭波",
            "üe": "乜斜",
            "ü": "一七",
            "uei": "灰堆",
            "iou": "由求",
            "üan": "言前",
            "ün": "人辰",
            "iang": "江阳",
            "iong": "中东",
        }
        self.assertEqual({final: rhyme_group_for_final(final) for final in cases}, cases)

    def test_alias_finals_are_normalized(self) -> None:
        self.assertEqual(rhyme_group_for_final("ui"), "灰堆")
        self.assertEqual(rhyme_group_for_final("iu"), "由求")
        self.assertEqual(rhyme_group_for_final("un"), "人辰")
        self.assertEqual(rhyme_group_for_final("v"), "一七")

    def test_target_final_is_normalized_to_standard_rhyme_group(self) -> None:
        self.assertEqual(normalize_rhyme_group("ang"), "江阳")
        self.assertEqual(normalize_rhyme_group("uang"), "江阳")
        self.assertEqual(normalize_rhyme_group("江阳"), "江阳")
        self.assertIsNone(normalize_rhyme_group("不存在的辙"))

        by_final = score_lyrics(GOOD_LYRICS, "晚风在等我 回家的路上", 30, "ang")
        by_group = score_lyrics(GOOD_LYRICS, "晚风在等我 回家的路上", 30, "江阳")
        self.assertEqual(by_final.total_score, by_group.total_score)
        self.assertFalse(
            any(issue.code == "unknown_target_rhyme_group" for issue in by_final.issues)
        )

    def test_normalized_target_is_enforced_instead_of_falling_back_to_local_pattern(self) -> None:
        mixed_lyrics = """[Verse]
晚风陪我走在路上
路灯照亮回家方向

[Chorus]
今晚陪我回到家
窗前有人等我呀

[Bridge]
抬头仍有温柔星光"""
        without_target = score_lyrics(mixed_lyrics, "回到家", 30)
        with_final_target = score_lyrics(mixed_lyrics, "回到家", 30, "ang")

        self.assertGreater(without_target.rhyme.score, with_final_target.rhyme.score)
        self.assertFalse(with_final_target.passed)
        self.assertEqual(
            with_final_target.rhyme.section_patterns["Chorus"], "目标辙：江阳"
        )
        self.assertIn(
            "rhyme_pattern_mismatch",
            {issue.code for issue in with_final_target.issues},
        )

    def test_aabb_and_abab_patterns_receive_full_consistency(self) -> None:
        aabb = """[Verse]
灯火照亮街巷
晚风掠过城墙
脚步停在窗外
心事放进人海"""
        abab = """[Verse]
灯火照亮街巷
脚步停在窗外
晚风掠过城墙
心事放进人海"""
        self.assertEqual(check_rhyme(aabb).section_patterns["Verse"], "AABB")
        self.assertEqual(check_rhyme(abab).section_patterns["Verse"], "ABAB")

    def test_good_lyrics_pass_and_bad_lyrics_fail(self) -> None:
        good = score_lyrics(GOOD_LYRICS, "晚风在等我 回家的路上", 30, "江阳")
        bad = score_lyrics(BAD_LYRICS, "不要说再见的光", 30)
        self.assertTrue(good.passed)
        self.assertGreater(good.total_score, bad.total_score)
        self.assertFalse(bad.passed)

    def test_missing_structure_is_reported(self) -> None:
        report = score_lyrics("[Verse]\n只有一行没有副歌", "没有副歌", 30)
        self.assertFalse(report.passed)
        self.assertIn("missing_section", {issue.code for issue in report.issues})

    def test_missing_hook_cannot_pass_only_on_total_score(self) -> None:
        report = score_lyrics(GOOD_LYRICS, "不存在的 Hook", 30, "江阳")
        self.assertGreaterEqual(report.total_score, 75)
        self.assertFalse(report.passed)
        self.assertIn("missing_hook", {issue.code for issue in report.issues})

    def test_weak_section_rhyme_reports_specific_mismatched_line(self) -> None:
        lyrics = """[Verse]
路灯照亮街巷
晚风掠过城墙
心事落进人海
脚步停在雨季"""
        report = check_rhyme(lyrics)
        mismatch_issues = [
            issue for issue in report.issues if issue.code == "rhyme_pattern_mismatch"
        ]
        self.assertTrue(mismatch_issues)
        self.assertTrue(all(issue.line_no is not None for issue in mismatch_issues))


if __name__ == "__main__":
    unittest.main()
