"""分段押韵影子分析的无网络测试。"""

import unittest

from scripts.rhyme_scope_shadow import (
    analyze_rhyme_scope,
    build_balanced_feedback,
    build_section_consistency_triage,
    choose_feedback_budget,
    parse_shadow_lines,
    section_change_metrics,
)


LYRICS = """[Verse]
风吹过旧城的长巷
我把旧信留在桌旁
[Chorus]
晚风陪我回到梦中
路灯照着回家的风
[Bridge]
故事转了方向
[Chorus]
晚风陪我回到梦中
路灯照着回家的风
[Outro]
天亮以后再出发
"""


class RhymeScopeShadowTests(unittest.TestCase):
    def test_shadow_parser_recognizes_outro_and_stanza_indexes(self) -> None:
        lines = parse_shadow_lines(
            "[Verse]\n第一段\n[Verse 2]\n第二段\n[Outro]\n尾声一句"
        )
        self.assertEqual(
            [(line.section, line.stanza_index) for line in lines],
            [("Verse", 1), ("Verse", 2), ("Outro", 1)],
        )

    def test_identical_choruses_share_template_but_keep_occurrences(self) -> None:
        report = analyze_rhyme_scope(LYRICS, "晚风陪我回到梦中", "中东", "chorus_target")
        self.assertEqual(len(report["chorus_templates"]), 1)
        template = report["chorus_templates"][0]
        self.assertEqual(template["occurrence_count"], 2)
        self.assertEqual(len(template["occurrences"]), 2)

    def test_one_changed_chorus_line_creates_a_distinct_template(self) -> None:
        changed = LYRICS.replace(
            "[Chorus]\n晚风陪我回到梦中\n路灯照着回家的风\n[Outro]",
            "[Chorus]\n晚风陪我回到梦中\n路灯照着回家的灯\n[Outro]",
        )
        report = analyze_rhyme_scope(changed, "晚风陪我回到梦中", "中东", "chorus_target")
        self.assertEqual(len(report["chorus_templates"]), 2)

    def test_scope_changes_target_application_without_formal_pass(self) -> None:
        global_report = analyze_rhyme_scope(LYRICS, "晚风陪我回到梦中", "中东", "global_target")
        chorus_report = analyze_rhyme_scope(LYRICS, "晚风陪我回到梦中", "中东", "chorus_target")
        global_sections = {
            item["section"]: item["target_applies"] for item in global_report["stanzas"]
        }
        chorus_sections = {
            item["section"]: item["target_applies"] for item in chorus_report["stanzas"]
        }
        self.assertTrue(global_sections["Verse"])
        self.assertFalse(chorus_sections["Verse"])
        self.assertTrue(chorus_sections["Chorus"])
        self.assertFalse(chorus_sections["Bridge"])
        self.assertFalse(chorus_sections["Outro"])

        def contains_passed(value: object) -> bool:
            if isinstance(value, dict):
                return "passed" in value or any(contains_passed(item) for item in value.values())
            if isinstance(value, list):
                return any(contains_passed(item) for item in value)
            return False

        self.assertFalse(contains_passed(chorus_report))

    def test_thresholds_are_descriptive_slices(self) -> None:
        report = analyze_rhyme_scope(LYRICS, "晚风陪我回到梦中", "中东", "chorus_target")
        chorus = next(item for item in report["stanzas"] if item["section"] == "Chorus")
        self.assertEqual(
            set(chorus["target_thresholds"]),
            {"gte_0_5", "gte_0_75", "gte_1_0"},
        )
        self.assertIn("do not define a pass gate", report["limitations"][1])

    def test_balanced_feedback_keeps_hook_and_all_problem_units(self) -> None:
        lyrics = """[Verse]
第一行向前
第二行向前
[Chorus]
副歌一句回家
副歌两句回家
[Bridge]
桥段走向远方
[Outro]
尾声仍在路上"""
        package = build_balanced_feedback(
            lyrics,
            "副歌一句回家",
            "中东",
            "global_target",
            issue_budget=10,
        )
        self.assertIn("Hook", package["covered_units"])
        self.assertEqual(package["uncovered_units"], [])
        self.assertIn("Verse 1", package["text"])
        self.assertIn("Chorus 1", package["text"])
        self.assertIn("Bridge 1", package["text"])
        self.assertIn("Outro 1", package["text"])

    def test_chorus_scope_does_not_force_bridge_or_outro_edits(self) -> None:
        package = build_balanced_feedback(
            LYRICS,
            "晚风陪我回到梦中",
            "中东",
            "chorus_target",
            issue_budget=12,
        )
        self.assertIn("Bridge 与 Outro 只报告", package["text"])
        bridge_issues = [
            unit for unit in package["problem_units"] if str(unit).startswith("Bridge")
        ]
        outro_issues = [
            unit for unit in package["problem_units"] if str(unit).startswith("Outro")
        ]
        self.assertEqual(bridge_issues, [])
        self.assertEqual(outro_issues, [])

    def test_budget_selection_uses_smallest_full_coverage_candidate(self) -> None:
        selection = choose_feedback_budget(
            [(LYRICS, "晚风陪我回到梦中", "中东")],
            budgets=(2, 10, 16),
        )
        full_coverage = [
            item
            for item in selection["measurements"]
            if item["uncovered_problem_units"] == 0
        ]
        self.assertTrue(full_coverage)
        self.assertEqual(
            selection["selected_issue_budget"],
            min(item["issue_budget"] for item in full_coverage),
        )

    def test_section_change_metrics_separates_non_target_edits(self) -> None:
        output = LYRICS.replace("故事转了方向", "故事转向远方").replace(
            "晚风陪我回到梦中", "晚风陪我走进梦中", 1
        )
        metrics = section_change_metrics(LYRICS, output, {"Chorus", "Hook"})
        self.assertEqual(metrics["changed_source_line_count"], 2)
        self.assertEqual(metrics["changed_source_lines_by_section"]["Bridge"], 1)
        self.assertEqual(metrics["changed_source_lines_by_section"]["Chorus"], 1)
        self.assertEqual(metrics["non_target_changed_source_line_rate"], 0.5)

    def test_system_target_is_metadata_only_for_section_consistency(self) -> None:
        report = analyze_rhyme_scope(
            LYRICS,
            "晚风陪我回到梦中",
            "中东",
            "section_consistency",
            target_origin="system_suggested",
        )
        self.assertEqual(report["target_origin"], "system_suggested")
        self.assertIn("advisory_metadata_only", report["target_policy"])
        self.assertFalse(
            any(stanza["target_applies"] for stanza in report["stanzas"])
        )
        self.assertFalse(report["hook"]["target_is_user_explicit"])

        feedback = build_balanced_feedback(
            LYRICS,
            "晚风陪我回到梦中",
            "中东",
            "section_consistency",
            issue_budget=10,
            target_origin="system_suggested",
        )
        self.assertIn("只作为记录", feedback["text"])
        self.assertNotIn("对应普通话韵母", feedback["text"])
        self.assertNotIn("例字", feedback["text"])

    def test_target_origin_rejects_contradictory_values(self) -> None:
        with self.assertRaises(ValueError):
            analyze_rhyme_scope(
                LYRICS,
                "晚风陪我回到梦中",
                "中东",
                "section_consistency",
                target_origin="none",
            )
        with self.assertRaises(ValueError):
            analyze_rhyme_scope(
                LYRICS,
                "晚风陪我回到梦中",
                None,
                "section_consistency",
                target_origin="user_explicit",
            )

    def test_section_consistency_triage_never_authorizes_rewrite(self) -> None:
        stable = build_section_consistency_triage(
            LYRICS,
            "晚风陪我回到梦中",
            "中东",
        )
        self.assertEqual(stable["triage_action"], "preserve_no_rewrite")
        self.assertFalse(stable["automatic_rewrite_authorized"])
        self.assertIsNone(stable["formal_pass_decision"])

        unstable_lyrics = LYRICS.replace("路灯照着回家的风", "路灯照着我回家")
        unstable = build_section_consistency_triage(
            unstable_lyrics,
            "晚风陪我回到梦中",
            "中东",
        )
        self.assertEqual(
            unstable["triage_action"], "human_review_before_any_rewrite"
        )
        self.assertFalse(unstable["automatic_rewrite_authorized"])


if __name__ == "__main__":
    unittest.main()
