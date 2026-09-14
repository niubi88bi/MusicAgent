"""第一阶段韵脚辅助实验的无网络测试。"""

import unittest

from scripts.evaluate_rhyme_assistance_stage1 import (
    B2_VARIANT,
    B3_VARIANT,
    B4_VARIANT,
    SECTION_CONSISTENCY_FAMILIES,
    SECTION_CONSISTENCY_VARIANT,
    SOURCE_FAMILIES,
    VARIANTS,
    _repeated_chorus_sync,
    build_aggregate,
    build_b2_aggregate,
    build_b2_mapping_note,
    build_b3_b4_aggregate,
    build_b3_b4_call_plan,
    build_b3_b4_plan_summary,
    build_section_consistency_plan_summary,
    build_section_consistency_review_markdown,
    build_call_plan,
    build_feedback,
    build_mapping_note,
    line_change_metrics,
    load_source_cases,
    load_section_consistency_cases,
    quality_metrics,
)


class RhymeAssistanceStage1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_source_cases()

    def test_locked_source_set_has_four_failed_drafts(self) -> None:
        self.assertEqual(len(self.cases), 4)
        for case in self.cases:
            self.assertEqual(case.requirement.target_rhyme_group, "中东")
            self.assertLess(quality_metrics(case.draft, case.requirement)["rhyme_score"], 70)

    def test_mapping_note_uses_deterministic_map_not_candidate_lexicon(self) -> None:
        note = build_mapping_note("中东")
        for final in ("eng", "ing", "ueng", "ong", "iong"):
            self.assertIn(final, note)
        self.assertIn("不要求直接写入歌词", note)

    def test_only_difference_is_appended_mapping_note(self) -> None:
        case = self.cases[0]
        current = build_feedback(case, "current_feedback")
        mapping = build_feedback(case, "fixed_mapping")
        self.assertTrue(mapping.startswith(current))
        self.assertEqual(mapping[len(current) :].strip(), build_mapping_note("中东"))

    def test_b2_allows_only_mismatched_hook_ending_to_change(self) -> None:
        note = build_b2_mapping_note("中东")
        self.assertIn("允许调整 Hook 句尾", note)
        self.assertIn("保留 Hook 的核心含义", note)
        self.assertNotIn("保留原主题、Hook、段落", note)
        feedback = build_feedback(self.cases[0], B2_VARIANT)
        self.assertTrue(feedback.endswith(note))

    def test_call_plan_has_eight_calls_and_alternates_order(self) -> None:
        plan = build_call_plan(self.cases)
        self.assertEqual(len(plan), 8)
        self.assertEqual([variant for _, variant in plan[:2]], list(VARIANTS))
        self.assertEqual([variant for _, variant in plan[2:4]], list(reversed(VARIANTS)))

    def test_identical_draft_has_no_changed_lines(self) -> None:
        case = self.cases[0]
        metrics = line_change_metrics(case.draft, case.draft)
        self.assertEqual(metrics["changed_line_count"], 0)
        self.assertEqual(metrics["line_sequence_similarity"], 1.0)
        self.assertTrue(metrics["original_hook_preserved_exactly"])

    def test_aggregate_counts_additional_rescue_separately(self) -> None:
        def result(variant: str, rhyme_score: float) -> dict[str, object]:
            return {
                "status": "completed",
                "source_id": "one",
                "variant": variant,
                "output_metrics": {
                    "rhyme_score": rhyme_score,
                    "passed": rhyme_score >= 70,
                    "target_match_rate": rhyme_score / 100,
                },
                "change_metrics": {"changed_line_count": 2},
            }

        aggregate = build_aggregate(
            [result("current_feedback", 60), result("fixed_mapping", 80)]
        )
        self.assertEqual(aggregate["mapping_additional_rescue_count"], 1)
        self.assertEqual(aggregate["human_naturalness_review_status"], "pending")

    def test_b2_aggregate_reuses_baseline_and_requires_three_rescues(self) -> None:
        def result(variant: str, source_id: str, rhyme_score: float) -> dict[str, object]:
            return {
                "status": "completed",
                "source_id": source_id,
                "variant": variant,
                "output_metrics": {
                    "rhyme_score": rhyme_score,
                    "passed": rhyme_score >= 70,
                    "target_match_rate": rhyme_score / 100,
                },
                "change_metrics": {"changed_line_count": 2, "hook_changed": True},
            }

        baseline = [result("current_feedback", str(index), 50) for index in range(4)]
        b2 = [
            result(B2_VARIANT, str(index), 80 if index < 3 else 60)
            for index in range(4)
        ]
        aggregate = build_b2_aggregate(baseline, b2)
        self.assertTrue(aggregate["baseline_reused_not_recalled"])
        self.assertEqual(aggregate["b2_additional_rescue_count"], 3)
        self.assertTrue(aggregate["automatic_gate_passed"])

    def test_b3_b4_plan_has_eight_interleaved_calls_and_full_feedback_coverage(self) -> None:
        call_plan = build_b3_b4_call_plan(self.cases)
        self.assertEqual(len(call_plan), 8)
        self.assertEqual(
            [variant for _, variant in call_plan[:4]],
            [B3_VARIANT, B4_VARIANT, B4_VARIANT, B3_VARIANT],
        )
        summary = build_b3_b4_plan_summary(self.cases)
        self.assertEqual(summary["logical_calls"], 8)
        self.assertFalse(summary["production_behavior_changed"])
        self.assertTrue(summary["shadow_only"])
        for item in summary["feedback_coverage"]:
            self.assertEqual(item["uncovered_units"], [], item["source_id"])

    def test_repeated_chorus_sync_checks_all_source_occurrences(self) -> None:
        source = {
            "chorus_templates": [
                {
                    "template_id": "source-template",
                    "occurrence_count": 2,
                    "occurrences": [
                        {"stanza_index": 1},
                        {"stanza_index": 2},
                    ],
                }
            ]
        }
        output = {
            "stanzas": [
                {"section": "Chorus", "stanza_index": 1, "template_id": "same"},
                {"section": "Chorus", "stanza_index": 2, "template_id": "same"},
            ]
        }
        result = _repeated_chorus_sync(source, output)
        self.assertEqual(result["repeated_template_groups_checked"], 1)
        self.assertEqual(result["repeated_template_groups_synchronized"], 1)

    def test_b3_b4_aggregate_forbids_formal_method_decision(self) -> None:
        aggregate = build_b3_b4_aggregate([])
        self.assertEqual(
            aggregate["formal_method_decision"],
            "not_allowed_from_development_screening",
        )
        self.assertEqual(aggregate["human_naturalness_review_status"], "pending")

    def test_section_consistency_plan_uses_heldout_system_targets_without_calls(self) -> None:
        cases = load_section_consistency_cases()
        self.assertEqual(len(cases), 8)
        self.assertEqual(
            {case.family_id for case in cases}, set(SECTION_CONSISTENCY_FAMILIES)
        )
        self.assertFalse({case.family_id for case in cases} & set(SOURCE_FAMILIES))

        plan = build_section_consistency_plan_summary(cases)
        self.assertEqual(plan["variant"], SECTION_CONSISTENCY_VARIANT)
        self.assertEqual(plan["planned_llm_calls_now"], 0)
        self.assertEqual(plan["preserve_no_rewrite_count"], 6)
        self.assertEqual(plan["human_review_before_any_rewrite_count"], 2)
        self.assertEqual(plan["maximum_future_calls_after_human_confirmation"], 2)
        self.assertEqual(
            set(plan["system_target_groups_for_reference_only"]),
            {"中东", "怀来", "江阳"},
        )
        self.assertFalse(plan["formal_threshold_selected"])
        self.assertEqual(
            plan["formal_method_decision"],
            "not_allowed_from_offline_preflight",
        )
        self.assertTrue(
            all(row["target_origin"] == "system_suggested" for row in plan["source_rows"])
        )
        review = build_section_consistency_review_markdown(cases, plan)
        self.assertEqual(review.count("## 1."), 1)
        self.assertEqual(review.count("## 2."), 1)
        self.assertIn("late-cafe-jazz--imagery", review)
        self.assertIn("night-running-electronic--imagery", review)
        self.assertIn("原稿控制：Z", review)
        self.assertIn("75% 只是筛查切片", review)


if __name__ == "__main__":
    unittest.main()
