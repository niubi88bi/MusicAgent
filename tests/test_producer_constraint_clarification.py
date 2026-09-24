"""Offline contracts for deterministic Producer clarification."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_producer_constraint_clarification_holdout import (
    ClarificationHoldoutManifest,
    build_manifest,
)
from scripts.evaluate_producer_constraint_clarification import run_evaluation
from tools.rag.producer_constraint_clarification import (
    PIANO_SCOPE,
    analyze_producer_constraint_clarification,
)


class ProducerConstraintClarificationTests(unittest.TestCase):
    def test_hard_piano_ban_without_replacement_requires_one_question(self) -> None:
        result = analyze_producer_constraint_clarification(
            "不要钢琴，做一首温暖克制的Deep House"
        )
        self.assertEqual(result.status, "clarification_required")
        self.assertEqual(result.replacement_mode, "unspecified")
        self.assertEqual(result.forbidden_scope, PIANO_SCOPE)
        self.assertEqual(len(result.questions), 1)
        self.assertEqual(len(result.questions[0].options), 5)
        self.assertEqual(
            result.questions[0].recommended_option, "auto_style_compatible"
        )

    def test_explicit_replacement_does_not_ask_again(self) -> None:
        result = analyze_producer_constraint_clarification(
            "禁止钢琴，改用木吉他承担伤感动机"
        )
        self.assertEqual(result.status, "no_clarification")
        self.assertEqual(result.replacement_mode, "explicit")
        self.assertEqual(result.explicit_replacement, "acoustic_guitar")
        self.assertEqual(result.questions, [])

    def test_hard_ban_with_explicit_use_word_is_recognized(self) -> None:
        result = analyze_producer_constraint_clarification(
            "禁止使用钢琴，做一首夜归主题中文歌"
        )
        self.assertEqual(result.status, "clarification_required")
        self.assertEqual(result.replacement_mode, "unspecified")

    def test_auto_and_no_replacement_authorizations_do_not_ask(self) -> None:
        automatic = analyze_producer_constraint_clarification(
            "不要钢琴，替代配器由系统按曲风选择"
        )
        none = analyze_producer_constraint_clarification(
            "不要钢琴，也不需要替代乐器"
        )
        self.assertEqual(automatic.replacement_mode, "auto_style_compatible")
        self.assertEqual(none.replacement_mode, "no_replacement")
        self.assertEqual(automatic.questions, [])
        self.assertEqual(none.questions, [])

    def test_soft_preference_is_advisory_not_blocking(self) -> None:
        for text in ("最好少用钢琴", "最好不用钢琴", "尽量避免钢琴"):
            with self.subTest(text=text):
                result = analyze_producer_constraint_clarification(text)
                self.assertEqual(result.status, "advisory_only")
                self.assertEqual(result.questions, [])

    def test_unrelated_request_does_not_create_constraints(self) -> None:
        result = analyze_producer_constraint_clarification(
            "做一首关于清晨渡船的Deep House中文歌"
        )
        self.assertEqual(result.status, "no_clarification")
        self.assertEqual(result.replacement_mode, "not_applicable")
        self.assertEqual(result.forbidden_instruments, [])

    def test_frozen_holdout_matches_all_expected_decisions(self) -> None:
        manifest = ClarificationHoldoutManifest.model_validate(build_manifest())
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            output_path = Path(directory) / "summary.json"
            manifest_path.write_text(
                manifest.model_dump_json(indent=2), encoding="utf-8"
            )
            summary = run_evaluation(
                manifest, manifest_path=manifest_path, output_path=output_path
            )
        self.assertEqual(summary["case_count"], 16)
        self.assertEqual(summary["passed_count"], 16)
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(summary["llm_calls"], 0)
        self.assertEqual(summary["runtime_agent_connected"], False)


if __name__ == "__main__":
    unittest.main()
