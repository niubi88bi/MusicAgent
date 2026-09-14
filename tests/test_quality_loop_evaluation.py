"""质量回炉评估的无网络测试。"""

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.evaluate_quality_loop import (
    _build_aggregate_metrics,
    _failure_payload,
    _select_controlled_record,
    _select_cases,
    _write_blind_review_pair,
    build_controlled_cases,
)
from services.config import Settings
from services.llm import DeepSeekLLM, LLMResponseError
from tools.lyrics_quality.scorer import score_lyrics


class QualityLoopEvaluationTests(unittest.TestCase):
    def test_case_filter_selects_only_requested_controlled_case(self) -> None:
        natural_cases, controlled_cases = _select_cases("controlled", "rhyme-mismatch")
        self.assertEqual(natural_cases, ())
        self.assertEqual([case.case_id for case in controlled_cases], ["rhyme-mismatch"])

    def test_case_filter_rejects_case_outside_requested_mode(self) -> None:
        with self.assertRaises(ValueError):
            _select_cases("controlled", "summer-night-home")

    def test_all_controlled_fixtures_trigger_their_named_defect(self) -> None:
        for case in build_controlled_cases():
            report = score_lyrics(
                case.draft.lyrics,
                case.draft.hook,
                case.requirement.duration_seconds,
                case.requirement.target_rhyme_group,
            )
            self.assertFalse(report.passed, case.case_id)
            self.assertIn(
                case.defect_type,
                {issue.code for issue in report.issues},
                case.case_id,
            )

    def test_capped_controlled_case_selects_highest_scoring_attempt(self) -> None:
        history = [
            {"attempt": 1, "scores": {"total": 80.0}},
            {"attempt": 2, "scores": {"total": 70.0}},
            {"attempt": 3, "scores": {"total": 75.0}},
        ]
        selected = _select_controlled_record(history, final_passed=False)
        self.assertEqual(selected["attempt"], 1)

    def test_blind_review_includes_theme_context_without_version_identity(self) -> None:
        case = build_controlled_cases()[0]
        selected = case.draft.model_copy(
            update={"lyrics": case.draft.lyrics + "\n雨后风声绕过长巷"}
        )
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            _write_blind_review_pair(
                output_dir,
                case.case_id,
                {
                    "theme": case.requirement.theme,
                    "emotion": case.requirement.emotion,
                    "genre": case.requirement.genre,
                },
                case.draft,
                selected,
                source_names=("structured", "imagery"),
                comparison="dual_lyricist_candidates",
            )
            review = (output_dir / "blind_review.txt").read_text(encoding="utf-8")
            key = (output_dir / "blind_review_key.json").read_text(encoding="utf-8")
        self.assertIn(case.requirement.theme, review)
        self.assertNotIn("structured", review)
        self.assertNotIn("imagery", review)
        self.assertIn("dual_lyricist_candidates", key)

    def test_aggregate_separates_rule_pass_from_target_defect_resolution(self) -> None:
        aggregate = _build_aggregate_metrics(
            [
                {
                    "feedback_triggered": True,
                    "initial_passed": False,
                    "quality_status": "passed",
                }
            ],
            [
                {
                    "target_defect_detected": True,
                    "target_defect_resolved": False,
                    "quality_status": "passed",
                }
            ],
            [],
        )
        self.assertEqual(aggregate["natural"]["feedback_trigger_rate"], 1.0)
        self.assertEqual(aggregate["controlled"]["quality_pass_cases"], 1)
        self.assertEqual(aggregate["controlled"]["target_defect_resolved_cases"], 0)

    def test_failure_payload_keeps_sanitized_llm_telemetry(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None), finish_reason="stop"
                )
            ],
            model="test-model",
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0),
        )
        create = unittest.mock.Mock(return_value=response)
        llm = DeepSeekLLM(Settings(DEEPSEEK_API_KEY="test-key", LLM_MAX_RETRIES=0))
        llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with patch("services.llm.time.sleep"):
            with self.assertRaises(LLMResponseError) as context:
                llm.chat_json(
                    [{"role": "user", "content": "test"}], operation="producer"
                )

        failure = _failure_payload(
            case_id="test-case",
            mode="natural_workflow",
            error=context.exception,
            llm=llm,
        )
        telemetry = failure["telemetry"]
        self.assertIsInstance(telemetry, dict)
        self.assertEqual(telemetry["failed_calls"], 1)
        self.assertEqual(telemetry["records"][0]["operation"], "producer")
        self.assertNotIn("content", telemetry["records"][0])


if __name__ == "__main__":
    unittest.main()
