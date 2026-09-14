"""命令行工作流质检证据落盘测试。"""

import json
from pathlib import Path
import unittest
from uuid import uuid4

from services.workflow_artifacts import write_quality_artifacts, write_text_artifacts


class PipelineArtifactTests(unittest.TestCase):
    def test_writes_dual_candidates_and_selection(self) -> None:
        state: dict[str, object] = {
            "requirement": {"theme": "夏夜回家"},
            "lyric_draft": {"title": "晚风"},
            "lyrics": "[Verse]\n测试歌词",
            "production_plan": {"tags": ["city pop"]},
            "lyric_candidates": [
                {"candidate_id": "structured", "scores": {"total": 90.0}},
                {"candidate_id": "imagery", "scores": {"total": 88.0}},
            ],
            "candidate_selection": {
                "method": "blind_subjective_judge_v1",
                "selected_candidate_id": "structured",
            },
            "subjective_review": {
                "method": "blind_subjective_judge_v1",
                "candidate_reviews": {
                    "structured": {"weighted_total": 88.0},
                    "imagery": {"weighted_total": 80.0},
                },
            },
            "subjective_review_history": [
                {"generation_round": 1, "regeneration_required": False}
            ],
            "quality_status": "passed",
            "attempts": 1,
            "selected_attempt": 1,
            "scores": {"rhyme": 90.0, "structure": 100.0, "total": 93.0},
            "quality_report": {"total_score": 93.0, "passed": True},
            "quality_history": [{"attempt": 1, "quality_status": "passed"}],
        }
        output_dir = Path("tmp/test-work") / uuid4().hex
        write_text_artifacts(output_dir, state)

        candidates = json.loads(
            (output_dir / "lyric_candidates.json").read_text(encoding="utf-8")
        )
        selection = json.loads(
            (output_dir / "candidate_selection.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(selection["selected_candidate_id"], "structured")
        self.assertTrue((output_dir / "subjective_review.json").is_file())
        self.assertTrue((output_dir / "subjective_review_history.json").is_file())

    def test_writes_final_report_and_revision_history(self) -> None:
        state: dict[str, object] = {
            "quality_status": "capped",
            "attempts": 3,
            "selected_attempt": 2,
            "scores": {"rhyme": 80.0, "structure": 100.0, "total": 86.0},
            "quality_report": {"total_score": 86.0, "passed": False},
            "quality_history": [
                {"attempt": 1, "quality_status": "retrying"},
                {"attempt": 2, "quality_status": "retrying"},
                {"attempt": 3, "quality_status": "capped"},
            ],
        }
        output_dir = Path("tmp/test-work") / uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=False)
        report_path, history_path = write_quality_artifacts(output_dir, state)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))

        self.assertEqual(report["quality_status"], "capped")
        self.assertEqual(report["selected_attempt"], 2)
        self.assertEqual(history[-1]["quality_status"], "capped")


if __name__ == "__main__":
    unittest.main()
