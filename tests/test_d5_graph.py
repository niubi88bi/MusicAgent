"""D5 LangGraph 线性图的无网络、无 GPU 测试。"""

from pathlib import Path
from threading import Barrier
import unittest

from agents.graph import build_music_graph
from agents.schemas import LyricDraft, ProductionPlan, SongRequirement
from services.music_gen import MusicGenerationResult


REQUIREMENT = SongRequirement(
    theme="夏夜回家",
    emotion="温暖、释然",
    genre="city pop",
    tempo="mid-tempo",
    duration_seconds=30,
    perspective="第一人称",
    vocal_type="温柔男声",
    bpm=96,
    emotion_intensity=3,
    keywords=["夏夜", "路灯", "晚风"],
    target_rhyme_group="江阳",
)

LYRIC_DRAFT = LyricDraft(
    title="晚风在等我",
    lyrics=(
        "[Verse]\n路灯把影子拉得很长\n晚风绕过熟悉的街巷\n"
        "我把疲惫放进微亮的窗\n回家的方向有温柔的光\n\n"
        "[Chorus]\n晚风在等我 回家的路上\n晚风在等我 回家的路上\n"
        "城市的心跳不再匆忙\n今晚的月色替我原谅\n\n"
        "[Bridge]\n远处的车灯像星河流淌\n我终于学会和自己相望"
    ),
    hook="晚风在等我 回家的路上",
    creative_note="以路灯和晚风承接下班后的释然。",
)

PRODUCTION_PLAN = ProductionPlan(
    style_prompt=(
        "City pop, warm midnight mood, modern Chinese pop production, "
        "electric piano and clean guitar, gentle male vocal, restrained verse "
        "building into a luminous chorus."
    ),
    tags=["city pop", "warm", "male vocal", "night drive"],
    vocal_direction="Close-mic gentle male vocal, relaxed phrasing.",
    arrangement_note="Use electric piano in verse and expand with guitar in chorus.",
)


class FakeRequirementAgent:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def run(self, theme: str) -> SongRequirement:
        self._calls.append(f"requirement:{theme}")
        return REQUIREMENT


class FakeLyricistAgent:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def run(self, requirement: SongRequirement) -> LyricDraft:
        self._calls.append(f"lyricist:{requirement.theme}")
        return LYRIC_DRAFT


class ParallelLyricistAgent:
    """只有两个分支同时进入 run 才能越过 Barrier。"""

    def __init__(self, persona_id: str, draft: LyricDraft, barrier: Barrier) -> None:
        self.persona_id = persona_id
        self.display_name = persona_id
        self._draft = draft
        self._barrier = barrier
        self.calls = 0

    def run(
        self,
        requirement: SongRequirement,
        revision_feedback: str | None = None,
        previous_draft: LyricDraft | None = None,
    ) -> LyricDraft:
        self.calls += 1
        if revision_feedback is None:
            self._barrier.wait(timeout=2)
        return self._draft


class FakeProducerAgent:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def run(
        self, requirement: SongRequirement, lyric_draft: LyricDraft
    ) -> ProductionPlan:
        self._calls.append(f"producer:{requirement.theme}:{lyric_draft.title}")
        return PRODUCTION_PLAN


class FakeMusicGenerator:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def generate(
        self, lyrics: str, tags: list[str], duration_seconds: int
    ) -> MusicGenerationResult:
        self._calls.append(f"generator:{duration_seconds}:{','.join(tags)}")
        output_path = Path("tmp/test-work/d5/song.mp3")
        return MusicGenerationResult(
            job_id="d5-test",
            output_path=output_path,
            lyrics_path=Path("tmp/test-work/d5/lyrics.txt"),
            tags_path=Path("tmp/test-work/d5/tags.txt"),
            duration_seconds=duration_seconds,
        )


def subjective_report(
    structured_score: float,
    imagery_score: float,
    *,
    structured_major: bool = False,
    imagery_major: bool = False,
) -> dict[str, object]:
    def review(score: float, major: bool) -> dict[str, object]:
        return {
            "weighted_total": score,
            "major_theme_issue": major,
            "issues": (
                [
                    {
                        "code": "theme_drift",
                        "severity": "major",
                        "evidence": "测试问题句",
                        "suggestion": "回到指定主题",
                    }
                ]
                if major
                else []
            ),
        }

    return {
        "method": "blind_subjective_judge_v1",
        "weights": {},
        "near_tie_points": 3.0,
        "anonymous_mapping": {"A": "structured", "B": "imagery"},
        "candidate_reviews": {
            "structured": review(structured_score, structured_major),
            "imagery": review(imagery_score, imagery_major),
        },
        "model_recommended_winner": "A",
        "confidence": "high",
        "comparison_summary": "离线评审结果。",
        "both_have_major_theme_issues": structured_major and imagery_major,
    }


class FakeLyricJudgeAgent:
    def __init__(self, reports: list[dict[str, object]]) -> None:
        self._reports = reports
        self.calls: list[list[dict[str, object]]] = []

    def run(
        self, requirement: SongRequirement, candidates: list[dict[str, object]]
    ) -> dict[str, object]:
        self.calls.append(candidates)
        return self._reports[min(len(self.calls) - 1, len(self._reports) - 1)]


class D5GraphTests(unittest.TestCase):
    def build_graph(self):
        calls: list[str] = []
        graph = build_music_graph(
            FakeRequirementAgent(calls),
            FakeLyricistAgent(calls),
            FakeProducerAgent(calls),
            FakeMusicGenerator(calls),
        )
        return graph, calls

    def test_compiles_with_quality_node(self) -> None:
        graph, _ = self.build_graph()
        node_names = set(graph.get_graph().nodes)
        self.assertTrue(
            {
                "validate_input",
                "requirement",
                "lyricist",
                "quality",
                "producer",
                "generator",
            }
            <= node_names
        )

    def test_runs_text_to_audio_state_in_order(self) -> None:
        graph, calls = self.build_graph()
        result = graph.invoke(
            {
                "theme": "  写一首夏夜下班回家的歌  ",
                "requested_duration_seconds": 45,
            }
        )

        self.assertEqual(result["theme"], "写一首夏夜下班回家的歌")
        self.assertEqual(result["requirement"].theme, REQUIREMENT.theme)
        self.assertEqual(result["requirement"].duration_seconds, 45)
        self.assertEqual(result["lyrics"], LYRIC_DRAFT.lyrics)
        self.assertEqual(result["production_plan"], PRODUCTION_PLAN)
        self.assertEqual(result["audio_path"], "tmp\\test-work\\d5\\song.mp3")
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(result["quality_status"], "passed")
        self.assertEqual(result["selected_attempt"], 1)
        self.assertEqual(len(result["quality_history"]), 1)
        self.assertEqual(result["quality_history"][0]["quality_status"], "passed")
        self.assertEqual(
            calls,
            [
                "requirement:写一首夏夜下班回家的歌",
                "lyricist:夏夜回家",
                "producer:夏夜回家:晚风在等我",
                "generator:45:city pop,warm,male vocal,night drive",
            ],
        )

    def test_rejects_invalid_theme_before_agent_calls(self) -> None:
        graph, calls = self.build_graph()
        with self.assertRaises(ValueError):
            graph.invoke({"theme": " "})
        self.assertEqual(calls, [])

    def test_dual_lyricists_run_in_parallel_and_rule_screen_selects_passed_candidate(self) -> None:
        from tests.test_d7_retry import BAD_DRAFT

        barrier = Barrier(2)
        structured = ParallelLyricistAgent("structured", BAD_DRAFT, barrier)
        imagery = ParallelLyricistAgent("imagery", LYRIC_DRAFT, barrier)
        graph = build_music_graph(
            FakeRequirementAgent([]),
            structured,
            FakeProducerAgent([]),
            secondary_lyricist_agent=imagery,
            text_only=True,
        )

        result = graph.invoke({"theme": "夏夜回家", "requested_duration_seconds": 30})

        self.assertEqual(structured.calls, 1)
        self.assertEqual(imagery.calls, 1)
        self.assertEqual(len(result["lyric_candidates"]), 2)
        self.assertEqual(result["selected_lyricist"], "imagery")
        self.assertEqual(
            result["candidate_selection"]["selected_candidate_id"], "imagery"
        )
        self.assertEqual(result["lyric_draft"], LYRIC_DRAFT)
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(
            result["quality_history"][0]["source_candidate_id"], "imagery"
        )

    def test_subjective_judge_can_override_rule_preference(self) -> None:
        barrier = Barrier(2)
        structured = ParallelLyricistAgent("structured", LYRIC_DRAFT, barrier)
        imagery = ParallelLyricistAgent(
            "imagery", LYRIC_DRAFT.model_copy(update={"title": "意象版"}), barrier
        )
        judge = FakeLyricJudgeAgent([subjective_report(90.0, 80.0)])
        graph = build_music_graph(
            FakeRequirementAgent([]),
            structured,
            FakeProducerAgent([]),
            secondary_lyricist_agent=imagery,
            lyric_judge_agent=judge,
            text_only=True,
        )

        result = graph.invoke({"theme": "夏夜回家", "requested_duration_seconds": 30})

        self.assertEqual(result["selected_lyricist"], "structured")
        self.assertEqual(result["candidate_selection"]["selection_basis"], "subjective_total")
        self.assertEqual(result["candidate_selection"]["method"], "blind_subjective_judge_v1")
        self.assertEqual(len(judge.calls), 1)

    def test_subjective_winner_can_continue_through_rule_rewrite_loop(self) -> None:
        from tests.test_d7_retry import BAD_DRAFT

        barrier = Barrier(2)
        structured = ParallelLyricistAgent("structured", BAD_DRAFT, barrier)
        imagery = ParallelLyricistAgent("imagery", BAD_DRAFT, barrier)
        judge = FakeLyricJudgeAgent([subjective_report(90.0, 80.0)])
        graph = build_music_graph(
            FakeRequirementAgent([]),
            structured,
            FakeProducerAgent([]),
            secondary_lyricist_agent=imagery,
            lyric_judge_agent=judge,
            text_only=True,
        )

        result = graph.invoke({"theme": "夏夜回家", "requested_duration_seconds": 30})

        self.assertEqual(result["selected_lyricist"], "structured")
        self.assertEqual(structured.calls, 3)
        self.assertEqual(imagery.calls, 1)
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["quality_status"], "capped")
        self.assertEqual(len(result["quality_history"]), 3)
        self.assertEqual(result["production_plan"], PRODUCTION_PLAN)

    def test_near_tie_uses_rule_result_instead_of_model_winner(self) -> None:
        from tests.test_d7_retry import BAD_DRAFT

        barrier = Barrier(2)
        structured = ParallelLyricistAgent("structured", BAD_DRAFT, barrier)
        imagery = ParallelLyricistAgent("imagery", LYRIC_DRAFT, barrier)
        judge = FakeLyricJudgeAgent([subjective_report(90.0, 88.0)])
        graph = build_music_graph(
            FakeRequirementAgent([]),
            structured,
            FakeProducerAgent([]),
            secondary_lyricist_agent=imagery,
            lyric_judge_agent=judge,
            text_only=True,
        )

        result = graph.invoke({"theme": "夏夜回家", "requested_duration_seconds": 30})

        self.assertEqual(result["selected_lyricist"], "imagery")
        self.assertEqual(
            result["candidate_selection"]["selection_basis"],
            "near_tie_rule_tiebreak",
        )

    def test_single_major_theme_issue_is_filtered_before_score_comparison(self) -> None:
        barrier = Barrier(2)
        structured = ParallelLyricistAgent("structured", LYRIC_DRAFT, barrier)
        imagery = ParallelLyricistAgent(
            "imagery", LYRIC_DRAFT.model_copy(update={"title": "意象版"}), barrier
        )
        judge = FakeLyricJudgeAgent(
            [subjective_report(99.0, 70.0, structured_major=True)]
        )
        graph = build_music_graph(
            FakeRequirementAgent([]),
            structured,
            FakeProducerAgent([]),
            secondary_lyricist_agent=imagery,
            lyric_judge_agent=judge,
            text_only=True,
        )

        result = graph.invoke({"theme": "夏夜回家", "requested_duration_seconds": 30})

        self.assertEqual(result["selected_lyricist"], "imagery")
        self.assertEqual(
            result["candidate_selection"]["selection_basis"],
            "major_theme_issue_filter",
        )
        self.assertEqual(result["candidate_regeneration_count"], 0)

    def test_both_major_theme_issues_regenerate_both_candidates_once(self) -> None:
        barrier = Barrier(2)
        structured = ParallelLyricistAgent("structured", LYRIC_DRAFT, barrier)
        imagery = ParallelLyricistAgent(
            "imagery", LYRIC_DRAFT.model_copy(update={"title": "意象版"}), barrier
        )
        judge = FakeLyricJudgeAgent(
            [
                subjective_report(
                    70.0,
                    68.0,
                    structured_major=True,
                    imagery_major=True,
                ),
                subjective_report(90.0, 80.0),
            ]
        )
        graph = build_music_graph(
            FakeRequirementAgent([]),
            structured,
            FakeProducerAgent([]),
            secondary_lyricist_agent=imagery,
            lyric_judge_agent=judge,
            text_only=True,
        )

        result = graph.invoke({"theme": "夏夜回家", "requested_duration_seconds": 30})

        self.assertEqual(structured.calls, 2)
        self.assertEqual(imagery.calls, 2)
        self.assertEqual(len(judge.calls), 2)
        self.assertEqual(result["candidate_regeneration_count"], 1)
        self.assertEqual(len(result["subjective_review_history"]), 2)
        self.assertEqual(len(result["lyric_candidates"]), 4)
        self.assertEqual(result["candidate_selection"]["generation_round"], 2)
        self.assertEqual(result["selected_lyricist"], "structured")

    def test_both_major_theme_issues_never_regenerate_more_than_once(self) -> None:
        barrier = Barrier(2)
        structured = ParallelLyricistAgent("structured", LYRIC_DRAFT, barrier)
        imagery = ParallelLyricistAgent(
            "imagery", LYRIC_DRAFT.model_copy(update={"title": "意象版"}), barrier
        )
        still_major = subjective_report(
            70.0,
            68.0,
            structured_major=True,
            imagery_major=True,
        )
        judge = FakeLyricJudgeAgent([still_major, still_major])
        graph = build_music_graph(
            FakeRequirementAgent([]),
            structured,
            FakeProducerAgent([]),
            secondary_lyricist_agent=imagery,
            lyric_judge_agent=judge,
            text_only=True,
        )

        result = graph.invoke({"theme": "夏夜回家", "requested_duration_seconds": 30})

        self.assertEqual(structured.calls, 2)
        self.assertEqual(imagery.calls, 2)
        self.assertEqual(len(judge.calls), 2)
        self.assertEqual(result["candidate_regeneration_count"], 1)
        self.assertTrue(result["candidate_selection"]["regeneration_exhausted"])
        self.assertEqual(result["candidate_selection"]["generation_round"], 2)


if __name__ == "__main__":
    unittest.main()
