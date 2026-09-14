"""D8.3 匿名主观评审的无网络契约测试。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import unittest

from pydantic import ValidationError

from agents.critic_agent import LyricJudgeAgent, weighted_subjective_score
from agents.schemas import (
    CandidateSubjectiveReview,
    LyricDraft,
    SongRequirement,
    SubjectiveDimensionScore,
)


REQUIREMENT = SongRequirement(
    theme="夏夜下班回家",
    emotion="温暖、放松",
    genre="City Pop",
    tempo="mid-tempo",
    duration_seconds=30,
    perspective="第一人称",
    vocal_type="温柔男声",
    bpm=96,
    emotion_intensity=3,
    keywords=["夏夜", "回家"],
    target_rhyme_group="江阳",
)

DRAFT = LyricDraft(
    title="晚风回家",
    lyrics=(
        "[Verse]\n路灯把影子拉得很长\n晚风绕过熟悉的街巷\n"
        "我把疲惫放进微亮的窗\n回家的方向有温柔的光\n\n"
        "[Chorus]\n晚风在等我 回家的路上\n晚风在等我 回家的路上\n"
        "城市的心跳不再匆忙\n今晚的月色替我原谅\n\n"
        "[Bridge]\n远处车灯像星河流淌\n我终于学会和自己相望"
    ),
    hook="晚风在等我 回家的路上",
    creative_note="测试歌词",
)


def dimension(score: float, evidence: str = "晚风在等我") -> dict[str, object]:
    return {"score": score, "evidence": evidence, "reason": "与歌曲需求能够对应"}


def candidate_review(score: float) -> dict[str, object]:
    return {
        "theme_fit": dimension(score),
        "naturalness": dimension(score),
        "emotional_control": dimension(score),
        "narrative_imagery_coherence": dimension(score),
        "hook_memorability": dimension(score),
        "strengths": ["主题表达具体"],
        "issues": [],
    }


class FakeJsonLLM:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[Sequence[Mapping[str, str]]] = []
        self.kwargs: list[dict[str, Any]] = []

    def chat_json(
        self, messages: Sequence[Mapping[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        self.calls.append(messages)
        self.kwargs.append(kwargs)
        return dict(self.response)


class D8CriticTests(unittest.TestCase):
    def test_weighted_score_is_calculated_by_program(self) -> None:
        review = CandidateSubjectiveReview.model_validate(candidate_review(5.0))
        self.assertEqual(weighted_subjective_score(review), 100.0)

    def test_dimension_score_rejects_non_half_point_value(self) -> None:
        with self.assertRaises(ValidationError):
            SubjectiveDimensionScore(
                score=4.2, evidence="具体歌词", reason="具体理由"
            )

    def test_judge_prompt_is_anonymous_and_excludes_rule_scores(self) -> None:
        response = {
            "candidate_a": candidate_review(4.5),
            "candidate_b": candidate_review(3.5),
            "recommended_winner": "A",
            "confidence": "high",
            "comparison_summary": "A 的主题方向和自然表达更加完整。",
        }
        llm = FakeJsonLLM(response)
        candidates = [
            {
                "candidate_id": "structured",
                "draft": DRAFT.model_dump(mode="json"),
                "scores": {"total": 91.234},
                "quality_report": {"passed": True},
            },
            {
                "candidate_id": "imagery",
                "draft": DRAFT.model_copy(update={"title": "街灯"}).model_dump(mode="json"),
                "scores": {"total": 98.765},
                "quality_report": {"passed": True},
            },
        ]

        report = LyricJudgeAgent(llm).run(REQUIREMENT, candidates)

        system_prompt = llm.calls[0][0]["content"]
        user_prompt = llm.calls[0][1]["content"]
        self.assertIn("遗漏一层不能给高分", system_prompt)
        self.assertIn("不能只凭一个命令句", system_prompt)
        self.assertNotIn("structured", user_prompt)
        self.assertNotIn("imagery", user_prompt)
        self.assertNotIn("candidate_id", user_prompt)
        self.assertNotIn("91.234", user_prompt)
        self.assertNotIn("98.765", user_prompt)
        self.assertEqual(set(report["candidate_reviews"]), {"structured", "imagery"})
        self.assertEqual(
            sorted(
                item["weighted_total"]
                for item in report["candidate_reviews"].values()
            ),
            [70.0, 90.0],
        )
        self.assertEqual(llm.kwargs[0]["operation"], "lyric_judge_initial")
        self.assertEqual(llm.kwargs[0]["temperature"], 0.2)
        self.assertEqual(llm.kwargs[0]["max_tokens"], 2400)
        self.assertTrue(callable(llm.kwargs[0]["payload_validator"]))

    def test_major_theme_issue_is_derived_from_structured_evidence(self) -> None:
        a = candidate_review(4.0)
        b = candidate_review(4.0)
        a["issues"] = [
            {
                "code": "theme_drift",
                "severity": "major",
                "evidence": "伞下隔阴阳",
                "suggestion": "改回久别重逢，不要写成生死相隔",
            }
        ]
        response = {
            "candidate_a": a,
            "candidate_b": b,
            "recommended_winner": "B",
            "confidence": "high",
            "comparison_summary": "A 出现重大主题偏移，B 保持原命题。",
        }
        report = LyricJudgeAgent(FakeJsonLLM(response)).run(
            REQUIREMENT,
            [
                {"candidate_id": "structured", "draft": DRAFT.model_dump(mode="json")},
                {"candidate_id": "imagery", "draft": DRAFT.model_dump(mode="json")},
            ],
        )

        flagged = [
            item
            for item in report["candidate_reviews"].values()
            if item["major_theme_issue"]
        ]
        self.assertEqual(len(flagged), 1)


if __name__ == "__main__":
    unittest.main()
