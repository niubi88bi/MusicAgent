"""D7 条件边、回炉上限与音频单次调用的离线测试。"""

from pathlib import Path
import unittest

from agents.graph import build_music_graph
from agents.schemas import LyricDraft, ProductionPlan, SongRequirement
from services.music_gen import MusicGenerationResult


REQUIREMENT = SongRequirement(
    theme="夏夜回家",
    emotion="温暖",
    genre="city pop",
    tempo="mid-tempo",
    duration_seconds=30,
    perspective="第一人称",
    vocal_type="温柔男声",
    bpm=96,
    emotion_intensity=3,
    keywords=["夏夜", "路灯"],
    target_rhyme_group="江阳",
)

GOOD_DRAFT = LyricDraft(
    title="晚风在等我",
    lyrics="""[Verse]
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
我终于学会和自己相望""",
    hook="晚风在等我 回家的路上",
    creative_note="测试用好歌词。",
)

BAD_DRAFT = LyricDraft(
    title="未对齐的句尾",
    lyrics="""[Verse]
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
这一刻想起你""",
    hook="不要说再见的光",
    creative_note="测试用坏歌词。",
)

WORSE_DRAFT = LyricDraft(
    title="更差的尾韵",
    lyrics="""[Verse]
路灯照亮安静的雨
故事停在旧书的页
我听见远方的海
脚步落在陌生的门

[Chorus]
不要说再见的雨
不要说再见的雨
明天会有新的海
我们走向陌生的门

[Bridge]
这一刻想起你""",
    hook="不要说再见的雨",
    creative_note="测试用更低分歌词。",
)


class FakeRequirementAgent:
    def run(self, theme: str) -> SongRequirement:
        return REQUIREMENT


class SequenceLyricist:
    def __init__(self, drafts: list[LyricDraft]) -> None:
        self.drafts = drafts
        self.calls: list[str | None] = []
        self.previous_drafts: list[LyricDraft | None] = []

    def run(
        self,
        requirement: SongRequirement,
        revision_feedback: str | None = None,
        previous_draft: LyricDraft | None = None,
    ) -> LyricDraft:
        self.calls.append(revision_feedback)
        self.previous_drafts.append(previous_draft)
        return self.drafts[min(len(self.calls) - 1, len(self.drafts) - 1)]


class FakeProducerAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, requirement: SongRequirement, lyric_draft: LyricDraft) -> ProductionPlan:
        self.calls += 1
        return ProductionPlan(
            style_prompt="Warm city pop with electric piano and gentle male vocal.",
            tags=["city pop", "warm", "male vocal"],
            vocal_direction="gentle male vocal",
            arrangement_note="electric piano in verse",
        )


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self, lyrics: str, tags: list[str], duration_seconds: int
    ) -> MusicGenerationResult:
        self.calls += 1
        return MusicGenerationResult(
            job_id="retry-test",
            output_path=Path("tmp/test-work/d7/song.mp3"),
            lyrics_path=Path("tmp/test-work/d7/lyrics.txt"),
            tags_path=Path("tmp/test-work/d7/tags.txt"),
            duration_seconds=duration_seconds,
        )


class D7RetryTests(unittest.TestCase):
    def build_graph(self, drafts: list[LyricDraft]):
        lyricist = SequenceLyricist(drafts)
        producer = FakeProducerAgent()
        generator = FakeGenerator()
        graph = build_music_graph(FakeRequirementAgent(), lyricist, producer, generator)
        return graph, lyricist, producer, generator

    def test_bad_first_draft_rewrites_once_before_audio_generation(self) -> None:
        graph, lyricist, producer, generator = self.build_graph([BAD_DRAFT, GOOD_DRAFT])
        result = graph.invoke({"theme": "夏夜回家"})

        self.assertEqual(len(lyricist.calls), 2)
        self.assertIsNone(lyricist.calls[0])
        self.assertIn("规则分", lyricist.calls[1])
        self.assertIsNone(lyricist.previous_drafts[0])
        self.assertEqual(lyricist.previous_drafts[1], BAD_DRAFT)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["quality_status"], "passed")
        self.assertEqual(
            [item["quality_status"] for item in result["quality_history"]],
            ["retrying", "passed"],
        )
        self.assertEqual(result["selected_attempt"], 2)
        self.assertEqual(result["quality_history"][1]["revision_source_attempt"], 1)
        self.assertEqual(producer.calls, 1)
        self.assertEqual(generator.calls, 1)

    def test_always_bad_draft_stops_after_three_attempts(self) -> None:
        graph, lyricist, producer, generator = self.build_graph([BAD_DRAFT])
        result = graph.invoke({"theme": "夏夜回家"})

        self.assertEqual(len(lyricist.calls), 3)
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["quality_status"], "capped")
        self.assertFalse(result["quality_report"]["passed"])
        self.assertEqual(len(result["quality_history"]), 3)
        self.assertEqual(result["quality_history"][-1]["quality_status"], "capped")
        self.assertEqual(producer.calls, 1)
        self.assertEqual(generator.calls, 1)

    def test_cap_keeps_the_highest_scored_draft(self) -> None:
        graph, lyricist, _, _ = self.build_graph(
            [BAD_DRAFT, WORSE_DRAFT, WORSE_DRAFT]
        )
        result = graph.invoke({"theme": "夏夜回家"})

        self.assertEqual(len(lyricist.calls), 3)
        self.assertEqual(result["quality_status"], "capped")
        self.assertEqual(result["lyric_draft"].title, BAD_DRAFT.title)
        self.assertEqual(result["selected_attempt"], 1)
        self.assertEqual(
            result["scores"]["total"], result["quality_report"]["total_score"]
        )


if __name__ == "__main__":
    unittest.main()
