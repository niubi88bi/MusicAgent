"""D3 Agent 的无网络契约测试。"""

import unittest
from typing import Any, Mapping, Sequence

from agents.lyricist_agent import LyricistAgent
from agents.producer_agent import ProducerAgent
from agents.requirement_agent import RequirementAgent
from agents.schemas import LyricDraft, SongRequirement, pydantic_validation_feedback


class FakeJsonLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self.calls: list[Sequence[Mapping[str, str]]] = []
        self.call_kwargs: list[dict[str, Any]] = []

    def chat_json(
        self, messages: Sequence[Mapping[str, str]], **_: Any
    ) -> dict[str, Any]:
        self.calls.append(messages)
        self.call_kwargs.append(dict(_))
        return self._responses.pop(0)


REQUIREMENT = {
    "theme": "夏夜回家",
    "emotion": "温暖、释然",
    "genre": "city pop",
    "tempo": "mid-tempo",
    "duration_seconds": 90,
    "perspective": "第一人称",
    "vocal_type": "温柔男声",
    "bpm": 96,
    "emotion_intensity": 3,
    "keywords": ["夏夜", "路灯", "晚风"],
    "target_rhyme_group": "江阳",
}

LYRIC_DRAFT = {
    "title": "晚风在等我",
    "lyrics": (
        "[Verse]\n路灯把影子拉得很长\n晚风绕过熟悉的街巷\n"
        "我把疲惫放进微亮的窗\n回家的方向有温柔的光\n\n"
        "[Chorus]\n晚风在等我 回家的路上\n晚风在等我 回家的路上\n"
        "城市的心跳不再匆忙\n今晚的月色替我原谅\n\n"
        "[Bridge]\n远处的车灯像星河流淌\n我终于学会和自己相望"
    ),
    "hook": "晚风在等我 回家的路上",
    "creative_note": "以路灯和晚风承接下班后的释然。",
}

PRODUCTION_PLAN = {
    "style_prompt": (
        "City pop, warm midnight mood, modern Chinese pop production, "
        "electric piano and clean guitar, gentle male vocal, restrained verse "
        "building into a luminous chorus."
    ),
    "tags": ["city pop", "warm", "male vocal", "night drive"],
    "vocal_direction": "Close-mic gentle male vocal, relaxed phrasing.",
    "arrangement_note": "Use electric piano in verse and expand with guitar in chorus.",
}


class D3AgentTests(unittest.TestCase):
    def test_three_agents_produce_valid_contracts(self) -> None:
        llm = FakeJsonLLM([REQUIREMENT, LYRIC_DRAFT, PRODUCTION_PLAN])
        requirement = RequirementAgent(llm).run("写一首关于夏夜回家的歌")
        lyric_draft = LyricistAgent(llm).run(requirement)
        production_plan = ProducerAgent(llm).run(requirement, lyric_draft)

        self.assertEqual(requirement.bpm, 96)
        self.assertIn("[Bridge]", lyric_draft.lyrics)
        self.assertEqual(lyric_draft.hook, "晚风在等我 回家的路上")
        self.assertIn("City pop", production_plan.style_prompt)
        self.assertEqual(len(llm.calls), 3)

    def test_lyric_contract_rejects_missing_hook(self) -> None:
        invalid = dict(LYRIC_DRAFT)
        invalid["hook"] = "不存在的 Hook"
        with self.assertRaises(ValueError):
            LyricDraft.model_validate(invalid)

    def test_lyric_contract_feedback_is_safe_and_specific(self) -> None:
        invalid = dict(LYRIC_DRAFT)
        invalid["hook"] = "不存在的 Hook"

        feedback = pydantic_validation_feedback(LyricDraft, invalid)

        self.assertIn("hook 必须原样出现在歌词正文中", feedback or "")
        self.assertIsNone(pydantic_validation_feedback(LyricDraft, LYRIC_DRAFT))

    def test_lyric_contract_normalizes_numbered_section_labels(self) -> None:
        numbered = dict(LYRIC_DRAFT)
        numbered["lyrics"] = numbered["lyrics"].replace("[Verse]", "[Verse 1]")
        numbered["lyrics"] = numbered["lyrics"].replace("[Chorus]", "[Chorus 1]")
        numbered["lyrics"] = numbered["lyrics"].replace("[Bridge]", "[Bridge 1]")
        draft = LyricDraft.model_validate(numbered)
        self.assertIn("[Verse]", draft.lyrics)
        self.assertNotIn("[Verse 1]", draft.lyrics)

    def test_lyric_rewrite_includes_previous_draft_and_feedback(self) -> None:
        requirement = SongRequirement.model_validate(REQUIREMENT)
        previous_draft = LyricDraft.model_validate(LYRIC_DRAFT)
        llm = FakeJsonLLM([LYRIC_DRAFT])
        LyricistAgent(llm).run(
            requirement,
            revision_feedback="第 8 行：请修正韵脚。",
            previous_draft=previous_draft,
        )
        user_message = llm.calls[0][1]["content"]
        self.assertIn(previous_draft.lyrics, user_message)
        self.assertIn("第 8 行", user_message)
        self.assertIn("保留未被要求修改的内容", user_message)
        self.assertIn("不能完全另起一稿", user_message)

    def test_requirement_rejects_invalid_theme(self) -> None:
        with self.assertRaises(ValueError):
            RequirementAgent(FakeJsonLLM([REQUIREMENT])).run(" ")

    def test_requirement_normalizes_final_to_standard_rhyme_group(self) -> None:
        requirement = dict(REQUIREMENT)
        requirement["target_rhyme_group"] = "ang"

        parsed = SongRequirement.model_validate(requirement)

        self.assertEqual(parsed.target_rhyme_group, "江阳")

    def test_requirement_rejects_unknown_rhyme_target(self) -> None:
        requirement = dict(REQUIREMENT)
        requirement["target_rhyme_group"] = "不存在的辙"

        with self.assertRaises(ValueError):
            SongRequirement.model_validate(requirement)

    def test_dual_personas_have_distinct_prompts_and_operation_names(self) -> None:
        structured_llm = FakeJsonLLM([LYRIC_DRAFT])
        imagery_llm = FakeJsonLLM([LYRIC_DRAFT])
        requirement = SongRequirement.model_validate(REQUIREMENT)

        LyricistAgent(structured_llm, persona="structured").run(requirement)
        LyricistAgent(imagery_llm, persona="imagery").run(requirement)

        structured_prompt = structured_llm.calls[0][0]["content"]
        imagery_prompt = imagery_llm.calls[0][0]["content"]
        self.assertIn("韵脚稳定", structured_prompt)
        self.assertIn("感官细节", imagery_prompt)
        self.assertNotEqual(structured_prompt, imagery_prompt)
        self.assertEqual(
            structured_llm.call_kwargs[0]["operation"],
            "lyricist_structured_initial",
        )
        self.assertEqual(
            imagery_llm.call_kwargs[0]["operation"],
            "lyricist_imagery_initial",
        )

    def test_producer_receives_lyrics(self) -> None:
        requirement = SongRequirement.model_validate(REQUIREMENT)
        lyric_draft = LyricDraft.model_validate(LYRIC_DRAFT)
        llm = FakeJsonLLM([PRODUCTION_PLAN])
        ProducerAgent(llm).run(requirement, lyric_draft)
        user_message = llm.calls[0][1]["content"]
        self.assertIn(lyric_draft.title, user_message)
        self.assertIn(lyric_draft.hook, user_message)

    def test_agent_system_prompts_include_json_structure_examples(self) -> None:
        llm = FakeJsonLLM([REQUIREMENT, LYRIC_DRAFT, PRODUCTION_PLAN])
        requirement = RequirementAgent(llm).run("写一首关于夏夜回家的歌")
        lyric_draft = LyricistAgent(llm).run(requirement)
        ProducerAgent(llm).run(requirement, lyric_draft)

        for messages in llm.calls:
            self.assertIn("JSON 输出结构示例", messages[0]["content"])
        self.assertEqual(
            [kwargs["operation"] for kwargs in llm.call_kwargs],
            ["requirement", "lyricist_initial", "producer"],
        )
        self.assertTrue(
            all(callable(kwargs["payload_validator"]) for kwargs in llm.call_kwargs)
        )


if __name__ == "__main__":
    unittest.main()
