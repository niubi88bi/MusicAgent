"""根据结构化需求创作可供音频模型使用的歌词草稿。"""

import json
from dataclasses import dataclass

from agents.schemas import (
    JsonLLM,
    LyricDraft,
    SongRequirement,
    pydantic_validation_feedback,
)


@dataclass(frozen=True)
class LyricistPersona:
    """双作词人的稳定身份与创作侧重点。"""

    persona_id: str
    display_name: str
    instruction: str


LYRICIST_PERSONAS = {
    "balanced": LyricistPersona(
        persona_id="balanced",
        display_name="综合作词人",
        instruction="兼顾中文表达、段落结构和韵脚自然度。",
    ),
    "structured": LyricistPersona(
        persona_id="structured",
        display_name="作词人 A · 押韵结构",
        instruction=(
            "优先保证段落结构、同段韵脚稳定和 Hook 可记忆；如果需求给出目标辙口，"
            "尽量自然使用该辙口，不能为了押韵堆砌生硬词语。"
        ),
    ),
    "imagery": LyricistPersona(
        persona_id="imagery",
        display_name="作词人 B · 意象表达",
        instruction=(
            "优先使用具体场景、感官细节和有推进的叙事，减少空泛套话；"
            "同时仍须满足段落标签、Hook 和基本押韵要求。"
        ),
    ),
}


class LyricistAgent:
    """支持综合、押韵结构和意象表达三种稳定 persona 的作词 Agent。"""

    def __init__(self, llm: JsonLLM, persona: str = "balanced") -> None:
        if persona not in LYRICIST_PERSONAS:
            raise ValueError(f"未知作词人 persona：{persona}")
        self._llm = llm
        self._persona = LYRICIST_PERSONAS[persona]

    @property
    def persona_id(self) -> str:
        return self._persona.persona_id

    @property
    def display_name(self) -> str:
        return self._persona.display_name

    def run(
        self,
        requirement: SongRequirement,
        revision_feedback: str | None = None,
        previous_draft: LyricDraft | None = None,
    ) -> LyricDraft:
        feedback = revision_feedback or "无"
        previous_lyrics = "无（这是首次创作）"
        revision_instruction = "请根据歌曲需求创作一版完整歌词。"
        if previous_draft is not None:
            previous_lyrics = (
                f"标题：{previous_draft.title}\n"
                f"Hook：{previous_draft.hook}\n"
                f"歌词：\n{previous_draft.lyrics}"
            )
            revision_instruction = (
                "这是回炉修改：必须以上一版歌词为基础，保留未被要求修改的内容、"
                "主题和可用意象；仅为解决反馈中的问题而改写，不能完全另起一稿。"
            )
        payload = self._llm.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是擅长中文流行音乐的作词人。必须只输出 JSON 对象，不要 Markdown。"
                        "lyrics 必须使用英文方括号段落标签，且至少各有一个 [Verse]、[Chorus]、[Bridge]。"
                        "hook 是一句短而好记的中文歌词，必须在 lyrics 中原样出现。"
                        "避免解释、避免真实歌手名、避免侵权歌词。"
                        f"你的创作侧重点：{self._persona.instruction}"
                        "JSON 输出结构示例（仅说明格式，示例内容不得照抄）："
                        '{"title":"示例标题","lyrics":"[Verse]\\n示例主歌\\n[Chorus]\\n示例 Hook\\n[Bridge]\\n示例桥段",'
                        '"hook":"示例 Hook","creative_note":"一句创作说明"}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "歌曲需求：\n"
                        f"{json.dumps(requirement.model_dump(), ensure_ascii=False)}\n\n"
                        f"上一版歌词：\n{previous_lyrics}\n\n"
                        f"修改反馈：{feedback}\n\n"
                        f"{revision_instruction}\n"
                        "请输出 JSON，字段必须为：title、lyrics、hook、creative_note。"
                    ),
                },
            ],
            temperature=0.85,
            max_tokens=2200,
            operation=self._operation_name(revision_feedback is not None),
            payload_validator=lambda payload: pydantic_validation_feedback(
                LyricDraft, payload
            ),
        )
        return LyricDraft.model_validate(payload)

    def _operation_name(self, is_rewrite: bool) -> str:
        suffix = "rewrite" if is_rewrite else "initial"
        if self.persona_id == "balanced":
            return f"lyricist_{suffix}"
        return f"lyricist_{self.persona_id}_{suffix}"
