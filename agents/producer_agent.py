"""将歌曲需求和歌词转换为音频模型使用的制作方案。"""

import json

from agents.schemas import (
    JsonLLM,
    LyricDraft,
    ProductionPlan,
    SongRequirement,
    pydantic_validation_feedback,
)


class ProducerAgent:
    """D3 制作 Agent，输出风格提示词和可检索的标签。"""

    def __init__(self, llm: JsonLLM) -> None:
        self._llm = llm

    def run(
        self,
        requirement: SongRequirement,
        lyric_draft: LyricDraft,
    ) -> ProductionPlan:
        payload = self._llm.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是音乐制作人，为文本到音乐模型设计提示词。必须只输出 JSON 对象。"
                        "style_prompt 用英文，覆盖 Genre、Mood、Era、Instruments、Vocal、Dynamics，"
                        "不要包含真实艺人名。tags 使用简短英文或中文标签。"
                        "JSON 输出结构示例（仅说明字段和类型，不得照抄内容）："
                        '{"style_prompt":"City pop, warm mood, electric piano, gentle vocal",'
                        '"tags":["city pop","warm","night"],"vocal_direction":"gentle close-mic vocal",'
                        '"arrangement_note":"start sparse and expand in chorus"}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "歌曲需求：\n"
                        f"{json.dumps(requirement.model_dump(), ensure_ascii=False)}\n\n"
                        "歌词草稿：\n"
                        f"标题：{lyric_draft.title}\n{lyric_draft.lyrics}\n\n"
                        "请输出 JSON，字段必须为：style_prompt、tags（至少 3 项）、"
                        "vocal_direction、arrangement_note。"
                    ),
                },
            ],
            temperature=0.55,
            max_tokens=1000,
            operation="producer",
            payload_validator=lambda payload: pydantic_validation_feedback(
                ProductionPlan, payload
            ),
        )
        return ProductionPlan.model_validate(payload)
