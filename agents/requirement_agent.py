"""将自然语言主题转为可执行的歌曲需求。"""

from agents.schemas import JsonLLM, SongRequirement, pydantic_validation_feedback


class RequirementAgent:
    """D3 需求理解 Agent。"""

    def __init__(self, llm: JsonLLM) -> None:
        self._llm = llm

    def run(self, theme: str) -> SongRequirement:
        clean_theme = theme.strip()
        if len(clean_theme) < 2:
            raise ValueError("主题至少需要两个字符")

        payload = self._llm.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是中文歌曲制作需求分析师。将用户主题转为可执行的歌曲需求。"
                        "必须只输出 JSON 对象，不要 Markdown。duration_seconds 取 30 到 240 的整数；"
                        "bpm 取 60 到 180；emotion_intensity 取 1 到 5。"
                        "target_rhyme_group 只能使用标准十三辙名称（如江阳、发花）或 null，"
                        "不能填写 ang、uang 等拼音韵母。"
                        "JSON 输出结构示例（仅说明字段和类型，不得照抄内容）："
                        '{"theme":"示例主题","emotion":"示例情绪","genre":"示例曲风",'
                        '"tempo":"mid-tempo","duration_seconds":90,"perspective":"第一人称",'
                        '"vocal_type":"温柔人声","bpm":96,"emotion_intensity":3,'
                        '"keywords":["关键词一","关键词二"],"target_rhyme_group":"江阳"}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"主题：{clean_theme}\n"
                        "请输出 JSON，字段必须为：theme、emotion、genre、tempo、"
                        "duration_seconds、perspective、vocal_type、bpm、emotion_intensity、"
                        "keywords（字符串数组，2-8 项）、target_rhyme_group（可为 null）。"
                    ),
                },
            ],
            temperature=0.4,
            max_tokens=800,
            operation="requirement",
            payload_validator=lambda payload: pydantic_validation_feedback(
                SongRequirement, payload
            ),
        )
        return SongRequirement.model_validate(payload)
