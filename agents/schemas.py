"""D3 文本创作链路的输入输出契约。"""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from tools.lyrics_quality.thirteen_rhymes import normalize_rhyme_group


class JsonLLM(Protocol):
    """Agent 所需的最小 LLM 能力，便于用 mock 测试。"""

    def chat_json(
        self,
        messages: Sequence[Mapping[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]: ...


def pydantic_validation_feedback(
    model_type: type[BaseModel], payload: dict[str, Any]
) -> str | None:
    """将模型契约校验转为可安全交给下一次 LLM 重试的简短反馈。"""
    try:
        model_type.model_validate(payload)
    except ValidationError as error:
        messages: list[str] = []
        for issue in error.errors()[:3]:
            location = ".".join(str(item) for item in issue["loc"]) or "整份输出"
            message = str(issue["msg"]).removeprefix("Value error, ")
            messages.append(f"{location}：{message}")
        return "；".join(messages)
    return None


class SongRequirement(BaseModel):
    """需求理解 Agent 输出的结构化歌曲需求。"""

    theme: str = Field(min_length=2, max_length=200)
    emotion: str = Field(min_length=1, max_length=40)
    genre: str = Field(min_length=1, max_length=60)
    tempo: str = Field(min_length=1, max_length=40)
    duration_seconds: int = Field(ge=30, le=240)
    perspective: str = Field(min_length=1, max_length=40)
    vocal_type: str = Field(min_length=1, max_length=60)
    bpm: int = Field(ge=60, le=180)
    emotion_intensity: int = Field(ge=1, le=5)
    keywords: list[str] = Field(min_length=2, max_length=8)
    target_rhyme_group: str | None = Field(default=None, max_length=20)

    @field_validator("keywords")
    @classmethod
    def keywords_must_be_nonempty(cls, keywords: list[str]) -> list[str]:
        cleaned = [keyword.strip() for keyword in keywords if keyword.strip()]
        if len(cleaned) < 2:
            raise ValueError("keywords 至少包含两个非空关键词")
        return cleaned

    @field_validator("target_rhyme_group")
    @classmethod
    def target_rhyme_group_must_be_standard(
        cls, target_rhyme_group: str | None
    ) -> str | None:
        if target_rhyme_group is None:
            return None
        normalized = normalize_rhyme_group(target_rhyme_group)
        if normalized is None:
            raise ValueError(
                "target_rhyme_group 必须为标准十三辙名称或可映射的普通话韵母"
            )
        return normalized


class LyricDraft(BaseModel):
    """作词 Agent 输出。歌词正文保留 HeartMuLa 所需的段落标签。"""

    title: str = Field(min_length=1, max_length=80)
    lyrics: str = Field(min_length=80, max_length=6000)
    hook: str = Field(min_length=2, max_length=120)
    creative_note: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_song_structure(self) -> "LyricDraft":
        # LLM 常输出 [Verse 1] / [Chorus 2]。统一为 HeartMuLa 兼容的基础标签，
        # 保留段落顺序，同时避免因为编号格式差异误判结构缺失。
        self.lyrics = re.sub(r"\[(Verse|Chorus|Bridge)\s+\d+\]", r"[\1]", self.lyrics)
        required_labels = ("[Verse]", "[Chorus]", "[Bridge]")
        missing = [label for label in required_labels if label not in self.lyrics]
        if missing:
            raise ValueError(f"歌词缺少结构标签：{', '.join(missing)}")
        if self.hook not in self.lyrics:
            raise ValueError("hook 必须原样出现在歌词正文中")
        return self


class SubjectiveDimensionScore(BaseModel):
    """主观评审单项分数；证据与理由必须同时存在。"""

    score: float = Field(ge=1.0, le=5.0)
    evidence: str = Field(min_length=2, max_length=240)
    reason: str = Field(min_length=2, max_length=300)

    @field_validator("score")
    @classmethod
    def score_uses_half_point_steps(cls, score: float) -> float:
        if abs(score * 2 - round(score * 2)) > 1e-9:
            raise ValueError("score 必须使用 0.5 分档")
        return score


class SubjectiveIssue(BaseModel):
    """主观问题标签，必须带歌词证据和可执行建议。"""

    code: Literal[
        "theme_drift",
        "emotion_overreach",
        "emotion_incomplete",
        "generic_expression",
        "imagery_mismatch",
        "narrative_gap",
        "hook_weakness",
        "other",
    ]
    severity: Literal["minor", "major"]
    evidence: str = Field(min_length=2, max_length=240)
    suggestion: str = Field(min_length=2, max_length=300)


class CandidateSubjectiveReview(BaseModel):
    """匿名候选的一组主观评分。"""

    theme_fit: SubjectiveDimensionScore
    naturalness: SubjectiveDimensionScore
    emotional_control: SubjectiveDimensionScore
    narrative_imagery_coherence: SubjectiveDimensionScore
    hook_memorability: SubjectiveDimensionScore
    strengths: list[str] = Field(min_length=1, max_length=3)
    issues: list[SubjectiveIssue] = Field(default_factory=list, max_length=5)

    @field_validator("strengths")
    @classmethod
    def strengths_must_be_nonempty(cls, strengths: list[str]) -> list[str]:
        cleaned = [item.strip() for item in strengths if item.strip()]
        if not cleaned:
            raise ValueError("strengths 至少包含一条非空优点")
        return cleaned


class LyricJudgeResponse(BaseModel):
    """评审 Agent 的匿名 A/B 输出；程序另行计算最终选择。"""

    candidate_a: CandidateSubjectiveReview
    candidate_b: CandidateSubjectiveReview
    recommended_winner: Literal["A", "B", "tie"]
    confidence: Literal["low", "medium", "high"]
    comparison_summary: str = Field(min_length=10, max_length=600)


class ProductionPlan(BaseModel):
    """制作 Agent 输出，供音频模型使用。"""

    style_prompt: str = Field(min_length=20, max_length=1200)
    tags: list[str] = Field(min_length=3, max_length=12)
    vocal_direction: str = Field(min_length=1, max_length=300)
    arrangement_note: str = Field(min_length=1, max_length=500)

    @field_validator("tags")
    @classmethod
    def tags_must_be_nonempty(cls, tags: list[str]) -> list[str]:
        cleaned = [tag.strip() for tag in tags if tag.strip()]
        if len(cleaned) < 3:
            raise ValueError("tags 至少包含三个非空标签")
        return cleaned
