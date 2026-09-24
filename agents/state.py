"""LangGraph 工作流共享状态。"""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import NotRequired, TypedDict

from agents.schemas import LyricDraft, ProductionPlan, SongRequirement


class MusicWorkflowState(TypedDict):
    """歌词候选、回炉、制作和音频阶段共享的 LangGraph 状态。"""

    theme: str
    requested_duration_seconds: NotRequired[int]
    production_constraints: NotRequired[dict[str, object]]
    requirement: NotRequired[SongRequirement]
    candidate_persona: NotRequired[str]
    candidate_generation_round: NotRequired[int]
    candidate_revision_feedback: NotRequired[str]
    candidate_previous_draft: NotRequired[LyricDraft]
    lyric_candidates: Annotated[list[dict[str, object]], operator.add]
    subjective_review: NotRequired[dict[str, object]]
    subjective_review_history: NotRequired[list[dict[str, object]]]
    candidate_regeneration_count: NotRequired[int]
    candidate_selection: NotRequired[dict[str, object]]
    selected_lyricist: NotRequired[str]
    lyric_draft: NotRequired[LyricDraft]
    lyrics: NotRequired[str]
    production_plan: NotRequired[ProductionPlan]
    scores: NotRequired[dict[str, float]]
    quality_report: NotRequired[dict[str, object]]
    quality_status: NotRequired[str]
    revision_feedback: NotRequired[str | None]
    revision_source_attempt: NotRequired[int | None]
    quality_history: NotRequired[list[dict[str, object]]]
    best_lyric_draft: NotRequired[LyricDraft]
    best_quality_report: NotRequired[dict[str, object]]
    best_quality_score: NotRequired[float]
    best_scores: NotRequired[dict[str, float]]
    best_attempt: NotRequired[int]
    selected_attempt: NotRequired[int]
    audio_path: NotRequired[str]
    attempts: NotRequired[int]
    error: NotRequired[str]
