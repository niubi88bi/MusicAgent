"""D5 的 LangGraph 线性工作流骨架。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.critic_agent import (
    MAX_CANDIDATE_REGENERATIONS,
    format_subjective_revision_feedback,
    select_subjective_candidate,
)
from agents.lyricist_agent import LyricistAgent
from agents.producer_agent import ProducerAgent
from agents.requirement_agent import RequirementAgent
from agents.schemas import LyricDraft
from agents.state import MusicWorkflowState
from tools.lyrics_quality.scorer import format_revision_feedback, score_lyrics


MAX_LYRIC_ATTEMPTS = 3


class MusicGenerator(Protocol):
    """图对音频服务的最小依赖，便于在无 GPU 测试中替换。"""

    def generate(
        self,
        lyrics: str,
        tags: list[str],
        duration_seconds: int,
    ) -> object: ...


class LyricJudge(Protocol):
    """图对匿名主观评审的最小依赖。"""

    def run(
        self,
        requirement: object,
        candidates: list[dict[str, object]],
    ) -> dict[str, object]: ...


def build_music_graph(
    requirement_agent: RequirementAgent,
    lyricist_agent: LyricistAgent,
    producer_agent: ProducerAgent,
    music_generator: MusicGenerator | None = None,
    *,
    secondary_lyricist_agent: LyricistAgent | None = None,
    lyric_judge_agent: LyricJudge | None = None,
    text_only: bool = False,
    on_event: Callable[[str, str, dict], None] | None = None,
):
    """编译歌词工作流；可选双作词与匿名评审，质检后最多回炉两次。"""
    if not text_only and music_generator is None:
        raise ValueError("音频模式必须提供 music_generator")

    dual_lyricists = secondary_lyricist_agent is not None
    if lyric_judge_agent is not None and not dual_lyricists:
        raise ValueError("主观评审只适用于双作词工作流")
    candidate_agents: dict[str, tuple[str, LyricistAgent]] = {}
    if dual_lyricists:
        for fallback_id, fallback_name, agent in (
            ("structured", "作词人 A · 押韵结构", lyricist_agent),
            ("imagery", "作词人 B · 意象表达", secondary_lyricist_agent),
        ):
            persona_id = str(getattr(agent, "persona_id", fallback_id))
            display_name = str(getattr(agent, "display_name", fallback_name))
            if persona_id in candidate_agents:
                raise ValueError("双作词人的 persona_id 必须不同")
            candidate_agents[persona_id] = (display_name, agent)
    candidate_order = {persona_id: index for index, persona_id in enumerate(candidate_agents)}

    def current_candidates(state: MusicWorkflowState) -> list[dict[str, object]]:
        """只返回最新一轮两个候选，同时保留旧轮次供落盘审计。"""

        candidates = list(state.get("lyric_candidates", []))
        if not candidates:
            raise RuntimeError("双作词人候选未返回")
        latest_round = max(int(item.get("generation_round", 1)) for item in candidates)
        current = [
            item
            for item in candidates
            if int(item.get("generation_round", 1)) == latest_round
        ]
        if len(current) != len(candidate_agents):
            raise RuntimeError("双作词人最新一轮候选未完整返回")
        return current

    def rule_rank(candidate: dict[str, object]) -> tuple[bool, float, float, float, int]:
        report = candidate["quality_report"]
        scores = candidate["scores"]
        if not isinstance(report, dict) or not isinstance(scores, dict):
            raise TypeError("作词候选缺少规则报告")
        return (
            bool(report.get("passed")),
            float(scores["total"]),
            float(scores["rhyme"]),
            float(scores["structure"]),
            -candidate_order[str(candidate["candidate_id"])],
        )

    def observed(name, function):
        # 回调只观察实际节点边界；不改变评分、重写次数或选优策略。
        def run(state):
            if on_event:
                on_event(name, "running", dict(state))
            try:
                update = function(state)
            except Exception:
                if on_event:
                    on_event(name, "failed", dict(state))
                raise
            if on_event:
                on_event(name, "done", {**state, **update})
            return update
        return run

    def validate_input_node(state: MusicWorkflowState) -> dict[str, object]:
        theme = state.get("theme", "")
        if not isinstance(theme, str) or len(theme.strip()) < 2:
            raise ValueError("主题至少需要两个字符")
        result: dict[str, object] = {
            "theme": theme.strip(),
            "attempts": state.get("attempts", 0),
        }
        requested_duration = state.get("requested_duration_seconds")
        if requested_duration is not None:
            if (
                isinstance(requested_duration, bool)
                or not isinstance(requested_duration, int)
                or not 1 <= requested_duration <= 240
            ):
                raise ValueError("requested_duration_seconds 必须在 1 到 240 之间")
            result["requested_duration_seconds"] = requested_duration
        return result

    def requirement_node(state: MusicWorkflowState) -> dict[str, object]:
        requirement = requirement_agent.run(state["theme"])
        requested_duration = state.get("requested_duration_seconds")
        if requested_duration is not None:
            # CLI 的显式时长优先于模型推断，确保歌词、制作方案与音频长度
            # 使用同一个约束；否则会出现“按 30 秒生成、需求记录却是 180 秒”的歧义。
            requirement = requirement.model_copy(
                update={"duration_seconds": requested_duration}
            )
        return {"requirement": requirement}

    def lyricist_node(state: MusicWorkflowState) -> dict[str, object]:
        feedback = state.get("revision_feedback")
        active_lyricist = lyricist_agent
        selected_persona = state.get("selected_lyricist")
        if dual_lyricists and selected_persona:
            active_lyricist = candidate_agents[selected_persona][1]
        if feedback:
            lyric_draft = active_lyricist.run(
                state["requirement"],
                revision_feedback=feedback,
                previous_draft=state["lyric_draft"],
            )
        else:
            lyric_draft = active_lyricist.run(state["requirement"])
        return {
            "lyric_draft": lyric_draft,
            "lyrics": lyric_draft.lyrics,
            "attempts": state.get("attempts", 0) + 1,
            "revision_feedback": None,
            "revision_source_attempt": state.get("attempts") if feedback else None,
        }

    def dispatch_lyricists(state: MusicWorkflowState) -> list[Send]:
        """通过 Send 将相同需求分发给两个独立 persona。"""
        return [
            Send(
                "lyric_candidate",
                {
                    "requirement": state["requirement"],
                    "requested_duration_seconds": state.get("requested_duration_seconds"),
                    "candidate_persona": persona_id,
                    "candidate_generation_round": 1,
                },
            )
            for persona_id in candidate_agents
        ]

    def lyric_candidate_node(state: MusicWorkflowState) -> dict[str, object]:
        persona_id = state["candidate_persona"]
        display_name, agent = candidate_agents[persona_id]
        stage_name = f"lyricist_{persona_id}"
        if on_event:
            on_event(stage_name, "running", dict(state))
        try:
            revision_feedback = state.get("candidate_revision_feedback")
            previous_draft = state.get("candidate_previous_draft")
            if revision_feedback:
                draft = agent.run(
                    state["requirement"],
                    revision_feedback=revision_feedback,
                    previous_draft=previous_draft,
                )
            else:
                draft = agent.run(state["requirement"])
            requirement = state["requirement"]
            report = score_lyrics(
                lyrics=draft.lyrics,
                hook=draft.hook,
                duration_seconds=state.get(
                    "requested_duration_seconds", requirement.duration_seconds
                ),
                target_rhyme_group=requirement.target_rhyme_group,
                theme=requirement.theme,
                theme_keywords=requirement.keywords,
            )
            candidate = {
                "candidate_id": persona_id,
                "display_name": display_name,
                "generation_round": int(state.get("candidate_generation_round", 1)),
                "draft": draft.model_dump(mode="json"),
                "scores": {
                    "rhyme": report.rhyme.score,
                    "structure": report.structure.score,
                    "total": report.total_score,
                },
                "quality_report": report.to_dict(),
            }
        except Exception:
            if on_event:
                on_event(stage_name, "failed", dict(state))
            raise
        if on_event:
            on_event(stage_name, "done", {**state, "lyric_candidates": [candidate]})
        return {"lyric_candidates": [candidate]}

    def subjective_judge_node(state: MusicWorkflowState) -> dict[str, object]:
        if lyric_judge_agent is None:
            raise RuntimeError("主观评审 Agent 未配置")
        candidates = current_candidates(state)
        review = lyric_judge_agent.run(state["requirement"], candidates)
        review = dict(review)
        regeneration_count = int(state.get("candidate_regeneration_count", 0))
        both_major = bool(review.get("both_have_major_theme_issues"))
        regeneration_required = (
            both_major and regeneration_count < MAX_CANDIDATE_REGENERATIONS
        )
        review["generation_round"] = max(
            int(item.get("generation_round", 1)) for item in candidates
        )
        review["regeneration_required"] = regeneration_required
        review["regeneration_exhausted"] = (
            both_major and regeneration_count >= MAX_CANDIDATE_REGENERATIONS
        )
        history = [*state.get("subjective_review_history", []), review]
        return {
            "subjective_review": review,
            "subjective_review_history": history,
            "candidate_regeneration_count": (
                regeneration_count + 1 if regeneration_required else regeneration_count
            ),
        }

    def route_after_subjective_judge(
        state: MusicWorkflowState,
    ) -> str | list[Send]:
        review = state.get("subjective_review")
        if not isinstance(review, dict):
            raise TypeError("主观评审未返回可路由结果")
        if not review.get("regeneration_required"):
            return "select_lyricist"

        next_round = int(review["generation_round"]) + 1
        candidates = current_candidates(state)
        reviews = review.get("candidate_reviews")
        if not isinstance(reviews, dict):
            raise TypeError("主观评审缺少候选反馈")
        return [
            Send(
                "lyric_candidate",
                {
                    "requirement": state["requirement"],
                    "requested_duration_seconds": state.get(
                        "requested_duration_seconds"
                    ),
                    "candidate_persona": str(candidate["candidate_id"]),
                    "candidate_generation_round": next_round,
                    "candidate_previous_draft": LyricDraft.model_validate(
                        candidate["draft"]
                    ),
                    "candidate_revision_feedback": format_subjective_revision_feedback(
                        reviews[str(candidate["candidate_id"])]
                    ),
                },
            )
            for candidate in candidates
        ]

    def select_lyricist_node(state: MusicWorkflowState) -> dict[str, object]:
        candidates = current_candidates(state)
        ordered = sorted(
            candidates, key=lambda item: candidate_order[str(item["candidate_id"])]
        )
        if lyric_judge_agent is None:
            selected = max(ordered, key=rule_rank)
            selection = {
                "method": "deterministic_rule_screen_v1",
                "priority": [
                    "quality_passed",
                    "total_score",
                    "rhyme_score",
                    "structure_score",
                    "stable_persona_order",
                ],
                "selected_candidate_id": str(selected["candidate_id"]),
                "selected_display_name": selected["display_name"],
                "reason": (
                    "先选择通过硬性规则的候选，再依次比较规则总分、押韵分、结构分；"
                    "完全同分时固定优先作词人 A。规则分不代表自然度或听感。"
                ),
            }
        else:
            review = state.get("subjective_review")
            if not isinstance(review, dict):
                raise TypeError("主观评审结果缺失")
            reviews = review.get("candidate_reviews")
            if not isinstance(reviews, dict):
                raise TypeError("主观评审候选分数缺失")

            selected, basis = select_subjective_candidate(
                ordered,
                review,
                candidate_order,
            )

            selected_id = str(selected["candidate_id"])
            selection = {
                "method": "blind_subjective_judge_v1",
                "priority": [
                    "major_theme_issue_filter",
                    "subjective_total_with_3_point_margin",
                    "quality_passed_for_near_tie",
                    "rule_total_for_near_tie",
                    "rhyme_score_for_near_tie",
                    "structure_score_for_near_tie",
                    "stable_persona_order",
                ],
                "selected_candidate_id": selected_id,
                "selected_display_name": selected["display_name"],
                "selection_basis": basis,
                "subjective_scores": {
                    candidate_id: candidate_review["weighted_total"]
                    for candidate_id, candidate_review in reviews.items()
                    if isinstance(candidate_review, dict)
                },
                "generation_round": review.get("generation_round", 1),
                "candidate_regeneration_count": state.get(
                    "candidate_regeneration_count", 0
                ),
                "regeneration_exhausted": bool(
                    review.get("regeneration_exhausted")
                ),
                "reason": (
                    "匿名主观评审先排除重大主题偏移，再比较五维加权分；"
                    "分差不足 3 分时才用规则通过状态和规则分决胜。"
                ),
            }
        selected_id = str(selected["candidate_id"])
        selected_draft = LyricDraft.model_validate(selected["draft"])
        return {
            "lyric_candidates": [],
            "candidate_selection": selection,
            "selected_lyricist": selected_id,
            "lyric_draft": selected_draft,
            "lyrics": selected_draft.lyrics,
            "attempts": 1,
            "revision_feedback": None,
            "revision_source_attempt": None,
        }

    def quality_node(state: MusicWorkflowState) -> dict[str, object]:
        requirement = state["requirement"]
        report = score_lyrics(
            lyrics=state["lyric_draft"].lyrics,
            hook=state["lyric_draft"].hook,
            duration_seconds=state.get(
                "requested_duration_seconds", requirement.duration_seconds
            ),
            target_rhyme_group=requirement.target_rhyme_group,
            theme=requirement.theme,
            theme_keywords=requirement.keywords,
        )
        report_dict = report.to_dict()
        scores = {
            "rhyme": report.rhyme.score,
            "structure": report.structure.score,
            "total": report.total_score,
        }
        result: dict[str, object] = {
            "quality_report": report_dict,
            "scores": scores,
        }
        history = [*state.get("quality_history", [])]
        attempt_record: dict[str, object] = {
            "attempt": state["attempts"],
            "title": state["lyric_draft"].title,
            "lyrics": state["lyric_draft"].lyrics,
            "hook": state["lyric_draft"].hook,
            "scores": scores,
            "quality_report": report_dict,
        }
        if state.get("selected_lyricist"):
            attempt_record["source_candidate_id"] = state["selected_lyricist"]
        revision_source_attempt = state.get("revision_source_attempt")
        if revision_source_attempt is not None:
            attempt_record["revision_source_attempt"] = revision_source_attempt
        previous_best = state.get("best_quality_score", -1.0)
        is_best = report.total_score >= previous_best
        if is_best:
            result.update(
                {
                    "best_lyric_draft": state["lyric_draft"],
                    "best_quality_report": report_dict,
                    "best_quality_score": report.total_score,
                    "best_scores": scores,
                    "best_attempt": state["attempts"],
                }
            )

        if report.passed:
            attempt_record["quality_status"] = "passed"
            history.append(attempt_record)
            result.update(
                {
                    "quality_history": history,
                    "quality_status": "passed",
                    "selected_attempt": state["attempts"],
                }
            )
            return result

        if state["attempts"] < MAX_LYRIC_ATTEMPTS:
            feedback = format_revision_feedback(report)
            attempt_record.update(
                {
                    "quality_status": "retrying",
                    "revision_feedback": feedback,
                }
            )
            history.append(attempt_record)
            result.update(
                {
                    "quality_history": history,
                    "quality_status": "retrying",
                    "revision_feedback": feedback,
                }
            )
            return result

        # 达到上限仍继续主链路，使用三版中规则分最高的歌词，避免音频阶段
        # 重复触发和工作流死循环；状态明确标记 capped，不能当作质检通过。
        best_draft = (
            state["lyric_draft"] if is_best else state["best_lyric_draft"]
        )
        best_report = report_dict if is_best else state["best_quality_report"]
        best_scores = scores if is_best else state["best_scores"]
        selected_attempt = state["attempts"] if is_best else state["best_attempt"]
        attempt_record["quality_status"] = "capped"
        history.append(attempt_record)
        result.update(
            {
                "lyric_draft": best_draft,
                "lyrics": best_draft.lyrics,
                "quality_report": best_report,
                "scores": best_scores,
                "quality_history": history,
                "quality_status": "capped",
                "selected_attempt": selected_attempt,
            }
        )
        return result

    def route_after_quality(state: MusicWorkflowState) -> str:
        if state.get("quality_status") in {"passed", "capped"}:
            return "producer"
        return "lyricist"

    def producer_node(state: MusicWorkflowState) -> dict[str, object]:
        production_plan = producer_agent.run(
            state["requirement"], state["lyric_draft"]
        )
        return {"production_plan": production_plan}

    def generator_node(state: MusicWorkflowState) -> dict[str, object]:
        duration_seconds = state.get(
            "requested_duration_seconds", state["requirement"].duration_seconds
        )
        result = music_generator.generate(
            lyrics=state["lyric_draft"].lyrics,
            tags=state["production_plan"].tags,
            duration_seconds=duration_seconds,
        )
        output_path = getattr(result, "output_path", None)
        if not isinstance(output_path, Path):
            raise TypeError("音频生成服务必须返回带 Path 类型 output_path 的结果")
        return {"audio_path": str(output_path)}

    workflow = StateGraph(MusicWorkflowState)
    workflow.add_node("validate_input", observed("validate_input", validate_input_node))
    workflow.add_node("requirement", observed("requirement", requirement_node))
    workflow.add_node("lyricist", observed("lyricist", lyricist_node))
    if dual_lyricists:
        workflow.add_node("lyric_candidate", lyric_candidate_node)
        if lyric_judge_agent is not None:
            workflow.add_node(
                "subjective_judge",
                observed("subjective_judge", subjective_judge_node),
            )
        workflow.add_node(
            "select_lyricist", observed("lyricist_selection", select_lyricist_node)
        )
    workflow.add_node("quality", observed("quality", quality_node))
    workflow.add_node("producer", observed("producer", producer_node))
    if not text_only:
        workflow.add_node("generator", observed("generator", generator_node))

    workflow.add_edge(START, "validate_input")
    workflow.add_edge("validate_input", "requirement")
    if dual_lyricists:
        workflow.add_conditional_edges("requirement", dispatch_lyricists)
        if lyric_judge_agent is not None:
            workflow.add_edge("lyric_candidate", "subjective_judge")
            workflow.add_conditional_edges(
                "subjective_judge", route_after_subjective_judge
            )
        else:
            workflow.add_edge("lyric_candidate", "select_lyricist")
        workflow.add_edge("select_lyricist", "quality")
    else:
        workflow.add_edge("requirement", "lyricist")
    # 双作词模式在选优后的规则质检也可能要求回炉。此时只让入选 persona
    # 改写，并必须重新进入 quality；否则图会在第一次改写后提前结束。
    workflow.add_edge("lyricist", "quality")
    workflow.add_conditional_edges(
        "quality",
        route_after_quality,
        {"lyricist": "lyricist", "producer": "producer"},
    )
    if text_only:
        workflow.add_edge("producer", END)
    else:
        workflow.add_edge("producer", "generator")
        workflow.add_edge("generator", END)
    return workflow.compile()
