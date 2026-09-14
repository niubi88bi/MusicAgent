"""固定主题的歌词质检回炉评估。

默认仅打印实验计划；只有显式传入 ``--execute`` 才会调用真实 DeepSeek，且全程
使用文本生成标记，不加载 HeartMuLa 或 GPU。自然主题与受控缺陷分开落盘，避免把
“规则修复成功”误写成“自然歌词质量已经提升”。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from agents.critic_agent import LyricJudgeAgent
from agents.graph import MAX_LYRIC_ATTEMPTS, build_music_graph
from agents.lyricist_agent import LyricistAgent
from agents.producer_agent import ProducerAgent
from agents.requirement_agent import RequirementAgent
from agents.schemas import LyricDraft, SongRequirement
from services.config import get_settings
from services.llm import DeepSeekLLM
from services.workflow_artifacts import write_quality_artifacts
from tools.lyrics_quality.scorer import format_revision_feedback, score_lyrics


@dataclass(frozen=True)
class NaturalCase:
    case_id: str
    theme: str
    duration_seconds: int = 30


@dataclass(frozen=True)
class ControlledCase:
    case_id: str
    defect_type: str
    description: str
    requirement: SongRequirement
    draft: LyricDraft


NATURAL_CASES = (
    NaturalCase("summer-night-home", "写一首夏夜下班回家的温暖 City Pop 歌曲"),
    NaturalCase("station-farewell", "写一首毕业后第一次离家、在车站与父母告别的克制民谣流行歌曲"),
    NaturalCase("old-town-reunion", "写一首暴雨后老城重逢、遗憾但不煽情的国风流行歌曲"),
)

# 评估配置必须显式记录当前各 Agent 的调用参数。若未来修改 Agent 参数，
# 同步更新此处，避免把不同温度的结果误当作同一实验条件。
EVALUATION_AGENT_PARAMETERS = {
    "requirement_agent": {"temperature": 0.4, "max_tokens": 800},
    "lyricist_agent": {"temperature": 0.85, "max_tokens": 2200},
    "lyric_judge_agent": {"temperature": 0.2, "max_tokens": 2400},
    "producer_agent": {"temperature": 0.55, "max_tokens": 1000},
}


def _controlled_requirement(theme: str) -> SongRequirement:
    return SongRequirement(
        theme=theme,
        emotion="克制、怀念",
        genre="流行民谣",
        tempo="mid-tempo",
        duration_seconds=30,
        perspective="第一人称",
        vocal_type="温柔女声",
        bpm=88,
        emotion_intensity=3,
        keywords=["雨后", "旧城", "重逢"],
        target_rhyme_group="江阳",
    )


def build_controlled_cases() -> tuple[ControlledCase, ...]:
    """构造可复现的项目原创缺陷草稿，专门覆盖三类规则反馈。"""

    requirement = _controlled_requirement("雨后旧城重逢")
    rhyme_mismatch = LyricDraft(
        title="雨停以后",
        lyrics="""[Verse]
雨后路灯照着旧巷
鞋跟敲过湿的石墙
我把名字写在玻璃上
等风吹散没说完的伤

[Chorus]
雨停以后我还在回望
雨停以后我还看着夜
旧城把沉默留给雨
你说再见时声音很轻
我把遗憾交给路人

[Bridge]
远处钟声穿过长巷""",
        hook="雨停以后我还在回望",
        creative_note="用于验证指定目标辙下的多行韵脚失配。",
    )
    hook_outside_chorus = LyricDraft(
        title="雨停以后",
        lyrics="""[Verse]
雨停以后我还在回望
雨后路灯照着旧巷
鞋跟敲过湿的石墙
等风吹散没说完的伤

[Chorus]
旧城把沉默藏进巷
你说再见时声音很轻
我把遗憾留在街旁
远处钟声穿过长巷

[Bridge]
我把名字留在街旁""",
        hook="雨停以后我还在回望",
        creative_note="用于验证 Hook 仅在主歌出现时的结构反馈。",
    )
    short_section = LyricDraft(
        title="雨停以后",
        lyrics="""[Verse]
雨后路灯照着旧巷

[Chorus]
雨停以后我仍站在原地回望
旧城把沉默慢慢藏进深巷
我把遗憾轻轻留在街旁
远处钟声穿过寂静长巷

[Bridge]
""",
        hook="雨停以后我仍站在原地回望",
        creative_note="用于验证短试听下 Verse 行数不足的结构反馈。",
    )
    return (
        ControlledCase(
            "rhyme-mismatch",
            "rhyme_pattern_mismatch",
            "目标为江阳辙，但副歌保留多行非江阳韵脚，确保稳定触发规则门槛。",
            requirement,
            rhyme_mismatch,
        ),
        ControlledCase(
            "hook-outside-chorus",
            "missing_hook",
            "Hook 原样存在，但只在 Verse 出现。",
            requirement,
            hook_outside_chorus,
        ),
        ControlledCase(
            "short-verse",
            "short_section",
            "30 秒试听的 Verse 只有一行，Bridge 为空，确保结构分未达通过门槛。",
            requirement,
            short_section,
        ),
    )


class TextOnlyEvaluationGenerator:
    """满足图的生成器协议，并把不调用音频模型写成明确产物。"""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def generate(
        self, lyrics: str, tags: list[str], duration_seconds: int
    ) -> object:
        marker = self._output_dir / "audio_skipped.json"
        _write_json(
            marker,
            {
                "mode": "quality_loop_evaluation",
                "audio_generated": False,
                "reason": "evaluation intentionally excludes HeartMuLa and GPU",
                "duration_seconds": duration_seconds,
                "tags": tags,
                "lyrics_characters": len(lyrics),
            },
        )
        return type("TextOnlyResult", (), {"output_path": marker})()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _failure_payload(
    *,
    case_id: str,
    mode: str,
    error: Exception,
    llm: DeepSeekLLM,
) -> dict[str, object]:
    """保存案例失败与已发生调用的脱敏遥测，避免失败证据只剩一句异常文本。"""
    return {
        "case_id": case_id,
        "mode": mode,
        "error_type": type(error).__name__,
        "message": str(error),
        "telemetry": DeepSeekLLM.summarize_call_records(llm.telemetry_snapshot()),
    }


def _draft_snapshot(draft: LyricDraft) -> dict[str, str]:
    return {
        "title": draft.title,
        "lyrics": draft.lyrics,
        "hook": draft.hook,
        "creative_note": draft.creative_note,
    }


def _quality_snapshot(draft: LyricDraft, requirement: SongRequirement) -> dict[str, object]:
    report = score_lyrics(
        draft.lyrics,
        draft.hook,
        duration_seconds=requirement.duration_seconds,
        target_rhyme_group=requirement.target_rhyme_group,
        theme=requirement.theme,
        theme_keywords=requirement.keywords,
    )
    return {
        "draft": _draft_snapshot(draft),
        "scores": {
            "rhyme": report.rhyme.score,
            "structure": report.structure.score,
            "total": report.total_score,
        },
        "quality_report": report.to_dict(),
    }


def _write_blind_review_pair(
    case_dir: Path,
    case_id: str,
    review_context: dict[str, str],
    initial: LyricDraft,
    selected: LyricDraft,
    *,
    source_names: tuple[str, str] = ("initial", "selected"),
    comparison: str = "revision",
) -> None:
    """把两份歌词稳定打乱为 A/B；评分正文不暴露候选来源。"""

    if initial.lyrics == selected.lyrics:
        _write_json(
            case_dir / "blind_review_key.json",
            {"case_id": case_id, "skipped": True, "reason": "lyrics are identical"},
        )
        return
    labels = source_names
    if hashlib.sha256(case_id.encode("utf-8")).digest()[0] % 2:
        labels = tuple(reversed(labels))
    drafts = {source_names[0]: initial, source_names[1]: selected}
    review_text = (
        f"案例：{case_id}\n\n"
        "歌曲需求（只用于主题契合评分，不透露版本来源）：\n"
        f"- 主题：{review_context['theme']}\n"
        f"- 情绪：{review_context['emotion']}\n"
        f"- 曲风：{review_context['genre']}\n\n"
        "请在不知道版本来源的前提下，为 A、B 分别按 1-5 分评价：\n"
        "自然度、主题契合、Hook 记忆点；并写一句是否有语义或意象损失。\n\n"
        f"[A]\n{drafts[labels[0]].lyrics}\n\n[B]\n{drafts[labels[1]].lyrics}\n"
    )
    (case_dir / "blind_review.txt").write_text(review_text, encoding="utf-8")
    _write_json(
        case_dir / "blind_review_key.json",
        {
            "case_id": case_id,
            "comparison": comparison,
            "A": labels[0],
            "B": labels[1],
        },
    )


def _issue_codes(record: dict[str, object]) -> list[str]:
    report = record["quality_report"]
    if not isinstance(report, dict):
        raise TypeError("质检记录缺少 quality_report")
    issues = report.get("issues")
    if not isinstance(issues, list):
        raise TypeError("质检报告缺少 issues")
    return [str(issue["code"]) for issue in issues if isinstance(issue, dict)]


def _select_controlled_record(
    history: list[dict[str, object]], final_passed: bool
) -> dict[str, object]:
    """与 LangGraph 一致：通过时选终版，达到上限时选历史最高规则分。"""

    if not history:
        raise ValueError("受控实验质量历史不能为空")
    if final_passed:
        return history[-1]
    return max(
        enumerate(history),
        key=lambda item: (float(item[1]["scores"]["total"]), item[0]),
    )[1]


def _build_aggregate_metrics(
    natural: list[dict[str, object]],
    controlled: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> dict[str, object]:
    """计算完成案例的描述性指标；失败案例单列，绝不伪装为零分或成功。"""

    def rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 3) if denominator else None

    natural_total = len(natural)
    natural_triggered = sum(bool(item["feedback_triggered"]) for item in natural)
    natural_capped = sum(item["quality_status"] == "capped" for item in natural)
    controlled_total = len(controlled)
    controlled_detected = sum(
        bool(item["target_defect_detected"]) for item in controlled
    )
    controlled_resolved = sum(
        bool(item["target_defect_resolved"]) for item in controlled
    )
    controlled_passed = sum(item["quality_status"] == "passed" for item in controlled)
    controlled_capped = sum(item["quality_status"] == "capped" for item in controlled)
    return {
        "completed_cases": natural_total + controlled_total,
        "failed_cases": len(failures),
        "natural": {
            "completed_cases": natural_total,
            "feedback_triggered_cases": natural_triggered,
            "feedback_trigger_rate": rate(natural_triggered, natural_total),
            "initial_pass_cases": sum(bool(item["initial_passed"]) for item in natural),
            "final_pass_cases": sum(item["quality_status"] == "passed" for item in natural),
            "capped_cases": natural_capped,
            "capped_rate": rate(natural_capped, natural_total),
        },
        "controlled": {
            "completed_cases": controlled_total,
            "target_defect_detected_cases": controlled_detected,
            "target_defect_detection_rate": rate(controlled_detected, controlled_total),
            "target_defect_resolved_cases": controlled_resolved,
            "target_defect_resolution_rate": rate(controlled_resolved, controlled_total),
            "quality_pass_cases": controlled_passed,
            "capped_cases": controlled_capped,
        },
    }


def _run_natural_case(
    case: NaturalCase, case_dir: Path, llm: DeepSeekLLM
) -> dict[str, object]:
    graph = build_music_graph(
        RequirementAgent(llm),
        LyricistAgent(llm, persona="structured"),
        ProducerAgent(llm),
        TextOnlyEvaluationGenerator(case_dir),
        secondary_lyricist_agent=LyricistAgent(llm, persona="imagery"),
        lyric_judge_agent=LyricJudgeAgent(llm),
    )
    started_at = time.perf_counter()
    state = graph.invoke(
        {"theme": case.theme, "requested_duration_seconds": case.duration_seconds}
    )
    wall_seconds = round(time.perf_counter() - started_at, 3)
    requirement = state["requirement"]
    final_draft = state["lyric_draft"]
    if not isinstance(requirement, SongRequirement) or not isinstance(final_draft, LyricDraft):
        raise TypeError("自然主题工作流未返回有效需求或歌词草稿")

    (case_dir / "requirement.json").write_text(
        requirement.model_dump_json(indent=2), encoding="utf-8"
    )
    (case_dir / "selected_lyrics.txt").write_text(final_draft.lyrics, encoding="utf-8")
    report_path, history_path = write_quality_artifacts(case_dir, state)
    history = state.get("quality_history", [])
    if not isinstance(history, list) or not history:
        raise TypeError("自然主题工作流未保存质量历史")
    initial = history[0]
    candidates = state.get("lyric_candidates", [])
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise TypeError("自然主题双作词工作流未返回候选")
    latest_round = max(int(candidate.get("generation_round", 1)) for candidate in candidates)
    latest_candidates = [
        candidate
        for candidate in candidates
        if int(candidate.get("generation_round", 1)) == latest_round
    ]
    if len(latest_candidates) != 2:
        raise TypeError("自然主题双作词最新一轮候选不完整")
    candidates_by_id = {
        str(candidate["candidate_id"]): candidate
        for candidate in latest_candidates
        if isinstance(candidate, dict)
    }
    if set(candidates_by_id) != {"structured", "imagery"}:
        raise TypeError("自然主题双作词候选身份不完整")
    structured_draft = LyricDraft.model_validate(candidates_by_id["structured"]["draft"])
    imagery_draft = LyricDraft.model_validate(candidates_by_id["imagery"]["draft"])
    _write_json(case_dir / "lyric_candidates.json", candidates)
    _write_json(case_dir / "subjective_review.json", state["subjective_review"])
    _write_json(
        case_dir / "subjective_review_history.json",
        state["subjective_review_history"],
    )
    _write_json(case_dir / "candidate_selection.json", state["candidate_selection"])
    _write_blind_review_pair(
        case_dir,
        case.case_id,
        {
            "theme": requirement.theme,
            "emotion": requirement.emotion,
            "genre": requirement.genre,
        },
        structured_draft,
        imagery_draft,
        source_names=("structured", "imagery"),
        comparison="dual_lyricist_candidates",
    )
    summary = {
        "case": asdict(case),
        "mode": "natural_workflow",
        "wall_seconds": wall_seconds,
        "quality_status": state.get("quality_status"),
        "attempts": state.get("attempts"),
        "selected_attempt": state.get("selected_attempt"),
        "initial_passed": bool(initial["quality_report"]["passed"]),
        "feedback_triggered": len(history) > 1,
        "initial_scores": initial.get("scores"),
        "selected_scores": state.get("scores"),
        "candidate_selection": state.get("candidate_selection"),
        "subjective_review": state.get("subjective_review"),
        "candidate_regeneration_count": state.get("candidate_regeneration_count", 0),
        "candidate_generation_rounds": latest_round,
        "candidate_scores": {
            candidate_id: candidate["scores"]
            for candidate_id, candidate in candidates_by_id.items()
        },
        "quality_report_path": report_path.name,
        "revision_history_path": history_path.name,
        "telemetry": DeepSeekLLM.summarize_call_records(llm.telemetry_snapshot()),
    }
    _write_json(case_dir / "case_summary.json", summary)
    return summary


def _run_controlled_case(
    case: ControlledCase,
    case_dir: Path,
    max_attempts: int,
    llm: DeepSeekLLM,
) -> dict[str, object]:
    started_at = time.perf_counter()
    lyricist = LyricistAgent(llm)
    current_draft = case.draft
    current_snapshot = _quality_snapshot(current_draft, case.requirement)
    history: list[dict[str, object]] = [
        {
            "attempt": 1,
            "source": "deterministic_controlled_fixture",
            **current_snapshot,
            "quality_status": (
                "passed" if current_snapshot["quality_report"]["passed"] else "retrying"
            ),
        }
    ]
    if current_snapshot["quality_report"]["passed"]:
        raise AssertionError(f"受控缺陷 {case.case_id} 未被规则检测到")

    while (
        not current_snapshot["quality_report"]["passed"]
        and len(history) < max_attempts
    ):
        report = score_lyrics(
            current_draft.lyrics,
            current_draft.hook,
            case.requirement.duration_seconds,
            case.requirement.target_rhyme_group,
            case.requirement.theme,
            case.requirement.keywords,
        )
        feedback = format_revision_feedback(report)
        next_draft = lyricist.run(
            case.requirement,
            revision_feedback=feedback,
            previous_draft=current_draft,
        )
        current_draft = next_draft
        current_snapshot = _quality_snapshot(current_draft, case.requirement)
        history[-1]["revision_feedback"] = feedback
        history.append(
            {
                "attempt": len(history) + 1,
                "source": "deepseek_rewrite",
                "revision_source_attempt": len(history),
                **current_snapshot,
                "quality_status": (
                    "passed" if current_snapshot["quality_report"]["passed"] else "retrying"
                ),
            }
        )

    final_passed = bool(current_snapshot["quality_report"]["passed"])
    final_status = "passed" if final_passed else "capped"
    history[-1]["quality_status"] = final_status
    selected_record = _select_controlled_record(history, final_passed)
    selected_draft = LyricDraft.model_validate(selected_record["draft"])
    selected_issue_codes = _issue_codes(selected_record)
    initial_issue_codes = _issue_codes(history[0])
    target_defect_detected = case.defect_type in initial_issue_codes
    target_defect_resolved = case.defect_type not in selected_issue_codes
    selected_report = selected_record["quality_report"]
    if not isinstance(selected_report, dict):
        raise TypeError("受控实验选中版本缺少质检报告")
    (case_dir / "requirement.json").write_text(
        case.requirement.model_dump_json(indent=2), encoding="utf-8"
    )
    (case_dir / "initial_lyrics.txt").write_text(case.draft.lyrics, encoding="utf-8")
    (case_dir / "selected_lyrics.txt").write_text(selected_draft.lyrics, encoding="utf-8")
    _write_json(
        case_dir / "audio_skipped.json",
        {
            "mode": "controlled_repair_evaluation",
            "audio_generated": False,
            "reason": "controlled repair intentionally excludes HeartMuLa and GPU",
        },
    )
    _write_json(
        case_dir / "quality_report.json",
        {
            "quality_status": final_status,
            "attempts": len(history),
            "selected_attempt": selected_record["attempt"],
            "quality_report": selected_report,
        },
    )
    _write_json(case_dir / "lyric_revision_history.json", history)
    _write_blind_review_pair(
        case_dir,
        case.case_id,
        {
            "theme": case.requirement.theme,
            "emotion": case.requirement.emotion,
            "genre": case.requirement.genre,
        },
        case.draft,
        selected_draft,
    )
    summary = {
        "case": {
            "case_id": case.case_id,
            "defect_type": case.defect_type,
            "description": case.description,
        },
        "mode": "controlled_repair",
        "wall_seconds": round(time.perf_counter() - started_at, 3),
        "quality_status": final_status,
        "attempts": len(history),
        "selected_attempt": selected_record["attempt"],
        "initial_scores": history[0]["scores"],
        "selected_scores": selected_record["scores"],
        "initial_issue_codes": initial_issue_codes,
        "selected_issue_codes": selected_issue_codes,
        "new_issue_codes": sorted(set(selected_issue_codes) - set(initial_issue_codes)),
        "target_defect_detected": target_defect_detected,
        "target_defect_resolved": target_defect_resolved,
        "telemetry": DeepSeekLLM.summarize_call_records(llm.telemetry_snapshot()),
    }
    _write_json(case_dir / "case_summary.json", summary)
    return summary


def _write_summary_markdown(experiment_dir: Path, summary: dict[str, object]) -> None:
    lines = [
        "# 歌词质检回炉评估（试点）",
        "",
        "本报告分开记录自然主题工作流与受控缺陷修复；不把规则分提升当作自然度提升证据。",
        "",
        "## 自然主题",
    ]
    for item in summary["natural"]:
        lines.append(
            f"- {item['case']['case_id']}：status={item['quality_status']}，"
            f"attempts={item['attempts']}，selected={item['selected_attempt']}。"
        )
    lines.extend(["", "## 受控缺陷修复"])
    for item in summary["controlled"]:
        lines.append(
            f"- {item['case']['case_id']}：status={item['quality_status']}，"
            f"attempts={item['attempts']}，初始问题={item['initial_issue_codes']}，"
            f"终版问题={item['selected_issue_codes']}。"
        )
    aggregate = summary["aggregate"]
    lines.extend(
        [
            "",
            "## 汇总指标",
            "",
            f"- 完成案例：{aggregate['completed_cases']}；失败案例：{aggregate['failed_cases']}。",
            f"- 自然主题回炉触发率：{aggregate['natural']['feedback_trigger_rate']}；"
            f"达到上限比例：{aggregate['natural']['capped_rate']}。",
            f"- 受控缺陷检测率：{aggregate['controlled']['target_defect_detection_rate']}；"
            f"指定缺陷修复率：{aggregate['controlled']['target_defect_resolution_rate']}；"
            f"整体规则通过数：{aggregate['controlled']['quality_pass_cases']}。",
        ]
    )
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "- 3 个自然主题和 3 个受控缺陷只是工程试点，不具备统计显著性。",
            "- 规则分、Token 与延迟由程序记录；自然度、主题贴合和 Hook 记忆点需使用每个案例的 blind_review.txt 进行人工 A/B 评分。",
            "- 若自然主题首稿全部通过，只能报告触发率，不能据此宣称回炉提升了自然歌词质量。",
        ]
    )
    (experiment_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_cases(
    mode: str,
    case_id: str | None,
) -> tuple[tuple[NaturalCase, ...], tuple[ControlledCase, ...]]:
    """按模式和可选案例标识筛选，避免真实冒烟意外运行整组案例。"""
    natural_cases = NATURAL_CASES if mode in {"all", "natural"} else ()
    controlled_cases = build_controlled_cases() if mode in {"all", "controlled"} else ()
    if case_id is None:
        return natural_cases, controlled_cases

    selected_natural = tuple(case for case in natural_cases if case.case_id == case_id)
    selected_controlled = tuple(
        case for case in controlled_cases if case.case_id == case_id
    )
    if not selected_natural and not selected_controlled:
        raise ValueError(f"案例 {case_id} 不属于当前 mode={mode}")
    return selected_natural, selected_controlled


def main() -> None:
    parser = argparse.ArgumentParser(description="运行固定主题的歌词质检回炉评估。")
    parser.add_argument(
        "--mode", choices=("all", "natural", "controlled"), default="all"
    )
    parser.add_argument(
        "--out", default="data/output/evaluations", help="评估产物根目录（默认 D 盘项目目录）"
    )
    parser.add_argument(
        "--case",
        choices=tuple(case.case_id for case in NATURAL_CASES)
        + tuple(case.case_id for case in build_controlled_cases()),
        help="只运行指定案例，用于低成本真实冒烟。",
    )
    parser.add_argument(
        "--execute", action="store_true", help="确认调用真实 DeepSeek；未提供时不会发起网络请求。"
    )
    args = parser.parse_args()
    try:
        natural_cases, controlled_cases = _select_cases(args.mode, args.case)
    except ValueError as error:
        parser.error(str(error))

    if not args.execute:
        print(
            "这是会调用真实 DeepSeek 的评估脚本。请确认后加 --execute；"
            "脚本不会加载 HeartMuLa 或 GPU。",
            flush=True,
        )
        return

    experiment_dir = Path(args.out) / f"quality-loop-pilot-{uuid4().hex}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    settings = get_settings()
    config = {
        "mode": args.mode,
        "max_lyric_attempts": MAX_LYRIC_ATTEMPTS,
        "natural_cases": [asdict(case) for case in natural_cases],
        "controlled_cases": [
            {"case_id": case.case_id, "defect_type": case.defect_type}
            for case in controlled_cases
        ],
        "audio_generated": False,
        "llm": {
            "provider": "DeepSeek",
            "base_url": settings.deepseek_base_url,
            "model": settings.deepseek_model,
            "timeout_seconds": settings.llm_timeout_seconds,
            "max_retries": settings.llm_max_retries,
            "json_thinking_mode": (
                "disabled" if settings.llm_json_disable_thinking else "default"
            ),
        },
        "agent_parameters": EVALUATION_AGENT_PARAMETERS,
        "limitations": [
            "Rule scores do not measure naturalness, topical fit, singability, or audio quality.",
            "Natural and controlled results must be reported separately.",
        ],
    }
    _write_json(experiment_dir / "experiment_config.json", config)
    natural_summaries: list[dict[str, object]] = []
    controlled_summaries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    if natural_cases:
        for case in natural_cases:
            print(f"运行自然主题：{case.case_id}", flush=True)
            case_dir = experiment_dir / "natural" / case.case_id
            case_dir.mkdir(parents=True, exist_ok=False)
            llm = DeepSeekLLM()
            try:
                natural_summaries.append(_run_natural_case(case, case_dir, llm))
            except Exception as error:
                failure = _failure_payload(
                    case_id=case.case_id,
                    mode="natural_workflow",
                    error=error,
                    llm=llm,
                )
                _write_json(case_dir / "failure.json", failure)
                failures.append(failure)
                print(f"案例失败：{case.case_id}，已保存 failure.json", flush=True)
            finally:
                llm.close()
    if controlled_cases:
        for case in controlled_cases:
            print(f"运行受控缺陷：{case.case_id}", flush=True)
            case_dir = experiment_dir / "controlled" / case.case_id
            case_dir.mkdir(parents=True, exist_ok=False)
            llm = DeepSeekLLM()
            try:
                controlled_summaries.append(
                    _run_controlled_case(
                        case, case_dir, MAX_LYRIC_ATTEMPTS, llm
                    )
                )
            except Exception as error:
                failure = _failure_payload(
                    case_id=case.case_id,
                    mode="controlled_repair",
                    error=error,
                    llm=llm,
                )
                _write_json(case_dir / "failure.json", failure)
                failures.append(failure)
                print(f"案例失败：{case.case_id}，已保存 failure.json", flush=True)
            finally:
                llm.close()
    summary = {
        "natural": natural_summaries,
        "controlled": controlled_summaries,
        "failures": failures,
        "aggregate": _build_aggregate_metrics(
            natural_summaries, controlled_summaries, failures
        ),
    }
    _write_json(experiment_dir / "summary.json", summary)
    _write_summary_markdown(experiment_dir, summary)
    print(f"评估完成；产物目录：{experiment_dir.resolve()}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
