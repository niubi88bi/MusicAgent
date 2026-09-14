"""真实 LLM 文本工作流验证：保留质检回炉，显式跳过 HeartMuLa。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from agents.critic_agent import LyricJudgeAgent
from agents.graph import build_music_graph
from agents.lyricist_agent import LyricistAgent
from agents.producer_agent import ProducerAgent
from agents.requirement_agent import RequirementAgent
from agents.schemas import LyricDraft, ProductionPlan, SongRequirement
from services.llm import DeepSeekLLM
from services.workflow_artifacts import write_quality_artifacts


@dataclass(frozen=True)
class TextOnlyGenerationResult:
    """满足图的生成器协议，但产物明确不是音频。"""

    output_path: Path


class TextOnlyGenerator:
    """为真实文本验证生成可追溯的“未调用音频”标记。"""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def generate(
        self, lyrics: str, tags: list[str], duration_seconds: int
    ) -> TextOnlyGenerationResult:
        marker = self._output_dir / "audio_skipped.json"
        marker.write_text(
            json.dumps(
                {
                    "mode": "text_only_validation",
                    "audio_generated": False,
                    "reason": "HeartMuLa was intentionally skipped",
                    "duration_seconds": duration_seconds,
                    "tags": tags,
                    "lyrics_characters": len(lyrics),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return TextOnlyGenerationResult(output_path=marker)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行真实 DeepSeek 文本工作流并验证歌词质检回炉，不调用 GPU。"
    )
    parser.add_argument("theme", help="歌曲主题")
    parser.add_argument("--duration", type=int, default=30, help="目标时长，秒")
    parser.add_argument(
        "--out", default="data/output/text-validation", help="验证产物根目录"
    )
    args = parser.parse_args()

    job_dir = Path(args.out) / uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=False)
    llm = DeepSeekLLM()
    graph = build_music_graph(
        RequirementAgent(llm),
        LyricistAgent(llm, persona="structured"),
        ProducerAgent(llm),
        TextOnlyGenerator(job_dir),
        secondary_lyricist_agent=LyricistAgent(llm, persona="imagery"),
        lyric_judge_agent=LyricJudgeAgent(llm),
    )

    final_state: dict[str, object] = {}
    for update in graph.stream(
        {
            "theme": args.theme,
            "requested_duration_seconds": args.duration,
        },
        stream_mode="updates",
    ):
        for node_name, changes in update.items():
            final_state.update(changes)
            if node_name == "quality":
                scores = changes.get("scores", {})
                print(
                    "质检节点完成："
                    f"status={changes.get('quality_status')}, "
                    f"total={scores.get('total') if isinstance(scores, dict) else None}",
                    flush=True,
                )
            else:
                print(f"LangGraph 节点完成：{node_name}", flush=True)

    requirement = final_state.get("requirement")
    lyric_draft = final_state.get("lyric_draft")
    production_plan = final_state.get("production_plan")
    if not isinstance(requirement, SongRequirement):
        raise TypeError("LangGraph 未返回有效 requirement")
    if not isinstance(lyric_draft, LyricDraft):
        raise TypeError("LangGraph 未返回有效 lyric_draft")
    if not isinstance(production_plan, ProductionPlan):
        raise TypeError("LangGraph 未返回有效 production_plan")

    (job_dir / "requirement.json").write_text(
        requirement.model_dump_json(indent=2), encoding="utf-8"
    )
    (job_dir / "lyrics.txt").write_text(lyric_draft.lyrics, encoding="utf-8")
    (job_dir / "production_plan.json").write_text(
        production_plan.model_dump_json(indent=2), encoding="utf-8"
    )
    for key, filename in (
        ("lyric_candidates", "lyric_candidates.json"),
        ("subjective_review", "subjective_review.json"),
        ("subjective_review_history", "subjective_review_history.json"),
        ("candidate_selection", "candidate_selection.json"),
    ):
        if key in final_state:
            (job_dir / filename).write_text(
                json.dumps(final_state[key], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    report_path, history_path = write_quality_artifacts(job_dir, final_state)
    (job_dir / "llm_telemetry.json").write_text(
        json.dumps(
            DeepSeekLLM.summarize_call_records(llm.telemetry_snapshot()),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("文本验证完成；未调用 HeartMuLa。", flush=True)
    print(f"产物目录：{job_dir.resolve()}", flush=True)
    print(f"质检报告：{report_path.resolve()}", flush=True)
    print(f"回炉记录：{history_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
