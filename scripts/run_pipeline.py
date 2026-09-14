"""W1 端到端脚本：主题 → 文本创作 → HeartMuLa 音频。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from agents.critic_agent import LyricJudgeAgent
from agents.graph import build_music_graph
from agents.lyricist_agent import LyricistAgent
from agents.producer_agent import ProducerAgent
from agents.requirement_agent import RequirementAgent
from agents.schemas import LyricDraft, ProductionPlan, SongRequirement
from services.llm import DeepSeekLLM
from services.music_gen import HeartMuLaGenerator
from services.workflow_artifacts import write_quality_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 MusicAgent W1 端到端链路")
    parser.add_argument("theme", nargs="?", help="歌曲主题")
    parser.add_argument("--duration", type=int, default=30, help="生成时长，秒")
    parser.add_argument("--out", default="data/output/pipeline", help="产物根目录")
    parser.add_argument(
        "--draft-dir",
        type=Path,
        help="复用已存在的 D3 草稿目录，跳过 LLM 文本创作阶段",
    )
    args = parser.parse_args()

    job_id = uuid4().hex
    job_dir = Path(args.out) / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    if args.draft_dir:
        requirement = SongRequirement.model_validate_json(
            (args.draft_dir / "requirement.json").read_text("utf-8")
        )
        lyric_draft = LyricDraft.model_validate_json(
            (args.draft_dir / "lyric_draft.json").read_text("utf-8")
        )
        production_plan = ProductionPlan.model_validate_json(
            (args.draft_dir / "production_plan.json").read_text("utf-8")
        )
        print(f"复用 D3 草稿：{args.draft_dir}", flush=True)
        (job_dir / "requirement.json").write_text(
            requirement.model_dump_json(indent=2), encoding="utf-8"
        )
        (job_dir / "lyrics.txt").write_text(lyric_draft.lyrics, encoding="utf-8")
        (job_dir / "production_plan.json").write_text(
            production_plan.model_dump_json(indent=2), encoding="utf-8"
        )

        print(f"开始 HeartMuLa {args.duration} 秒音频生成…", flush=True)
        generator = HeartMuLaGenerator.from_settings(output_dir=job_dir / "audio")
        result = generator.generate(
            lyrics=lyric_draft.lyrics,
            tags=production_plan.tags,
            duration_seconds=args.duration,
            job_id="music",
        )
    else:
        if not args.theme:
            parser.error("未提供 --draft-dir 时必须提供 theme")

        llm = DeepSeekLLM()
        generator = HeartMuLaGenerator.from_settings(output_dir=job_dir / "audio")
        graph = build_music_graph(
            RequirementAgent(llm),
            LyricistAgent(llm, persona="structured"),
            ProducerAgent(llm),
            generator,
            secondary_lyricist_agent=LyricistAgent(llm, persona="imagery"),
            lyric_judge_agent=LyricJudgeAgent(llm),
        )
        print("开始 LangGraph 线性工作流…", flush=True)
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
                print(f"LangGraph 节点完成：{node_name}", flush=True)

        requirement = final_state["requirement"]
        lyric_draft = final_state["lyric_draft"]
        production_plan = final_state["production_plan"]
        audio_path = final_state["audio_path"]
        if not isinstance(requirement, SongRequirement):
            raise TypeError("LangGraph 未返回有效 requirement")
        if not isinstance(lyric_draft, LyricDraft):
            raise TypeError("LangGraph 未返回有效 lyric_draft")
        if not isinstance(production_plan, ProductionPlan):
            raise TypeError("LangGraph 未返回有效 production_plan")
        if not isinstance(audio_path, str):
            raise TypeError("LangGraph 未返回有效 audio_path")
        report_path, history_path = write_quality_artifacts(job_dir, final_state)
        quality_status = final_state.get("quality_status")
        attempts = final_state.get("attempts")
        selected_attempt = final_state.get("selected_attempt")
        print(
            "歌词质检完成："
            f"status={quality_status}, attempts={attempts}, "
            f"selected_attempt={selected_attempt}",
            flush=True,
        )
        print(f"质检报告：{report_path.resolve()}", flush=True)
        print(f"回炉记录：{history_path.resolve()}", flush=True)
        if quality_status == "capped":
            print("警告：歌词达到回炉上限，当前音频使用规则分最高草稿。", flush=True)

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
        (job_dir / "llm_telemetry.json").write_text(
            json.dumps(
                DeepSeekLLM.summarize_call_records(llm.telemetry_snapshot()),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        result = None

    if result is None:
        print(f"音频已生成：{audio_path}")
        return
    print(f"音频已生成：{result.output_path.resolve()}")


if __name__ == "__main__":
    main()
