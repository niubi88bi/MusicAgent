"""D3 文本创作链路：主题 → 需求 → 歌词 → 制作方案。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.lyricist_agent import LyricistAgent
from agents.producer_agent import ProducerAgent
from agents.requirement_agent import RequirementAgent
from agents.schemas import LyricDraft, ProductionPlan, SongRequirement
from services.llm import DeepSeekLLM


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 MusicAgent 的 D3 文本创作链路")
    parser.add_argument("theme", help="歌曲主题，例如：写一首关于夏夜的歌")
    parser.add_argument("--out", default="data/output/drafts", help="产物目录")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用输出目录中已完成的阶段产物，避免重复调用 LLM",
    )
    parser.add_argument(
        "--stop-after",
        choices=("requirement", "lyrics", "production"),
        default="production",
        help="完成指定阶段后退出，便于分段调试长链路",
    )
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    requirement_path = output_dir / "requirement.json"
    lyrics_path = output_dir / "lyrics.txt"
    lyric_draft_path = output_dir / "lyric_draft.json"
    production_plan_path = output_dir / "production_plan.json"

    llm = DeepSeekLLM()

    if args.resume and requirement_path.exists():
        requirement = SongRequirement.model_validate_json(requirement_path.read_text("utf-8"))
        print("复用需求理解产物", flush=True)
    else:
        print("开始需求理解…", flush=True)
        requirement = RequirementAgent(llm).run(args.theme)
        requirement_path.write_text(requirement.model_dump_json(indent=2), encoding="utf-8")
        print("需求理解完成", flush=True)
    if args.stop_after == "requirement":
        return

    if args.resume and lyric_draft_path.exists():
        lyric_draft = LyricDraft.model_validate_json(lyric_draft_path.read_text("utf-8"))
        print("复用歌词草稿", flush=True)
    else:
        print("开始歌词创作…", flush=True)
        lyric_draft = LyricistAgent(llm).run(requirement)
        lyrics_path.write_text(lyric_draft.lyrics, encoding="utf-8")
        lyric_draft_path.write_text(lyric_draft.model_dump_json(indent=2), encoding="utf-8")
        print("歌词创作完成", flush=True)
    if args.stop_after == "lyrics":
        return

    if args.resume and production_plan_path.exists():
        production_plan = ProductionPlan.model_validate_json(
            production_plan_path.read_text("utf-8")
        )
        print("复用制作方案", flush=True)
    else:
        print("开始制作方案…", flush=True)
        production_plan = ProducerAgent(llm).run(requirement, lyric_draft)
        production_plan_path.write_text(
            production_plan.model_dump_json(indent=2), encoding="utf-8"
        )
        print("制作方案完成", flush=True)

    print(f"草稿已生成：{output_dir.resolve()}")
    print(f"标题：{lyric_draft.title}")


if __name__ == "__main__":
    main()
