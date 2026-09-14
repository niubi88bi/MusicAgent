"""受控的 HeartMuLa 子进程入口，只读取已完成文本任务的结构化产物。"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from agents.schemas import LyricDraft, ProductionPlan
from services.music_gen import HeartMuLaGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--duration", type=int, default=30)
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-f]{32}", args.job_id):
        parser.error("job-id 必须是 32 位十六进制任务 ID")
    if args.duration != 30:
        parser.error("当前 Web 试听只支持 30 秒")
    source_dir = args.source_dir.resolve()
    project_root = Path(__file__).resolve().parents[1]
    output_root = (project_root / "data/output/web").resolve()
    if source_dir.parent != output_root:
        parser.error("source-dir 必须是 Web 任务目录")

    lyric_draft = LyricDraft.model_validate_json((source_dir / "lyric_draft.json").read_text(encoding="utf-8"))
    production = ProductionPlan.model_validate_json((source_dir / "production_plan.json").read_text(encoding="utf-8"))
    generator = HeartMuLaGenerator.from_settings(output_dir=source_dir / "audio")
    result = generator.generate(
        lyrics=lyric_draft.lyrics,
        tags=production.tags,
        duration_seconds=30,
        job_id=args.job_id,
    )
    print(f"audio={result.output_path}", flush=True)


if __name__ == "__main__":
    main()
