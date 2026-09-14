"""用真实 DeepSeek 生成 D9.4 多样化文本样本；不调用评审、制作或音频模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.lyricist_agent import LyricistAgent
from agents.requirement_agent import RequirementAgent
from services.llm import DeepSeekLLM


def _read_plan(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("来源计划必须是 JSON 数组")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("来源计划的每一项必须是 JSON 对象")
        family_id = str(item.get("family_id", "")).strip()
        theme = str(item.get("theme", "")).strip()
        if not family_id or not theme:
            raise ValueError("每个来源计划必须包含 family_id 和 theme")
        if family_id in seen:
            raise ValueError(f"family_id 重复：{family_id}")
        seen.add(family_id)
        rows.append({"family_id": family_id, "theme": theme})
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/d9_4_source_plan.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过已有完成记录的歌词家族。",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=args.resume)
    llm = DeepSeekLLM()
    requirement_agent = RequirementAgent(llm)
    lyricists = (
        LyricistAgent(llm, persona="structured"),
        LyricistAgent(llm, persona="imagery"),
    )
    completed: list[str] = []
    failed: list[str] = []
    for item in _read_plan(args.plan):
        family_id = item["family_id"]
        family_dir = args.out / family_id
        completion_path = family_dir / "family_summary.json"
        if args.resume and completion_path.exists():
            print(f"跳过已完成家族：{family_id}", flush=True)
            completed.append(family_id)
            continue
        family_dir.mkdir(parents=True, exist_ok=args.resume)
        try:
            requirement = requirement_agent.run(item["theme"])
            candidates = []
            for lyricist in lyricists:
                draft = lyricist.run(requirement)
                candidates.append(
                    {
                        "persona_id": lyricist.persona_id,
                        "display_name": lyricist.display_name,
                        "draft": draft.model_dump(),
                    }
                )
            (family_dir / "requirement.json").write_text(
                requirement.model_dump_json(indent=2), encoding="utf-8"
            )
            _write_json(family_dir / "lyric_candidates.json", candidates)
            _write_json(
                completion_path,
                {
                    "family_id": family_id,
                    "requested_theme": item["theme"],
                    "candidate_count": len(candidates),
                    "llm_used": True,
                    "audio_model_used": False,
                    "purpose": "d9_4_internal_annotation_source",
                },
            )
            completed.append(family_id)
            print(f"完成歌词家族：{family_id}", flush=True)
        except Exception as error:  # noqa: BLE001 - 批量采样要保留其他家族
            _write_json(
                family_dir / "failure.json",
                {
                    "family_id": family_id,
                    "error_type": type(error).__name__,
                    "message": str(error)[:1000],
                },
            )
            failed.append(family_id)
            print(f"歌词家族失败：{family_id} ({type(error).__name__})", flush=True)

    _write_json(
        args.out / "generation_summary.json",
        {
            "method": "d9_4_source_pool_v1",
            "completed": completed,
            "failed": failed,
            "llm_used": True,
            "audio_model_used": False,
            "llm_telemetry": DeepSeekLLM.summarize_call_records(
                llm.telemetry_snapshot()
            ),
        },
    )
    if failed:
        raise RuntimeError(f"仍有 {len(failed)} 个歌词家族生成失败，可用 --resume 重试")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
