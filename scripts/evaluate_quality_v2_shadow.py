"""对已保存歌词运行 D9 影子评估，不调用 LLM 或音频模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from agents.schemas import SongRequirement
from tools.lyrics_quality.rhyme_checker import check_rhyme
from tools.lyrics_quality.scorer import (
    MIN_RHYME_SCORE,
    QUALITY_PASS_THRESHOLD,
    score_lyrics,
)
from tools.lyrics_quality.structure_checker import check_structure


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_selected_draft(case_dir: Path) -> tuple[str, str]:
    lyrics_path = case_dir / "selected_lyrics.txt"
    if not lyrics_path.exists():
        lyrics_path = case_dir / "lyrics.txt"
    lyrics = lyrics_path.read_text(encoding="utf-8").strip() if lyrics_path.exists() else ""

    draft_path = case_dir / "lyric_draft.json"
    if draft_path.exists():
        draft = _read_json(draft_path)
        if not isinstance(draft, dict):
            raise ValueError(f"{draft_path} 不是 JSON 对象")
        return lyrics or str(draft["lyrics"]), str(draft["hook"])

    history_path = case_dir / "lyric_revision_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"{case_dir} 缺少 lyric_draft.json 或修订历史")
    history = _read_json(history_path)
    attempts = history if isinstance(history, list) else [history]
    if not attempts:
        raise ValueError(f"{history_path} 没有草稿记录")

    selected_attempt: int | None = None
    summary_path = case_dir / "quality_report.json"
    if summary_path.exists():
        summary = _read_json(summary_path)
        if isinstance(summary, dict) and summary.get("selected_attempt") is not None:
            selected_attempt = int(summary["selected_attempt"])
    selected = next(
        (
            item
            for item in attempts
            if isinstance(item, dict)
            and selected_attempt is not None
            and int(item.get("attempt", -1)) == selected_attempt
        ),
        attempts[-1],
    )
    if not isinstance(selected, dict):
        raise ValueError(f"{history_path} 的选中草稿格式无效")
    return lyrics or str(selected["lyrics"]), str(selected["hook"])


def _old_result(case_dir: Path) -> dict[str, object] | None:
    report_path = case_dir / "quality_report.json"
    if not report_path.exists():
        return None
    payload = _read_json(report_path)
    if not isinstance(payload, dict):
        return None
    report = payload.get("quality_report", payload)
    if not isinstance(report, dict):
        return None
    return {
        "total_score": report.get("total_score"),
        "passed": report.get("passed"),
    }


def evaluate_case(case_id: str, case_dir: Path) -> dict[str, object]:
    requirement = SongRequirement.model_validate(_read_json(case_dir / "requirement.json"))
    lyrics, hook = _load_selected_draft(case_dir)
    report = score_lyrics(
        lyrics=lyrics,
        hook=hook,
        duration_seconds=requirement.duration_seconds,
        target_rhyme_group=requirement.target_rhyme_group,
        theme=requirement.theme,
        theme_keywords=requirement.keywords,
    )
    # 独立重算 D6-D7 的原公式，证明 D9 报告没有暗中进入总分或门槛。
    baseline_rhyme = check_rhyme(
        lyrics,
        target_rhyme_group=requirement.target_rhyme_group,
    )
    baseline_structure = check_structure(
        lyrics,
        hook,
        duration_seconds=requirement.duration_seconds,
    )
    baseline_total = round(
        baseline_rhyme.score * 0.7 + baseline_structure.score * 0.3,
        1,
    )
    baseline_passed = (
        baseline_total >= QUALITY_PASS_THRESHOLD
        and baseline_rhyme.score >= MIN_RHYME_SCORE
        and baseline_structure.score >= QUALITY_PASS_THRESHOLD
        and baseline_structure.hook_present
    )
    historical = _old_result(case_dir)
    return {
        "case_id": case_id,
        "source_dir": str(case_dir.resolve()),
        "historical_result": historical,
        "recomputed_d6_d7_baseline": {
            "total_score": baseline_total,
            "passed": baseline_passed,
        },
        "current_result": {
            "total_score": report.total_score,
            "passed": report.passed,
        },
        "d9_score_and_gate_unchanged": (
            baseline_total == report.total_score and baseline_passed == report.passed
        ),
        "historical_result_matches_current": (
            None
            if historical is None
            else (
                historical["total_score"] == report.total_score
                and historical["passed"] == report.passed
            )
        ),
        "rhythm": report.rhythm.to_dict(),
        "theme_signal": report.theme_signal.to_dict(),
    }


def _parse_case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--case 格式必须是 名称=目录")
    case_id, raw_path = value.split("=", 1)
    if not case_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--case 的名称和目录都不能为空")
    return case_id.strip(), Path(raw_path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        type=_parse_case,
        help="可重复传入，格式为 名称=已保存案例目录",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/output/evaluations"),
    )
    args = parser.parse_args()

    run_dir = args.out / f"quality-v2-shadow-{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    results = [evaluate_case(case_id, case_dir) for case_id, case_dir in args.case]
    payload = {
        "method": "d9_quality_shadow_v1",
        "advisory_only": True,
        "score_formula_changed": False,
        "llm_called": False,
        "audio_model_called": False,
        "cases": results,
    }
    _write_json(run_dir / "summary.json", payload)

    lines = [
        "# D9 质检影子评估",
        "",
        "本次只读取已保存歌词；未调用 LLM 或音频模型，未修改歌词，未改变评分公式和通过门槛。",
        "",
    ]
    for result in results:
        rhythm = result["rhythm"]
        theme = result["theme_signal"]
        historical_match = result["historical_result_matches_current"]
        historical_text = (
            "无历史报告"
            if historical_match is None
            else str(historical_match)
        )
        lines.extend(
            [
                f"## {result['case_id']}",
                "",
                (
                    f"- D9 未改变当前 D6-D7 总分/门槛："
                    f"{result['d9_score_and_gate_unchanged']}；"
                    f"当前总分 {result['current_result']['total_score']}，"
                    f"通过={result['current_result']['passed']}"
                ),
                (
                    f"- 与历史产物记录一致：{historical_text}"
                ),
                f"- 节奏参考分：{rhythm['score']}；建议数：{len(rhythm['issues'])}",
                (
                    f"- 主题词面命中：{len(theme['matched_keywords'])}/"
                    f"{len(theme['keywords'])}；Hook 命中："
                    f"{('、'.join(theme['hook_matches']) or '无直接命中')}"
                ),
                "",
            ]
        )
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(run_dir.resolve())


if __name__ == "__main__":
    main()
