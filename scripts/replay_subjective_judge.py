"""用已保存的双作词候选回放匿名主观评审，不重新生成歌词或音频。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from agents.critic_agent import LyricJudgeAgent, select_subjective_candidate
from agents.schemas import SongRequirement
from services.llm import DeepSeekLLM


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _latest_candidates(case_dir: Path) -> list[dict[str, object]]:
    payload = _read_json(case_dir / "lyric_candidates.json")
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"{case_dir} 缺少双作词候选")
    candidates = [item for item in payload if isinstance(item, dict)]
    latest_round = max(int(item.get("generation_round", 1)) for item in candidates)
    latest = [
        item
        for item in candidates
        if int(item.get("generation_round", 1)) == latest_round
    ]
    candidate_ids = {str(item.get("candidate_id")) for item in latest}
    if candidate_ids != {"structured", "imagery"}:
        raise ValueError(f"{case_dir} 最新一轮候选身份不完整：{candidate_ids}")
    return latest


def _human_preference(case_dir: Path) -> str | None:
    path = case_dir / "human_blind_review.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return None
    agreement = payload.get("agreement")
    if not isinstance(agreement, dict):
        return None
    preferred = agreement.get("human_preferred_candidate_id")
    return str(preferred) if preferred in {"structured", "imagery"} else None


def _rule_preference(case_dir: Path) -> str | None:
    path = case_dir / "candidate_selection.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return None
    selected = payload.get("selected_candidate_id")
    return str(selected) if selected in {"structured", "imagery"} else None


def _write_summary_markdown(run_dir: Path, summary: dict[str, object]) -> None:
    lines = [
        "# 主观评审 Agent 历史候选回放",
        "",
        "本回放只读取既有 A/B 歌词，不重新作词、不加载 HeartMuLa 或 GPU。",
        "",
        "## 案例结果",
        "",
    ]
    for item in summary["cases"]:
        lines.append(
            f"- {item['case_id']}：评审选 {item['selected_candidate_id']}，"
            f"人工选 {item['human_preferred_candidate_id']}，"
            f"一致={item['matches_human']}，依据={item['selection_basis']}。"
        )
    aggregate = summary["aggregate"]
    lines.extend(
        [
            "",
            "## 汇总",
            "",
            f"- 有人工标签案例：{aggregate['human_labeled_cases']}。",
            f"- 与本轮人工偏好一致：{aggregate['human_agreement_cases']}。",
            f"- 描述性一致率：{aggregate['human_agreement_rate']}。",
            "",
            "## 限制",
            "",
            "- 只有三条历史主题和一名人工评审者，不能视为准确率或统计结论。",
            "- 本结果只评价歌词文本，不评价旋律、演唱或音频质量。",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="回放已有双作词候选的匿名主观评审。"
    )
    parser.add_argument(
        "case_dirs",
        nargs="+",
        type=Path,
        help="包含 requirement.json 与 lyric_candidates.json 的案例目录。",
    )
    parser.add_argument(
        "--out",
        default="data/output/evaluations",
        type=Path,
        help="回放结果根目录。",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="确认调用真实 DeepSeek；未提供时不会发起网络请求。",
    )
    args = parser.parse_args()

    case_dirs = [path.resolve() for path in args.case_dirs]
    for case_dir in case_dirs:
        if not case_dir.is_dir():
            parser.error(f"案例目录不存在：{case_dir}")
        for required_name in ("requirement.json", "lyric_candidates.json"):
            if not (case_dir / required_name).is_file():
                parser.error(f"案例缺少 {required_name}：{case_dir}")

    if not args.execute:
        print(
            "该命令只回放既有歌词，但会调用真实 DeepSeek。确认后请加 --execute；"
            "不会加载 HeartMuLa 或 GPU。",
            flush=True,
        )
        return

    run_dir = args.out / f"subjective-judge-replay-{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    llm = DeepSeekLLM()
    judge = LyricJudgeAgent(llm)
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for case_dir in case_dirs:
        case_id = case_dir.name
        output_dir = run_dir / case_id
        output_dir.mkdir(parents=True, exist_ok=False)
        before_calls = len(llm.telemetry_snapshot())
        try:
            requirement_payload = _read_json(case_dir / "requirement.json")
            requirement = SongRequirement.model_validate(requirement_payload)
            candidates = _latest_candidates(case_dir)
            review = judge.run(requirement, candidates)
            selected, basis = select_subjective_candidate(
                candidates,
                review,
                {"structured": 0, "imagery": 1},
            )
            selected_id = str(selected["candidate_id"])
            human_preferred = _human_preference(case_dir)
            rule_preferred = _rule_preference(case_dir)
            call_records = llm.telemetry_snapshot()[before_calls:]
            result = {
                "case_id": case_id,
                "source_case_dir": str(case_dir),
                "selected_candidate_id": selected_id,
                "selection_basis": basis,
                "human_preferred_candidate_id": human_preferred,
                "rule_preferred_candidate_id": rule_preferred,
                "matches_human": (
                    selected_id == human_preferred
                    if human_preferred is not None
                    else None
                ),
                "subjective_review": review,
                "telemetry": DeepSeekLLM.summarize_call_records(call_records),
            }
            _write_json(output_dir / "review_result.json", result)
            results.append(result)
        except Exception as error:
            call_records = llm.telemetry_snapshot()[before_calls:]
            failure = {
                "case_id": case_id,
                "source_case_dir": str(case_dir),
                "error_type": type(error).__name__,
                "message": str(error),
                "telemetry": DeepSeekLLM.summarize_call_records(call_records),
            }
            _write_json(output_dir / "failure.json", failure)
            failures.append(failure)

    labeled = [item for item in results if item["matches_human"] is not None]
    agreement_count = sum(item["matches_human"] is True for item in labeled)
    summary = {
        "mode": "saved_candidate_subjective_judge_replay",
        "audio_generated": False,
        "lyrics_regenerated": False,
        "cases": results,
        "failures": failures,
        "aggregate": {
            "completed_cases": len(results),
            "failed_cases": len(failures),
            "human_labeled_cases": len(labeled),
            "human_agreement_cases": agreement_count,
            "human_agreement_rate": (
                round(agreement_count / len(labeled), 3) if labeled else None
            ),
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_summary_markdown(run_dir, summary)
    print(run_dir, flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
