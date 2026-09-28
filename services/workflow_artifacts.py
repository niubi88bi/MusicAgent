"""保存工作流中的可解释文本产物。"""

from __future__ import annotations

import json
import os
import re
from time import sleep
from pathlib import Path
from typing import Mapping
from uuid import uuid4
from pydantic import BaseModel


_SNAPSHOT_REPLACE_RETRIES = 4
_SNAPSHOT_RETRY_DELAY_SECONDS = 0.05


def load_persisted_task_snapshots(root: Path) -> list[dict]:
    """读取终态快照供历史查看；不恢复运行中的任务。"""
    if not root.is_dir():
        return []
    snapshots: list[dict] = []
    for job_dir in root.iterdir():
        if not job_dir.is_dir() or not re.fullmatch(r"[0-9a-f]{32}", job_dir.name):
            continue
        try:
            snapshot = json.loads((job_dir / "task.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (isinstance(snapshot, dict) and snapshot.get("task_id") == job_dir.name
                and snapshot.get("status") in {"done", "failed"}):
            snapshots.append(snapshot)
    return sorted(snapshots, key=lambda item: item.get("created_at", ""), reverse=True)


def public_text_state(state: Mapping[str, object]) -> dict:
    """只导出文本产物白名单；不把内部磁盘路径作为可播放音频。"""
    keys = ("theme", "production_constraints", "requirement", "lyric_candidates", "subjective_review",
            "subjective_review_history", "candidate_regeneration_count", "candidate_selection",
            "selected_lyricist", "lyric_draft", "lyrics", "production_plan",
            "scores", "quality_report", "quality_status", "quality_history",
            "attempts", "selected_attempt")
    result = {}
    for key in keys:
        value = state.get(key)
        if value is not None:
            result[key] = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    result["audio_url"] = None
    return result


def write_task_snapshot(job_dir: Path, snapshot: dict) -> None:
    """原子替换最新快照，失败时仍保留已完成的歌词和逐轮报告。"""
    job_dir.mkdir(parents=True, exist_ok=True)
    target = job_dir / "task.json"
    # 不使用固定的 task.json.tmp：Windows 下多个写入者共享同一个临时文件
    # 会互相覆盖或在 replace 时触发 PermissionError。
    temporary = job_dir / f".task.json.{uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for attempt in range(_SNAPSHOT_REPLACE_RETRIES):
            try:
                os.replace(temporary, target)
                return
            except PermissionError:
                if attempt == _SNAPSHOT_REPLACE_RETRIES - 1:
                    raise
                sleep(_SNAPSHOT_RETRY_DELAY_SECONDS * (attempt + 1))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_text_artifacts(job_dir: Path, state: Mapping[str, object]) -> None:
    """完成文本任务时保存现有脚本可读取的独立产物。"""
    job_dir.mkdir(parents=True, exist_ok=True)
    exported = public_text_state(state)
    for key, filename in (("requirement", "requirement.json"),
                          ("lyric_draft", "lyric_draft.json"),
                          ("production_plan", "production_plan.json")):
        (job_dir / filename).write_text(json.dumps(exported[key], ensure_ascii=False, indent=2), encoding="utf-8")
    (job_dir / "lyrics.txt").write_text(str(exported["lyrics"]), encoding="utf-8")
    if "lyric_candidates" in exported:
        (job_dir / "lyric_candidates.json").write_text(
            json.dumps(exported["lyric_candidates"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if "candidate_selection" in exported:
        (job_dir / "candidate_selection.json").write_text(
            json.dumps(exported["candidate_selection"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if "subjective_review" in exported:
        (job_dir / "subjective_review.json").write_text(
            json.dumps(exported["subjective_review"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if "subjective_review_history" in exported:
        (job_dir / "subjective_review_history.json").write_text(
            json.dumps(
                exported["subjective_review_history"], ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
    write_quality_artifacts(job_dir, state)


def write_quality_artifacts(
    job_dir: Path, final_state: Mapping[str, object]
) -> tuple[Path, Path]:
    """保存最终质检结论和每次回炉记录，供演示与复现查看。"""

    quality_report = final_state.get("quality_report")
    quality_history = final_state.get("quality_history")
    if not isinstance(quality_report, dict) or not isinstance(quality_history, list):
        raise TypeError("LangGraph 未返回可保存的歌词质检记录")

    summary = {
        "quality_status": final_state.get("quality_status"),
        "attempts": final_state.get("attempts"),
        "selected_attempt": final_state.get("selected_attempt"),
        "scores": final_state.get("scores"),
        "quality_report": quality_report,
    }
    report_path = job_dir / "quality_report.json"
    history_path = job_dir / "lyric_revision_history.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    history_path.write_text(
        json.dumps(quality_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report_path, history_path
