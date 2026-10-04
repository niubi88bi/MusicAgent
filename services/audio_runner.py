"""HeartMuLa 按需试听任务：单任务执行，并用子进程隔离 GPU 模型。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import BoundedSemaphore, RLock
from typing import Callable
from uuid import uuid4


class AudioQueueFullError(RuntimeError):
    """当前已有试听任务占用唯一执行名额。"""


class AudioTaskNotFoundError(KeyError):
    """指定文本任务没有试听记录。"""


AudioExecutor = Callable[[Path, str, int], Path]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execute_audio_subprocess(source_dir: Path, audio_task_id: str, duration: int) -> Path:
    """调用受控脚本；HTTP 输入不能指定磁盘路径或命令。"""
    project_root = Path(__file__).resolve().parents[1]
    log_path = source_dir / "audio_process.log"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        "-m",
        "scripts.run_audio_task",
        "--source-dir",
        str(source_dir),
        "--job-id",
        audio_task_id,
        "--duration",
        str(duration),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60 * 60 if duration == 240 else 30 * 60,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"HeartMuLa 子进程退出码 {completed.returncode}")
    return source_dir / "audio" / audio_task_id / "song.mp3"


class AudioRunner:
    """每次只运行一个试听；终态写入源文本任务目录。"""

    def __init__(self, output_root: Path, *, execute: AudioExecutor = execute_audio_subprocess):
        self.output_root = output_root
        self.execute = execute
        self._lock = RLock()
        self._slot = BoundedSemaphore(1)
        self._records: dict[str, dict] = {}
        self._executor: ThreadPoolExecutor | None = None

    def start(self) -> None:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="music-audio")

    def close(self) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor:
            executor.shutdown(wait=True)

    def submit(
        self,
        source_task_id: str,
        *,
        lyrics: str,
        tags: list[str],
        selected_attempt: int,
        quality_status: str,
        duration_seconds: int = 30,
    ) -> dict:
        if duration_seconds not in {30, 240}:
            raise ValueError("Web 音频生成上限只能是 30 或 240 秒")
        with self._lock:
            if self._executor is None:
                raise RuntimeError("音频执行器未启动")
            existing = self.get_optional(source_task_id)
            if existing and existing["status"] in {"pending", "running"}:
                return existing
            if existing and existing["status"] == "done" and self.audio_path(existing).is_file():
                return existing
            if not self._slot.acquire(blocking=False):
                raise AudioQueueFullError()

            record = {
                "audio_task_id": uuid4().hex,
                "source_task_id": source_task_id,
                "status": "pending",
                "backend": "heartmula-local",
                "duration_seconds": duration_seconds,
                "duration_semantics": "generation_upper_bound",
                "selected_attempt": selected_attempt,
                "quality_status": quality_status,
                "created_at": _now(),
                "updated_at": _now(),
                "started_at": None,
                "finished_at": None,
                "audio_url": None,
                "size_bytes": None,
                "error": None,
            }
            self._records[source_task_id] = record
            self._write(record)
            try:
                future = self._executor.submit(self._run, deepcopy(record), lyrics, list(tags))
                future.add_done_callback(lambda _: self._slot.release())
            except Exception:
                self._slot.release()
                record.update(status="failed", error="试听任务无法调度，请稍后再试。", updated_at=_now(), finished_at=_now())
                self._records[source_task_id] = record
                self._write(record)
                raise
            return deepcopy(record)

    def get(self, source_task_id: str) -> dict:
        record = self.get_optional(source_task_id)
        if record is None:
            raise AudioTaskNotFoundError(source_task_id)
        return record

    def get_optional(self, source_task_id: str) -> dict | None:
        with self._lock:
            record = self._records.get(source_task_id)
            if record is None:
                record = self._read(source_task_id)
                if record is None:
                    return None
                # 进程重启后没有对应 worker，旧的运行态只能标记为中断。
                if record["status"] in {"pending", "running"}:
                    record.update(
                        status="failed",
                        error="试听生成因服务重启而中断，可以重新生成。",
                        updated_at=_now(),
                        finished_at=_now(),
                    )
                    self._write(record)
                self._records[source_task_id] = record
            return deepcopy(record)

    def audio_path(self, record: dict) -> Path:
        return self.output_root / record["source_task_id"] / "audio" / record["audio_task_id"] / "song.mp3"

    def _run(self, record: dict, lyrics: str, tags: list[str]) -> None:
        source_task_id = record["source_task_id"]
        try:
            self._update(source_task_id, status="running", started_at=_now())
            path = self.execute(
                self.output_root / source_task_id,
                record["audio_task_id"],
                record["duration_seconds"],
            )
            expected = self.audio_path(record).resolve()
            if path.resolve() != expected or not expected.is_file() or expected.stat().st_size == 0:
                raise RuntimeError("HeartMuLa 未生成有效的预期 MP3")
            self._update(
                source_task_id,
                status="done",
                finished_at=_now(),
                audio_url=f"/api/tasks/{source_task_id}/audio",
                size_bytes=expected.stat().st_size,
                error=None,
            )
        except subprocess.TimeoutExpired:
            minutes = 60 if record["duration_seconds"] == 240 else 30
            self._update(source_task_id, status="failed", finished_at=_now(), error=f"音频生成超过 {minutes} 分钟，已终止。")
        except Exception:
            self._update(source_task_id, status="failed", finished_at=_now(), error="HeartMuLa 试听生成失败，请查看本地任务日志后重试。")

    def _update(self, source_task_id: str, **changes) -> None:
        with self._lock:
            record = self._records[source_task_id]
            record.update(changes, updated_at=_now())
            self._write(record)

    def _record_path(self, source_task_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", source_task_id):
            raise AudioTaskNotFoundError(source_task_id)
        return self.output_root / source_task_id / "audio_task.json"

    def _read(self, source_task_id: str) -> dict | None:
        try:
            record = json.loads(self._record_path(source_task_id).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AudioTaskNotFoundError):
            return None
        if not isinstance(record, dict) or record.get("source_task_id") != source_task_id:
            return None
        if not re.fullmatch(r"[0-9a-f]{32}", str(record.get("audio_task_id", ""))):
            return None
        if record.get("status") not in {"pending", "running", "done", "failed"}:
            return None
        return record

    def _write(self, record: dict) -> None:
        path = self._record_path(record["source_task_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
