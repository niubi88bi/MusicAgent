"""Demo 阶段使用的线程安全内存任务表。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any
from uuid import uuid4


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class TaskNotFoundError(KeyError):
    """任务 ID 不存在。"""


class InvalidTaskTransitionError(ValueError):
    """任务状态转换不符合生命周期约束。"""


_ALLOWED_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.RUNNING, TaskStatus.FAILED},
    TaskStatus.RUNNING: {TaskStatus.DONE, TaskStatus.FAILED},
    TaskStatus.DONE: set(),
    TaskStatus.FAILED: set(),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskRecord:
    task_id: str
    status: TaskStatus
    input_data: dict[str, Any]
    created_at: str
    updated_at: str
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "input": deepcopy(self.input_data),
            "result": deepcopy(self.result),
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress": deepcopy(self.progress),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class TaskStore:
    """线程安全的任务快照存储；进程重启后数据会丢失。"""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = RLock()

    def create(self, input_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """创建 pending 任务并返回独立快照。"""
        timestamp = _now()
        record = TaskRecord(
            task_id=uuid4().hex,
            status=TaskStatus.PENDING,
            input_data=deepcopy(input_data or {}),
            created_at=timestamp,
            updated_at=timestamp,
        )
        with self._lock:
            self._tasks[record.task_id] = record
            return record.to_dict()

    def get(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            return self._get_record(task_id).to_dict()

    def list(self) -> list[dict[str, Any]]:
        """按创建时间倒序返回任务快照。"""
        with self._lock:
            records = sorted(
                self._tasks.values(), key=lambda task: task.created_at, reverse=True
            )
            return [record.to_dict() for record in records]

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """更新状态，并限制 pending → running → done/failed 生命周期。"""
        with self._lock:
            record = self._get_record(task_id)
            if status not in _ALLOWED_TRANSITIONS[record.status]:
                raise InvalidTaskTransitionError(
                    f"不允许任务从 {record.status.value} 转为 {status.value}"
                )
            if status is TaskStatus.DONE and error is not None:
                raise ValueError("成功任务不能携带 error")
            if status is TaskStatus.FAILED and not error:
                raise ValueError("失败任务必须提供 error")

            record.status = status
            record.updated_at = _now()
            if status is TaskStatus.RUNNING:
                record.started_at = record.updated_at
            if status in {TaskStatus.DONE, TaskStatus.FAILED}:
                record.finished_at = record.updated_at
            record.result = deepcopy(result) if result is not None else None
            record.error = error
            return record.to_dict()

    def update_progress(self, task_id: str, progress: dict[str, Any]) -> None:
        """只允许运行中的任务更新进度，外部快照不能污染内部记录。"""
        with self._lock:
            record = self._get_record(task_id)
            if record.status is not TaskStatus.RUNNING:
                raise InvalidTaskTransitionError("只有运行中的任务可以更新进度")
            record.progress = deepcopy(progress)
            record.updated_at = _now()

    def _get_record(self, task_id: str) -> TaskRecord:
        try:
            return self._tasks[task_id]
        except KeyError as error:
            raise TaskNotFoundError(f"任务不存在：{task_id}") from error
