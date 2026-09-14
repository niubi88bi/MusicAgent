"""单进程、有界并发的文本任务执行器，不导入音频生成服务。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from threading import BoundedSemaphore, Lock

from pydantic import ValidationError

from agents.critic_agent import LyricJudgeAgent
from agents.graph import build_music_graph
from agents.lyricist_agent import LyricistAgent
from agents.producer_agent import ProducerAgent
from agents.requirement_agent import RequirementAgent
from services.llm import DeepSeekLLM, LLMConfigurationError, LLMRequestError, LLMResponseError
from services.task_store import TaskStatus, TaskStore
from services.workflow_artifacts import public_text_state, write_task_snapshot, write_text_artifacts


class TaskQueueFullError(RuntimeError):
    """运行名额和等待名额均已占用。"""


def execute_text(theme, on_event):
    """每个任务独立使用 LLM 连接，文本模式图没有 generator 节点。"""
    llm = DeepSeekLLM()
    try:
        graph = build_music_graph(
            RequirementAgent(llm),
            LyricistAgent(llm, persona="structured"),
            ProducerAgent(llm),
            secondary_lyricist_agent=LyricistAgent(llm, persona="imagery"),
            lyric_judge_agent=LyricJudgeAgent(llm),
            text_only=True,
            on_event=on_event,
        )
        return graph.invoke({"theme": theme, "requested_duration_seconds": 30})
    finally:
        llm.close()


def safe_error(error):
    # 不透传供应商报错、模型原始返回、配置值或内部路径。
    if isinstance(error, LLMConfigurationError):
        return "未配置文本模型密钥，请在本地配置 DEEPSEEK_API_KEY 后重启服务。"
    if isinstance(error, LLMRequestError):
        return "文本模型请求失败或超时，请稍后重新提交。已有文本仍保留。"
    if isinstance(error, (ValidationError, LLMResponseError)):
        return "模型返回的内容未满足格式要求。已有文本仍保留，请重新提交。"
    if isinstance(error, OSError):
        return "任务产物保存失败，请检查本地磁盘空间和目录权限。"
    return "文本任务执行失败，已有结果仍可查看。请根据任务编号检查本地服务。"


class WorkflowRunner:
    """一个运行位和两个等待位；满载时拒绝，不创建悬空任务。"""

    def __init__(self, store: TaskStore, output_dir: Path, *, execute=execute_text, waiting_slots=2):
        self.store = store
        self.output_dir = output_dir
        self.execute = execute
        self._slots = BoundedSemaphore(1 + waiting_slots)
        self._lock = Lock()
        self._executor = None

    def start(self):
        with self._lock:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="music-text")

    def close(self):
        with self._lock:
            executor, self._executor = self._executor, None
        if executor:
            # 优雅关闭等待已接收任务完成；强制杀进程不支持自动恢复。
            executor.shutdown(wait=True)

    def submit(self, theme: str) -> dict:
        with self._lock:
            if self._executor is None:
                raise RuntimeError("任务执行器未启动")
            if not self._slots.acquire(blocking=False):
                raise TaskQueueFullError()
            task = None
            try:
                task = self.store.create({"theme": theme, "mode": "text", "duration_seconds": 30})
                future = self._executor.submit(self._run, task["task_id"], theme)
                future.add_done_callback(lambda _: self._slots.release())
                return task
            except Exception:
                self._slots.release()
                if task is not None:
                    self.store.update_status(task["task_id"], TaskStatus.FAILED, error="无法调度任务，请稍后再试。")
                raise

    def _run(self, task_id, theme):
        job_dir = self.output_dir / task_id
        self.store.update_status(task_id, TaskStatus.RUNNING)
        events = []
        event_lock = Lock()

        def observe(stage, status, state):
            # 双作词分支会并发上报；串行化事件、任务快照和磁盘替换，避免覆盖。
            with event_lock:
                attempt = state.get("attempts", 0)
                if stage == "lyricist" and status in {"running", "failed"}:
                    attempt += 1
                elif stage in {"lyricist_structured", "lyricist_imagery"}:
                    attempt = state.get("candidate_generation_round", 1)
                events.append({"stage": stage, "status": status,
                               "at": datetime.now(timezone.utc).isoformat(),
                               "attempt": attempt})
                self.store.update_progress(task_id, {"stage": stage, "events": [*events],
                                                     "text": public_text_state(state)})
                write_task_snapshot(job_dir, self.store.get(task_id))

        try:
            state = self.execute(theme, observe)
            write_text_artifacts(job_dir, state)
            # 完成状态先落盘，再向轮询读者发布，避免保存失败却显示成功。
            result = public_text_state(state)
            snapshot = self.store.get(task_id)
            finished = datetime.now(timezone.utc).isoformat()
            snapshot.update(status="done", result=result, finished_at=finished, updated_at=finished)
            write_task_snapshot(job_dir, snapshot)
            self.store.update_status(task_id, TaskStatus.DONE, result=result)
        except Exception as error:
            logging.getLogger(__name__).error("Text task %s failed (%s)", task_id, type(error).__name__)
            # 只在本地保存字段位置和错误类型，避免缺少定位信息又泄露模型输入。
            diagnostic = {"error_type": type(error).__name__}
            if isinstance(error, ValidationError):
                diagnostic["fields"] = [{"location": list(item["loc"]), "type": item["type"]}
                                        for item in error.errors(include_input=False, include_context=False)]
            message = safe_error(error)
            try:
                snapshot = self.store.get(task_id)
                finished = datetime.now(timezone.utc).isoformat()
                snapshot.update(status="failed", error=message, finished_at=finished, updated_at=finished)
                write_task_snapshot(job_dir, snapshot)
                (job_dir / "failure.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                logging.getLogger(__name__).error("Could not persist task %s", task_id)
            finally:
                self.store.update_status(task_id, TaskStatus.FAILED, error=message)
