"""MusicAgent 的最小 FastAPI 服务入口。"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from services.config import get_settings
from services.task_store import TaskStore, TaskNotFoundError
from services.workflow_runner import WorkflowRunner, TaskQueueFullError
from services.workflow_artifacts import load_persisted_task_snapshots
from services.audio_runner import AudioQueueFullError, AudioRunner, AudioTaskNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TextTaskInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    theme: str = Field(min_length=2, max_length=200, strict=True)


def create_app(*, store=None, runner=None, audio_runner=None, include_persisted: bool = True) -> FastAPI:
    """创建应用实例，便于单元测试和后续部署复用。"""
    task_store = store if store is not None else TaskStore()
    runner = runner if runner is not None else WorkflowRunner(task_store, PROJECT_ROOT / "data/output/web")
    audio_runner = audio_runner if audio_runner is not None else AudioRunner(runner.output_dir)

    def find_text_task(task_id: str) -> dict:
        try:
            return task_store.get(task_id)
        except TaskNotFoundError:
            if include_persisted:
                archived = next((item for item in load_persisted_task_snapshots(runner.output_dir)
                                 if item["task_id"] == task_id), None)
                if archived is not None:
                    return archived
            raise HTTPException(404, "任务不存在或服务已重启，当前不能恢复该任务。")

    def attach_audio(task: dict) -> dict:
        snapshot = dict(task)
        snapshot["audio"] = audio_runner.get_optional(task["task_id"])
        return snapshot

    @asynccontextmanager
    async def lifespan(application):
        runner.start()
        audio_runner.start()
        try:
            yield
        finally:
            await asyncio.gather(
                asyncio.to_thread(runner.close),
                asyncio.to_thread(audio_runner.close),
            )

    app = FastAPI(title="MusicAgent API", version="0.2.0", lifespan=lifespan)
    app.state.task_store = task_store

    @app.get("/health")
    def health() -> dict[str, object]:
        """返回服务存活状态，不触发外部 LLM 调用。"""
        settings = get_settings()
        return {
            "status": "ok",
            "service": "music-agent",
            "llm_configured": settings.is_llm_configured,
            "text_tasks_enabled": True,
            "audio_backend": "heartmula-local",
        }

    @app.get("/api/tasks")
    def list_tasks() -> dict[str, object]:
        """返回内存任务和磁盘终态快照；磁盘快照只用于查看。"""
        items = task_store.list()
        ids = {item["task_id"] for item in items}
        if include_persisted:
            items.extend(item for item in load_persisted_task_snapshots(runner.output_dir)
                         if item["task_id"] not in ids)
        return {"items": [attach_audio(item) for item in items]}

    @app.post("/api/tasks", status_code=202)
    def create_task(body: TextTaskInput):
        if not get_settings().is_llm_configured:
            raise HTTPException(503, "未配置文本模型密钥，请在本地配置后重启服务。")
        try:
            return runner.submit(body.theme)
        except TaskQueueFullError:
            raise HTTPException(429, "创作队列已满（1 个运行、最多 2 个等待），请稍后再试。", headers={"Retry-After": "5"})

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str):
        return attach_audio(find_text_task(task_id))

    @app.post("/api/tasks/{task_id}/audio", status_code=202)
    def create_audio(task_id: str):
        task = find_text_task(task_id)
        result = task.get("result")
        if task.get("status") != "done" or not isinstance(result, dict):
            raise HTTPException(409, "只有已完成的文本任务可以生成试听。")
        lyrics = result.get("lyrics")
        production = result.get("production_plan")
        tags = production.get("tags") if isinstance(production, dict) else None
        selected_attempt = result.get("selected_attempt")
        quality_status = result.get("quality_status")
        if not isinstance(lyrics, str) or not isinstance(tags, list) or not tags:
            raise HTTPException(409, "文本任务缺少歌词或制作标签，不能生成试听。")
        if not isinstance(selected_attempt, int) or quality_status not in {"passed", "capped"}:
            raise HTTPException(409, "文本任务缺少最终采用版本，不能生成试听。")
        try:
            return audio_runner.submit(
                task_id,
                lyrics=lyrics,
                tags=tags,
                selected_attempt=selected_attempt,
                quality_status=quality_status,
            )
        except AudioQueueFullError:
            raise HTTPException(429, "当前已有试听任务在生成，请完成后再试。", headers={"Retry-After": "10"})

    @app.get("/api/tasks/{task_id}/audio/status")
    def get_audio_status(task_id: str):
        find_text_task(task_id)
        try:
            return audio_runner.get(task_id)
        except AudioTaskNotFoundError:
            raise HTTPException(404, "该文本任务还没有试听记录。")

    @app.get("/api/tasks/{task_id}/audio")
    def get_audio(task_id: str):
        find_text_task(task_id)
        try:
            record = audio_runner.get(task_id)
        except AudioTaskNotFoundError:
            raise HTTPException(404, "该文本任务还没有试听记录。")
        if record["status"] != "done":
            raise HTTPException(409, record.get("error") or "试听仍在生成中。")
        path = audio_runner.audio_path(record).resolve()
        allowed_root = runner.output_dir.resolve()
        if allowed_root not in path.parents or not path.is_file():
            raise HTTPException(404, "试听文件不存在。")
        return FileResponse(path, media_type="audio/mpeg")

    # API 在前、静态目录在后；只能访问 frontend，不能暴露项目根目录。
    app.mount("/", StaticFiles(directory=PROJECT_ROOT / "frontend", html=True), name="frontend")

    return app


app = create_app()
