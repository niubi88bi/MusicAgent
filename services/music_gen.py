"""HeartMuLa 本地音频生成服务。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Sequence
from uuid import uuid4

from services.config import Settings, get_settings


class MusicGenerationError(RuntimeError):
    """本地音频生成失败。"""


@dataclass(frozen=True)
class MusicGenerationResult:
    """一次本地生成的可追溯产物。"""

    job_id: str
    output_path: Path
    lyrics_path: Path
    tags_path: Path
    duration_seconds: int


PipelineFactory = Callable[[Path], Any]


class HeartMuLaGenerator:
    """将 HeartMuLa 调用隔离在服务层。

    4060 8GB 使用 lazy_load=True，避免 HeartMuLa 与 HeartCodec 同时常驻时 OOM。
    Pipeline 在当前进程内复用；后续接入任务队列时，可将此服务放到专用 worker 进程。
    """

    def __init__(
        self,
        model_path: Path,
        output_dir: Path,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self._model_path = model_path
        self._output_dir = output_dir
        self._pipeline_factory = pipeline_factory or self._create_pipeline
        self._pipeline: Any | None = None
        self._pipeline_lock = Lock()
        self._generation_lock = Lock()

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        output_dir: Path | None = None,
    ) -> "HeartMuLaGenerator":
        settings = settings or get_settings()
        return cls(
            model_path=settings.heartmula_model_path,
            output_dir=output_dir or settings.music_output_dir,
        )

    def generate(
        self,
        lyrics: str,
        tags: Sequence[str],
        duration_seconds: int,
        *,
        job_id: str | None = None,
    ) -> MusicGenerationResult:
        """写入模型输入、串行调用 GPU，并返回音频与输入文件路径。"""
        if not lyrics.strip():
            raise ValueError("lyrics 不能为空")
        clean_tags = [tag.strip() for tag in tags if tag.strip()]
        if not clean_tags:
            raise ValueError("tags 至少需要一个非空标签")
        if not 1 <= duration_seconds <= 240:
            raise ValueError("duration_seconds 必须在 1 到 240 之间")

        resolved_job_id = job_id or uuid4().hex
        job_dir = self._output_dir / resolved_job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        lyrics_path = job_dir / "lyrics.txt"
        tags_path = job_dir / "tags.txt"
        output_path = job_dir / "song.mp3"
        lyrics_path.write_text(lyrics.strip() + "\n", encoding="utf-8")
        tags_path.write_text(", ".join(clean_tags), encoding="utf-8")

        # 单张 GPU 下避免两个任务同时争抢显存；任务队列将在 W3 接管排队。
        with self._generation_lock:
            pipeline = self._get_pipeline()
            try:
                import torch

                with torch.no_grad():
                    pipeline(
                        {"lyrics": str(lyrics_path), "tags": str(tags_path)},
                        max_audio_length_ms=duration_seconds * 1000,
                        save_path=str(output_path),
                    )
            except Exception as error:
                raise MusicGenerationError(f"HeartMuLa 生成失败：{error}") from error

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise MusicGenerationError("HeartMuLa 未生成有效音频文件")

        return MusicGenerationResult(
            job_id=resolved_job_id,
            output_path=output_path,
            lyrics_path=lyrics_path,
            tags_path=tags_path,
            duration_seconds=duration_seconds,
        )

    def _get_pipeline(self) -> Any:
        with self._pipeline_lock:
            if self._pipeline is None:
                self._pipeline = self._pipeline_factory(self._model_path)
            return self._pipeline

    @staticmethod
    def _create_pipeline(model_path: Path) -> Any:
        if not model_path.is_dir():
            raise MusicGenerationError(f"HeartMuLa 权重目录不存在：{model_path}")

        try:
            import torch
            from heartlib import HeartMuLaGenPipeline
        except ImportError as error:
            raise MusicGenerationError("缺少 HeartMuLa 或 Torch 依赖") from error

        if not torch.cuda.is_available():
            raise MusicGenerationError("未检测到可用 CUDA GPU")

        return HeartMuLaGenPipeline.from_pretrained(
            str(model_path),
            device={"mula": torch.device("cuda"), "codec": torch.device("cuda")},
            dtype={"mula": torch.bfloat16, "codec": torch.float32},
            version="3B",
            lazy_load=True,
        )
