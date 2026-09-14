"""HeartMuLa 服务的无 GPU 单元测试。"""

from pathlib import Path
import unittest
from uuid import uuid4

from services.music_gen import HeartMuLaGenerator, MusicGenerationError


TEST_TEMP_ROOT = Path("tmp/test-work")


def project_temp_dir() -> Path:
    """测试临时文件留在项目 D 盘，避免 Windows 临时目录 ACL 干扰。"""
    path = TEST_TEMP_ROOT / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, inputs: dict[str, str], **kwargs: object) -> None:
        self.calls.append({"inputs": inputs, **kwargs})
        Path(str(kwargs["save_path"])).write_bytes(b"fake mp3")


class MusicGenerationTests(unittest.TestCase):
    def test_writes_inputs_and_reuses_pipeline(self) -> None:
        pipeline = FakePipeline()
        factory_calls = 0

        def factory(_: Path) -> FakePipeline:
            nonlocal factory_calls
            factory_calls += 1
            return pipeline

        generator = HeartMuLaGenerator(
            model_path=Path("unused-in-test"),
            output_dir=project_temp_dir(),
            pipeline_factory=factory,
        )
        first = generator.generate(
            "[Verse]\n测试歌词", ["city pop", "warm"], 30, job_id="first"
        )
        second = generator.generate(
            "[Verse]\n第二段歌词", ["ballad"], 30, job_id="second"
        )

        self.assertTrue(first.output_path.is_file())
        self.assertEqual(first.tags_path.read_text("utf-8"), "city pop, warm")
        self.assertEqual(second.duration_seconds, 30)
        self.assertEqual(factory_calls, 1)
        self.assertEqual(len(pipeline.calls), 2)

    def test_rejects_empty_audio_output(self) -> None:
        class EmptyPipeline:
            def __call__(self, *_: object, **__: object) -> None:
                return None

        generator = HeartMuLaGenerator(
            model_path=Path("unused-in-test"),
            output_dir=project_temp_dir(),
            pipeline_factory=lambda _: EmptyPipeline(),
        )
        with self.assertRaises(MusicGenerationError):
            generator.generate("[Verse]\n测试歌词", ["pop"], 30)


if __name__ == "__main__":
    unittest.main()
