"""任务快照在 Windows 文件替换语义下的回归测试。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from unittest.mock import patch

import services.workflow_artifacts as workflow_artifacts


class WorkflowArtifactSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1] / "tmp/test-work"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(dir=root)
        self.addCleanup(self.temporary.cleanup)

    def test_concurrent_writers_do_not_share_temporary_file(self) -> None:
        job_dir = Path(self.temporary.name) / "job"
        errors: list[Exception] = []

        def write_many(worker: int) -> None:
            for index in range(100):
                try:
                    workflow_artifacts.write_task_snapshot(
                        job_dir, {"worker": worker, "index": index}
                    )
                except Exception as error:  # pragma: no cover - assertion below reports it
                    errors.append(error)

        threads = [Thread(target=write_many, args=(worker,)) for worker in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertTrue((job_dir / "task.json").is_file())
        json.loads((job_dir / "task.json").read_text(encoding="utf-8"))
        self.assertEqual(list(job_dir.glob(".task.json.*.tmp")), [])

    def test_retries_transient_permission_error_and_cleans_temporary_file(self) -> None:
        job_dir = Path(self.temporary.name) / "job"
        original_replace = workflow_artifacts.os.replace
        attempts = 0

        def flaky_replace(source, target):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise PermissionError("simulated Windows file lock")
            return original_replace(source, target)

        with patch.object(workflow_artifacts.os, "replace", side_effect=flaky_replace):
            workflow_artifacts.write_task_snapshot(job_dir, {"status": "done"})

        self.assertEqual(attempts, 3)
        self.assertEqual(
            json.loads((job_dir / "task.json").read_text(encoding="utf-8")),
            {"status": "done"},
        )
        self.assertEqual(list(job_dir.glob(".task.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
