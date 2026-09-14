"""按需试听接口离线测试；用假 MP3 替代 HeartMuLa，不访问 GPU。"""

from pathlib import Path
import shutil
from threading import Event
import time
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import create_app
from services.audio_runner import AudioRunner
from services.task_store import TaskStatus, TaskStore
from services.workflow_runner import WorkflowRunner


RESULT = {
    "lyrics": "[Verse]\n测试歌词足够用于离线接口验证",
    "production_plan": {"tags": ["city pop", "warm", "male vocal"]},
    "selected_attempt": 1,
    "quality_status": "passed",
}


class AudioApiTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1] / "tmp/test-work"
        base.mkdir(parents=True, exist_ok=True)
        # Windows TemporaryDirectory 会收紧 ACL，隔离账户下子线程无法创建任务目录。
        self.output = base / f"audio-api-{uuid4().hex}"
        self.output.mkdir()
        self.addCleanup(shutil.rmtree, self.output, True)

    def completed_task(self, store: TaskStore, result=None) -> str:
        task_id = store.create({"theme": "测试主题", "mode": "text"})["task_id"]
        store.update_status(task_id, TaskStatus.RUNNING)
        store.update_status(task_id, TaskStatus.DONE, result=result or RESULT)
        return task_id

    def client(self, store, audio_execute):
        text_runner = WorkflowRunner(store, self.output, execute=lambda *_: {})
        audio_runner = AudioRunner(self.output, execute=audio_execute)
        return TestClient(create_app(store=store, runner=text_runner, audio_runner=audio_runner, include_persisted=False))

    def wait_audio(self, client, source_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            response = client.get(f"/api/tasks/{source_id}/audio/status")
            if response.status_code == 200 and response.json()["status"] in {"done", "failed"}:
                return response.json()
            time.sleep(.01)
        self.fail("试听任务未结束")

    @staticmethod
    def fake_audio(source_dir, audio_task_id, duration):
        path = source_dir / "audio" / audio_task_id / "song.mp3"
        path.parent.mkdir(parents=True, exist_ok=False)
        path.write_bytes(b"ID3-fake-audio")
        return path

    def test_completed_text_creates_playable_idempotent_audio(self):
        store = TaskStore()
        source_id = self.completed_task(store)
        with self.client(store, self.fake_audio) as client:
            self.assertEqual(client.get(f"/api/tasks/{source_id}/audio/status").status_code, 404)
            response = client.post(f"/api/tasks/{source_id}/audio")
            self.assertEqual(response.status_code, 202)
            audio = self.wait_audio(client, source_id)
            self.assertEqual(audio["backend"], "heartmula-local")
            self.assertEqual(audio["duration_seconds"], 30)
            self.assertEqual(audio["selected_attempt"], 1)
            self.assertEqual(client.get(f"/api/tasks/{source_id}/audio").content, b"ID3-fake-audio")
            duplicate = client.post(f"/api/tasks/{source_id}/audio").json()
            self.assertEqual(duplicate["audio_task_id"], audio["audio_task_id"])
            self.assertEqual(client.get(f"/api/tasks/{source_id}").json()["audio"]["status"], "done")

    def test_incomplete_or_malformed_text_is_rejected(self):
        store = TaskStore()
        pending_id = store.create({"theme": "等待", "mode": "text"})["task_id"]
        malformed_id = self.completed_task(store, {"lyrics": "歌词"})
        with self.client(store, self.fake_audio) as client:
            self.assertEqual(client.post(f"/api/tasks/{pending_id}/audio").status_code, 409)
            self.assertEqual(client.post(f"/api/tasks/{malformed_id}/audio").status_code, 409)
            self.assertEqual(client.post("/api/tasks/not-found/audio").status_code, 404)

    def test_failure_keeps_text_task_done_and_can_retry(self):
        calls = 0
        def flaky(source_dir, audio_task_id, duration):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("private D:/secret")
            return self.fake_audio(source_dir, audio_task_id, duration)
        store = TaskStore()
        source_id = self.completed_task(store)
        with self.client(store, flaky) as client:
            client.post(f"/api/tasks/{source_id}/audio")
            failed = self.wait_audio(client, source_id)
            self.assertEqual(failed["status"], "failed")
            self.assertNotIn("secret", str(failed))
            self.assertEqual(client.get(f"/api/tasks/{source_id}").json()["status"], "done")
            client.post(f"/api/tasks/{source_id}/audio")
            self.assertEqual(self.wait_audio(client, source_id)["status"], "done")

    def test_only_one_audio_task_runs_at_once(self):
        entered, release = Event(), Event()
        def blocked(source_dir, audio_task_id, duration):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("测试未释放")
            return self.fake_audio(source_dir, audio_task_id, duration)
        store = TaskStore()
        first = self.completed_task(store)
        second = self.completed_task(store)
        with self.client(store, blocked) as client:
            try:
                self.assertEqual(client.post(f"/api/tasks/{first}/audio").status_code, 202)
                self.assertTrue(entered.wait(1))
                self.assertEqual(client.post(f"/api/tasks/{second}/audio").status_code, 429)
            finally:
                release.set()
            self.assertEqual(self.wait_audio(client, first)["status"], "done")
            self.assertEqual(client.post(f"/api/tasks/{second}/audio").status_code, 202)

    def test_completed_audio_is_read_after_runner_restart(self):
        store = TaskStore()
        source_id = self.completed_task(store)
        with self.client(store, self.fake_audio) as client:
            client.post(f"/api/tasks/{source_id}/audio")
            completed = self.wait_audio(client, source_id)
        with self.client(store, self.fake_audio) as client:
            restored = client.get(f"/api/tasks/{source_id}/audio/status").json()
            self.assertEqual(restored["status"], "done")
            self.assertEqual(restored["audio_task_id"], completed["audio_task_id"])
            self.assertEqual(client.get(f"/api/tasks/{source_id}/audio").status_code, 200)


if __name__ == "__main__":
    unittest.main()
