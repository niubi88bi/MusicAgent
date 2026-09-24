"""文本 API 的真实图离线测试；替换 Agent，不访问网络或 GPU。"""

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import create_app
from agents.graph import build_music_graph
from services.config import Settings
from services.task_store import TaskStore
from services.workflow_runner import WorkflowRunner
from test_d7_retry import (BAD_DRAFT, GOOD_DRAFT, WORSE_DRAFT, FakeRequirementAgent,
                           FakeProducerAgent, SequenceLyricist, FakeGenerator)


def fake_execute(drafts, producer=None):
    def execute(theme, observer):
        graph = build_music_graph(FakeRequirementAgent(), SequenceLyricist(drafts),
                                  producer or FakeProducerAgent(), text_only=True, on_event=observer)
        return graph.invoke({"theme": theme, "requested_duration_seconds": 30})
    return execute


class TextTaskApiTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "tmp/test-work"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.settings_patch = patch("app.main.get_settings", return_value=Settings(DEEPSEEK_API_KEY="offline-test"))
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)

    def client(self, execute):
        store = TaskStore()
        runner = WorkflowRunner(store, self.output, execute=execute)
        return TestClient(create_app(store=store, runner=runner))

    def await_done(self, client, task_id):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            task = client.get(f"/api/tasks/{task_id}").json()
            if task["status"] in {"done", "failed"}:
                return task
            time.sleep(.01)
        self.fail("任务未在测试期限内结束")

    def test_real_graph_text_mode_has_no_generator_and_preserves_rewrite(self):
        generator = FakeGenerator()
        graph = build_music_graph(FakeRequirementAgent(), SequenceLyricist([BAD_DRAFT, GOOD_DRAFT]),
                                  FakeProducerAgent(), generator, text_only=True)
        result = graph.invoke({"theme": "夏夜回家"})
        self.assertNotIn("generator", graph.get_graph().nodes)
        self.assertNotIn("audio_path", result)
        self.assertEqual(generator.calls, 0)
        self.assertEqual(result["attempts"], 2)

    def test_submission_saves_versions_and_real_event_order(self):
        with self.client(fake_execute([BAD_DRAFT, GOOD_DRAFT])) as client:
            response = client.post("/api/tasks", json={"theme": " 夏夜回家 "})
            self.assertEqual(response.status_code, 202)
            task = self.await_done(client, response.json()["task_id"])
            self.assertEqual(task["status"], "done")
            self.assertEqual(task["input"]["theme"], "夏夜回家")
            self.assertIsNone(task["result"]["audio_url"])
            self.assertEqual(task["result"]["selected_attempt"], 2)
            self.assertEqual(len(task["result"]["quality_history"]), 2)
            events = task["progress"]["events"]
            self.assertEqual([e["status"] for e in events], ["running", "done"] * 7)
            self.assertTrue(task["started_at"] <= task["finished_at"])
            for filename in ("task.json", "lyrics.txt", "lyric_draft.json", "production_plan.json", "quality_report.json", "lyric_revision_history.json"):
                self.assertTrue((self.output / task["task_id"] / filename).is_file())

    def test_piano_preflight_blocks_then_answers_create_constrained_task(self):
        received_constraints = []

        def execute(theme, observer, production_constraints=None):
            received_constraints.append(production_constraints)
            state = fake_execute([GOOD_DRAFT])(theme, observer)
            state["production_constraints"] = production_constraints
            return state

        with self.client(execute) as client:
            request = {"theme": "不要钢琴，做一首夜归主题中文歌"}
            preflight = client.post("/api/tasks/preflight", json=request)
            self.assertEqual(preflight.status_code, 200)
            self.assertEqual(preflight.json()["status"], "clarification_required")
            self.assertEqual(client.get("/api/tasks").json()["items"], [])

            blocked = client.post("/api/tasks", json=request)
            self.assertEqual(blocked.status_code, 409)
            self.assertEqual(client.get("/api/tasks").json()["items"], [])

            answer = {
                **request,
                "clarification_answers": [{
                    "code": "forbidden_instrument_without_replacement",
                    "option_code": "acoustic_guitar",
                }],
            }
            ready = client.post("/api/tasks/preflight", json=answer)
            self.assertEqual(ready.status_code, 200)
            self.assertEqual(ready.json()["status"], "ready")
            self.assertEqual(
                ready.json()["production_constraints"]["replacement_instrument"],
                "acoustic_guitar",
            )
            response = client.post("/api/tasks", json=answer)
            self.assertEqual(response.status_code, 202)
            task = self.await_done(client, response.json()["task_id"])
            self.assertEqual(task["input"]["production_constraints"]["replacement_mode"], "acoustic_guitar")
            self.assertEqual(task["result"]["production_constraints"]["replacement_instrument"], "acoustic_guitar")
            self.assertEqual(received_constraints[0]["replacement_instrument"], "acoustic_guitar")

    def test_cap_selects_earlier_best_but_done_does_not_mean_passed(self):
        with self.client(fake_execute([BAD_DRAFT, WORSE_DRAFT, WORSE_DRAFT])) as client:
            task_id = client.post("/api/tasks", json={"theme":"夏夜回家"}).json()["task_id"]
            task = self.await_done(client, task_id)
            self.assertEqual(task["status"], "done")
            self.assertEqual(task["result"]["quality_status"], "capped")
            self.assertEqual(task["result"]["selected_attempt"], 1)
            self.assertFalse(task["result"]["quality_report"]["passed"])

    def test_failure_keeps_text_and_redacts_exception(self):
        class BrokenProducer:
            def run(self, *_):
                raise RuntimeError("secret-key D:/private/config")
        with self.client(fake_execute([GOOD_DRAFT], BrokenProducer())) as client:
            task_id = client.post("/api/tasks", json={"theme":"夏夜回家"}).json()["task_id"]
            task = self.await_done(client, task_id)
            self.assertEqual(task["status"], "failed")
            self.assertEqual(task["progress"]["text"]["lyrics"], GOOD_DRAFT.lyrics)
            self.assertEqual(task["progress"]["events"][-1]["status"], "failed")
            self.assertNotIn("secret-key", str(task))
            self.assertNotIn("private/config", (self.output / task_id / "task.json").read_text(encoding="utf-8"))
            self.assertNotIn("secret-key", (self.output / task_id / "failure.json").read_text(encoding="utf-8"))

    def test_queue_is_bounded_and_releases_capacity(self):
        entered, release = Event(), Event()
        def execute(theme, observer):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("测试未及时释放")
            return fake_execute([GOOD_DRAFT])(theme, observer)
        with self.client(execute) as client:
            try:
                first = client.post("/api/tasks", json={"theme":"夏夜回家"})
                self.assertEqual(first.status_code, 202)
                self.assertTrue(entered.wait(2))
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.post("/api/tasks", json={"theme":"第二个任务"}).status_code, 202)
                self.assertEqual(client.post("/api/tasks", json={"theme":"第三个任务"}).status_code, 202)
                self.assertEqual(client.post("/api/tasks", json={"theme":"第四个任务"}).status_code, 429)
                self.assertEqual(len(client.get("/api/tasks").json()["items"]), 3)
            finally:
                release.set()
            for task in client.get("/api/tasks").json()["items"]:
                self.await_done(client, task["task_id"])
            self.assertEqual(client.post("/api/tasks", json={"theme":"容量已释放"}).status_code, 202)

    def test_invalid_unknown_and_static_directory_boundaries(self):
        with self.client(fake_execute([GOOD_DRAFT])) as client:
            for body in ({"theme":"  "}, {"theme":True}, {"theme":"歌"}, {"theme":"长"*201}, {"theme":"正常主题","path":".env"}):
                self.assertEqual(client.post("/api/tasks", json=body).status_code, 422)
            self.assertEqual(client.get("/api/tasks/unknown").status_code, 404)
            self.assertEqual(client.get("/").status_code, 200)
            for path in ("/.env", "/data/ckpt/test", "/%2e%2e/.env"):
                self.assertEqual(client.get(path).status_code, 404)
            self.assertEqual(client.get("/api/tasks").json()["items"], [])

    def test_missing_key_does_not_create_task(self):
        with self.client(fake_execute([GOOD_DRAFT])) as client:
            with patch("app.main.get_settings", return_value=Settings(DEEPSEEK_API_KEY="")):
                self.assertEqual(client.post("/api/tasks", json={"theme":"夏夜回家"}).status_code, 503)
            self.assertEqual(client.get("/api/tasks").json()["items"], [])

    def test_restart_lists_terminal_snapshot_but_does_not_resume_execution(self):
        with self.client(fake_execute([GOOD_DRAFT])) as client:
            task_id = client.post("/api/tasks", json={"theme":"夏夜回家"}).json()["task_id"]
            self.await_done(client, task_id)
        with self.client(fake_execute([GOOD_DRAFT])) as client:
            response = client.get(f"/api/tasks/{task_id}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "done")
            self.assertEqual(client.get("/api/tasks").json()["items"][0]["task_id"], task_id)


if __name__ == "__main__":
    unittest.main()
