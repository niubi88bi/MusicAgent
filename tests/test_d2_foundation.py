"""D2 服务骨架的无网络单元测试。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import RateLimitError
from fastapi.testclient import TestClient
from services.config import Settings
from services.llm import DeepSeekLLM, LLMResponseError

from app.main import create_app
from services.task_store import (
    InvalidTaskTransitionError,
    TaskStatus,
    TaskStore,
)


class TaskStoreTests(unittest.TestCase):
    def test_task_lifecycle_and_snapshot_isolation(self) -> None:
        store = TaskStore()
        created = store.create({"theme": "夏天"})
        task_id = created["task_id"]

        created["input"]["theme"] = "被外部修改"
        self.assertEqual(store.get(task_id)["input"]["theme"], "夏天")

        running = store.update_status(task_id, TaskStatus.RUNNING)
        self.assertEqual(running["status"], "running")
        done = store.update_status(
            task_id, TaskStatus.DONE, result={"lyrics": "测试歌词"}
        )
        self.assertEqual(done["status"], "done")
        self.assertEqual(done["result"]["lyrics"], "测试歌词")

    def test_invalid_transition_is_rejected(self) -> None:
        store = TaskStore()
        task_id = store.create()["task_id"]
        with self.assertRaises(InvalidTaskTransitionError):
            store.update_status(task_id, TaskStatus.DONE)


class ApiTests(unittest.TestCase):
    def test_health_and_empty_tasks(self) -> None:
        client = TestClient(create_app())
        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertIsInstance(health.json()["llm_configured"], bool)

        # 单元测试隔离项目已有演示产物；生产 app 默认会列出终态快照。
        client = TestClient(create_app(include_persisted=False))
        tasks = client.get("/api/tasks")
        self.assertEqual(tasks.status_code, 200)
        self.assertEqual(tasks.json()["items"], [])


class SettingsTests(unittest.TestCase):
    def test_explicit_proxy_configuration(self) -> None:
        settings = Settings(DEEPSEEK_PROXY_URL="http://127.0.0.1:7897")
        self.assertEqual(settings.deepseek_proxy_url, "http://127.0.0.1:7897")

    def test_json_thinking_is_disabled_by_default_and_configurable(self) -> None:
        self.assertTrue(Settings().llm_json_disable_thinking)
        self.assertFalse(
            Settings(LLM_JSON_DISABLE_THINKING=False).llm_json_disable_thinking
        )


class LlmRetryTests(unittest.TestCase):
    def test_parallel_first_use_creates_only_one_client(self) -> None:
        constructor_entered = Event()
        release_constructor = Event()
        fake_client = SimpleNamespace(close=unittest.mock.Mock())

        def construct_client(**_: object) -> SimpleNamespace:
            constructor_entered.set()
            if not release_constructor.wait(timeout=2):
                raise RuntimeError("测试未及时释放客户端构造")
            return fake_client

        llm = DeepSeekLLM(Settings(DEEPSEEK_API_KEY="test-key"))
        with (
            patch("services.llm.httpx2.Client", return_value=object()),
            patch("services.llm.OpenAI", side_effect=construct_client) as openai,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            first = executor.submit(llm._get_client)
            self.assertTrue(constructor_entered.wait(timeout=1))
            second = executor.submit(llm._get_client)
            release_constructor.set()
            self.assertIs(first.result(timeout=2), fake_client)
            self.assertIs(second.result(timeout=2), fake_client)

        self.assertEqual(openai.call_count, 1)

    def test_empty_response_is_retried(self) -> None:
        def response(content: str | None) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                model="test-model",
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        create = unittest.mock.Mock(side_effect=[response(None), response("{}")])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        settings = Settings(DEEPSEEK_API_KEY="test-key", LLM_MAX_RETRIES=1)
        llm = DeepSeekLLM(settings)
        llm._client = client  # 注入 fake client，确保测试不访问网络。

        with patch("services.llm.time.sleep"):
            result = llm.chat([{"role": "user", "content": "test"}])

        self.assertEqual(result.content, "{}")
        self.assertEqual(create.call_count, 2)
        summary = llm.summarize_call_records(llm.telemetry_snapshot())
        self.assertEqual(summary["api_attempts"], 2)

    def test_rate_limit_error_is_retried_before_generic_status_error(self) -> None:
        def response(content: str) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                model="test-model",
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        rate_limit = RateLimitError(
            "rate limited",
            response=httpx.Response(
                429,
                request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"),
            ),
            body=None,
        )
        create = unittest.mock.Mock(side_effect=[rate_limit, response("{}")])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        settings = Settings(DEEPSEEK_API_KEY="test-key", LLM_MAX_RETRIES=1)
        llm = DeepSeekLLM(settings)
        llm._client = client

        with patch("services.llm.time.sleep"):
            result = llm.chat([{"role": "user", "content": "test"}])

        self.assertEqual(result.content, "{}")
        self.assertEqual(create.call_count, 2)
        summary = llm.summarize_call_records(llm.telemetry_snapshot())
        self.assertEqual(summary["api_attempts"], 2)

    def test_telemetry_records_tokens_elapsed_time_and_retry_count(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
            model="test-model",
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5),
        )
        create = unittest.mock.Mock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        llm = DeepSeekLLM(Settings(DEEPSEEK_API_KEY="test-key"))
        llm._client = client

        llm.chat([{"role": "user", "content": "test"}], json_mode=True)
        summary = llm.summarize_call_records(llm.telemetry_snapshot())

        self.assertEqual(summary["successful_calls"], 1)
        self.assertEqual(summary["api_attempts"], 1)
        self.assertEqual(summary["total_tokens"], 8)
        self.assertTrue(summary["records"][0]["json_mode"])

    def test_invalid_json_is_retried_within_the_same_total_limit(self) -> None:
        def response(content: str) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content), finish_reason="stop"
                    )
                ],
                model="test-model",
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
            )

        create = unittest.mock.Mock(side_effect=[response('{"broken":'), response("{}")])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        llm = DeepSeekLLM(Settings(DEEPSEEK_API_KEY="test-key", LLM_MAX_RETRIES=1))
        llm._client = client

        with patch("services.llm.time.sleep"):
            payload = llm.chat_json(
                [{"role": "user", "content": "test"}], operation="requirement"
            )

        self.assertEqual(payload, {})
        self.assertEqual(create.call_count, 2)
        summary = llm.summarize_call_records(llm.telemetry_snapshot())
        self.assertEqual(summary["successful_calls"], 1)
        self.assertEqual(summary["failed_calls"], 0)
        self.assertEqual(summary["api_attempts"], 2)
        self.assertEqual(summary["total_tokens"], 10)
        self.assertEqual(summary["records"][0]["operation"], "requirement")
        self.assertEqual(summary["records"][0]["thinking_mode"], "disabled")
        self.assertEqual(summary["records"][0]["retry_reasons"], ["LLMResponseError"])
        self.assertEqual(
            create.call_args.kwargs["extra_body"], {"thinking": {"type": "disabled"}}
        )

    def test_contract_error_is_retried_with_specific_feedback(self) -> None:
        def response(content: str) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content), finish_reason="stop"
                    )
                ],
                model="test-model",
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
            )

        create = unittest.mock.Mock(
            side_effect=[response('{"hook": "遗漏的 Hook"}'), response('{"hook": "合格 Hook"}')]
        )
        llm = DeepSeekLLM(Settings(DEEPSEEK_API_KEY="test-key", LLM_MAX_RETRIES=1))
        llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        def validate_hook(payload: dict[str, object]) -> str | None:
            if payload["hook"] != "合格 Hook":
                return "整份输出：hook 必须原样出现在歌词正文中"
            return None

        with patch("services.llm.time.sleep"):
            payload = llm.chat_json(
                [{"role": "user", "content": "test"}],
                operation="lyricist_initial",
                payload_validator=validate_hook,
            )

        self.assertEqual(payload, {"hook": "合格 Hook"})
        self.assertEqual(create.call_count, 2)
        retry_message = create.call_args_list[1].kwargs["messages"][-1]["content"]
        self.assertIn("hook 必须原样出现在歌词正文中", retry_message)
        summary = llm.summarize_call_records(llm.telemetry_snapshot())
        self.assertEqual(summary["records"][0]["retry_reasons"], ["LLMContractError"])

    def test_plain_chat_does_not_override_thinking_mode(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            model="test-model",
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )
        create = unittest.mock.Mock(return_value=response)
        llm = DeepSeekLLM(Settings(DEEPSEEK_API_KEY="test-key"))
        llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        llm.chat([{"role": "user", "content": "test"}])

        self.assertNotIn("extra_body", create.call_args.kwargs)

    def test_exhausted_empty_response_is_saved_as_failure_telemetry(self) -> None:
        def response() -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None), finish_reason="stop"
                    )
                ],
                model="test-model",
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=0),
            )

        create = unittest.mock.Mock(side_effect=[response(), response()])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        llm = DeepSeekLLM(Settings(DEEPSEEK_API_KEY="test-key", LLM_MAX_RETRIES=1))
        llm._client = client

        with patch("services.llm.time.sleep"):
            with self.assertRaises(LLMResponseError):
                llm.chat_json(
                    [{"role": "user", "content": "test"}],
                    operation="lyricist_rewrite",
                )

        summary = llm.summarize_call_records(llm.telemetry_snapshot())
        self.assertEqual(summary["successful_calls"], 0)
        self.assertEqual(summary["failed_calls"], 1)
        self.assertEqual(summary["api_attempts"], 2)
        record = summary["records"][0]
        self.assertEqual(record["outcome"], "failed")
        self.assertEqual(record["operation"], "lyricist_rewrite")
        self.assertEqual(record["last_content_length"], None)
        self.assertEqual(record["error_type"], "LLMResponseError")


if __name__ == "__main__":
    unittest.main()
