"""DeepSeek 的统一 LLM 调用封装。"""

import json
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Mapping, Sequence

import httpx2
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from services.config import Settings, get_settings


class LLMConfigurationError(RuntimeError):
    """调用 LLM 前发现必要配置缺失。"""


class LLMRequestError(RuntimeError):
    """LLM 请求在重试耗尽后仍失败。"""


class LLMResponseError(RuntimeError):
    """LLM 返回空内容或无法解析为预期结构。"""


class LLMContractError(LLMResponseError):
    """LLM 返回的 JSON 可解析，但未满足调用方声明的输出契约。"""


PayloadValidator = Callable[[dict[str, Any]], str | None]


@dataclass(frozen=True)
class ChatResult:
    """屏蔽供应商响应细节后的统一结果。"""

    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True)
class LLMCallRecord:
    """一条逻辑 LLM 调用的可汇总遥测记录，不保存提示词、响应正文或密钥。"""

    model: str
    json_mode: bool
    thinking_mode: str | None
    operation: str | None
    outcome: str
    elapsed_seconds: float
    api_attempts: int
    prompt_tokens: int | None
    completion_tokens: int | None
    missing_usage_attempts: int
    finish_reason: str | None
    last_content_length: int | None
    error_type: str | None
    retry_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        total_tokens = (
            self.prompt_tokens + self.completion_tokens
            if self.prompt_tokens is not None and self.completion_tokens is not None
            else None
        )
        return {
            "model": self.model,
            "json_mode": self.json_mode,
            "thinking_mode": self.thinking_mode,
            "operation": self.operation,
            "outcome": self.outcome,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "api_attempts": self.api_attempts,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": total_tokens,
            "missing_usage_attempts": self.missing_usage_attempts,
            "finish_reason": self.finish_reason,
            "last_content_length": self.last_content_length,
            "error_type": self.error_type,
            "retry_reasons": list(self.retry_reasons),
        }


class DeepSeekLLM:
    """基于 OpenAI 兼容 SDK 调用 DeepSeek Chat Completions。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: OpenAI | None = None
        self._client_lock = Lock()
        self._telemetry_lock = Lock()
        self._call_records: list[LLMCallRecord] = []

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        operation: str | None = None,
    ) -> ChatResult:
        """发送非流式对话请求，并对瞬时网络或服务端故障有限重试。"""
        return self._request_chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            operation=operation,
            validate_json_object=False,
            payload_validator=None,
        )

    def _request_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
        operation: str | None,
        validate_json_object: bool,
        payload_validator: PayloadValidator | None,
    ) -> ChatResult:
        """执行一次逻辑调用；所有可恢复失败共享同一个总重试上限。"""
        if not messages:
            raise ValueError("messages 不能为空")

        request: dict[str, Any] = {
            "model": model or self._settings.deepseek_model,
            "messages": [dict(message) for message in messages],
        }
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        thinking_mode = (
            "disabled"
            if json_mode and self._settings.llm_json_disable_thinking
            else None
        )
        if thinking_mode == "disabled":
            # DeepSeek 的 Chat Completions 需通过 extra_body 传递 thinking 开关。
            # 仅影响结构化输出，普通 chat() 保持模型默认行为。
            request["extra_body"] = {"thinking": {"type": "disabled"}}

        started_at = time.perf_counter()
        api_attempts = 0
        prompt_tokens_total = 0
        completion_tokens_total = 0
        missing_usage_attempts = 0
        finish_reason: str | None = None
        last_content_length: int | None = None
        retry_reasons: list[str] = []
        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                api_attempts += 1
                response = self._get_client().chat.completions.create(**request)
                choice = response.choices[0] if response.choices else None
                message = choice.message if choice else None
                content = message.content if message else None
                finish_reason = getattr(choice, "finish_reason", None)
                last_content_length = len(content) if isinstance(content, str) else None
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                if prompt_tokens is None or completion_tokens is None:
                    missing_usage_attempts += 1
                else:
                    prompt_tokens_total += prompt_tokens
                    completion_tokens_total += completion_tokens
                if not isinstance(content, str) or not content.strip():
                    raise LLMResponseError("LLM 返回了空内容")
                if validate_json_object:
                    payload = self._parse_json_object(content)
                    if payload_validator is not None:
                        validation_feedback = payload_validator(payload)
                        if validation_feedback:
                            raise LLMContractError(
                                "JSON 内容不满足输出契约：" + validation_feedback
                            )

                result = ChatResult(
                    content=content,
                    model=getattr(response, "model", None) or request["model"],
                    prompt_tokens=(
                        prompt_tokens_total if missing_usage_attempts == 0 else None
                    ),
                    completion_tokens=(
                        completion_tokens_total if missing_usage_attempts == 0 else None
                    ),
                )
                self._append_call_record(
                    LLMCallRecord(
                        model=result.model,
                        json_mode=json_mode,
                        thinking_mode=thinking_mode,
                        operation=operation,
                        outcome="success",
                        elapsed_seconds=time.perf_counter() - started_at,
                        api_attempts=api_attempts,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        missing_usage_attempts=missing_usage_attempts,
                        finish_reason=finish_reason,
                        last_content_length=last_content_length,
                        error_type=None,
                        retry_reasons=tuple(retry_reasons),
                    )
                )
                return result
            except LLMResponseError as error:
                # DeepSeek 的 JSON 模式可能偶发返回空内容；把它视为瞬时错误，
                # 无效 JSON 同样在这里重试。两者共用总上限，避免成倍增加 API 成本。
                retry_reasons.append(type(error).__name__)
                if attempt == self._settings.llm_max_retries:
                    self._append_failure_record(
                        model=request["model"],
                        json_mode=json_mode,
                        thinking_mode=thinking_mode,
                        operation=operation,
                        started_at=started_at,
                        api_attempts=api_attempts,
                        prompt_tokens=prompt_tokens_total,
                        completion_tokens=completion_tokens_total,
                        missing_usage_attempts=missing_usage_attempts,
                        finish_reason=finish_reason,
                        last_content_length=last_content_length,
                        error=error,
                        retry_reasons=retry_reasons,
                    )
                    raise
                self._append_retry_feedback(request, error)
            # RateLimitError 是 APIStatusError 的子类；必须先捕获，否则 429 会
            # 被下面的通用状态码分支直接抛出，配置的有限重试永远不会生效。
            except RateLimitError as error:
                retry_reasons.append(type(error).__name__)
                if attempt == self._settings.llm_max_retries:
                    final_error = LLMRequestError("LLM 请求在重试后仍被限流")
                    self._append_failure_record(
                        model=request["model"],
                        json_mode=json_mode,
                        thinking_mode=thinking_mode,
                        operation=operation,
                        started_at=started_at,
                        api_attempts=api_attempts,
                        prompt_tokens=prompt_tokens_total,
                        completion_tokens=completion_tokens_total,
                        missing_usage_attempts=missing_usage_attempts,
                        finish_reason=finish_reason,
                        last_content_length=last_content_length,
                        error=final_error,
                        retry_reasons=retry_reasons,
                    )
                    raise final_error from error
            except APIStatusError as error:
                retry_reasons.append(type(error).__name__)
                if error.status_code < 500 or attempt == self._settings.llm_max_retries:
                    final_error = LLMRequestError(
                        f"LLM 请求失败，HTTP 状态码 {error.status_code}"
                    )
                    self._append_failure_record(
                        model=request["model"],
                        json_mode=json_mode,
                        thinking_mode=thinking_mode,
                        operation=operation,
                        started_at=started_at,
                        api_attempts=api_attempts,
                        prompt_tokens=prompt_tokens_total,
                        completion_tokens=completion_tokens_total,
                        missing_usage_attempts=missing_usage_attempts,
                        finish_reason=finish_reason,
                        last_content_length=last_content_length,
                        error=final_error,
                        retry_reasons=retry_reasons,
                    )
                    raise final_error from error
            except (APIConnectionError, APITimeoutError) as error:
                retry_reasons.append(type(error).__name__)
                if attempt == self._settings.llm_max_retries:
                    final_error = LLMRequestError("LLM 请求在重试后仍失败")
                    self._append_failure_record(
                        model=request["model"],
                        json_mode=json_mode,
                        thinking_mode=thinking_mode,
                        operation=operation,
                        started_at=started_at,
                        api_attempts=api_attempts,
                        prompt_tokens=prompt_tokens_total,
                        completion_tokens=completion_tokens_total,
                        missing_usage_attempts=missing_usage_attempts,
                        finish_reason=finish_reason,
                        last_content_length=last_content_length,
                        error=final_error,
                        retry_reasons=retry_reasons,
                    )
                    raise final_error from error

            # 仅对可能恢复的错误退避，避免无限重试和不可控成本。
            time.sleep(2**attempt)

        raise AssertionError("重试循环不应到达此处")

    @staticmethod
    def _append_retry_feedback(request: dict[str, Any], error: LLMResponseError) -> None:
        """给下一次同预算重试提供脱敏的格式或契约反馈，不回传模型原文。"""
        request["messages"].append(
            {
                "role": "user",
                "content": (
                    "上一次输出未通过校验，原因："
                    f"{error}。请修正该问题，并且只输出一个 JSON 对象，不要解释。"
                ),
            }
        )

    def _append_failure_record(
        self,
        *,
        model: str,
        json_mode: bool,
        thinking_mode: str | None,
        operation: str | None,
        started_at: float,
        api_attempts: int,
        prompt_tokens: int,
        completion_tokens: int,
        missing_usage_attempts: int,
        finish_reason: str | None,
        last_content_length: int | None,
        error: Exception,
        retry_reasons: list[str],
    ) -> None:
        """保存失败摘要，供评估脚本定位问题；不持久化正文和敏感请求内容。"""
        self._append_call_record(
            LLMCallRecord(
                model=model,
                json_mode=json_mode,
                thinking_mode=thinking_mode,
                operation=operation,
                outcome="failed",
                elapsed_seconds=time.perf_counter() - started_at,
                api_attempts=api_attempts,
                prompt_tokens=(prompt_tokens if missing_usage_attempts == 0 else None),
                completion_tokens=(
                    completion_tokens if missing_usage_attempts == 0 else None
                ),
                missing_usage_attempts=missing_usage_attempts,
                finish_reason=finish_reason,
                last_content_length=last_content_length,
                error_type=type(error).__name__,
                retry_reasons=tuple(retry_reasons),
            )
        )

    def telemetry_snapshot(self) -> tuple[LLMCallRecord, ...]:
        """返回线程安全快照，调用方可按实验范围自行切片汇总。"""

        with self._telemetry_lock:
            return tuple(self._call_records)

    @staticmethod
    def summarize_call_records(
        records: Sequence[LLMCallRecord],
    ) -> dict[str, object]:
        """把一次工作流范围内的成功、失败逻辑调用汇总成可序列化指标。"""

        prompt_tokens = sum(record.prompt_tokens or 0 for record in records)
        completion_tokens = sum(record.completion_tokens or 0 for record in records)
        successful_records = [record for record in records if record.outcome == "success"]
        failed_records = [record for record in records if record.outcome == "failed"]
        return {
            "successful_calls": len(successful_records),
            "failed_calls": len(failed_records),
            "api_attempts": sum(record.api_attempts for record in records),
            "elapsed_seconds": round(sum(record.elapsed_seconds for record in records), 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "calls_missing_token_usage": sum(
                record.missing_usage_attempts > 0 for record in records
            ),
            "api_attempts_missing_token_usage": sum(
                record.missing_usage_attempts for record in records
            ),
            "records": [record.to_dict() for record in records],
        }

    def _append_call_record(self, record: LLMCallRecord) -> None:
        with self._telemetry_lock:
            self._call_records.append(record)

    def chat_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        operation: str | None = None,
        payload_validator: PayloadValidator | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """请求 JSON Object 并解析为字典；无效 JSON 与空内容共享有限重试上限。"""
        result = self._request_chat(
            messages,
            json_mode=True,
            operation=operation,
            validate_json_object=True,
            payload_validator=payload_validator,
            model=kwargs.get("model"),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
        )
        return self._parse_json_object(result.content)

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise LLMResponseError("LLM 返回内容不是有效 JSON") from error
        if not isinstance(payload, dict):
            raise LLMResponseError("LLM JSON 根节点必须是对象")
        return payload

    def close(self) -> None:
        """任务结束后释放连接；不影响尚未初始化的客户端。"""
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        # 双作词人会从两个 LangGraph 分支同时首次调用客户端；双重检查避免
        # 重复创建连接池，也确保 close() 不与首次初始化交错。
        with self._client_lock:
            if self._client is not None:
                return self._client
            if not self._settings.deepseek_api_key:
                raise LLMConfigurationError(
                    "缺少 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 后填写密钥。"
                )

            # 不继承终端或宿主环境的代理变量，避免错误代理影响服务运行。
            # 需要代理时只读取项目显式配置的 DEEPSEEK_PROXY_URL。
            http_client = httpx2.Client(
                proxy=self._settings.deepseek_proxy_url,
                timeout=self._settings.llm_timeout_seconds,
                trust_env=False,
            )
            self._client = OpenAI(
                api_key=self._settings.deepseek_api_key,
                base_url=self._settings.deepseek_base_url,
                timeout=self._settings.llm_timeout_seconds,
                # SDK 内置重试关闭，统一由本类控制重试次数和退避策略。
                max_retries=0,
                http_client=http_client,
            )
            return self._client
