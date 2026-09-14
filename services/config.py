"""集中读取运行配置，避免业务代码直接访问环境变量。"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用运行配置。

    API key 可为空，以便健康检查和本地单元测试无需真实外部凭证。
    只有真正调用 LLM 时，服务层才会拒绝缺失配置。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepseek_api_key: str | None = Field(
        default=None,
        validation_alias="DEEPSEEK_API_KEY",
        repr=False,
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias="DEEPSEEK_BASE_URL",
    )
    deepseek_model: str = Field(
        default="deepseek-v4-flash",
        validation_alias="DEEPSEEK_MODEL",
    )
    deepseek_proxy_url: str | None = Field(
        default=None,
        validation_alias="DEEPSEEK_PROXY_URL",
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        validation_alias="LLM_TIMEOUT_SECONDS",
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        validation_alias="LLM_MAX_RETRIES",
    )
    llm_json_disable_thinking: bool = Field(
        default=True,
        validation_alias="LLM_JSON_DISABLE_THINKING",
    )
    heartmula_model_path: Path = Field(
        default=Path("data/ckpt/HeartMuLaGen"),
        validation_alias="HEARTMULA_MODEL_PATH",
    )
    music_output_dir: Path = Field(
        default=Path("data/output/generated"),
        validation_alias="MUSIC_OUTPUT_DIR",
    )

    @property
    def is_llm_configured(self) -> bool:
        """仅报告是否配置，绝不返回密钥值。"""
        return bool(self.deepseek_api_key)


@lru_cache
def get_settings() -> Settings:
    """缓存配置，避免每次请求重复读取环境变量和 .env。"""
    return Settings()
