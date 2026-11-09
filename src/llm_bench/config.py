"""配置文件解析与校验。

配置只放模型、价格和请求参数；API key 永远从环境变量读取，
不写进配置文件，也不出现在命令行参数里。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """配置文件缺失、JSON 非法或字段不合法。"""


@dataclass(frozen=True)
class Price:
    """每百万 token 的价格（美元）。"""

    input_per_1m: float
    output_per_1m: float

    def cost(self, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
        """token 数缺失时返回 None，而不是把成本当成 0。"""
        if prompt_tokens is None or completion_tokens is None:
            return None
        return (
            prompt_tokens / 1_000_000 * self.input_per_1m
            + completion_tokens / 1_000_000 * self.output_per_1m
        )


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_url: str
    model: str
    api_key_env: str
    price: Price | None = None


@dataclass(frozen=True)
class BenchConfig:
    models: tuple[ModelConfig, ...]
    timeout: float = 60.0
    temperature: float = 0.0
    max_tokens: int | None = 512


_TOP_KEYS = {"models", "timeout", "temperature", "max_tokens"}
_MODEL_KEYS = {"name", "base_url", "model", "api_key_env", "price"}
_PRICE_KEYS = {"input_per_1m", "output_per_1m"}
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _reject_unknown(raw: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"{where} 有未知字段: {', '.join(unknown)}")


def _string(raw: dict[str, Any], key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} 的 {key} 必须是非空字符串")
    return value.strip()


def _number(raw: dict[str, Any], key: str, where: str, *, minimum: float | None = None) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{where} 的 {key} 必须是数字")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ConfigError(f"{where} 的 {key} 不能小于 {minimum:g}")
    return number


def _optional_number(
    raw: dict[str, Any], key: str, where: str, default: float, *, minimum: float | None = None
) -> float:
    if key not in raw:
        return default
    return _number(raw, key, where, minimum=minimum)


def _price(raw: Any, where: str) -> Price:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} 必须是 JSON 对象")
    _reject_unknown(raw, _PRICE_KEYS, where)
    return Price(
        input_per_1m=_number(raw, "input_per_1m", where, minimum=0),
        output_per_1m=_number(raw, "output_per_1m", where, minimum=0),
    )


def _max_tokens(raw: dict[str, Any]) -> int | None:
    if "max_tokens" not in raw:
        return 512
    value = raw["max_tokens"]
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError("max_tokens 必须是正整数或 null")
    return value


def load_config(path: Path) -> BenchConfig:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"读不到配置文件 {path}: {exc.strerror or exc}") from None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{path} 不是合法 JSON：第 {exc.lineno} 行第 {exc.colno} 列：{exc.msg}"
        ) from None
    if not isinstance(raw, dict):
        raise ConfigError("配置根节点必须是 JSON 对象")
    _reject_unknown(raw, _TOP_KEYS, "配置")

    timeout = _optional_number(raw, "timeout", "配置", 60.0, minimum=0)
    if timeout == 0:
        raise ConfigError("配置的 timeout 必须大于 0")
    temperature = _optional_number(raw, "temperature", "配置", 0.0, minimum=0)
    max_tokens = _max_tokens(raw)

    models_raw = raw.get("models")
    if not isinstance(models_raw, list) or not models_raw:
        raise ConfigError("models 必须是非空数组")

    models: list[ModelConfig] = []
    seen: set[str] = set()
    for index, item in enumerate(models_raw, 1):
        where = f"models[{index}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{where} 必须是 JSON 对象")
        _reject_unknown(item, _MODEL_KEYS, where)

        name = _string(item, "name", where)
        if name in seen:
            raise ConfigError(f"模型名重复: {name}")
        seen.add(name)

        base_url = _string(item, "base_url", where)
        if not base_url.startswith(("http://", "https://")):
            raise ConfigError(f"{where} 的 base_url 必须以 http:// 或 https:// 开头")

        api_key_env = _string(item, "api_key_env", where)
        if not _ENV_NAME.fullmatch(api_key_env):
            raise ConfigError(f"{where} 的 api_key_env 必须是合法的环境变量名")

        models.append(
            ModelConfig(
                name=name,
                base_url=base_url,
                model=_string(item, "model", where),
                api_key_env=api_key_env,
                price=_price(item["price"], f"{where}.price") if "price" in item else None,
            )
        )

    return BenchConfig(
        models=tuple(models), timeout=timeout, temperature=temperature, max_tokens=max_tokens
    )
