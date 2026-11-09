from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_bench.config import ConfigError, Price, load_config


def _model(**overrides: object) -> dict[str, object]:
    model: dict[str, object] = {
        "name": "demo",
        "base_url": "https://api.example.com/v1",
        "model": "demo-1",
        "api_key_env": "DEMO_API_KEY",
        "price": {"input_per_1m": 0.5, "output_per_1m": 1.5},
    }
    model.update(overrides)
    return model


class LoadConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "bench.json"

    def write(self, payload: object) -> Path:
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        return self.path

    def test_loads_models_and_defaults(self) -> None:
        config = load_config(self.write({"models": [_model()]}))

        self.assertEqual(config.timeout, 60.0)
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.max_tokens, 512)
        model = config.models[0]
        self.assertEqual(model.name, "demo")
        self.assertEqual(model.api_key_env, "DEMO_API_KEY")
        self.assertEqual(model.price, Price(0.5, 1.5))

    def test_max_tokens_null_means_unlimited(self) -> None:
        config = load_config(self.write({"models": [_model()], "max_tokens": None}))
        self.assertIsNone(config.max_tokens)

    def test_price_optional(self) -> None:
        data = _model()
        data.pop("price")
        config = load_config(self.write({"models": [data]}))
        self.assertIsNone(config.models[0].price)

    def test_rejects_unknown_fields(self) -> None:
        with self.assertRaisesRegex(ConfigError, "未知字段: extra"):
            load_config(self.write({"models": [_model()], "extra": 1}))

    def test_rejects_unknown_model_fields(self) -> None:
        with self.assertRaisesRegex(ConfigError, "未知字段: temperature"):
            load_config(self.write({"models": [_model(temperature=0.2)]}))

    def test_rejects_duplicate_model_names(self) -> None:
        with self.assertRaisesRegex(ConfigError, "模型名重复: demo"):
            load_config(self.write({"models": [_model(), _model()]}))

    def test_rejects_bad_base_url(self) -> None:
        with self.assertRaisesRegex(ConfigError, "base_url"):
            load_config(self.write({"models": [_model(base_url="api.example.com")]}))

    def test_rejects_bad_env_name(self) -> None:
        with self.assertRaisesRegex(ConfigError, "api_key_env"):
            load_config(self.write({"models": [_model(api_key_env="not-a-name")]}))

    def test_rejects_incomplete_price(self) -> None:
        with self.assertRaisesRegex(ConfigError, "output_per_1m"):
            load_config(
                self.write({"models": [_model(price={"input_per_1m": 1.0})]})
            )

    def test_rejects_empty_models(self) -> None:
        with self.assertRaisesRegex(ConfigError, "非空数组"):
            load_config(self.write({"models": []}))

    def test_rejects_zero_timeout(self) -> None:
        with self.assertRaisesRegex(ConfigError, "timeout"):
            load_config(self.write({"models": [_model()], "timeout": 0}))

    def test_rejects_negative_temperature(self) -> None:
        with self.assertRaisesRegex(ConfigError, "temperature"):
            load_config(self.write({"models": [_model()], "temperature": -1}))

    def test_reports_json_position(self) -> None:
        self.path.write_text('{"models": [', encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "第 1 行"):
            load_config(self.path)

    def test_missing_file(self) -> None:
        with self.assertRaisesRegex(ConfigError, "读不到配置文件"):
            load_config(Path(self._tmp.name) / "nope.json")


if __name__ == "__main__":
    unittest.main()
