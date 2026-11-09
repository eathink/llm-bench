"""执行基准测试并聚合结果。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .client import ApiError, chat
from .config import ModelConfig
from .tasks import Task


@dataclass(frozen=True)
class RunResult:
    model: str
    task_id: str
    task_title: str
    repeat: int
    passed: bool
    detail: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    error: str | None = None


@dataclass(frozen=True)
class ModelSummary:
    model: str
    runs: int
    passed: int
    errors: int
    pass_rate: float
    avg_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _save(directory: Path, model: str, task_id: str, repeat: int, text: str) -> None:
    name = f"{_safe_name(model)}__{_safe_name(task_id)}__{repeat}.txt"
    (directory / name).write_text(text, encoding="utf-8")


def run_bench(
    *,
    models: Sequence[ModelConfig],
    tasks: Sequence[Task],
    repeats: int,
    environment: Mapping[str, str],
    timeout: float,
    temperature: float,
    max_tokens: int | None,
    save_dir: Path | None = None,
) -> list[RunResult]:
    """逐模型、逐任务、逐轮执行。单个请求失败只记录错误，不中断整轮。"""
    results: list[RunResult] = []
    for model in models:
        api_key = environment[model.api_key_env]
        for task in tasks:
            for repeat in range(1, repeats + 1):
                try:
                    completion = chat(
                        base_url=model.base_url,
                        api_key=api_key,
                        model=model.model,
                        prompt=task.prompt,
                        timeout=timeout,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except ApiError as exc:
                    message = str(exc)
                    results.append(
                        RunResult(
                            model=model.name,
                            task_id=task.id,
                            task_title=task.title,
                            repeat=repeat,
                            passed=False,
                            detail="请求失败",
                            latency_ms=exc.latency_ms,
                            prompt_tokens=None,
                            completion_tokens=None,
                            cost_usd=None,
                            error=message,
                        )
                    )
                    if save_dir is not None:
                        _save(save_dir, model.name, task.id, repeat, f"ERROR: {message}\n")
                    continue

                check = task.check(completion.content)
                cost = (
                    model.price.cost(completion.prompt_tokens, completion.completion_tokens)
                    if model.price is not None
                    else None
                )
                results.append(
                    RunResult(
                        model=model.name,
                        task_id=task.id,
                        task_title=task.title,
                        repeat=repeat,
                        passed=check.passed,
                        detail=check.detail,
                        latency_ms=completion.latency_ms,
                        prompt_tokens=completion.prompt_tokens,
                        completion_tokens=completion.completion_tokens,
                        cost_usd=cost,
                    )
                )
                if save_dir is not None:
                    _save(save_dir, model.name, task.id, repeat, completion.content)
    return results


def summarize(results: Sequence[RunResult], models: Sequence[ModelConfig]) -> list[ModelSummary]:
    """按模型顺序聚合；成本只要有一轮未知就标为 None。"""
    summaries: list[ModelSummary] = []
    for model in models:
        rows = [row for row in results if row.model == model.name]
        runs = len(rows)
        passed = sum(1 for row in rows if row.passed)
        errors = sum(1 for row in rows if row.error is not None)
        latencies = [row.latency_ms for row in rows if row.error is None]
        costs = [row.cost_usd for row in rows]
        cost = (
            sum(cost for cost in costs if cost is not None)
            if rows and all(cost is not None for cost in costs)
            else None
        )
        summaries.append(
            ModelSummary(
                model=model.name,
                runs=runs,
                passed=passed,
                errors=errors,
                pass_rate=passed / runs if runs else 0.0,
                avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
                prompt_tokens=sum(row.prompt_tokens or 0 for row in rows),
                completion_tokens=sum(row.completion_tokens or 0 for row in rows),
                cost_usd=cost,
            )
        )
    return summaries
