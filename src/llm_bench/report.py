"""报告渲染：table / markdown / csv / json。"""

from __future__ import annotations

import csv
import io
import json
import unicodedata
from dataclasses import asdict
from typing import Sequence

from .bench import ModelSummary, RunResult


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.6f}"


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def render_table(summaries: Sequence[ModelSummary]) -> str:
    headers = ["模型", "通过", "通过率", "平均延迟", "输入 tok", "输出 tok", "成本"]
    rows = [
        [
            summary.model,
            f"{summary.passed}/{summary.runs}",
            f"{summary.pass_rate:.0%}",
            f"{summary.avg_latency_ms:.0f} ms",
            str(summary.prompt_tokens),
            str(summary.completion_tokens),
            _money(summary.cost_usd),
        ]
        for summary in summaries
    ]
    widths = [max(_width(row[index]) for row in [headers, *rows]) for index in range(len(headers))]
    lines = [
        "  ".join(_pad(cell, widths[index]) for index, cell in enumerate(headers)).rstrip(),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(_pad(cell, widths[index]) for index, cell in enumerate(row)).rstrip()
        for row in rows
    )
    return "\n".join(lines)


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(summaries: Sequence[ModelSummary], results: Sequence[RunResult]) -> str:
    lines = [
        "# llm-bench 报告",
        "",
        "## 汇总",
        "",
        "| 模型 | 通过 | 通过率 | 平均延迟 | 输入 tok | 输出 tok | 成本 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {_cell(summary.model)} | {summary.passed}/{summary.runs} | "
            f"{summary.pass_rate:.0%} | {summary.avg_latency_ms:.0f} ms | "
            f"{summary.prompt_tokens} | {summary.completion_tokens} | {_money(summary.cost_usd)} |"
        )

    lines += [
        "",
        "## 明细",
        "",
        "| 模型 | 任务 | 次数 | 结果 | 延迟 | 输入 | 输出 | 成本 | 说明 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        outcome = "error" if result.error is not None else ("pass" if result.passed else "fail")
        note = result.error if result.error is not None else result.detail
        lines.append(
            f"| {_cell(result.model)} | {_cell(result.task_id)} | {result.repeat} | {outcome} | "
            f"{result.latency_ms:.0f} ms | {result.prompt_tokens or 'n/a'} | "
            f"{result.completion_tokens or 'n/a'} | {_money(result.cost_usd)} | {_cell(note)} |"
        )
    return "\n".join(lines)


def render_csv(results: Sequence[RunResult]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "model",
            "task",
            "repeat",
            "passed",
            "latency_ms",
            "prompt_tokens",
            "completion_tokens",
            "cost_usd",
            "error",
            "detail",
        ]
    )
    for result in results:
        writer.writerow(
            [
                result.model,
                result.task_id,
                result.repeat,
                "true" if result.passed else "false",
                f"{result.latency_ms:.1f}",
                "" if result.prompt_tokens is None else result.prompt_tokens,
                "" if result.completion_tokens is None else result.completion_tokens,
                "" if result.cost_usd is None else f"{result.cost_usd:.6f}",
                result.error or "",
                result.detail,
            ]
        )
    return buffer.getvalue().rstrip("\n")


def render_json(summaries: Sequence[ModelSummary], results: Sequence[RunResult]) -> str:
    payload = {
        "summary": [asdict(summary) for summary in summaries],
        "results": [asdict(result) for result in results],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render(fmt: str, summaries: Sequence[ModelSummary], results: Sequence[RunResult]) -> str:
    if fmt == "table":
        return render_table(summaries)
    if fmt == "markdown":
        return render_markdown(summaries, results)
    if fmt == "csv":
        return render_csv(results)
    if fmt == "json":
        return render_json(summaries, results)
    raise ValueError(f"未知输出格式: {fmt}")
