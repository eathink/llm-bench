"""命令行入口。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Sequence, TypeVar

from . import __version__
from .bench import run_bench, summarize
from .config import ConfigError, load_config
from .report import render
from .tasks import TASKS

T = TypeVar("T")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-bench",
        description="在多个 OpenAI 兼容模型上跑固定任务，对比延迟、成本与正确率。",
    )
    parser.add_argument("--version", action="version", version=f"llm-bench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="运行基准测试")
    run.add_argument("--config", required=True, type=Path, help="JSON 配置文件")
    run.add_argument("--models", help="逗号分隔的模型名，默认全部")
    run.add_argument("--tasks", help="逗号分隔的任务 id，默认全部")
    run.add_argument("--repeat", type=int, default=1, help="每个任务重复次数（默认 1）")
    run.add_argument(
        "--format", choices=["table", "markdown", "csv", "json"], default="table", help="输出格式"
    )
    run.add_argument("--output", type=Path, help="写入文件而不是标准输出")
    run.add_argument("--timeout", type=float, help="单次请求超时秒数，覆盖配置")
    run.add_argument("--save-responses", type=Path, help="保存每次原始回答的目录")
    run.add_argument("--exit-zero", action="store_true", help="有失败也返回 0")
    run.add_argument("--dry-run", action="store_true", help="只列出模型与任务，不发请求")

    sub.add_parser("tasks", help="列出内置任务")
    return parser


def _filter(items: Sequence[T], names: str | None, key: Callable[[T], str], label: str) -> list[T]:
    if not names:
        return list(items)
    wanted = list(dict.fromkeys(name.strip() for name in names.split(",") if name.strip()))
    index = {key(item): item for item in items}
    missing = [name for name in wanted if name not in index]
    if missing:
        raise ConfigError(f"未知{label}: {', '.join(missing)}")
    return [index[name] for name in wanted]


def _cmd_tasks() -> int:
    for task in TASKS:
        first_line = task.prompt.strip().splitlines()[0]
        print(f"{task.id:10} {task.title} — {first_line}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    models = _filter(config.models, args.models, lambda model: model.name, "模型")
    tasks = _filter(TASKS, args.tasks, lambda task: task.id, "任务")
    if args.repeat < 1:
        raise ConfigError("--repeat 至少为 1")

    timeout = config.timeout if args.timeout is None else args.timeout
    if timeout <= 0:
        raise ConfigError("--timeout 必须大于 0")

    if args.dry_run:
        print(f"模型（{len(models)}）:")
        for model in models:
            state = "已设置" if os.environ.get(model.api_key_env) else "未设置"
            print(
                f"  {model.name}: {model.model} @ {model.base_url} "
                f"key={model.api_key_env}（{state}）"
            )
        print(f"任务（{len(tasks)}）:")
        for task in tasks:
            print(f"  {task.id}: {task.title}")
        print(f"重复 {args.repeat} 次；不会发起网络请求。")
        return 0

    missing = [model.api_key_env for model in models if not os.environ.get(model.api_key_env)]
    if missing:
        raise ConfigError("缺少环境变量: " + ", ".join(dict.fromkeys(missing)))

    if args.save_responses is not None:
        args.save_responses.mkdir(parents=True, exist_ok=True)

    results = run_bench(
        models=models,
        tasks=tasks,
        repeats=args.repeat,
        environment=os.environ,
        timeout=timeout,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        save_dir=args.save_responses,
    )
    summaries = summarize(results, models)
    text = render(args.format, summaries, results)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"已写入 {args.output}", file=sys.stderr)
    else:
        print(text)

    failed = any(result.error is not None or not result.passed for result in results)
    return 0 if args.exit_zero or not failed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "tasks":
            return _cmd_tasks()
        return _cmd_run(args)
    except (ConfigError, ValueError) as exc:
        print(f"llm-bench: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("llm-bench: 已中断", file=sys.stderr)
        return 130
