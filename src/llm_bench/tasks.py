"""内置任务与确定性评分。

刻意不执行模型返回的代码：评分只看文本和 JSON 结构，避免把不可信输出
直接跑在本机。代价是代码题只能验证关键表达式，做不到真正的行为测试。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    detail: str


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    prompt: str
    check: Callable[[str], CheckResult]


_FENCE = re.compile(r"```(?:[A-Za-z0-9_+-]+)?[ \t]*\n(.*?)```", re.S)


def _answer_body(text: str) -> str:
    """有代码块时取最后一个代码块，否则用整段输出。"""
    blocks = _FENCE.findall(text)
    return blocks[-1] if blocks else text


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_CODE_FIX_PROMPT = """下面这个 Python 函数本意是把两个数相加，但现在有 bug。
请只返回修复后的完整函数代码，放在一个 ```python 代码块里，不要解释。

def add(a, b):
    return a - b
"""


def check_code_fix(text: str) -> CheckResult:
    code = _compact(_answer_body(text))
    fixed = "return a + b" in code or "return b + a" in code
    buggy = "return a - b" in code
    if buggy:
        return CheckResult(False, "仍保留了 return a - b")
    if not fixed:
        return CheckResult(False, "没找到 return a + b 或 return b + a")
    return CheckResult(True, "返回值已改成加法")


_EXTRACT_PROMPT = """从下面的发票文本中抽取字段，只返回一个 JSON 对象，不要解释。
字段：invoice_no（字符串）、total（数字）、currency（字符串，三位大写）。

发票号 INV-2026-0917，金额 1280.50 元，币种 CNY。"""

_EXPECTED_INVOICE = {
    "invoice_no": "INV-2026-0917",
    "total": 1280.5,
    "currency": "CNY",
}


def _json_object(text: str) -> dict[str, object] | None:
    source = _answer_body(text)
    start = source.find("{")
    end = source.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(source[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def check_extract(text: str) -> CheckResult:
    data = _json_object(text)
    if data is None:
        return CheckResult(False, "没有找到合法 JSON 对象")

    problems: list[str] = []
    for key, expected in _EXPECTED_INVOICE.items():
        if key not in data:
            problems.append(f"缺字段 {key}")
            continue
        value = data[key]
        if key == "total":
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or abs(float(value) - float(expected)) > 1e-6
            ):
                problems.append(f"total={value!r}")
        elif value != expected:
            problems.append(f"{key}={value!r}")

    if problems:
        return CheckResult(False, "；".join(problems))
    return CheckResult(True, "三个字段都正确")


_SUMMARY_PROMPT = """用不超过 3 句话、100 字以内概括下面的文字，必须提到“延迟”“成本”“缓存”三个词，只返回摘要。

延迟、成本和缓存是长上下文推理服务的三个约束。请求越长，prefill 阶段占用的算力越多，首 token 延迟随之上升；并发上来以后，批处理能把吞吐拉高，但排队时间也会增加尾延迟。缓存分两类：前缀缓存复用相同开头的 KV，命中后能省掉重复 prefill；结果缓存按输入哈希保存完整回答，适合重复问题。缓存不是免费的，它要占显存或内存，也要处理失效。工程上通常先做前缀缓存，再按命中率决定是否引入结果缓存；同时用分位延迟而不是平均值来评估体验。成本方面，输入 token 通常比输出 token 便宜，但长上下文会把输入成本放大，所以压缩提示词和限制输出长度往往比换更小的模型更有效。"""

_SUMMARY_KEYWORDS = ("延迟", "成本", "缓存")


def check_summary(text: str) -> CheckResult:
    body = _answer_body(text).strip()
    compact = re.sub(r"\s+", "", body)
    problems: list[str] = []

    missing = [word for word in _SUMMARY_KEYWORDS if word not in body]
    if missing:
        problems.append("缺少关键词 " + "、".join(missing))
    if len(compact) < 20:
        problems.append(f"太短（{len(compact)} 字）")
    if len(compact) > 100:
        problems.append(f"超过 100 字（{len(compact)} 字）")

    sentences = [part for part in re.split(r"[。！？!?]+", body) if part.strip()]
    if not 1 <= len(sentences) <= 3:
        problems.append(f"{len(sentences)} 句，超过 3 句")

    if problems:
        return CheckResult(False, "；".join(problems))
    return CheckResult(True, f"摘要 {len(compact)} 字 / {len(sentences)} 句，关键词齐全")


TASKS: tuple[Task, ...] = (
    Task("code-fix", "修复加法函数", _CODE_FIX_PROMPT, check_code_fix),
    Task("extract", "抽取发票字段", _EXTRACT_PROMPT, check_extract),
    Task("summarize", "压缩长文摘要", _SUMMARY_PROMPT, check_summary),
)


def get_tasks(ids: list[str] | None = None) -> list[Task]:
    """按给定顺序取任务；未知 id 抛 ValueError。"""
    if not ids:
        return list(TASKS)
    index = {task.id: task for task in TASKS}
    missing = [task_id for task_id in ids if task_id not in index]
    if missing:
        raise ValueError("未知任务: " + ", ".join(missing))
    return [index[task_id] for task_id in ids]
