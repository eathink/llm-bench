# llm-bench

**在多个 OpenAI 兼容模型上跑同一组固定任务，对比延迟、成本和正确率。**

换模型时凭感觉挑最麻烦：贵的未必对，快的未必稳。`llm-bench` 把几个小任务固定下来，对每个模型跑同样的 prompt，用确定性规则判分，最后给一张可直接贴进周报的对比表。

[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Zero deps](https://img.shields.io/badge/dependencies-0-brightgreen)](#设计取舍)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![CI](https://github.com/eathink/llm-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/eathink/llm-bench/actions/workflows/ci.yml)

## 快速开始

```bash
pipx install git+https://github.com/eathink/llm-bench

# 先看会跑哪些模型和任务，不发请求、不需要 key
llm-bench run --config examples/bench.example.json --dry-run

# 真正跑一轮
export OPENAI_API_KEY=...
export DEEPSEEK_API_KEY=...
llm-bench run --config examples/bench.example.json --format markdown
```

```console
$ llm-bench run --config bench.json
模型  通过  通过率  平均延迟  输入 tok  输出 tok  成本
----  ----  ------  --------  --------  --------  ---------
gpt-4o-mini  3/3   100%    842 ms    612       141       $0.000176
deepseek-chat  2/3   67%     1204 ms   598       156       $0.000333
```

## 配置

JSON 配置只放模型和请求参数；API key 从环境变量读，**不写进配置文件**。

```json
{
  "models": [
    {
      "name": "gpt-4o-mini",
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-4o-mini",
      "api_key_env": "OPENAI_API_KEY",
      "price": { "input_per_1m": 0.15, "output_per_1m": 0.60 }
    },
    {
      "name": "deepseek-chat",
      "base_url": "https://api.deepseek.com/v1",
      "model": "deepseek-chat",
      "api_key_env": "DEEPSEEK_API_KEY"
    }
  ],
  "timeout": 60,
  "temperature": 0,
  "max_tokens": 512
}
```

| 字段 | 说明 |
| --- | --- |
| `models[].name` | 报告里显示的模型名，必须唯一 |
| `models[].base_url` | OpenAI 兼容端点，自动补 `/chat/completions` |
| `models[].model` | 请求里的 `model` 字段 |
| `models[].api_key_env` | 存放 key 的环境变量名 |
| `models[].price` | 可选，每百万 token 的美元价格；不填时成本显示 `n/a` |
| `timeout` | 单次请求超时秒数，默认 60 |
| `temperature` | 默认 0 |
| `max_tokens` | 默认 512；设为 `null` 则不在请求里带该字段 |

> 示例价格只用于演示算法，实际价格以厂商官网为准。

## 内置任务

| id | 任务 | 判定方式 |
| --- | --- | --- |
| `code-fix` | 修复一个把加法写成减法的函数 | 输出代码里出现 `return a + b`（或 `b + a`），且不再出现 `return a - b` |
| `extract` | 从发票文本抽取 `invoice_no` / `total` / `currency` | 解析输出里的 JSON，三个字段完全匹配（数字允许浮点误差） |
| `summarize` | 用不超过 3 句、100 字概括一段长文 | 长度、句数、三个关键词“延迟/成本/缓存”同时满足 |

```bash
llm-bench tasks          # 列出任务与 prompt 首行
llm-bench run --config bench.json --tasks code-fix,extract
llm-bench run --config bench.json --repeat 3
```

评分全部是确定性规则，不需要再请一个模型当裁判——同一份回答永远得到同一个结论。

## 输出

```bash
llm-bench run --config bench.json --format table      # 默认，终端表格
llm-bench run --config bench.json --format markdown   # 汇总 + 明细
llm-bench run --config bench.json --format csv        # 每次运行一行
llm-bench run --config bench.json --format json
llm-bench run --config bench.json --output report.md --format markdown
```

`--save-responses DIR` 会把每次原始回答写成一个文本文件（文件名含模型、任务和轮次），方便复盘判分为何失败。回答可能包含敏感内容，保存前注意目录权限。

### 参数

| 参数 | 说明 |
| --- | --- |
| `--config FILE` | 必填，JSON 配置 |
| `--models a,b` | 只跑指定模型，默认全部 |
| `--tasks id,...` | 只跑指定任务，默认全部 |
| `--repeat N` | 每个任务重复 N 次，默认 1 |
| `--format ...` | `table` / `markdown` / `csv` / `json` |
| `--output FILE` | 写入文件，报告不再打到标准输出 |
| `--timeout SEC` | 覆盖配置里的超时 |
| `--save-responses DIR` | 保存原始回答 |
| `--exit-zero` | 有失败也返回 0 |
| `--dry-run` | 只列模型与任务，不发请求、不检查 key |

### 退出码

| 码 | 含义 |
| --- | --- |
| 0 | 全部通过（或用了 `--exit-zero`） |
| 1 | 至少一个任务未通过或请求失败 |
| 2 | 配置、参数或环境变量错误 |
| 130 | 被 Ctrl+C 中断 |

## 安全边界

- API key 只从 `api_key_env` 指向的环境变量读取，配置文件、命令行和报告里都不会出现 key
- 报告不会打印请求头；HTTP 错误只保留状态码和最多 300 字节响应体
- **不执行模型输出**：代码题只做文本匹配，避免把不可信代码跑在本机

## 设计取舍

- **确定性判分，不用 LLM 裁判**：可复现、零额外成本，代价是只能覆盖有明确答案的小任务
- **不执行模型返回的代码**：安全优先，代码题因此只能验证关键表达式，不能验证边界行为
- **零依赖**：HTTP 用标准库 `urllib`，报告用 `csv` / `json`，装完即用
- **串行执行**：MVP 一次只发一个请求，延迟数字不受并发干扰；比总耗时更重要的是可比性
- **成本未知就显示 `n/a`**：缺 token 数或没配价格时，宁可不给数字，也不把未知当 0

## 已知边界

- 只支持 OpenAI Chat Completions 协议，不覆盖 Responses API / Anthropic 原生协议
- 没有重试和并发；429 或 5xx 直接记为一次失败
- 任务集固定为三个内置任务，暂不支持自定义任务文件
- 延迟是端到端墙钟时间，包含网络往返；跨地区对比时要注意网络差异

## 开发

```bash
git clone https://github.com/eathink/llm-bench
cd llm-bench

# 仅用标准库
PYTHONPATH=src python3 -m unittest discover -s tests -v

# 本地运行
PYTHONPATH=src python3 -m llm_bench run --config examples/bench.example.json --dry-run
```

测试里有一个本地假 OpenAI 服务，覆盖成功、HTTP 错误、坏 JSON、缺字段和超时；不需要真实 API key。

## License

[MIT](LICENSE)
