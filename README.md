# coding-agent

一个不依赖 LangChain、LlamaIndex、Agents SDK、AutoGen、CrewAI 等 Agent
框架的最小 Coding Agent。模型只负责生成文本和原生 tool calling；对话历史、
工具执行、参数校验、路径权限、循环终止、重试、SQLite 历史和事件记录均由本地代码实现。

当前版本提供文件读取/搜索/枚举、工作区修改、只读 Git、受控命令执行和项目验证。
`list_files` / `glob_files` 遵循 `.gitignore`；`git_status`、`git_diff`、
`git_log` 不开放 Git 写操作；文件变更与进程执行统一经过权限和用户审批策略。

## 架构

```text
用户请求
  -> Agent.run（本地循环与终止策略）
  -> DeepSeekV4ProClient（OpenAI 兼容传输层）
  -> 模型返回原生 tool_calls
  -> ToolRegistry（本地 Schema 校验 + PermissionRequest）
  -> 读取 / Git / 修改 / 命令 / 验证工具（路径沙箱 + 用户审批）
  -> assistant/tool 状态事务写入本地 SQLite
  -> ContextBuilder 从 SQLite 投影模型消息
  -> 再次调用模型，直到得到最终文本
```

SQLite 是会话状态的事实来源；JSONL 只用于观察和排错。模型厂商的服务端会话
ID 不作为本地状态的事实来源。

发送给模型的持久化上下文默认使用约 24,000 Token 的软预算。短会话仍发送完整
历史；超出预算后，`ContextBuilder` 按完整用户轮次保留最近内容，并把更早内容压缩
为结构化摘要。完整原始消息不会删除，摘要单独保存在 SQLite 的
`context_summary` 表中，因此可以重新生成。工具调用与对应结果始终作为一个整体
保留，避免产生孤立的 `tool` 消息。

SQLite FTS5 会为用户文本、助手文本、推理内容及工具参数/结果建立全文索引，并使用
trigram 分词支持中文片段及代码标识符子串检索。
索引由数据库触发器随 `part` 的增删改自动同步；升级已有数据库时会自动回填。
长会话发生压缩时，`ContextBuilder` 使用最新用户问题检索已被移出窗口的历史，
并优先把相关命中放入结构化摘要。可通过 Python API 直接搜索：

```python
matches = store.search_history(
    "schema migration",
    session_id=session_id,
    before_seq=100,
    limit=10,
)
```

## 环境

- Python 3.11+
- DeepSeek API Key

核心 Agent 代码只使用 Python 标准库。DeepSeek V4 Pro 使用官方文档推荐的
OpenAI Python 客户端作为 HTTP/API 传输层，不使用任何 Agent SDK：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[deepseek]"
```

设置模型配置：

```powershell
$env:DEEPSEEK_API_KEY = "your-api-key"

# 可选：使用代理或兼容网关时覆盖，默认是 https://api.deepseek.com
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
```

仓库也提供 `.env.example` 作为变量名称示例，但程序不会自动读取 `.env`，避免
偷偷引入配置框架；请通过进程环境或你自己的密钥管理系统注入密钥。

运行任务：

```powershell
coding-agent --workspace . "读取 README.md，并说明这个项目的用途"
```

默认配置固定使用 `deepseek-v4-pro`、思考模式和 `high` reasoning effort。可以
按任务切换：

```powershell
coding-agent --reasoning-effort max --workspace . "分析项目架构"
coding-agent --no-thinking --workspace . "读取 README.md"
coding-agent --max-context-tokens 32000 --context-summary-tokens 3000 `
  --workspace . "继续长会话"
coding-agent --history-search-limit 8 --workspace . "回顾之前的数据库决策"
coding-agent --approval-mode deny --workspace . "只读分析这个仓库"
coding-agent --approval-mode allow --workspace . "运行测试并修复失败"
```

`--approval-mode` 默认是 `ask`：每次文件写入、补丁、永久删除或命令执行都会在
终端显示具体操作，并只允许单次确认。`deny` 适合完全只读的无人值守分析；`allow`
仅适合已经信任任务和仓库的自动化环境。读取、文件枚举和专用只读 Git 工具不提示。

思考模式产生工具调用时，适配器会把 DeepSeek 返回的 `reasoning_content`
原样保存在 assistant 消息中，并在下一轮请求中带回。Agent 不解析或执行其中内容。

也可以使用模块入口：

```powershell
python -m coding_agent --workspace . "读取 README.md"
```

模型可调用 `search_text` 在整个工作区或指定相对路径下递归搜索代码。工具默认执行
不区分大小写的字面量搜索，也可启用 `regex`，用 `include_patterns` / `exclude_patterns`
组合多个 glob，并通过 `context_lines` 获取上下文。每处命中分别返回相对文件路径、
匹配文本、从 1 开始的行号及起止列号；长行会返回围绕命中位置的有界片段。旧的单个
`file_pattern` 参数继续兼容。示例工具参数：

```json
{
  "query": "def\\s+create_",
  "path": "src",
  "regex": true,
  "include_patterns": ["*.py"],
  "exclude_patterns": ["tests/*", "generated/*"],
  "context_lines": 2,
  "max_results": 50
}
```

结果还包含扫描文件数及按 glob、二进制、越界链接、读取错误和文件大小上限分类的
跳过统计。二进制文件、版本控制目录、依赖目录和常见缓存目录默认跳过；路径穿越与
绝对路径仍由工作区沙箱拦截。

文件发现使用两个独立工具：

- `list_files` 列出目录的直接文件或全部后代，返回相对路径和字节数；
- `glob_files` 用一到多个 glob 筛选递归文件集合。

在 Git 工作区中，两者使用 `git ls-files --cached --others --exclude-standard`
作为可见文件的事实来源，因此同时支持根目录和嵌套 `.gitignore`、全局 excludes，
并始终保留已跟踪文件。非 Git 目录使用本地 `.gitignore` 兼容回退。

## Git、命令与验证

专用 `git_status`、`git_diff`、`git_log` 只暴露有界的读取参数。Git pager、交互式
凭据提示和外部 diff 被禁用，输出有大小上限。模型无法通过这些工具调用
`commit`、`push`、`reset` 等状态变更操作。

`run_command` 接收字符串数组形式的 `argv`，不把模型输出隐式交给 shell 解析。
执行程序必须在开发工具白名单中，工作目录必须位于工作区，单次运行默认 120 秒、
最长 300 秒，stdout/stderr 分别有上限。环境中的 API Key 不会传给子进程。需要
PowerShell 时必须显式使用 `powershell` 或 `pwsh`，并且删除、下载、动态执行等
高风险 cmdlet 会在审批前直接拒绝。通过 `run_command` 调用的 Git 也会拒绝写操作
和网络操作。

`verify_project` 根据仓库标记生成非交互验证计划，并按失败即停执行：

- Python：unittest/pytest、`python -m build`、配置过的 Ruff/Black check 模式；
- Node.js：只运行 `package.json` 中存在的 test/build/format-check 脚本；
- Rust：Cargo test/build 和 `cargo fmt --check`；
- Go：`go test ./...` 与 `go build ./...`。

验证类别是 `test`、`build`、`format_check` 或 `all`。格式化只使用检查模式，不会
自动重写源文件；未检测到相应配置时通过 `skipped_checks` 明确报告，`complete` 也会
保持为 false。每次验证计划作为一个完整操作请求用户审批。

文件变更工具遵循以下约束：

- `write_file` 创建 UTF-8 文件；只有显式传入 `overwrite=true` 才会替换已有文件，
  缺失的父目录也必须用 `create_parent_dirs=true` 明确允许创建。
- `apply_patch` 使用 `old_text` / `new_text` 做精确文本替换，默认要求旧文本恰好出现
  一次；`expected_replacements` 可明确指定次数。计数不符时文件保持不变。
- `delete_file` 只永久删除单个普通文件，不删除目录、符号链接或工作区外路径；可传入
  `expected_sha256`，避免删除内容与预期不符的文件。

创建和修改先在目标目录写入临时文件，再原子发布；单次写入默认限制为 1 MiB。
`.git` 仓库元数据和 `.coding-agent` 会话状态目录禁止通过变更工具修改。
`create_read_only_registry` 可供只读嵌入场景使用；CLI 默认使用
`create_workspace_registry`。嵌入方可以注入自定义 `ApprovalPolicy`，或使用内置的
`InteractiveApprovalPolicy`、`DenyApprovalPolicy`、`AllowAllApprovalPolicy`。
不注入策略时保持 Python API 的向后兼容行为；CLI 始终显式注入所选策略。

默认事件日志写入 `.coding-agent/events.jsonl`，该目录不会提交到 Git。
会话历史默认写入工作区下的 `.coding-agent/history.sqlite3`。数据库把历史拆为
`session -> message -> part`，工具 part 使用
`pending -> running -> completed/error` 状态机；进程异常退出后，未完成调用会被
标记为 `interrupted`，不会自动重放可能带副作用的工具。

每次运行会输出结果并返回一个持久化 `session_id`（Python API 的
`AgentResult.session_id`）。后续可以在同一工作区继续该会话：

```powershell
coding-agent --workspace . --session-id <session-id> "继续检查测试"
```

也可以用 `--db` 和 `--event-log` 覆盖默认位置。SQLite 文件用于恢复上下文，
JSONL 文件不参与恢复。

## 测试

测试使用假模型，不需要 API Key 或网络：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

测试覆盖：

- 工具调用、结果回传和最终响应
- DeepSeek `reasoning_content` 的保留
- `tool_call_id` 关联
- 模型请求重试
- 最大步数终止
- 重复工具调用检测
- 路径穿越和绝对路径拦截
- 代码文本/正则递归搜索、多 glob、上下文、精确位置、结果截断和扫描统计
- 遵循 `.gitignore` 的文件列举与 glob 筛选
- 只读 Git status/diff/log
- 文件创建、显式覆盖、精确补丁、原子写入和单文件删除保护
- 读操作免审批、状态变更审批/拒绝，以及拒绝后零副作用
- 结构化 argv、程序白名单、PowerShell 高危操作拦截和输出捕获
- 仓库验证策略检测与测试执行
- 工具参数 Schema 校验
- SQLite 重启后上下文恢复
- 上下文预算、完整轮次裁剪和结构化摘要持久化
- FTS5 迁移回填、自动索引同步、会话/序号过滤和相关历史召回
- 工具调用状态转换、幂等领取和中断恢复

## 当前边界

- CLI 支持使用 `--session-id` 继续历史会话；尚未提供会话列表命令
- 命令白名单和审批降低风险，但不是操作系统级容器；不可信仓库仍应在隔离环境运行
- 非 Git 目录的 `.gitignore` 回退覆盖常见规则，完整 Git 语义以 Git 工作区为准
- 上下文 Token 数使用无第三方 tokenizer 的保守估算，不是厂商精确计数
- 结构化摘要是确定性的历史压缩，尚未使用模型提炼长期语义记忆
- FTS5 是关键词检索，尚未提供向量语义检索或混合排序
- trigram 查询至少需要 3 个字符；更短的缩写不适合单独作为检索词
- DeepSeek V4 Pro 当前通过 OpenAI 兼容的 Chat Completions 格式调用

下一阶段可以增加操作系统级进程隔离、可持久化的审批规则，以及更细粒度的网络权限；
历史量明显增长且关键词召回不足后，再评估向量语义检索与混合排序。
