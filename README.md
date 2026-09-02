# coding-agent

一个不依赖 LangChain、LlamaIndex、Agents SDK、AutoGen、CrewAI 等 Agent
框架的最小 Coding Agent。模型只负责生成文本和原生 tool calling；对话历史、
工具执行、参数校验、路径权限、循环终止、重试、SQLite 历史和事件记录均由本地代码实现。

当前版本提供文件读取/搜索/枚举、工作区修改、只读 Git、受控命令执行和项目验证。
`list_files` / `glob_files` 遵循 `.gitignore`；`git_status`、`git_diff`、
`git_log` 不开放 Git 写操作；文件变更与进程执行统一经过权限和用户审批策略。
可选的 Textual 终端界面提供多轮输入、工具活动、Markdown 回答和审批弹窗，同时
保留适合脚本与 CI 的一次性文本模式。

安全边界、残余风险和回归要求见 [`SECURITY.md`](SECURITY.md)。

## 架构

```text
用户请求
  -> plain CLI / Textual TUI（同一套 Agent API）
  -> Agent.run（本地循环与终止策略）
  -> DeepSeekV4ProClient（OpenAI 兼容传输层）
  -> 模型返回原生 tool_calls
  -> ToolRegistry（本地 Schema 校验 + PermissionRequest）
  -> 读取 / Git / 修改 / 命令 / 验证工具（路径沙箱 + 用户审批）
  -> assistant/tool 状态事务写入本地 SQLite
  -> ContextBuilder 从 SQLite 投影模型消息
  -> EventSink 同步到审计日志和终端界面
  -> 再次调用模型，直到得到最终文本
```

SQLite 是会话状态的事实来源；JSONL 只用于观察和排错。模型厂商的服务端会话
ID 不作为本地状态的事实来源。

发送给模型的持久化上下文默认使用约 128K Token 的软预算。短会话仍发送完整
历史；超出预算后，`ContextBuilder` 按完整用户轮次保留最近内容，并把更早内容压缩
为结构化摘要。完整原始消息不会删除，摘要单独保存在 SQLite 的
`context_summary` 表中，因此可以重新生成。工具调用与对应结果始终作为一个整体
保留，避免产生孤立的 `tool` 消息。
压缩记录会被明确标注为低权限、不可信历史数据，不会把仓库或工具输出提升为
`system` 指令。

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
- Docker 或 Podman（CLI 默认要求，用于命令的 OS 级隔离）

核心 Agent 代码只使用 Python 标准库。DeepSeek V4 Pro 使用官方文档推荐的
OpenAI Python 客户端作为 HTTP/API 传输层，不使用任何 Agent SDK：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[deepseek,tui]"
```

只使用通用 OpenAI 兼容适配器时也可以安装 `.[openai]`。参与开发和运行完整测试、
构建检查时使用：

```powershell
python -m pip install -e ".[dev]"
```

设置模型配置：

```powershell
$env:DEEPSEEK_API_KEY = "your-api-key"

# 可选：使用代理或兼容网关时覆盖，默认是 https://api.deepseek.com
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
```

`web_search` 通过 Exa 的远程 MCP 服务查询公开网页。Exa 公共 MCP 端点可在不配置
Key 时使用；需要独立配额或更稳定的生产使用时，可选设置：

```powershell
$env:EXA_API_KEY = "your-exa-api-key"
```

搜索请求固定发送到 `https://mcp.exa.ai/mcp`，返回内容会标记为不可信数据；
`fetch_webpage` 继续使用带 DNS/IP 复查和响应大小限制的受限 HTTPS 读取。
使用 `--no-web-access` 可以同时禁用这两个工具。

CLI 默认以 fail-closed 方式要求本地 OCI 容器运行时和一个明确指定的可信镜像。
仓库提供仅含 Python 的最小示例；构建不会由 Agent 自动执行，也不会在运行任务时
自动拉取镜像：

```powershell
docker build --pull -f sandbox/Dockerfile -t coding-agent-sandbox:python sandbox
$env:CODING_AGENT_SANDBOX_IMAGE = "coding-agent-sandbox:python"

# 可选；默认自动选择 docker 或 podman
$env:CODING_AGENT_SANDBOX_RUNTIME = "docker"
```

示例镜像适合本仓库的标准库测试。Node、Rust、Go、.NET 等项目应从受信任并经过
审查的基础镜像构建自己的工具链镜像，再通过 `--sandbox-image` 指定。镜像必须已经
存在于本地；Agent 使用 `--pull=never`，避免执行期间产生隐式网络和供应链变化。

仓库也提供 `.env.example` 作为变量名称示例，但程序不会自动读取 `.env`，避免
偷偷引入配置框架；请通过进程环境或你自己的密钥管理系统注入密钥。

不提供任务参数时进入交互式终端界面：

```powershell
coding-agent --workspace .
```

提供任务参数时保持一次性文本模式：

```powershell
coding-agent --workspace . "读取 README.md，并说明这个项目的用途"
```

也可以带着初始任务进入终端界面，或显式要求纯文本输出：

```powershell
coding-agent --interactive --workspace . "先检查当前 Git 改动"
coding-agent --plain --workspace . "只输出项目摘要"
```

默认配置固定使用 `deepseek-v4-pro`、思考模式和 `high` reasoning effort。可以
按任务切换：

```powershell
coding-agent --reasoning-effort max --workspace . "分析项目架构"
coding-agent --no-thinking --workspace . "读取 README.md"
coding-agent --max-context-tokens 262144 --context-summary-tokens 16384 `
  --workspace . "继续长会话"
coding-agent --history-search-limit 8 --workspace . "回顾之前的数据库决策"
coding-agent --max-steps 40 --max-total-steps 160 --step-extension 20 `
  --workspace . "执行特别长的重构任务"
coding-agent --approval-mode deny --workspace . "只读分析这个仓库"
coding-agent --approval-mode ask --workspace . "逐项审查这个不可信仓库"
coding-agent --approval-timeout-s 300 --workspace . "运行需要审批的任务"
coding-agent --verification-mode strict --workspace . "执行 CI 级严格验证"
coding-agent --approval-mode allow --workspace . "运行测试并修复失败"
coding-agent --sandbox-image coding-agent-sandbox:python --workspace . "运行测试"
coding-agent --model-timeout-s 180 --max-model-retries 5 `
  --retry-base-delay-s 1 --workspace . "继续长任务"
```

持久会话默认使用约 128K（131072）Token 的上下文预算，其中最多约 8K（8192）
用于压缩后的历史摘要。Token 数是本地保守估算值；可以通过上述参数为特别长的任务
临时调整。

模型连接中断、响应体读取失败或请求超时时，Agent 会在同一模型步骤内按指数退避重试，
不会重复已经完成的本地工具调用。上述三个参数可用于高延迟模型或不稳定的兼容网关；
持续出现 `stream disconnected before completion` 时还应检查代理/VPN、网关健康状态和
`--base-url` 是否指向预期端点。
HTTP 408/409/425/429、5xx 和没有 HTTP 状态码的传输错误会重试；400、401、403、404、
422 等永久请求错误会立即停止，并在 CLI/TUI 中显示简洁的可操作错误信息。

每个 CLI 顶层任务初始使用 30 个模型步骤；到达当前软上限时，如果自上次预算检查后
出现了新的成功工具调用，预算会自动增加 15 步，硬上限默认为 100 步。可分别使用
`--max-steps`、`--step-extension` 和 `--max-total-steps` 调整。直接使用 Python API 时，
`AgentConfig.max_total_steps` 默认为 `None`，因此保持固定 `max_steps` 的原有行为；设置
硬上限后才启用自动续期。

没有新成功结果、检测到无进展或达到硬上限时，最后一个步骤不再提供任何工具，而是
要求模型输出纯文本交接，明确已完成、未完成、验证状态和下一步，因此不会在预算耗尽前
的最后一轮留下新的未验证修改。若此前存在工作区变更，核心会在最终交接前主动运行一次
`verify_project`。正常完成状态为 `completed`；预算耗尽或检测到无进展时为 `partial`；
验证或外部条件无法继续时为 `blocked`；用户中断和不可恢复故障分别记录为
`interrupted`、`failed`。持久会话可以在 `partial` 或 `blocked` 后继续。

无进展检测除了连续的完全相同调用，还会识别短窗口内的交替工具调用周期、同一工具
反复产生相同错误，以及连续工具失败。触发后剩余工具调用会被安全跳过，下一模型步骤
只生成部分交接，不会继续扩大副作用。

`--approval-mode` 默认是 `workspace`：工作区内的文件创建/修改和 OS 沙箱内的命令、
验证自动执行；专用 `delete_file` 操作和脱离 OS 沙箱的命令仍要求审批。沙箱命令仍可
修改或删除工作区内容，因此该模式信任工作区边界，而不是保证逐文件无破坏。`ask` 保留原有的逐项审批，
`deny` 适合完全只读的无人值守分析，`allow` 仅适合已经信任任务和仓库的自动化环境。
读取、文件枚举和专用只读 Git 工具不提示。

CLI 审批提示和 TUI 弹窗均支持“允许一次”与“本任务内允许”。任务授权只复用于同一
次顶层 Agent 任务的受限 scope，例如工作区写入、删除或同一命令程序；下一次
`Agent.run()` 开始时自动清空。权限规则可以同时按操作类型、工具名、资源 glob、命令
前缀和沙箱状态匹配；冲突时固定采用 `deny > ask > allow`，因此宽泛允许规则不能覆盖
更具体的拒绝或询问规则。
TUI 审批默认等待 300 秒，超时后自动拒绝；可用 `--approval-timeout-s` 调整。

## 交互式终端界面

初版 TUI 保持单栏布局，以便在普通 80×24 终端和 Windows Terminal 中使用：

```text
Header：coding-agent 与时钟
Context：workspace、model、approval、session
Conversation：用户消息、Markdown 回答、reasoning 摘要、工具状态
Activity：模型轮次、当前工具和该阶段已用时间
Prompt：多行任务编辑器
Footer：快捷键
```

快捷键和本地命令：

- `Ctrl+S`：发送任务；部分支持增强键盘协议的终端也可使用 `Ctrl+Enter`；
- `Ctrl+Y`：复制最近一条 Agent 回答的原始 Markdown（代码块保持原样）；
- 点击代码块标题栏右侧的 `⧉ Copy`：只复制该代码块内容；
- 鼠标选中会话文本后按 `Ctrl+C`：复制选中的局部文本；
- `Ctrl+L`：清空当前界面，不删除 SQLite 历史；
- `Ctrl+Q`：退出；
- `/new`：开始一个新的持久化会话；
- `/copy`：复制最近一条 Agent 回答；
- `/clear`、`/help`、`/exit`：清屏、帮助和退出。

同步的 `Agent.run` 在 Textual 线程 Worker 中执行，模型和工具事件通过
`TuiEventSink` 回到 UI 线程；`TuiApprovalPolicy` 只阻塞 Agent Worker，并用 Future
等待审批弹窗。每次模型请求和工具执行都会先发送 started 事件，因此等待完整响应期间
状态栏仍会显示当前模型轮次、工具名称和持续时间。审计事件仍同时写入 JSONL。这个桥接
层接受任意字符串事件类型，后续加入 `text_delta`、`tool_output_delta` 和取消事件时无需
重写页面启动逻辑。

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

`search_text` 会自动探测 PATH 中位于工作区外的 `rg`，优先通过 ripgrep 的 JSON 协议执行搜索，
因此大型 Git 工作区能够利用并行遍历、`.gitignore` 和 ripgrep 的高性能匹配；参数
校验、路径沙箱、文件大小及结果大小上限仍由 Python 层控制。如果未安装 ripgrep 或
启动失败，字面量搜索会自动回退到 Python 实现。正则搜索必须使用 ripgrep；不支持的
正则会返回可操作错误，不再回退到无法安全中断的 Python 正则引擎。结果中的
`search_backend` 表示实际后端；ripgrep 无法提供逐类跳过文件计数时，
`search_statistics_complete` 为 `false`，这些分类计数保守返回 `0`。
两种后端共享默认 30 秒的总搜索时限；达到时限后会终止 ripgrep 或停止 Python 遍历，
返回已经找到的部分结果，并在 `truncation_reasons` 中包含 `time_limit`，避免 TUI 因
磁盘、目录或子进程异常而无限停留在 `Searching code`。

文件发现使用两个独立工具：

- `list_files` 列出目录的直接文件或全部后代，返回相对路径和字节数；
- `glob_files` 用一到多个 glob 筛选递归文件集合。

`read_file`、`list_files` 和 `glob_files` 共享默认 30 秒文件操作时限。达到时限时读取或
枚举会返回已有部分结果，并用 `truncation_reasons=["time_limit"]` 明确标记。

在 Git 工作区中，两者使用 `git ls-files --cached --others --exclude-standard`
作为可见文件的事实来源，因此同时支持根目录和嵌套 `.gitignore`、全局 excludes，
并始终保留已跟踪文件。非 Git 目录使用本地 `.gitignore` 兼容回退。

## Git、命令与验证

专用 `git_status`、`git_diff`、`git_log` 只暴露有界的读取参数。Git pager、交互式
凭据提示和外部 diff 被禁用，输出有大小上限。模型无法通过这些工具调用
`commit`、`push`、`reset` 等状态变更操作。

`run_command` 接收字符串数组形式的 `argv`，不把模型输出隐式交给 shell 解析。
执行程序必须在开发工具白名单中，工作目录必须位于工作区，单次运行默认 120 秒、
最长 300 秒，stdout/stderr 分别有上限。子进程使用最小环境白名单，不会继承 API
Key、Token、密码和任意自定义环境变量；超时会终止为该命令创建的进程组。需要
PowerShell 时必须显式使用 `powershell` 或 `pwsh`，并且删除、下载、动态执行等
高风险 cmdlet 会在审批前直接拒绝。通过 `run_command` 调用的 Git 也会拒绝写操作
和网络操作。
命令超时后的进程树终止和输出回收还有独立的 5 秒清理截止；清理未完全结束时结果会
设置 `cleanup_incomplete=true`，不会再次无限阻塞 Agent Worker。

CLI 默认把 `run_command` 与 `verify_project` 放入 Docker/Podman Linux 容器执行：

- 只把工作区绑定到 `/workspace`，不传递宿主 API Key、Token 或任意自定义环境变量；
- 容器根文件系统只读，工作区保持可写以支持构建和测试；`.git` 额外只读挂载，
  `.coding-agent` 用不可访问的临时文件系统遮蔽；
- `--network=none`、`--cap-drop=ALL`、`no-new-privileges`，并限制 PID、内存、CPU、
  打开文件数、执行时长及输出；
- 使用非 root 数字用户；超时时按唯一容器名强制删除容器，再终止客户端进程组；
- 运行时和镜像启动前预检。缺失、守护进程不可用或镜像不在本地时直接退出，绝不
  静默退回宿主执行。

`--sandbox off` 是面向明确可信环境的高风险逃生开关，会恢复受控但非 OS 隔离的
宿主进程执行。审批与 OS 隔离是两层独立控制：容器命令仍然遵循 `--approval-mode`。
Docker Desktop 在 Windows 上通过其 Linux VM 提供上述容器边界；本实现不声称使用
Windows AppContainer。文件读取和精确修改工具仍在 Agent 进程内执行，依赖路径、
符号链接、受保护目录和审批策略，不属于容器隔离范围。

`verify_project` 根据仓库标记生成非交互验证计划，并按失败即停执行：

- Python：unittest/pytest，以及配置过的 Ruff/Black check 模式；
- Node.js：只运行 `package.json` 中存在的 test/build/format-check 脚本；
- Rust：Cargo test/build 和 `cargo fmt --check`；
- Go：`go test ./...` 与 `go build ./...`。

仓库可以提供受保护的 `.coding-agent-verification.toml`，用显式命令覆盖自动发现：

```toml
version = 1

[[commands]]
kind = "test"
argv = ["python", "-m", "unittest", "discover", "-s", "../tests", "-v"]
cwd = "src"
```

每个 `commands` 项的 `kind` 必须是 `test`、`build` 或 `format_check`，`argv`
仍受命令白名单约束，`cwd` 必须是工作区内现有目录。配置文件存在时是验证计划的
唯一来源；未声明的类别会报告为 skipped。核心文件工具禁止模型修改该策略文件，
容器执行时也会把它单独只读挂载。需要改变策略时应由用户在 Agent 外部审查并修改。
Python 自动发现不再仅因存在 `pyproject.toml` 就假定已安装 `python -m build`；需要
构建检查的仓库应在显式配置中声明。

验证类别是 `test`、`build`、`format_check` 或 `all`。每条命令默认最多 180 秒，整个
验证计划默认最多 300 秒，可分别用 `timeout_s` 和 `total_timeout_s` 调整。格式化只使用检查模式，不会
自动重写源文件；未检测到相应配置时通过 `skipped_checks` 明确报告，`complete` 也会
保持为 false。每次验证计划作为一个完整权限操作处理；默认 `workspace` 模式会自动
执行沙箱内验证，`ask` 模式仍会请求用户审批。

验证门禁默认使用 `--verification-mode available`：发现验证命令时仍必须执行并通过；
仓库完全没有可用验证命令时不再抛出异常终止，而是让模型基于明确的未验证状态完成
交接，任务状态记为 `partial`。CI 或高风险仓库可使用 `--verification-mode strict`，
此时没有验证命令仍会抛出 `VerificationRequiredError`。验证工具缺失、审批被拒、沙箱
不可用和已执行检查失败不属于“未配置”，两种模式都会继续阻断完成或要求模型修复。

成功执行 `write_file`、`apply_patch`、`delete_file` 或可能改变工作区的
`run_command` 后，Agent 核心会设置强制验证状态。模型给出最终回答时，核心先自动
执行一次 `verify_project(kind="all")`：通过后才接受并保存最终回答；检查失败会把
结果写回对话供模型继续修复；没有可用检查时按上述 verification mode 处理，验证权限
被拒绝或验证工具不可用时仍抛出 `VerificationRequiredError`。该状态和未验证原因可从
SQLite 工具历史恢复，因此重启进程或继续持久会话不会丢失验证状态。

文件变更工具遵循以下约束：

- `write_file` 创建 UTF-8 文件；只有显式传入 `overwrite=true` 才会替换已有文件，
  缺失的父目录也必须用 `create_parent_dirs=true` 明确允许创建。
- `apply_patch` 使用 `old_text` / `new_text` 做精确文本替换，默认要求旧文本恰好出现
  一次；`expected_replacements` 可明确指定次数。计数不符时文件保持不变。
- `delete_file` 只永久删除单个普通文件，不删除目录、符号链接或工作区外路径；可传入
  `expected_sha256`，避免删除内容与预期不符的文件。删除默认拒绝超过 64 MiB 的文件，
  且删除前哈希遵守 30 秒文件操作时限。

`web_search` 和 `fetch_webpage` 使用默认 15 秒端到端时限，覆盖 DNS、HTTPS 传输、响应
读取和全部重定向，而不是为每个阶段重新计时。

创建和修改先在目标目录写入临时文件，再原子发布；单次写入默认限制为 1 MiB。
`.git` 仓库元数据、`.coding-agent` 会话状态目录和
`.coding-agent-verification.toml` 验证策略禁止通过变更工具修改。
`create_read_only_registry` 可供只读嵌入场景使用；CLI 默认使用
`create_workspace_registry`。嵌入方可以注入自定义 `ApprovalPolicy`，或使用内置的
`create_approval_policy`、`PermissionRuleEngine`、`InteractiveApprovalPolicy`、
`DenyApprovalPolicy`、`AllowAllApprovalPolicy`。规则引擎只在至少一条规则匹配时比较
匹配结果；没有匹配时使用策略默认值。
不注入策略时保持 Python API 的向后兼容行为；CLI 始终显式注入所选策略。

默认事件日志写入 `.coding-agent/events.jsonl`，该目录不会提交到 Git。JSONL 边界会
递归脱敏常见凭据字段和文本，并默认移除 `reasoning_content`；实时 TUI 事件不受影响。
可以离线汇总模型/工具调用、失败、重试、终止原因、耗时和 Token usage：

```powershell
coding-agent-report .coding-agent/events.jsonl
```

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

未安装 `tui` 可选依赖时，核心测试仍可运行，TUI 测试会跳过；安装后使用 Textual
`run_test()` 在无头终端中验证布局、后台 Worker、消息渲染和审批弹窗。

测试覆盖：

- 工具调用、结果回传和最终响应
- DeepSeek `reasoning_content` 的保留
- `tool_call_id` 关联
- 模型请求重试
- 非正常 `finish_reason` 拒绝、响应 ID/Token usage 记录
- 最大步数终止
- 重复工具调用检测
- 路径穿越和绝对路径拦截
- 代码文本/正则递归搜索、多 glob、上下文、精确位置、结果截断和扫描统计
- 遵循 `.gitignore` 的文件列举与 glob 筛选
- 只读 Git status/diff/log
- 文件创建、显式覆盖、精确补丁、原子写入和单文件删除保护
- 读操作免审批、状态变更审批/拒绝，以及拒绝后零副作用
- 结构化 argv、程序白名单、PowerShell 高危操作拦截和输出捕获
- 子进程最小环境、超时进程组终止和非沙箱状态标记
- 审计日志凭据脱敏及 reasoning 移除
- 仓库验证策略检测与测试执行
- 工具参数 Schema 校验
- Exa MCP JSON/SSE 搜索响应、可选 API Key 和搜索错误处理
- SQLite 重启后上下文恢复
- 上下文预算、完整轮次裁剪和结构化摘要持久化
- 压缩历史不提升为 system 权限
- FTS5 迁移回填、自动索引同步、会话/序号过滤和相关历史召回
- 工具调用状态转换、幂等领取和中断恢复
- plain/interactive CLI 参数兼容
- Textual 页面挂载、后台执行、响应渲染、本地命令和审批弹窗

## 当前边界

- CLI 支持使用 `--session-id` 继续历史会话；尚未提供会话列表命令
- TUI 已显示模型轮次、当前工具和阶段耗时；尚未提供 token/命令输出流式显示和进程取消
- CLI 命令默认使用 OS 级容器隔离；显式 `--sandbox off` 或直接使用未注入容器后端的
  Python API 时，结果会返回 `sandboxed=false`，此模式不应用于不可信仓库
- 容器不是虚拟机安全边界；仍需信任 Docker/Podman、OCI 运行时、内核与指定镜像，
  高风险场景应使用专用主机或一次性 VM，并优先使用 rootless 运行时
- 非 Git 目录的 `.gitignore` 回退覆盖常见规则，完整 Git 语义以 Git 工作区为准
- 上下文 Token 数使用无第三方 tokenizer 的保守估算，不是厂商精确计数
- 结构化摘要是确定性的历史压缩，尚未使用模型提炼长期语义记忆
- FTS5 是关键词检索，尚未提供向量语义检索或混合排序
- trigram 查询至少需要 3 个字符；更短的缩写不适合单独作为检索词
- DeepSeek V4 Pro 当前通过 OpenAI 兼容的 Chat Completions 格式调用

下一阶段优先增加真实任务评测基线、镜像 SBOM/签名校验与更细粒度的只读工作区模式，
然后增加模型/命令流式事件、取消令牌和会话选择器；之后再增加可持久化审批规则和
按域名/操作授权的临时网络权限。历史量明显增长且关键词召回不足
后，再评估向量语义检索与混合排序。
