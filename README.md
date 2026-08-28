# coding-agent

一个不依赖 LangChain、LlamaIndex、Agents SDK、AutoGen、CrewAI 等 Agent
框架的最小 Coding Agent。模型只负责生成文本和原生 tool calling；对话历史、
工具执行、参数校验、路径权限、循环终止、重试、SQLite 历史和事件记录均由本地代码实现。

当前版本是只读 MVP，唯一工具是 `read_file`。这让端到端闭环可以在加入 Shell、
文件写入和复杂上下文管理前先被可靠验证。

## 架构

```text
用户请求
  -> Agent.run（本地循环与终止策略）
  -> DeepSeekV4ProClient（OpenAI 兼容传输层）
  -> 模型返回原生 tool_calls
  -> ToolRegistry（本地 Schema 校验）
  -> read_file（本地路径沙箱）
  -> assistant/tool 状态事务写入本地 SQLite
  -> ContextBuilder 从 SQLite 投影模型消息
  -> 再次调用模型，直到得到最终文本
```

SQLite 是会话状态的事实来源；JSONL 只用于观察和排错。模型厂商的服务端会话
ID 不作为本地状态的事实来源。

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

运行只读任务：

```powershell
coding-agent --workspace . "读取 README.md，并说明这个项目的用途"
```

默认配置固定使用 `deepseek-v4-pro`、思考模式和 `high` reasoning effort。可以
按任务切换：

```powershell
coding-agent --reasoning-effort max --workspace . "分析项目架构"
coding-agent --no-thinking --workspace . "读取 README.md"
```

思考模式产生工具调用时，适配器会把 DeepSeek 返回的 `reasoning_content`
原样保存在 assistant 消息中，并在下一轮请求中带回。Agent 不解析或执行其中内容。

也可以使用模块入口：

```powershell
python -m coding_agent --workspace . "读取 README.md"
```

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
- 工具参数 Schema 校验
- SQLite 重启后上下文恢复
- 工具调用状态转换、幂等领取和中断恢复

## 当前边界

- CLI 支持使用 `--session-id` 继续历史会话；尚未提供会话列表命令
- 只有 `read_file`，不支持写文件和 Shell
- SQLite 保存完整本地历史，尚未实现 Token 预算、滑动窗口和摘要
- DeepSeek V4 Pro 当前通过 OpenAI 兼容的 Chat Completions 格式调用

下一阶段建议依次增加 `list_files`、`search_text`、补丁写入、受控 Shell，之后再
实现 Token 预算与结构化摘要。
