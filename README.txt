Git 仓库地址
https://github.com/GloryMCU/coding-agent

项目简介
coding-agent 是不依赖 LangChain、LlamaIndex 等 Agent 框架的轻量编程智能体。模型负责推理和原生 tool calling；本地 Python 代码管理工具、权限、重试、终止、持久化与审计，并通过 OpenAI 兼容接口调用 DeepSeek V4 Pro。

工作流程
用户通过命令行或 Textual 界面提交任务。Agent 调用模型、执行工具并回传结果，直至完成、受阻、中断或达到步骤上限。SQLite 保存会话状态，JSONL 用于排错和审计。

运行环境
需要 Python 3.11+、DeepSeek API Key，以及 Docker

安装步骤（PowerShell）
1. 克隆仓库并进入目录：

```powershell
git clone https://github.com/GloryMCU/coding-agent.git
Set-Location coding-agent
```

2. 创建、激活虚拟环境并升级 pip：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

3. 安装模型和终端界面依赖。开发者可改用第二条命令：

```powershell
python -m pip install -e ".[deepseek,tui]"
# python -m pip install -e ".[dev]"
```

4. 配置模型。程序不会自动加载 .env，请在当前进程或密钥系统中设置：

```powershell
$env:DEEPSEEK_API_KEY = "your-api-key"
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
```

5. 构建可信沙箱镜像并指定运行时：

```powershell
docker build --pull -f sandbox/Dockerfile -t coding-agent-sandbox:python sandbox
$env:CODING_AGENT_SANDBOX_IMAGE = "coding-agent-sandbox:python"
$env:CODING_AGENT_SANDBOX_RUNTIME = "docker"
docker image inspect coding-agent-sandbox:python
```

6. 启动交互界面或执行单次任务：

```powershell
coding-agent --workspace .
```

核心能力
1. 文件与代码：支持遵循 .gitignore 的文件枚举、glob 筛选、文本或正则搜索、精确补丁、原子写入和受保护的单文件删除。路径穿越、绝对路径、越界链接以及受保护目录会被拦截。
2. Git 与命令：git_status、git_diff、git_log 仅开放有界读取；run_command 使用结构化 argv、程序白名单、最小环境和超时控制，不把模型输出隐式交给 shell。
3. 沙箱与审批：命令默认在禁网、只读根文件系统、最小权限的容器中运行，只挂载工作区且不传递 API Key 等敏感变量。approval-mode 支持 workspace、ask、deny、allow，规则冲突按拒绝优先处理。
4. 自动验证：修改后自动运行测试、构建和格式检查。available 模式在未配置检查时以 partial 状态交接；strict 模式会阻止完成，检查失败则继续修复。
5. 长会话：默认使用约 128K Token 的上下文软预算。超出预算后按完整用户轮次保留近期内容，将更早历史压缩为可重建的结构化摘要；原始消息不会删除。FTS5 全文索引支持中文片段和代码标识符检索，可在压缩后召回相关历史。
6. 可靠执行：模型请求遇到超时、断流、限流或服务端错误时按指数退避重试，不重复已经完成的本地工具调用。系统还能识别重复调用、短周期循环和连续失败；预算耗尽或无进展时输出已完成事项、未完成事项、验证状态和下一步。
7. 双界面与恢复：Textual TUI 提供多轮输入、Markdown 回答、工具活动和审批弹窗；纯文本模式适合脚本和 CI。历史默认保存在 .coding-agent/history.sqlite3，可通过 session-id 继续中断或部分完成的会话。

更多参数、安全说明和测试范围见 README.md 与 SECURITY.md。
