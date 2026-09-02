coding-agent

Git 仓库地址
https://github.com/GloryMCU/coding-agent

项目简介
这是一个不依赖 LangChain、LlamaIndex、Agents SDK、AutoGen、CrewAI 等 Agent 框架的编程智能体。模型只负责推理并生成原生 tool calling；对话管理、工具调度、参数校验、权限控制、本地执行、错误处理和循环终止均由本项目自行实现。

运行环境
Python 3.11+；DeepSeek API Key；Docker 或 Podman。命令执行默认要求本地已有可信沙箱镜像。

安装与运行（PowerShell）
1. python -m venv .venv
2. .venv\Scripts\Activate.ps1
3. python -m pip install -e ".[deepseek,tui]"
4. docker build --pull -f sandbox/Dockerfile -t coding-agent-sandbox:python sandbox
5. 设置环境变量 DEEPSEEK_API_KEY，并将 CODING_AGENT_SANDBOX_IMAGE 设置为 coding-agent-sandbox:python
6. 交互模式：coding-agent --workspace <目标项目目录>
   单次任务：coding-agent --workspace <目标项目目录> "你的编程任务"

特色功能
1. 自主执行闭环：模型根据工具结果持续读取、搜索、修改代码并运行命令，直至完成任务或触发终止条件。
2. 完整本地工具：支持文件读取、搜索与精确修改，只读 Git 查询、受控命令执行、网页检索和项目验证。
3. 安全边界：限制工作区路径，校验工具参数，按策略审批操作；命令默认在禁网、最小权限的容器中运行，不向子进程传递 API Key 等敏感变量。
4. 验证门禁：代码修改后记录待验证状态，结束前自动运行可用检查。available 模式在没有验证命令时以 partial 状态交接；strict 模式则阻止完成。实际检查失败时两种模式都会要求继续修复。
5. 持久会话：SQLite 保存完整对话和工具状态，长会话支持历史压缩及全文检索；程序中断后可以继续原会话。
6. 可靠终止：设置最大步骤数，并检测重复工具调用、调用周期和连续失败，避免无限循环；达到预算时输出明确的部分交接。
7. 双界面：提供带工具活动、Markdown 回答和审批弹窗的终端界面，同时支持适合脚本与 CI 的单次文本模式。

详细参数、安全设计及测试说明见仓库中的 README.md 和 SECURITY.md。
