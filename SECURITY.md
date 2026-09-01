# Security model

`coding-agent` treats model output, repository contents, tool output, and compacted
history as untrusted data. The local Python process and the human approval policy
are trusted.

## Enforced controls

- File tools reject absolute paths, workspace traversal, protected agent/Git state,
  and unsafe symbolic-link targets.
- The default `workspace` policy automatically allows bounded workspace writes and
  sandboxed commands, while the dedicated `delete_file` tool and unsandboxed execution
  require approval. Sandboxed commands can still modify or delete files inside the
  writable workspace. File-tool mutations use atomic writes, and immutable path
  controls remain enforced regardless of approval mode.
- The CLI fails closed unless commands can run in a pre-existing trusted Docker or
  Podman image. The container has no network, a read-only root filesystem, no added
  capabilities or privilege escalation, a non-root numeric user, bounded resources,
  and only the workspace mounted from the host. Agent and Git state are masked or
  mounted read-only.
- Commands additionally use structured argv, an executable allowlist, a
  workspace-local cwd, bounded output and duration, a minimal environment, and a
  dedicated process group.
- Read-only Git operations disable pagers, credential prompts, hooks, and external
  diff commands where applicable.
- Audit JSONL redacts common credential fields and patterns and omits model reasoning
  by default.
- Compacted repository and tool data is never inserted with the `system` role.
- Truncated or content-filtered model responses are not accepted as successful final
  answers.
- The repository verification policy is protected from file-tool mutation and is
  mounted read-only into command containers. Interrupted mutation calls conservatively
  retain the mandatory verification requirement after process recovery.

## Residual risks

`--sandbox off` and the default Python embedding API use the legacy host runner. In
that mode, an approved Python, PowerShell, package-manager, build, or test process can
use the permissions of the agent user outside the workspace. The executable allowlist
does not change that fact. Never use that mode for an untrusted repository.

Containers reduce host exposure but are not equivalent to a VM. Trust remains in the
container engine, OCI runtime, host kernel, and administrator-selected image. A Docker
daemon commonly has powerful host privileges; prefer rootless Docker/Podman and a
disposable VM or dedicated host for hostile workloads. The workspace remains writable
inside the container, so an approved command can alter its source files. Review the
complete argv shown by the approval prompt and use the stricter `ask` mode for
  untrusted input. Task-scoped approvals are cleared before the next top-level task;
  permission-rule conflicts resolve to the most restrictive result (`deny > ask >
  allow`).

Images are never pulled automatically. Build or obtain them through a controlled
supply-chain process, pin their base layers by digest where practical, scan them, and
set `CODING_AGENT_SANDBOX_IMAGE` (or `--sandbox-image`) explicitly.

Redaction is defense in depth, not a secret-management system. Do not paste secrets
into prompts or store them in the repository. Delete compromised logs and rotate any
credential that may have been exposed.

The durable `history.sqlite3` database intentionally stores complete conversation and
tool state so a session can resume after restart. Unlike the diagnostic JSONL log, it
is not content-redacted or encrypted. Protect the workspace and database with OS file
permissions, and delete the database when its retained context is no longer needed.

## Security regression expectations

Every security change should include a test that fails before the fix. At minimum,
the suite must continue to cover path traversal, symbolic links, protected metadata,
permission denial with zero side effects, command environment scrubbing, container
argument ordering and isolation flags, fail-closed runtime discovery, timeout
termination, audit redaction, history role separation, and abnormal model completion.

Run the complete offline suite with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```
