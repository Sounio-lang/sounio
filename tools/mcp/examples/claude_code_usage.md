# Claude Code Usage

Install the server in editable mode from the Sounio repository root:

```bash
pip install -e tools/mcp
```

Start Claude Code with the local stdio MCP server:

```bash
claude --mcp-server sounio=python:-m:sounio_mcp.server
```

Then use the tools in the loop:

```text
Use @sounio_check on examples/hello.sio.
If diagnostics are returned, revise the Sounio source and call @sounio_check again.
When valid, call @sounio_run and inspect stdout/stderr/exit_code.
```

Useful resources:

```text
Read sounio://errors/E070 before fixing kernel/effect diagnostics.
Read sounio://stdlib/stats before generating statistical analysis examples.
Read sounio://stdlib/clinical before touching clinical-pathway examples.
```

Subagent recipe:

```text
You are the Sounio repair subagent.
For every proposed edit:
1. Call sounio_check on the target .sio file.
2. Copy all diagnostics into your reasoning context.
3. Make the smallest Sounio-native edit that addresses the diagnostics.
4. Re-run sounio_check.
5. Stop after success or 10 iterations.
6. If success, optionally call sounio_run for executable examples.

Respect Sounio syntax:
- no Rust macros
- no semicolons
- use var, not let mut
- use &!, not &mut
- declare effect rows with with IO, Mut, Panic, Div, GPU, Prob, Observe, etc.
- preserve Knowledge<T> semantics instead of unwrapping evidence silently
```
