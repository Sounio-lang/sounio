# CC-1 Iterative Convergence Log

Per the CC-1 charter (`docs/...` Sounio LSP brief), every substantive
decision goes through ≥ 3 draft → adversarial-review → revise cycles
before landing. This log records those cycles for the 14-day W1–W4
arc, anchored to the commit hashes that ended each cycle.

Three protocol-level decisions converged across the session. Each was
re-considered in light of empirical results from the previous cycle.

## Cycle 1 — How the LSP reaches the type-checker

### Draft 1.0 (plan as approved)

In-process Sounio library: extend `self-hosted/compiler/api.sio`
with `pub fn check_source(...)`, `inspect_at(...)`, etc. The LSP
imports them and links into one binary. Fastest hover/diagnostics, no
IPC, reuses existing `self-hosted/lsp/*.sio` modules.

Captured in `cc-1-claude-iridescent-diffie.md` plan, section
"Prerequisite compiler work, step 3".

### Adversarial review 1.0

- The 13 `self-hosted/lsp/*.sio` modules were never linked into
  anything. All 13 fail standalone `souc check` (`error: no main`).
  That's expected for library files, but it means there is no public
  `pub fn` boundary today.
- Exposing a stable API surface requires invasive changes to
  `ir_pipeline_v2.sio` whose internals are not designed as a library.
- The lean_single self-host fixed point is fragile. Any new pub fn
  must not break stage1==stage2==stage3 bit-identicality.

### Draft 1.1

Subprocess approach: shell out to `souc check` via `fork(57)` +
`execve(59)` + capture stderr to a temp file. Reuse existing souc
binary as-is; no compiler API changes; LSP stays decoupled.

Trade-off: per-keystroke diagnostics latency = process fork cost,
which on Linux is ~1 ms — invisible to the user. Worth it to keep
the compiler-side surface frozen.

### Final (committed `514c56a3`)

Subprocess wins. Diagnostics flow via fork+execve+wait4; stderr
parsed for `error: <msg> at line <N>` patterns. The library API
plan remains as a future option, recorded as plan compiler-prereq #3.

## Cycle 2 — How `read_line()` segfaults break the LSP

### Draft 2.0

Initial bring-up (commit `748acf61`) used the `read_line()` builtin
to read framed LSP headers. server.sio compiles clean. Smoke test
crashes with SIGSEGV inside `read_line()` after the first
`Content-Length:` frame.

### Adversarial review 2.0

Bisected with two near-identical sources differing only in string
literal length (`artifacts/lsp_bringup/sb6.sio` vs `sb7.sio`). One
crashes, the other doesn't. The "long string" hypothesis fit at
first: claimed it was a codegen / BSS-layout bug.

### Counter-evidence

Re-bisected on input rather than source. `printf 'hi\n' | sb6`
passes; `printf 'hi' | sb6` segfaults. The bug is runtime, not
codegen. The compiler emits `read_line()` with two short-jump
displacements (`jle 0x08`, `jne 0x02`) that don't actually skip
the right number of bytes:
- `jne 0x02` jumps mid-`dec rax` when the read doesn't end in '\n'.
  Result: PC lands on `0xC8`, which decodes as `enter`, corrupts
  RBP/RSP, segfaults.
- `jle 0x08` lands mid-`movsx` on EOF — latent until the LSP fed
  it Content-Length-framed input that didn't end in a newline.

### Final (committed `f22f1a6f`)

Patched lean_single.sio `emit_read_line_x86`:
- `jle 0x08` → `jle 0x13` (skip the entire 19-byte newline-strip
  block).
- `jne 0x02` → `jne 0x03` (skip the 3-byte `dec rax`).

stage1==stage2==stage3 fixed point holds at md5 `a44826…` after the
patch. `bin/souc-linux-x86_64` rebuilt in place; previous binary
preserved at `bin/souc-linux-x86_64.prev`.

Lesson: an isolated source-level repro doesn't isolate the input
dimension. The next time a binary "depends on string content,"
also vary the *runtime input*, not just the source.

## Cycle 3 — Publication path

### Draft 3.0

OpenVSX publication under `sounio` publisher (matches the `publisher`
field that was in `package.json` from before this session).
`npx ovsx create-namespace sounio -p $PAT && npx ovsx publish ...`.

### Adversarial review 3.0

The `package.json` `repository.url` points at
`github.com/sounio-lang/sounio` — the GitHub org is `sounio-lang`,
not `sounio`. OpenVSX uses the repo URL for namespace
auto-verification. A namespace called `sounio` cannot auto-link to
a repo under `sounio-lang`. Result: the upload lands in the
"inactive-pending" state and never becomes visible.

Confirmed empirically:

```
❌ Extension sounio.sounio-vscode 1.1.0 is already published,
   but currently isn't active and therefore not visible.
```

### Draft 3.1

Republish under the existing `sounio-lang` namespace (which already
had v0.2.0 on the registry from 2026-03-24). Bump to 1.1.1 since
1.1.0 is now occupied by the inactive entry under `sounio`. Same
code; flip `publisher: "sounio"` → `"sounio-lang"`, repackage,
publish.

### Final (committed `df303044`)

`sounio-lang.sounio-vscode@1.1.1` live at
https://open-vsx.org/extension/sounio-lang/sounio-vscode

API confirmed indexed at 2026-05-16T22:40:00Z. Download URL
returned 302 redirect to the .vsix.

Lesson: a publisher field that pre-exists in tree is not
necessarily a publisher the user actually controls. Verify against
the established namespace on the target registry before claiming a
new one.

## Smaller convergence moments

- **`let X: string = "..."` constants don't survive `as i64` casts.**
  Found while wiring diagnostics. First impl used module-level
  `let TMP_DOC_PATH: string = "/tmp/...sio"` and passed that to
  `write_file()` / `execve()`. write_file returned -1 silently. Bisect
  pointed at the constant binding; inlining the literal at each call
  site fixed it. Recorded in code comments; the underlying compiler
  quirk is left as future work.
- **`match` is a reserved keyword in Sounio.** server.sio v0 named a
  loop-local `var match: i64`; `souc check` reported "if condition
  must be bool" on a totally unrelated line. Renamed to `ok`; root
  cause was the lexer eating `match` as a keyword token. Adversarial
  review: when the compiler diagnostic line number looks wrong,
  treat it as a token-classification problem first.
- **Background `tail -f` Monitor patterns.** While polling the
  OpenVSX index endpoint for `1.1.1`, an `until ...; do sleep 5;
  done` Bash `run_in_background` worked. The earlier 30-second
  chained sleep was blocked by the harness; the `until` pattern is
  the documented escape hatch.

## What's deliberately out of this log

- Routine compile/run/check loops with no design content.
- Other agents' commits to `lean_single.sio` (effect-scan bytewise
  matcher; ZD gate hardening). Those were independent tracks.
- The R.2.1 `park_miller.sio` commit (`111a2ea5`) landed inside this
  branch from a parallel agent and is not part of CC-1.
