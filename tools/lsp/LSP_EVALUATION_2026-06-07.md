# Sounio LSP — evaluation & revival assessment (2026-06-07)

Scope: evaluate the existing Sounio LSP, determine whether precise diagnostics
are achievable, and — if a usable error position exists — land a minimal,
additive diagnostics proof in `tools/lsp/`. No changes to `self-hosted/**`,
`docs/`, or other branches. Branch: `claude/lsp-revival` (worktree
`/workspace/sounio-lsp-work`, off `011b9c3dca`).

Toolchain built for this evaluation: `./bin/souc self-hosted/compiler/main.sio
/tmp/lsp_mc.elf` → `BUILD_OK` (86 MB, Madáres v0.80.0).

---

## 1. Inventory — what already exists

| Artefact | State |
|---|---|
| `self-hosted/lsp/server.sio` (216 KB) | The real LSP: pure-Sounio JSON-RPC over stdio. Single-file. |
| `self-hosted/lsp/{hover,completions,goto_def,rename,diagnostics,…}.sio` | Feature modules (hover 139 KB, goto_def 112 KB, completions 94 KB, …). |
| `bin/sounio-lsp` (266 KB) | **Prebuilt server binary — runs.** |
| `bin/souc-linux-x86_64` (2.2 MB) | `mini_native`. The binary the server `fork+execve`s for checks. |
| `tools/lsp/sounio-lsp.sh` + `.deprecated` | Legacy bash launcher (kept). |
| `tools/lsp/parse_diagnostics.sh` | **Deprecated** bash+jq diagnostic parser. |
| `tools/lsp/test_protocol.sh`, `test_smoke.sh` | Protocol/smoke harnesses. |
| `scripts/ci/lsp_smoke_gate.sh` | CI gate around `test_protocol.sh`. |
| `tools/shared/diagnostic_schema.json` | `sounio.diagnostic.v1` spec — **has no producer** (see §4). |

### Does `bin/sounio-lsp` run? — YES.
`initialize` returns a full, honest capability set (hover, definition,
references, rename+prepare, documentSymbol, documentHighlight, formatting,
codeActions `[quickfix, source.fixAll, refactor.extract]`, semanticTokens,
codeLens, executeCommand `[sounio.run, sounio.check]`). serverInfo
`sounio-lsp 0.3.0`.

Driven end-to-end with `initialize` → `initialized` → `didOpen` (a file whose
call passes a `string` where `i64` is expected), the server **does publish a
diagnostic**:

```json
{"severity":1,"source":"sounio:type","code":"E001",
 "range":{"start":{"line":0,"character":0},"end":{"line":0,"character":0}},
 "message":"error[E001]: Type mismatch in call argument … at <main>:3"}
```

So message + code + severity are correct, but **every diagnostic lands at
0:0** — the line is lost. That is the single defect (§3).

### Does the smoke gate pass? — NO.
- Default path tries to fetch a pinned `v1.0.0-beta.5/souc-linux-x86_64`
  release asset → **HTTP 404**.
- With `SOUC_BIN=$(pwd)/bin/souc-linux-x86_64`, the gate fails at
  `building lsp binary`: **`FAIL: build of self-hosted/lsp/server.sio`**.
  `server.sio` no longer type-checks on the current toolchain — modular
  `--check` reports `expected [i8; 16] found [i64; 16]` (the `POS_ARG` buffer).
  The prebuilt `bin/sounio-lsp` works because it was built by an earlier
  compiler; the source can no longer be rebuilt in-tree.

---

## 2. The decisive question — does the compiler emit a usable error position?

The task gate (item 2): does **`mc --check`** emit line:column? **No.** It emits
a coarse byte span (`start` always 0), no column, and ignores `--json`. By the
literal gate that is **blocker #1, stop**. I proceeded because a usable *line*
is extractable from a *different* binary — `mini_native` — which item 3's
"(ou der pra extrair)" permits, but flagged: `mini_native` is the
operator's **retire target**. This was a deliberate judgement call, not an
assumption that the path is fine.

The **intended** format settles the fix direction. `tools/shared/diagnostic_schema.json`
(`sounio.diagnostic.v1`) specifies the real contract: a `Range` of
`Position{line, character}` (zero-based, LSP convention), emitted by
`souc check --json` and consumed by both LSP and MCP. So the compiler was
*designed* to emit line:col JSON; the `parse_diagnostics()` text-scrape of
`at line N` was always a stopgap. **The principled fix is compiler-side — make
`--check --json` actually emit `sounio.diagnostic.v1`** — not "teach the LSP to
chase mini_native's `at <func>:N`" (that is only an interim band-aid).

Findings:

**Two compilers, two diagnostic formats:**

| Compiler | Invocation | Error format | Position quality |
|---|---|---|---|
| `mini_native` (`bin/souc-linux-x86_64`, the binary the LSP execs) | `<src> <out>` | `error[E001]: <msg> at <func>:<LINE>` | **Real, correct line** — verified `:3` and `:5` against errors on source lines 3 and 5. |
| modular (`souc main.sio`, `/tmp/lsp_mc.elf`) | `--check <src>` | `error[Exxx] at <start>..<end>: <msg>` | **Coarse byte span**: `start` is **always `0`**, `end` is unreliable (`0..92` for a 95-byte file, `0..150` of 153 bytes, but `0..2` for a 40-byte file). The constant-0 `start` is what kills usability — not convertible to a precise location. |

Two further facts about the modular `--check` path:
- `--json` is **ignored** — `--check --json` and `check --json` emit the same
  plain text. There is no `sounio.diagnostic.v1` JSON producer.
- Its diagnostic is **newline-mangled**: `print_int` appends a newline, so one
  error prints across several physical lines (`error[E009\n] at 0\n..92\n: …`).
  Any line-oriented parser must re-join first.

**Verdict on position:** a usable position **exists, but only from
`mini_native`** (line-level, `at <func>:N`). The modular `--check` output gives
no usable location and is not JSON. The only good source is the compiler the
operator has directed to **retire**
(`memory: project_retire_legacy_compilers`). This is the strategic catch: LSP
diagnostics are revivable *today* on the legacy binary; on the modular compiler
they are not, until it emits real spans (or line:col) on a single line.

Because a usable position exists, the "stop" branch is **not** triggered — a
minimal proof was built (§3).

---

## 3. Blocker #1 (revival) and the minimal proof

**Blocker #1 — a format-string mismatch, not a missing position.**
`parse_diagnostics()` in `self-hosted/lsp/server.sio` scans for the literal
`" at line N"`. **No current compiler emits that string.** `mini_native` emits
`at <func>:N`; the modular compiler emits `at S..E`. So the parser matches the
message but never the number, and pins every diagnostic to line 0. Fixing the
LSP is a **one-locus change in `parse_diagnostics()`** (parse `at <func>:N`).
That file is in `self-hosted/**` and therefore **out of scope here** — recorded
as a follow-up dispatch below.

**Minimal proof landed in `tools/lsp/` (additive only):**

- `tools/lsp/diag_bridge.sh` — runs the compiler in check mode on one `.sio`
  file and emits an LSP `publishDiagnostics` params object. Pure bash (no jq,
  no python — does not regress the "pure-Sounio, no hybrid" posture; the
  deprecated `parse_diagnostics.sh` used jq). It auto-detects the compiler,
  handles **both** formats (`at <func>:N` → exact line; `at S..E` → byte-offset
  → line:col via the source), and re-joins the modular newline-mangling.
- `tools/lsp/test_diag_bridge.sh` — self-test. **Exercises the mini_native
  (line-accurate) path only**; the modular path is coarse (line-0) and
  manually verified, not asserted by the self-test.

Proof (real cases, reproducible):

```
$ tools/lsp/diag_bridge.sh err_on_line5.sio ./bin/souc-linux-x86_64
{"uri":"file://…","diagnostics":[{"severity":1,"source":"sounio:check",
 "code":"E001","range":{"start":{"line":4,"character":0}, …},
 "message":"error[E001]: Type mismatch in call argument …"}]}     # line 4 = src line 5 ✓

$ tools/lsp/test_diag_bridge.sh
[PASS] type-error-line5 (line=4)
[PASS] type-error-line3 (line=2)
[PASS] clean-file (no diagnostics)
diag_bridge self-test: all passed
```

The bridge puts the diagnostic on the **correct line** — exactly what the live
`bin/sounio-lsp` fails to do (0:0) — demonstrating that the defect is purely the
parser format and that position extraction works.

**Honesty bounds of the bridge:**
- Line-level only — no column (neither compiler emits a usable column; the
  mini_native form has none, the modular span start is unreliable).
- It depends on `mini_native` for correct lines (the retire-target binary).
- The modular path is intentionally kept but yields only a coarse line-0
  diagnostic — documented as inferior, not recommended.
- It checks single-file documents (matches the server's v0 single-buffer model).

---

## 4. README / capability claims vs. measured reality

| Claim (`tools/lsp/README.md`, `CLAUDE.md` §5) | Reality |
|---|---|
| `souc check --json` → `{"schema":"sounio.diagnostic.v1",…}` | **Overclaim.** `--json` is ignored by the modular compiler; no JSON producer exists. `diagnostic_schema.json` is a spec without an emitter. |
| `souc inspect --pos L:C` returns inferred signature | **Overclaim.** `inspect` is not a working subcommand on the modular binary — it just compiles the file and reports the type error. |
| Diagnostics "sourced from a real `souc check` subprocess" | True in spirit, but the parser format is stale → positions dropped (§3). |
| Capabilities advertised "match what the server answers" | The capability *set* is honest; the diagnostics it answers are mislocated. |

(Consistent with the standing memory note that `bin/souc` subcommands are a
known README overclaim.)

---

## 5. Recommendations (follow-up dispatches — not done here)

1. **Interim revive (cheap):** in `self-hosted/lsp/server.sio`, update
   `parse_diagnostics()` to read `at <func>:N` (and optionally the modular
   `at S..E` after re-joining). One locus; turns every diagnostic from 0:0 into
   the correct line. `diag_bridge.sh` is the working reference parser. **Caveat:
   this hard-couples the LSP to mini_native, the retire target — treat as a
   band-aid, not the destination (see #4, the per-`diagnostic_schema.json`
   fix).**
2. **Make `server.sio` rebuildable:** the smoke gate's `build of server.sio`
   step fails on `[i8;16]` vs `[i64;16]` at `POS_ARG`. **Triage first — this may
   be a real source bug *or* a modular-checker false-reject** (the checker has
   documented false-rejects); don't edit `server.sio` for a checker problem.
3. **Repair the smoke gate's souc resolution:** the pinned `beta.5` asset 404s;
   point it at an in-tree binary or `SOUC_BIN`.
4. **Compiler-side (the principled fix, per the existing contract):** make
   `souc check --json` actually emit `sounio.diagnostic.v1`
   (`tools/shared/diagnostic_schema.json` already specifies `Range` of
   `Position{line, character}`). This is what the LSP *and* MCP were designed to
   consume; it lets the LSP drop both the text-scrape and the dependency on the
   retire-target `mini_native`, and is the only path to precise *column*
   diagnostics.

## 6. What landed on `claude/lsp-revival` (additive, `tools/lsp/` only)

- `tools/lsp/diag_bridge.sh` — minimal compiler-diagnostics → LSP bridge.
- `tools/lsp/test_diag_bridge.sh` — self-test (3/3 pass).
- `tools/lsp/LSP_EVALUATION_2026-06-07.md` — this report.

Not pushed.
