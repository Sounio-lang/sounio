# RECERT_RESULT — current-main recertification of the ModuleGraph semantic fidelity witness

> **Date completed:** 2026-07-19.
> **Recertifying agent:** opencode session in workspace `/tmp/sounio-modgraph-recert-20260719`.
> **Canonical source worktree (preserved untouched):** `/tmp/sounio-modgraph-witness-20260719` at commit `b0a8af6e96ec9e29dbe4565af0712017de0b05f4`.
> **Canonical receipt of record:** `module_graph_facade_vertical_20260719T132620Z.json`.

---

## 1. Git SHA exato do origin/main testado

| Item | Value |
|---|---|
| `origin/main` SHA at test time | `fcf7b521f1674793813f67f06e93ec898316a4f5` |
| Branch in recert worktree | `recert/modulegraph-facade-current-main-20260719` |
| Recert worktree path | `/tmp/sounio-modgraph-recert-20260719` |
| Commits ahead of canonical `b0a8af6e9` | 8 |
| Commits between (subjects) | #1176 epistemic materialization corrections; #1193 bigframe unique/drop_duplicates; #1196 midazolam CYP3A DDI E2E; #1197 bigframe value_counts; #1199 GPU novelty map; #1200 bigframe cumsum/count; #1202 bigframe cummax/cummin; #1204 synthetic non-associativity benchmark |

## 2. Madaros source-fresh identity

| Property | Value |
|---|---|
| Path | `artifacts/self-hosted/madaros` |
| SHA-256 | `0facbca77dc64dc0d6d652a0aa0294a5f261b4dc92db78bf3f359633b71fd40f` |
| Size | 99,129,162 bytes |
| mtime (UTC) | 2026-07-19T13:43:52Z |
| Build command | `make build-madaros` (resolves to `bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros`) |
| Build duration | 3 min 26 s (single-threaded, workspace pod) |
| Build output | `fns=10396 code=99125066 main=fn1439 patches=54259 gates[direct=28179 guarded=21357]` |
| Build seed | derived from `lean_single.sio` via `bin/souc-linux-x86_64` (script trace shown in build log) |
| Identity string | `Madaros v0.80.0 -- the Sounio self-hosted compiler` |

**Prebuilt vs source-fresh divergence:** the checked prebuilt `bin/madaros-linux-x86_64` (SHA-256 `11e7730f01f5382f1f8a5afc3599d7069b3d917f6972e6e47ffb57aa6bf4421e`) is byte-distinct from the source-fresh build. The prebuilt is the binary the canonical `20260719T132620Z.json` receipt was produced against; the recert uses only the source-fresh build.

## 3. Fixture SHA-256

| Fixture | SHA-256 | Identical to canonical? |
|---|---|---|
| `tests/compiler/module_graph_facade_vertical_witness/leaf.sio` | `2c938ae48b5296febe584ae3a1cda921605a94c6ec463690a589bcb46ecb9c71` | yes |
| `tests/compiler/module_graph_facade_vertical_witness/facade.sio` | `93c4c0216192fe59318c0cef5bbff7569ae37e3ee1ca405ce232db1073fff89a` | yes |
| `tests/compiler/module_graph_facade_vertical_witness/main.sio` | `f1e2960a77653d213b2fdb8979e2ba3e34387a93b3da6e43fd371b1f343ae2dd` | yes |
| `examples/projects/hello_pkg/src/greet.sio` (CONTROL) | unchanged from `origin/main` HEAD | n/a (canonical fixture) |
| `examples/projects/hello_pkg/src/main.sio` (CONTROL) | unchanged from `origin/main` HEAD | n/a (canonical fixture) |

## 4. Per-mutation matrix (8 cases × 2 runs)

For each mutation: `compile_rc`, `exec_rc`, `stdout`, `stderr` SHA-256, and `ELF` SHA-256. Both witness runs used the source-fresh Madaros.

| Case | compile_rc | exec_rc | stdout | stderr SHA-16 | ELF SHA-16 |
|---|---:|---:|---:|---|---|
| CONTROL_A greet=42 (run #1) | 0 | 0 | `42` | `e9d032500c44923e` | `553ea15fd8fb88da` |
| CONTROL_A greet=42 (run #2) | 0 | 0 | `42` | `e9d032500c44923e` | `553ea15fd8fb88da` |
| CONTROL_B greet=999 (run #1) | 0 | 0 | `42` ❌ | `e9d032500c44923e` | `4aaa5bf9bc018f87` |
| CONTROL_B greet=999 (run #2) | 0 | 0 | `42` ❌ | `e9d032500c44923e` | `4aaa5bf9bc018f87` |
| PROBE_A leaf=42 (run #1) | 0 | 0 | `42` | `071887e0b799d050` | `553ea15fd8fb88da` |
| PROBE_A leaf=42 (run #2) | 0 | 0 | `42` | `071887e0b799d050` | `553ea15fd8fb88da` |
| PROBE_B leaf=7 (run #1) | 0 | 0 | `42` ❌ | `071887e0b799d050` | `2514a484042546cd` |
| PROBE_B leaf=7 (run #2) | 0 | 0 | `42` ❌ | `071887e0b799d050` | `2514a484042546cd` |

Stderr SHA is identical between mutation A and mutation B within each closure — the compiler's stderr does not mention the mutated file's contents at all. The resolver validates the existence of `greet.sio`/`leaf.sio` (without them `error: visibility preflight failed` fires) but does not propagate their bodies into the IR.

Stderr tail (last line, identical for all 4 cases):
```
science-boundary: mode=advisory verdict=OK
```

The science boundary attests the program as `OK` while silently dropping the imported function bodies — operational success with false semantics.

## 5. Two consecutive receipts with identical classification

| Receipt | verdict | result_control.status | result_control.class | result_probe.status | result_probe.class |
|---|---|---|---|---|---|
| `module_graph_facade_vertical_20260719T134414Z.json` | BLOCKED | BLOCKED | `closure_not_consumed_runtime` | BLOCKED | `closure_not_consumed_runtime` |
| `module_graph_facade_vertical_20260719T134425Z.json` | BLOCKED | BLOCKED | `closure_not_consumed_runtime` | BLOCKED | `closure_not_consumed_runtime` |

Reproducibility: identical verdict, identical classes, identical ELF SHA-256 across both runs and across both commits.

## 6. Factual delta vs canonical `20260719T132620Z.json`

| Property | Canonical | Recert | Same? |
|---|---|---|---|
| Git commit | `b0a8af6e9` | `fcf7b521f` | **different** (8 commits ahead) |
| Madaros raw_elf SHA-256 | `11e7730f01...` (checked prebuilt) | `0facbca77d...` (source-fresh from current main) | **different** |
| Madaros raw_elf size | 99,097,305 bytes | 99,129,162 bytes | **different** (+31,857 bytes) |
| Madaros raw_elf mtime | 2026-07-19T12:38:00Z | 2026-07-19T13:43:52Z | **different** (rebuilt) |
| CONTROL_A ELF SHA | `553ea15fd8fb88da...` | `553ea15fd8fb88da...` | **identical** |
| CONTROL_B ELF SHA | `4aaa5bf9bc018f87...` | `4aaa5bf9bc018f87...` | **identical** |
| PROBE_A ELF SHA | `553ea15fd8fb88da...` | `553ea15fd8fb88da...` | **identical** |
| PROBE_B ELF SHA | `2514a484042546cd...` | `2514a484042546cd...` | **identical** |
| CONTROL_A stdout | `42` | `42` | identical |
| CONTROL_B stdout (expected `999`) | `42` | `42` | identical (still wrong) |
| PROBE_A stdout | `42` | `42` | identical |
| PROBE_B stdout (expected `7`) | `42` | `42` | identical (still wrong) |
| CONTROL classification | `closure_not_consumed_runtime` | `closure_not_consumed_runtime` | identical |
| PROBE classification | `closure_not_consumed_runtime` | `closure_not_consumed_runtime` | identical |
| Final verdict | `BLOCKED` | `BLOCKED` | identical |

**Headline finding:** despite rebuilding Madaros from a different commit and producing a byte-distinct compiler binary, the resulting ELFs for the witness fixtures are bit-identical to those produced by the canonical run. The silent corruption is **deterministic and unaffected by any commit on `origin/main` between `b0a8af6e9` and `fcf7b521f`** (8 commits including #1176 epistemic materialization corrections).

## 7. Final classification

# **BLOCKED**

`closure_not_consumed_runtime` for both CONTROL (2-module direct use) and PROBE (3-module `pub use` transitive re-export), reproduced deterministically across two consecutive runs against a source-fresh Madaros built from `origin/main@fcf7b521f`.

The default Madaros compiler on current `main` still produces silently wrong semantics for any cross-module import: compilation and execution complete with `rc=0`, but mutations in the body of imported functions do not reach the runtime stdout.

## 8. Exact commands executed

```bash
# Setup
cd /tmp/sounio-modgraph-witness-20260719
git fetch origin main
git worktree add -b recert/modulegraph-facade-current-main-20260719 \
  /tmp/sounio-modgraph-recert-20260719 origin/main

# Transport artifacts (script + fixtures + brief only)
cd /tmp/sounio-modgraph-recert-20260719
cp /tmp/sounio-modgraph-witness-20260719/scripts/dev/module_graph_facade_vertical_witness.sh \
   scripts/dev/module_graph_facade_vertical_witness.sh
mkdir -p tests/compiler/module_graph_facade_vertical_witness
cp /tmp/sounio-modgraph-witness-20260719/tests/compiler/module_graph_facade_vertical_witness/{leaf,facade,main}.sio \
   tests/compiler/module_graph_facade_vertical_witness/
cp /tmp/sounio-modgraph-witness-20260719/tests/compiler/module_graph_facade_vertical_witness/RECERT_BRIEF.md \
   tests/compiler/module_graph_facade_vertical_witness/RECERT_BRIEF.md
chmod +x scripts/dev/module_graph_facade_vertical_witness.sh

# Build Madaros source-fresh from origin/main@fcf7b521f
make build-madaros                       # 3min 26s on workspace pod

# Verify source-fresh binary identity
sha256sum artifacts/self-hosted/madaros  # 0facbca77d...
./bin/souc info                          # routes to artifacts/self-hosted/madaros

# Two consecutive witness runs
bash scripts/dev/module_graph_facade_vertical_witness.sh
# → MODULE_GRAPH_FACADE_WITNESS_BLOCKED receipt=...20260719T134414Z.json

bash scripts/dev/module_graph_facade_vertical_witness.sh
# → MODULE_GRAPH_FACADE_WITNESS_BLOCKED receipt=...20260719T134425Z.json

# Stderr capture (not serialised in receipt JSON) — 4 manual compiles via tee
# (script in this directory; see recert stderr-capture artefacts)

# Confirm canonical worktree untouched
git -C /tmp/sounio-modgraph-witness-20260719 status --short
git -C /tmp/sounio-modgraph-witness-20260719 diff --stat
ls -la /tmp/sounio-modgraph-witness-20260719/artifacts/witnesses/
```

## Boundary compliance

- [x] Zero changes under `self-hosted/compiler/`, `parser/`, `check/`, `ir/`, `native/` (recert worktree has 0 tracked files modified)
- [x] Zero changes to binaries, resolvers, Makefile, CI, governance
- [x] No claim that the merge is the cause — boundary strings preserved as `primary suspect ... pending differential IR hashes`
- [x] Runtime observation separated from architectural suspicion in all reported boundaries
- [x] Four historical receipts in canonical worktree untouched (`git diff --stat` empty; mtimes preserved)
- [x] No new roadmap opened
- [x] No promotion to CI attempted

## What this answers

> **O `main` de hoje ainda produz semântica silenciosamente errada?**

**Sim.** Em `origin/main@fcf7b521f` com Madaros source-fresh (`0facbca7...`), o closure `use facade::{leaf_value}` (e o closure canônico `examples/projects/hello_pkg/src/main.sio` com `use greet::{answer}`) ainda produzem um ELF que sempre imprime `42` independentemente do corpo da função importada. A perda ocorre após a construção da closure e antes da observação em runtime; o merge multi-módulo (`module_frontend.sio:4290-4722`) permanece o principal suspeito, ainda pendente de comparação diferencial do IR imediatamente antes e depois de `fn_remap`.

O agente principal está liberado para conduzir a próxima prova cirúrgica (`H_pre` / `H_post` / `H_driver`) e a correção de `module_frontend` sem risco de sobrepor este recert — worktree do recert é independente e o canonical está preservado.
