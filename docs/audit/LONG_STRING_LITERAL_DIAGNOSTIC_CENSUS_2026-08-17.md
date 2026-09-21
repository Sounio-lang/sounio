<!-- docs:meta
topic_id: repo.docs.audit.long-string-literal-diagnostic-census-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.long-string-literal-diagnostic-census-2026-08-17
-->

# Long string-literal census — diagnostic honesty impact

**Date:** 2026-08-17  
**Branch context:** post #1784 (silent truncation refuse)  
**Question:** Of string literals with content length > 127 in `self-hosted/` and `stdlib/`, how many are user-facing compiler diagnostics, and were any being cut before the Name/arena fix?

**Pre-fix emission rule (measured on Madaros, #1784):**
- content length ≤ 127 → printed in full
- content length ≥ 128 → printed as **127 characters**, rc=0, warning only (or silent after arena clamp)
- Root clamps: `Name.buf` 128 (incl. quotes at parse), `ir_normalize_string_literal_name` `out.len < 128`, `ir_arena_put_name` `len > 128 → 128`, native-v2 rodata from arena name

**Method:** Scan all `*.sio` under `self-hosted/` and `stdlib/` for double-quoted string literals; expand common escapes (`\n`, `\t`, …); keep content length > 127; classify by path + call-site + content shape.

**Count note:** Earlier hand-wave was “324 / 14 files”. This pass finds **328** literals in **16** files. The bulk is PTX/data in one GPU file, not diagnostics. Kind labels are conservative: only `print`/`println` sites with error/help/note/ZD-message shape (or the `code == NNN` message table) count as **diagnostic**.

## By kind

| Kind | Count | Content >127 (would be cut pre-fix) |
|---|---:|---:|
| `data` | 277 | 277 |
| `diagnostic` | 33 | 33 |
| `data_runtime_print` | 11 | 11 |
| `internal_debug` | 4 | 4 |
| `usage_cli` | 2 | 2 |
| `test_fixture` | 1 | 1 |
| **TOTAL** | **328** | **328** |

### Kind definitions

| Kind | Meaning |
|---|---|
| `diagnostic` | User-facing compiler error/help/note text emitted via `print`/`println` from check/IR/compiler |
| `internal_debug` | Compiler receipts, self-test PASS lines, optimize/transaction traces |
| `usage_cli` | `Usage: …` CLI help |
| `test_fixture` | Embedded oracle blobs (e.g. PTX text in tests) |
| `data` | Non-printed payloads (templates, embedded source, GPU tables) |
| `data_runtime_print` | Long `println` from stdlib/science scenarios (user-visible app output, not compiler diagnostics) |
| `doc_comment` | Inside `//` comments only |

## Answer on diagnostics

**User-facing diagnostic literals longer than 127 characters: `33` — not zero.**

**All 33 were subject to pre-fix truncation** whenever that literal was lowered through the Madaros string path (content ≥ 128 → emit 127). These are not dead strings: they sit on live `print(...)` paths in the typechecker error-code table, help/note printers, IR lowering fatals, and one preflight note.

### Why this taints the audit trail

Diagnostic text is itself a string literal **inside the compiler sources**. When the seed builds Madaros (or Madaros rebuilds), those literals pass through the same Name → normalize → arena → rodata path that truncated user programs. So a shipped Madaros binary could only carry the **first 127 characters** of each of these messages in `.rodata`. Every user (and every CI log) that hit those codes saw a **quietly shortened** explanation — the missing tail included the load-bearing clauses (e.g. *why* ZD is required, the full help list of body-less builtins, closure-escape guidance).

Aggregate: if each of the 33 messages were shown once at the old cap, **1488 characters of diagnostic text would be missing** from those emissions.

### Diagnostics by file

| File | Count | Max len | Max cut |
|---|---:|---:|---:|
| `self-hosted/check/check.sio` | 29 | 333 | 206 |
| `self-hosted/ir/lower.sio` | 2 | 153 | 26 |
| `self-hosted/compiler/main.sio` | 1 | 129 | 2 |
| `self-hosted/compiler/lean_single.sio` | 1 | 128 | 1 |

### Full diagnostic inventory (sorted by length)

| Len | Cut | Location | Preview |
|---:|---:|---|---|
| 333 | 206 | `self-hosted/check/check.sio:13100` | `   \|\n   = help: only these body-less names are implemented: print, print_int, print_char, print_f64, get_arg, get_arg_count, str_len, str_char_at, str_eq, str_s` |
| 312 | 185 | `self-hosted/check/check.sio:12806` | `Composable<T> requires ZD effect: ZD-orthogonal model merge (G8) relies on the 4D annihilator complement — two models can only be composed without mutual pertur` |
| 285 | 158 | `self-hosted/check/check.sio:12807` | `Audited<T> requires ZD and Witness effects: witness-bearing surgery (G9) must emit a machine-checkable Lean derivation of the ZD-kernel action. `with ZD` alone ` |
| 285 | 158 | `self-hosted/check/check.sio:12808` | `Revivable<T> requires ZD and Temporal effects: time-bounded reversible surgery (G10) needs both the ZD annihilator algebra and an explicit Temporal window in wh` |
| 248 | 121 | `self-hosted/check/check.sio:12809` | `Interpretable<T> requires ZD effect: the 168-class canonical basis is the projective ZD-graph of sedenions. Without the ZD effect the compiler cannot guarantee ` |
| 212 | 85 | `self-hosted/check/check.sio:12803` | `ExactlyPrivate<T> requires ZD effect: exact data-contribution removal is only possible via sedenion annihilation. Differential privacy (epsilon-bounded) does no` |
| 211 | 84 | `self-hosted/check/check.sio:12804` | `Editable<T> requires ZD effect: locality-bounded edits (G5) require ZD-based kernel annihilation. Ripple-free editing is not achievable via rank-1 or mask-based` |
| 205 | 78 | `self-hosted/check/check.sio:12805` | `CapabilityGated<T> requires ZD effect: capability containment requires exact ZD annihilation of the gated subspace. Activation patching and masking leave residu` |
| 201 | 74 | `self-hosted/check/check.sio:12802` | `Forgettable type requires ZD effect: exact annihilation is only possible via zero-divisor algebra (sedenions). Division algebras cannot satisfy this guarantee. ` |
| 195 | 68 | `self-hosted/check/check.sio:12319` | ` must provide aggregate_divergence plus latent_state_divergence, causal_decomposition_divergence, intervention_response_divergence, parameter_role_divergence, a` |
| 163 | 36 | `self-hosted/check/check.sio:12140` | `   = help: attach latent-state, causal-decomposition, intervention-response, parameter-role, and structural explanation diagnostics before projecting disagreeme` |
| 156 | 29 | `self-hosted/check/check.sio:13277` | `   = note: the self-hosted compiler now tracks selected Contest/Robust metadata, but malformed wrapper arity and unknown level/scope tags still fail closed\n` |
| 155 | 28 | `self-hosted/check/check.sio:12988` | `   \|\n   = help: use Model<I, O>, Contest<T, Family, Policy>, Robust<T, Level, Scope>, or the legacy single-argument migration forms with the correct arity\n` |
| 153 | 26 | `self-hosted/ir/lower.sio:16771` | `error: native-v2 capturing closure literals are not yet supported as values; bind and call directly, use a named function, or use a noncapturing closure\n` |
| 147 | 20 | `self-hosted/check/check.sio:13046` | `   \|\n   = help: ensure the causal DAG has observed confounders that block all backdoor paths from cause to effect, or use an instrumental variable\n` |
| 144 | 17 | `self-hosted/check/check.sio:4151` | `   = note: functions with `with MultiTest` must record an explicit multiple-testing correction once they perform more than one statistical test\n` |
| 144 | 17 | `self-hosted/check/check.sio:12378` | `   = note: functions with `with MultiTest` must record an explicit multiple-testing correction once they perform more than one statistical test\n` |
| 139 | 12 | `self-hosted/ir/lower.sio:16337` | `error: capturing closure cannot be used as a value (escaping closures are not yet supported); only a direct call of the closure is allowed\n` |
| 138 | 11 | `self-hosted/check/check.sio:10815` | `   = note: the current parser/checker slice preserves disagreement structure but does not infer diagnostic metrics from raw contest terms\n` |
| 138 | 11 | `self-hosted/check/check.sio:12994` | `   \|\n   = help: ensure the contest was realized through an explicit Contest<T, Family, Policy> annotation before calling witness builtins\n` |
| 137 | 10 | `self-hosted/check/check.sio:11011` | `   \|\n   = help: use the result of prove_robust(contest_value, PolicyName) directly, instead of a bare Robust<T, Level, Scope> annotation\n` |
| 137 | 10 | `self-hosted/check/check.sio:13027` | `   \|\n   = help: counterexample { ... } must provide model_name, subgroup_name, metric_name, observed_value, threshold_value, and summary\n` |
| 136 | 9 | `self-hosted/check/check.sio:11065` | `   \|\n   = help: use the result of prove_robust(contest_value, PolicyName) directly instead of a bare Robust<T, Level, Scope> annotation\n` |
| 136 | 9 | `self-hosted/check/check.sio:13039` | `   \|\n   = help: use the result of prove_robust(contest_value, PolicyName) directly instead of a bare Robust<T, Level, Scope> annotation\n` |
| 135 | 8 | `self-hosted/check/check.sio:11261` | `   \|\n   = help: use the result of defer_action(value, evidence, DeferralPolicyName) directly, instead of a bare Deferred<T> annotation\n` |
| 131 | 4 | `self-hosted/check/check.sio:13445` | `   = note: the effect was INFERRED from the body (spec 7.2.1); callers outside this module read the declaration, not the inference\n` |
| 130 | 3 | `self-hosted/check/check.sio:12395` | `   = note: Hypothesis currently applies this penalty as an advisory warning; automatic Knowledge<T> confidence scaling is pending\n` |
| 130 | 3 | `self-hosted/check/check.sio:13088` | `   \|\n   = help: add `with Epistemic` to your function signature, or use `acknowledge(k, reason)` to explicitly accept uncertainty\n` |
| 129 | 2 | `self-hosted/check/check.sio:10852` | `   = help: narrow the admissible model family with explicit justification, or strengthen the policy diagnostics before promotion\n` |
| 129 | 2 | `self-hosted/check/check.sio:13307` | `   = note: posterior weights must preserve the declared model set instead of laundering disagreement through partial weight maps\n` |
| 129 | 2 | `self-hosted/compiler/main.sio:5310` | `note: preflight covers module-closure parsing AND the full multi-module typecheck; see the diagnostics above for the actual cause` |
| 128 | 1 | `self-hosted/check/check.sio:13000` | `   \|\n   = help: attach a mechanistic report before calling disagreement(contest_value) or mechanistic_divergence(contest_value)\n` |
| 128 | 1 | `self-hosted/compiler/lean_single.sio:17444` | `error: Hessian AD over a non-associative algebra — `reassociate: fano_selective` required (formal/NonAssocHessian.lean) at line ` |

### Highest-impact cluster: ZD effect messages (codes 200–207)

Eight primary messages in `self-hosted/check/check.sio` (the error-code → text table) explain why epistemic wrapper types require the `ZD` effect. Lengths 201–312. Under the old cap each lost **74–185 characters** — typically the clause that distinguishes exact annihilation from DP/masking/rank-1 approximations. Those tails are exactly the scientific content of the diagnostic.

Largest single diagnostic: **333 characters** at `check.sio:13100` — help text listing implemented body-less builtins (code 219). Cut by **206** characters pre-fix; users would see only the first ~127 chars of the allowed-name list.

### Non-diagnostic bulk

- **`self-hosted/gpu/nvidia_bare.sio`**: 272 long literals classified `data` (embedded PTX/tables) — dominates the census, not user diagnostics.
- **stdlib science prints**: 11 `data_runtime_print` (e.g. darwin PBPK scenario narratives) — app output honesty issue if run under pre-fix Madaros, but not compiler E-codes.
- **usage_cli**: 2 — CLI usage strings also truncated if printed via the same path.
- **internal_debug**: 4 — compiler self-receipts / PASS lines.

## What #1784 changes

After source-built Madaros with Name/arena/rodata capacity 384 and E220 refuse:

- These 33 diagnostics fit (max 333 < 382 content with quotes under 384 Name).
- Anything still past capacity **refuses to compile** rather than shipping a shortened lie.
- **Historical logs and pre-#1784 binaries remain untrustworthy** for the tails of these messages; re-run affected scenarios on a rebuilt compiler before citing diagnostic wording as evidence.

## Reproduction

```bash
# inventory (this census)
python3 -c '...'  # see session artifacts .scratch/long_literal_classification.json

# pre-fix truncation witness (still true of bin/madaros-linux-x86_64 until rebuild)
MADAROS_RAW_BIN=./bin/madaros-linux-x86_64 bash scripts/ci/string_literal_truncation_gate.sh
# → FAIL on 128/129/200 print lengths

# post-fix
bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros
MADAROS_RAW_BIN=./artifacts/self-hosted/madaros bash scripts/ci/string_literal_truncation_gate.sh
# → PASS
```

## Bottom line

| Question | Answer |
|---|---|
| Long literals total (`self-hosted/` + `stdlib/`) | **328** in **16** files |
| User-facing **diagnostics** among them | **33** |
| Of those, cut pre-fix? | **33 / 33 — all of them** |
| Zero? | **No** |
| Trust impact | Every emission of these messages on a pre-fix Madaros (or any binary built through the 128-cap path) was a **partial diagnostic**. Audit trails that quote those messages are incomplete unless re-verified on a rebuilt compiler. |

Machine-readable companion: `.scratch/long_literal_classification.json` (session-local).
