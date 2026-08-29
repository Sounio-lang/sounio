<!-- docs:meta
topic_id: repo.docs.audit.madaros-native-multimodule-scale-2026-07-14
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-native-multimodule-scale-2026-07-14
-->

# Madaros v0.80.0 — native compile fails on large multi-module import graphs

**Date:** 2026-07-14
**Toolchain:** `./bin/souc` → Madaros v0.80.0 (default engine)
**Owner:** CODEX-2 (`self-hosted/` native/imported codegen)
**Class:** compiler-semantics · **Severity:** B1 (feature-rich stdlib modules can't native-compile)
Forensic dispatch per CLAUDE.md §8.

## Symptom

A program that `use`s a stdlib module whose **transitive** dependency graph is large fails native
compilation, even though it **type-checks cleanly**. Found via `prob::distributions`, which imports
`special::gamma`, `special::igamma`, `special::erf` (→ ~210 merged functions):

```sounio
use prob::distributions::*
fn main() -> i32 with IO, Mut, Div, Panic {
    print("m="); println(uniform_mean(0.0, 10.0))   // uniform_mean is trivial
    return 0
}
```
```
run_check_mode: verdict=0        # type-check OK
...
Merged IR: 210 functions
Native compilation failed: imported_simple_ir_emit_failed
module_native_driver: compact IR ELF write failed; rc=1     # compact-IR path fails
...                                                          # full-IR fallback also fails
error: multimodule native thin-link compilation failed      # rc=12
```
Both the compact-IR path and the full-IR fallback fail; `souc run` fails identically (it native-compiles).
The failing function (`uniform_mean`) is trivial — the blocker is the **graph size/link**, not the code.

## Not a small graph problem

Single self-contained modules native-compile fine (`epistemic::gum`, `units::lib`, `linalg::matnm`,
`special::erf`, `special::gamma`, `special::igamma` each import+run natively). It is the **combined**
graph (main + distributions + 3 special modules, ~210 fns) that exceeds the native path.

## Workaround (in use)

The **`lean_single` engine compiles the same program** and it runs correctly:
```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile prog.sio -o prog.elf
chmod +x prog.elf      # lean_single does not set the exec bit
./prog.elf             # runs; distribution values are correct
```
Verified: `prob::distributions` gives pdf(0)=0.398942, cdf(0)=0.5, cdf(1.96)=0.975002, exp_mean(2)=0.5,
unif_mean=5, pois_var(3)=3 — all textbook. `stdlib/prob/*` run-proof (`tests/stdlib/prob/test_prob_stdlib.sio`)
passes under lean_single.

## Impact

Any feature-rich stdlib module with a deep dependency graph (distributions, likely large stats/ODE/PBPK
compositions) cannot be native-compiled under the default Madaros engine — only under `lean_single`. This
is a real ceiling on real-world usability of composed stdlib code.

## Acceptance gate

`souc compile` (default Madaros) of the `prob::distributions` probe above produces a runnable ELF.

## Update 2026-07-22 — filed fail class **CLOSED** (Wave15C)

Measured on `origin/main` post into-acc (#1402) + specialized-list DCE (#1397) under stock
Madaros v0.80.0: the probe compiles (`Merged IR` ~73 after into-acc; historically ~210),
runs `m=5.000000`, and the textbook science graph / `tests/stdlib/prob/test_prob_stdlib.sio`
print `PROB_TEXTBOOK_OK` / `PROB_STDLIB_OK` under **default Madaros** (no lean_single pin).

Hard gate:

```bash
bash scripts/madaros_native_multimodule_scale_901_gate.sh
# → MADAROS_NATIVE_MULTIMODULE_SCALE_901_GATE_OK
```

Closeout audit: [`MADAROS_WAVE15C_ISSUE901_SCALE_CLOSEOUT_2026-07-22.md`](MADAROS_WAVE15C_ISSUE901_SCALE_CLOSEOUT_2026-07-22.md).
Related: #921 thin-link pairing closed Wave14D; compact opt-in residual classified there.

**Remaining (orthogonal to this fail class):** stats OLS multi-mod still red with `E019`
(method calls), not thin-link/scale; exclusive-ref / memory-wall fragile chains elsewhere.

## Next-Action

~~Fix the imported/native codegen so large merged IR graphs thin-link successfully~~ — **done**
for the filed `prob::distributions` scale class (gate above). Residual work is non-scale D3
(OLS E019, exclusive-ref chains) and optional compact-emitter rewrite (not fail-open).
