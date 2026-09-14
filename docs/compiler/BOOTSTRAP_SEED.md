<!-- docs:meta
topic_id: repo.docs.compiler.bootstrap-seed
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.bootstrap-seed
-->

# Bootstrap seed (`lean_single.sio`)

Moved out of `docs/compiler/KNOWN_LIMITATIONS.md` on 2026-09-11 (KL-0). The
seed's open defects are ledger rows KL-9 and KL-16 in that file; this
document is the standing policy and the #1494 record.

**Status:** bootstrap seed and escape hatch. Not a bug; a maturity-stage reality that contributors must know about before editing compiler logic.

### What the situation actually is

The preserved bootstrap compiler binary (`bin/souc-linux-x86_64`, also available through `SOUNIO_SOUC_ENGINE=lean_single`) is produced from a **single self-hosted source file**:

- `self-hosted/compiler/lean_single.sio`

The modular directory layout most readers expect —

- `self-hosted/lexer/`
- `self-hosted/parser/`
- `self-hosted/check/`
- `self-hosted/types/`
- `self-hosted/ir/`
- `self-hosted/native/`

— is now the source for the checked x86-64 Madaros prebuilt (`bin/madaros-linux-x86_64`). The 2-stage bootstrap recipe below remains the legacy seed/escape-hatch path and uses `lean_single.sio` exclusively:

```bash
./bin/souc-linux-x86_64 self-hosted/compiler/lean_single.sio /tmp/souc-stage1
/tmp/souc-stage1 self-hosted/compiler/lean_single.sio /tmp/souc-stage2
cp /tmp/souc-stage2 bin/souc-linux-x86_64
```

### Implication for contributors

Changes to the default user-facing compiler path must land in the modular tree and be proven through the Madaros gates before refreshing `bin/madaros-linux-x86_64`. Changes needed for the legacy seed or explicit `SOUNIO_SOUC_ENGINE=lean_single` path still need the corresponding `lean_single.sio` update.

Examples of this pattern in recent history:

- 2026-04-20 — surgical type gates (`ExactlyPrivate`, `Editable`, `CapabilityGated`) and error codes `E201`–`E203` added to `lean_single.sio`; modular files updated in parallel.
- 2026-04-29 — extended surgical type gates (`Composable`, `Audited`, `Revivable`, `Interpretable`), new effect bit-flags (`Witness=32768`, `Temporal=65536`, `Learn=131072`), and error codes `E204`–`E207` added to `lean_single.sio`; 2-stage bootstrap executed; `bin/souc-linux-x86_64` rebuilt.

### Risk of silent divergence

Because the modular compiler and legacy seed are no longer the same source file, reviewers should check which lane a PR affects. Default Madaros changes need the modular-source build plus named Madaros gates; legacy-seed changes still need a `lean_single.sio` bootstrap proof.

### Parity status and planned resolution

1. **Parity harness is workflow-reachable since KL-10 (2026-09-11).**
   `scripts/ci/engine_parity_gate.sh` compares compile/run outcomes and stdout
   for both engines against `tests/engine_parity_baseline.txt` (1007
   classified rows). `.github/workflows/engine-parity-nightly.yml` runs it
   nightly and on dispatch against a Madaros built from the checked-out
   source (the gate refuses a compiler older than any `self-hosted/` file, so
   a committed prebuilt can never certify parity in a fresh checkout), and
   runs `scripts/ci/lean_single_fixed_point_gate.sh` on every pull request
   that touches the seed, the baseline or the two gates. `ci.yml` still runs
   the narrower `epsilon_engine_parity_gate.sh` on every PR.
2. **Bootstrap retirement remains long term.** Retire `lean_single.sio` as an
   escape hatch only after the modular compiler has sufficient fixed-point and
   parity evidence.

Until that lands, treat the modular tree as the source for the default Madaros prebuilt and `lean_single.sio` as the seed/escape-hatch source.

## Bootstrap seed: imported-module typecheck errors are non-fatal (#1494)

**Status: documented and frozen as a known property of the seed. Not fixed. Owner decision, 2026-07-27.**

The `lean_single` seed (`self-hosted/compiler/lean_single.sio`) tolerates a
typecheck error inside a module reached via `import`: the error is reported,
and the build continues and still emits an ELF. The same construct, compiled
as a standalone program instead of via an imported module, correctly refuses
to emit (exit 2, no ELF) — the tolerance is specific to the imported-module
path.

The mechanism under the `CONVERGENCE FIX` block in `lean_single.sio` only rewinds
and stubs an imported function with a clean `return 0` placeholder when that
function accumulates **more than 10** typecheck errors
(`fn_err_count > 10 && fn_is_import`). Below that threshold,
the function's already-emitted, partially-broken codegen is left in the
binary as-is — not stubbed, not refused.

This is not a hypothetical severity concern: #1494 was filed while
root-causing #1471, where exactly this tolerance let a typecheck error that
the checker had already reported (an unresolvable assignment place) reach
codegen anyway. The resulting store had no valid address; inside Madaros's
~8 MB `Checker` struct it landed in mapped memory instead of faulting, and
silently corrupted name resolution — surfacing as three unrelated spurious
`error[E137]` diagnostics in a different subsystem, on a different source
file. Diagnosing that took roughly seven full rebuild cycles before the
tolerated error (printed inline, scrolled past in the build log) was
identified as the actual cause. As of #1494's filing, the current `main`
build log already carries three such tolerated errors, from `lower.sio`,
`imports.sio`, and `opt_cleanup.sio`.

#1494 poses this as a policy decision among three options: (1) make
imported-module typecheck errors fatal outright (correct in principle, but
would fail the build on the three currently-tolerated errors, which would
need repairing first, with blast radius measured against
`tests/madaros_corpus_baseline.txt`); (2) keep them non-fatal but refuse to
emit code for the specific construct that failed, rather than emitting
something with no valid address; (3), stated in the issue as *the minimum*
acceptable outcome — if the tolerance is load-bearing for the bootstrap
chain (plausible, given three such errors already sit in the current
build), say so explicitly in the source and here, **and** make the build
print a prominent end-of-build summary of every tolerated error, rather
than leaving it inline where it is lost.

**What this entry does, and does not, close.** This documents the behaviour
and its severity — the "say so explicitly" half of option 3. It does **not**
implement the prominent end-of-build summary that #1494 names as the other
half of the minimum acceptable outcome, and does not choose between options
1/2/3 as a permanent policy. The seed (`lean_single`) is the frozen
bootstrap artifact whose guarantee is bit-identical fixed-point
self-regeneration, not per-construct correctness enforcement — that
guarantee lives in Madaros, which type-checks through a different,
modular checker (`self-hosted/check/`) and is not affected by this specific
mechanism. Changing `lean_single.sio` risks perturbing that fixed point and
was judged out of scope for this measurability pass; #1494 stays open for
whoever picks option 1, 2, or the remainder of option 3.
