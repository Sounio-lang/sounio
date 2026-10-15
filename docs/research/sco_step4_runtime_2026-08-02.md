<!-- docs:meta
topic_id: repo.docs.research.sco-step4-runtime-2026-08-02
authority: historical
audience: researchers
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sco-step4-runtime-2026-08-02
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# SCO Step IV — Runtime discharge path

> Reconciliation boundary (2026-10-12): this is a historical research proposal, not a theorem or a certificate that SCO is satisfied. V1/V2 are exact runtime sentinels. The bounded V3 now uses an imported `Unobserved<i64>` payload: source `aef6c07081514e3dc1d8adaf2716a9f0df60fcdb`, Foundry job13624, fresh Madaros SHA256 `aeaf8126524ed0362f7bfaaea86980792f139c321c674a5fef9b5da9ba12c65c` passed exact stdout `1`/`7001` and the matching missing-Observe negative in both check and compile with E036. This is native i64 coverage, not generic or floating Unobserved certification; later source changes require their own validation. A labeled enforce/ablation strategy change is strategy-selection coverage only; scientific non-vacuity requires an observable entitlement failure or legality fault and remains pending. Historical “discharged” wording below does not supersede these limits. No Python research exemption exists under ADR009.


**Historical status (superseded by the reconciliation boundary above):** **discharged on this worktree** — `SCO_STEP4_RUNTIME_GATE_OK` (2026-08-02)
**Date:** 2026-08-02\
**Depends on:** Step I–III\
**Runtime ELF:** `artifacts/self-hosted/madaros-sco` (~101 MB, rebuilt from this branch’s `lower.sio`)\

---

## 1. Goal

Step III installed \(E(\tau)\) in **source** (`lower.sio`).\
Step IV puts that install into a **running Madaros binary** and keeps:

1. SCO corpus green under the **rebuilt** compiler.\
2. Ablation switch `SOUNIO_SCO_ABLATION=1` present in source (skips scientific strategy elevation).\
3. Explicit honesty: without rebuild, Step III cannot claim runtime.

---

## 2. Rebuild

```bash
cd /tmp/sounio-novel-compiler-research
bash scripts/ci/build_modular_madaros.sh \
  artifacts/self-hosted/madaros-sco
```

Then point the public entrypoint at that ELF (or set `SOUNIO_SOUC_ENGINE` / wrapper policy for this worktree only).

Gate helper: `scripts/ci/sco_step4_runtime_gate.sh`

---

## 3. Ablation (clause 6 — partial)

| Mode | Env | Expected strategy for `with NonAssoc` ordinary return |
|------|-----|--------------------------------------------------------|
| Enforce (default) | unset / not `1` | `PRECISION_PRESERVING` (2) |
| Ablation | `SOUNIO_SCO_ABLATION=1` | fall through to return-type logic → typically `STANDARD` (0) |

Source marker: `SCO_ETAU_ABLATION_ENV` in `self-hosted/ir/lower.sio`.

**Measured non-vacuity after rebuild:**\
compile the same `with NonAssoc` unit twice (enforce vs ablation) and confirm strategy differs *or* that ablation is at least wired (trace/probe). Full IR dump probe is optional; corpus must stay green under enforce.

---

## 4. Acceptance

| Check | Command / evidence |
|-------|-------------------|
| Install still gated | `sco_etau_install_gate.sh` |
| Rebuild produced ELF | `artifacts/self-hosted/madaros-sco` executable, source-tree and ELF SHA256 receipt verified |
| Corpus under rebuilt SOUC | `SCO_CORPUS_GATE_OK` with `SOUC=.../madaros-sco` |
| Ablation marker | grep `SCO_ETAU_ABLATION_ENV` |

---

## 5. Historical strategy probe (selection coverage; non-vacuity pending)

```bash
bash scripts/ci/sco_strategy_probe_gate.sh
# → SCO_STRATEGY_PROBE_GATE_OK
# enforce:  SCO_STRATEGY=2  (PRECISION_PRESERVING)
# ablation: SCO_STRATEGY=0  (STANDARD)
```

Probe unit: `examples/sco_corpus/strategy_probe/probe_nonassoc.sio`\
Requires compile (not check-only) + rebuilt Madaros with `SCO_ETAU_STRATEGY_TRACE`.

## 6. Still open (not this step)

- Lean lemma for non-contraction.\
- mfi multi-module scientific peels.\
- Full EqSat activation (2a).\
- First-class CLI flag for strategy dump (today: env `SOUNIO_SCO_STRATEGY_TRACE=1`).

End of Step IV note.
