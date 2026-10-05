<!-- docs:meta
topic_id: repo.docs.audit.madaros-epistemic-payload-gate-2026-08-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-epistemic-payload-gate-2026-08-20
-->

# Madaros `Epistemic(N)` payload gate receipt

Date: 2026-08-20

## Source boundary

- Consumer branch base: `0e523d0643320b7d5a55f3574e4a65339b702ebc`
- Required payload branch: `origin/feat/effect-payload` at the same commit
- Comparison target at measurement time: `origin/main` at
  `67aa2aec127020122ff961480b83b36c09e91432`
- Final rebase target: `origin/main` at
  `e021ce8a3` (revalidated 2026-08-21 before this receipt-only update)
- The four witnesses under `tests/compiler/epistemic_payload_gate/` are
  byte-identical after replacing the integer in `Epistemic(N)` with `N`.

The consumer stores the parsed floor in `FnSig.epistemic_min_confidence` in
both function-signature collection spines and both function-type lowerers. The
`Knowledge` constructor keeps its literal confidence in the semantic type.
While checking a function with a parameterised `Epistemic(N)` effect, Madaros
emits E215 when a literal confidence is below the floor, and treats a
non-literal confidence as unknown and therefore insufficient.

## Before: measured engine divergence

Command, using the checked-in Madaros prebuilt only as the pre-fix control:

```bash
MADAROS_RAW_BIN="$PWD/bin/madaros-linux-x86_64" \
SOUNIO_WITNESS_GLOB='tests/compiler/epistemic_payload_gate/*.sio' \
bash scripts/ci/madaros_epistemic_payload_gate.sh
```

| N | Madaros prebuilt | lean_single |
|---:|---|---|
| 400 | accept, rc 0, ELF yes | accept, rc 0, ELF yes |
| 401 | **incorrectly accepts**, rc 0, ELF yes | reject, rc 1, E215 text, no ELF |
| 950 | **incorrectly accepts**, rc 0, ELF yes | reject, rc 1, E215 text, no ELF |
| 999 | **incorrectly accepts**, rc 0, ELF yes | reject, rc 1, E215 text, no ELF |

## Non-vacuous positive control

The in-place comparison was temporarily changed from:

```sio
if minimum > 0 && confidence_milli < minimum
```

to unconditional rejection whenever a parameterised floor was present:

```sio
if minimum > 0
```

The compiler was rebuilt from that live source on Slurm. The previously green
Madaros N=400 cell then failed with rc 1, no ELF, and:

```text
error[E215] in epistemic_payload_gate/n400::gated_value at 0..146: EpistemicComplete violation
```

In the same run lean_single still accepted N=400. This distinguishes a live
consumer path from a gate that merely reports green without observing it. The
temporary comparison was then restored; it is not present in the committed
diff.

## After: current-source side-by-side oracle

Exact command:

```bash
env SLURM_CONF=/tmp/slurm-direct.conf \
  SOUNIO_WITNESS_GLOB='tests/compiler/epistemic_payload_gate/*.sio' \
  bash scripts/dev/souc-build-remote.sh \
    --partition gpu-orangefs \
    --node gpuorangefs-r770-proxmox \
    --cpus 32 \
    --gate witness
```

Build receipt:

```text
REMOTE: host=gpuorangefs-r770-proxmox nproc=32 unpacked=196M
REMOTE: build rc=0 elapsed=227s
REMOTE: elf bytes=100642004
```

| Engine | N | Expected | rc | ELF | Diagnostic | Verdict |
|---|---:|---|---:|---|---|---|
| Madaros, rebuilt from source | 400 | accept | 0 | yes | n/a | PASS |
| Madaros, rebuilt from source | 401 | reject | 1 | no | E215 text present | PASS |
| Madaros, rebuilt from source | 950 | reject | 1 | no | E215 text present | PASS |
| Madaros, rebuilt from source | 999 | reject | 1 | no | E215 text present | PASS |
| lean_single seed | 400 | accept | 0 | yes | n/a | PASS |
| lean_single seed | 401 | reject | 1 | no | EpistemicComplete text present | PASS |
| lean_single seed | 950 | reject | 1 | no | EpistemicComplete text present | PASS |
| lean_single seed | 999 | reject | 1 | no | EpistemicComplete text present | PASS |

Final gate output:

```text
epistemic-payload-gate: PASS
REMOTE: witness_gate rc=0
```

## Post-merge addendum: negative payloads are not modeled

Measured on 2026-08-21 after #2048 merged as `2ca46ce2b`. The modular parser
uses `EffectRef.payload == -1` to mean that an effect carries no payload. A
negative literal is tokenized as `Minus`, `IntLit`, so
`parse_effect_payload` does not take its single-`IntLit` branch. It skips the
balanced parentheses and returns the same `-1` sentinel. Consequently,
`with Epistemic(-1)` and bare `with Epistemic` are indistinguishable after
parsing.

This was checked with one source whose only relevant signature is:

```sio
fn reserved_negative_payload() -> i64 with Epistemic(-1) { 1 }
fn main() -> i64 with Epistemic { reserved_negative_payload() }
```

Both engines accepted it and emitted an ELF:

| Engine | rc | ELF | Named negative-payload diagnostic |
|---|---:|---|---|
| checked-in Madaros | 0 | yes | no |
| `SOUNIO_SOUC_ENGINE=lean_single` | 0 | yes | no |

**Known boundary:** negative effect payloads are not modeled; `-1` is the
absence sentinel and collides with them. The #2048 consumer activates a floor
only for positive payloads, so it does not reinterpret the colliding `-1` as a
valid confidence floor. A future implementation must either reserve negative
payload syntax with the same named refusal in both engines, or replace the
sentinel with an option-like representation. It must not make `-1` a valid
floor.

## Claim boundary

This receipt proves the four-rung literal-constructor oracle. It does not claim
that Madaros now implements lean_single's complete general
`EXPR_CONF` algebra, cross-call `FN_EFF_CONF` certificates, or confidence
propagation through every expression kind. Those broader propagation surfaces
remain separate work and are not needed to distinguish or close the measured
payload-consumer divergence above.
