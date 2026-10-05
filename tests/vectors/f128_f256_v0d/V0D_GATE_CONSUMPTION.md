<!-- docs:meta
topic_id: repo.tests.vectors.f128-f256-v0d.gate-consumption
authority: repo_only
audience: internal+codex
last_validated: 2026-09-06
validated_by: cursor-agent
source_of_truth: scripts/ci/madaros_f128_f256_v0d_softfloat_gate.sh
-->

# V0-D gate consumption of grok-cli1 arith_hard corpus

**Gate:** `bash scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0d`  
**Corpus:** `arith_hard_f128.jsonl` (53) + `arith_hard_f256.jsonl` (50), MPFR 4.2.1 RNE.

## Engine

| Path | Engine |
|---|---|
| Scaffold probes (descriptor / payload) | **lean_single seed ELF** |
| V0-D green | **Compiler-owned softfloat limb routines** bit-identical to MPFR `result` on the hard corpus — not host libm, not widen-f64 |

**Consumer (required):** `scripts/dev/ws_g_v0d_softfloat_corpus_runner.py` driving `scripts/dev/softfloat_limb.py` (limb bigint, RNE). Oracle exit 0 requires runner bit-identity — file presence alone is insufficient.

## Correctness bar (beyond plausible values)

Green is **not** “ops return something near the right magnitude on easy inputs.”

A consumer must:

1. Bit-identity `result.limbs` for **every** hard row under RNE.
2. Treat `MUST_TRAP_IDS` as mandatory (halfway / sticky / cancel / rump).
3. For `family=rump`, compare to `result` (MPFR), **never** `f64_result`. `f*_rump1988` is evaluated at extended precision then packed once (matches MPFR EXT corpus), not stepwise binaryN AST (which loses the residual).

## Corpus note (sticky sub wire fix)

Rows `f128_arith_0010` / `f256_arith_0063` store **IEEE results on encoded limbs**. The generator’s exact MPFR inputs `1+ulp/2+tiny` are not binaryN-representable; `encode(a)` rounded before the op, so the old `result` was unreachable by any correct wire softfloat. Results updated 2026-09-06; MD5 pins in the oracle.

## Widen-f64 shortcut: which entries catch it

Classification from `scripts/dev/ws_g_v0d_softfloat_corpus_oracle.py` (re-run with the gate).

### CATCH (wrong if implementation is decode→f64→op→widen)

| Family | Why it catches |
|---|---|
| `halfway_tie_even` | Tie at binaryN ulp/2; f64 cannot see the midpoint structure |
| `sticky_bit` | Bits below half-ulp set sticky; f64 loses them |
| `catastrophic_cancel` | Guard digits past f64 mantissa; residual wrong |
| `rump` (`f*_rump1988`) | Ill-conditioned poly; host double ~1e21 wrong; `f64_result` ≠ `result` (`f64_bits_differ`) |

### MISS / weak alone (may still pass under widen-f64)

| Family | Why weak alone |
|---|---|
| `sqrt_hard` | Some exact squares (`sqrt(4)=2`) match f64 then widen |
| `overflow_underflow` | Overflow-to-inf can agree with f64 path |

These rows are still required for **full** bit-identity green; they are not sufficient as the only tests.

## Green receipt

```
PASS f128_f256_v0d_softfloat ops=add/sub/mul/div/cmp/sqrt/rump limb_routines=green const_fold=exact rounded=ieee754-2019 must_trap=… catch_widen_f64=… miss_widen_f64=… bit_identity=all_hard_rows
PASS madaros_f128_f256_ladder_gate stage=v0d
```
