<!-- docs:meta
topic_id: repo.docs.audit.lean-single-nan-semantics-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-nan-semantics-2026-07-05
-->

# Audit: lean_single f64 NaN/Inf comparison semantics deviate from IEEE-754

- Date: 2026-07-05
- Engine: `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run` (canonical fixed-point ELF)
- Context: EISA E1 (`stdlib/eisa/format.sio`) needed a runtime finiteness
  predicate for `.eisax` loader validation (error 10: non-finite constant).
  The E1 subagent reported that synthesising Inf/NaN at runtime was
  "hang-sensitive" and dropped the error-10 witness. This audit measured the
  actual semantics and produced a portable detector; the witness was restored.

## Measured behaviour (probes, 2026-07-05)

Values synthesised through memory loads with runtime indices, so the e-graph
cannot constant-fold them (`buf[1] * buf[2]` → +Inf, `z / z` → NaN,
`inf - inf` → NaN).

| Expression | IEEE-754 | lean_single (measured) |
|---|---|---|
| `nan == nan` | false | **true** |
| `nan != nan` | true | **false** |
| `nan == 0.0` | false | **true** |
| `nan == 1.0` | false | **true** |
| `nan < 1.0` | false | **true** |
| `nan > 1.0` | false | false |
| `nan >= 0.0` | false | false |
| `inf > DBL_MAX` | true | true |
| `inf <= DBL_MAX` | false | false |
| `1.0 / 0.0` (runtime) | +Inf | +Inf (ordered comparisons correct) |
| `DBL_MAX * 2.0` | +Inf | +Inf |

Interpretation: the equality lowering treats the *unordered* comparison result
as equal (consistent with using the ZF flag of `ucomisd` without a parity
check), so **NaN compares equal to every operand**, and `<`/`<=` inherit the
same flag confusion (NaN "less than" anything). Ordered comparisons involving
Inf are correct. Arithmetic itself produces genuine IEEE Inf/NaN bit patterns;
only the comparison lowering is unsound.

Consequences for the classic idioms:

- `x != x` — constant **false** on this engine, even for genuine NaN.
- `(x - x) == 0.0` — **true for every x** (and with the e-graph fold
  `x - x → 0` it is also compile-time true), so it is not a finiteness test.
- `x <= DBL_MAX` — unreliable for NaN (NaN `<=` anything is true).

## Portable detector (dual-semantics)

Adopted in `stdlib/eisa/format.sio` (`is_finite_f64`, `f64_is_nan`) and
`stdlib/eisa/isa.sio` (`is_finite_f64`, `is_nan_f64`):

```
fn is_finite_f64(x: f64) -> bool {
    if x != x { return false }              // fires under real IEEE
    if x == 0.0 && x == 1.0 { return false } // fires under measured semantics
    let ax = if x < 0.0 { 0.0 - x } else { x }
    ax <= 1.7976931348623157e308             // rejects +/-Inf under both
}
```

Rationale: no real number equals both 0.0 and 1.0, so the second clause is
inert under IEEE and detects NaN exactly under the measured semantics.
Verified by probe on the corpus {1.5, 0.0, 1.0, -2.5, DBL_MAX, 1e-300, +Inf,
-Inf, 0/0, Inf-Inf, -NaN}: 11/11 correct.

NaN sign is **not observable** on this engine (ordered comparisons with NaN
are unsound), so `f64_parts`/`f64_decompose` canonicalise NaN to
`s0 e2047 m1` in receipts.

## Blocker classification

- Class: compiler codegen (comparison lowering, `lean_single` x86 backend);
  likely `ucomisd` flag mapping without unordered/parity handling.
- Severity: MEDIUM for general code (silent logic inversion around NaN);
  LOW for EISA (worked around with the dual-semantics detector).
- Not yet a named blocker; candidate `BLK-LEANSINGLE-NAN-CMP`. The Madaros
  lane was not measured (blocked by BLK-MADAROS-CHECK-HANG-STRLIB for these
  modules).
- Next action: forensic dispatch against
  `self-hosted/native/codegen_x86_linux.sio` f64 comparison emission; add a
  conformance witness for IEEE unordered comparisons.

## Evidence

Probes run 2026-07-05 (files in `/tmp`, reproducible from the tables above):
`probe_nan_cmp.sio`, `probe_nan_cmp2.sio`, `probe_finite3..6.sio`,
`probe_div0.sio`, `probe_detector.sio`. Post-fix suite status on lean_single:

```
ALL PASS: eisa isa P1 P2 P3 P4 P5
ALL PASS: eisa core W1 W2 W3 W4 W5
ALL PASS: eisax F1 F2 F3 F4 F5 F6 F7
```

(F7 is the restored error-10 witness: runtime Inf and NaN constants are both
rejected with code 10 before the hash check.)
