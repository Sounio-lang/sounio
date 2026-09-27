<!-- docs:meta
topic_id: repo.docs.audit.ns-n1-grok46-mathreview-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.ns-n1-grok46-mathreview-2026-08-23
-->

# NS N1 — grok-4.6 math-review (xai, reasoning_effort=high)

Date: 2026-08-23. Target: self-hosted/check/noise_sets.sio (NS refinement invariant).
Model: grok-4.6 via api.x.ai. Bug caught: fail-open at ns_bit(63)/OOR interning a 0-mask;
fixed (ns_singleton: never intern a 0-mask -> unknown). Probe: (1 as i64)<<63 is NONZERO on
Madaros v0.80.0, so the concrete fail-open did not fire; the guard is defensive (any shift-UB).
Control tests/run-pass/ns_handle_validity.sio: 9/9 PASS incl src=63.

## Full review

```
[OK] handle encoding: -1 = top, 0 = empty, h>0 ↔ masks[h-1]
  intern stores at idx=count and returns idx+1; deref/valid use h-1 / h<=count; first intern is handle 1 ↔ masks[0]; last live is handle=count ↔ masks[count-1].

[OK] ns_unknown / ns_empty / ns_cap / ns_table_new
  0-1 = -1; empty is 0; cap 4096 matches [i64;4096]; new table is all-zero with count=0.

[OK] ns_intern: 0→empty; hit→existing handle; miss→append; full→-1
  Search is before the cap check (duplicates still found when full). Write-then-count++ cannot expose a zero slot. Full→top over-approximates (fail-closed).

[OK] ns_singleton: src∉[0,63] → unknown; else intern(1<<src)
  Range is rejected before ns_bit; does not intern a collapsed 0-mask for OOR ids.

[OK] ns_handle_valid iff h=0 or h∈[1,count]
  Negatives (incl. -1) are not “valid interned”. Out-of-range positives are invalid. Matches deref’s live range: (h-1)<count.

[OK] ns_union: top absorbs; invalid⁺ → top; empty is identity only after those checks
  Order is load-bearing: union(invalid, 0) and union(0, invalid) return -1, not the other operand. Returning the leftover handle would drop unknown sources (understate union = fail-open). ma|mb then intern is exact ∪ or top if the table is full.

[OK] ns_disjoint: -1 and invalid⁺ → false; then empty→true; else (ma&mb)==0
  Matches the certificate contract. Validity/top checks precede the empty shortcut, so disjoint(0, bogus) is false, not true. After those guards, deref cannot see an OOB slot.

[OK] empty ∩ X = ∅, so disjoint(0, valid X)=true is sound for ep_add
  Deterministic values carry no source identity; certifying independence cannot hide a shared interned bit.

[TIGHTENABLE] ns_bit: ((1 as i64)<<src) for src=63 (sign bit)
  Soundness of the whole domain needs 1<<src injective and nonzero on {0,…,63}. If the shift is UB, wraps to 0, or saturates, singleton(63) becomes handle 0.
  Concrete fail-open if ns_bit(63)==0: h=ns_singleton(t,63)=0; ns_disjoint(t,h,h)==true (a source certified independent of itself); ns_union(t,h,S)=S (source 63 dropped).
  Correction: build the mask in unsigned 64, or special-case src==63 to the min-signed bit pattern; never feed a 0-mask through intern for a real source.

[TIGHTENABLE] ns_bit OOR returns 0 (empty mask), not a distinguished failure
  Private today and guarded by ns_singleton, but intern(ns_bit(x)) for x∉[0,63] is empty, hence disjoint from every valid handle (fail-open). Return a Option/sentinel and do not intern.

[TIGHTENABLE] ns_deref maps top, invalid⁺, and empty all to mask 0
  Confuses ⊤ with ⊥. Current union/disjoint special-case first, so they are safe. Any N2 caller that derefs then ORs/ANDs without the same guards understates sets and can return disjoint=true.
  Correction: deref unknown/invalid to a fail-closed path (no mask); never hand back 0 for ⊤.

[TIGHTENABLE] zero-value handle is empty (most permissive), not unknown
  Default/memset/unseeded i64 is 0 → ns_disjoint(t,0,S)=true. Fail-open once N2 wires ep_add. Safer encoding: 0=top, and reserve another sentinel for empty.

[TIGHTENABLE] ns_union reimplements validity as `h > count` instead of ns_handle_valid
  Equivalent now (after h<0 checks) but will drift. One helper.

[TIGHTENABLE] cap 4096 is duplicated (array type vs ns_cap)
  cap>4096 would OOB-read a “valid” handle and feed a garbage mask to (ma&mb)==0 (fail-open). Single named constant; optionally also require h<=ns_cap() in ns_handle_valid.

[OVERREACH] comment: “empty is disjoint from anything”
  Code correctly returns false for unknown and invalid⁺ (fail-closed). Empty is disjoint from every *valid known* set, not from top.

[OK] no residual fail-open in union/disjoint *as written*, assuming (i) 1<<src is a distinct nonzero bit for src=0..63, (ii) count≤4096, (iii) only this module mutates the table
  Wrongly returning disjoint=true or a strict subset-union does not occur on those paths: invalid/top cannot take the empty shortcut; intern never stores 0; bitwise ∪/∩ are on masks, not handles.
```
