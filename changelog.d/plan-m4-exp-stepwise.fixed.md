M4 bis_v4 root cause, owned: the host backend does not evaluate the
nested Horner form in math::pure::exp op-by-op (1-ulp deviations at
boundary-sensitive inputs; 16/58 L1 e-cells, 12/48 att cells, first
diff at flat index 4 — the historical device-vs-host signature).
Dots, max, scale, sums and the weighted accumulation are proven
IEEE-faithful on the host (bit-exact vs an independent Python
oracle); only the deep nested mul-add chain diverges. exp is now
written stepwise in evaluation order (same literals, same order as
the paged_attn_f64.ptx device kernel, effect-free let-chain, no
signature change). Fixed host matches the stepwise oracle 48/48 on
the L1 rows. GPU bit-exact confirmation (devdump/need12) is queued
behind the current scheduler stall. Backlog: specify the backend FP
evaluation-order contract (order-faithful vs reassociation allowed).
