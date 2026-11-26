EPI-006, step 2. The struct-literal field `Knowledge { epsilon: x }` (or
`confidence: x`) is a stated confidence, the same slot the lowering writes.
The checker used to read it as an error bound. With `epsilon: 0.01`, six
run-pass tests met `epsilon < 0.05` on values whose GUM variance was 4.04,
72.36 or 76.6. Those annotations now read `ε >= 0.01`, and their printed values
are unchanged. An unannotated `let k = Knowledge(v, e=c)` is now typed
Knowledge in lowering. Before, `k.epsilon` was resolved by name, and a struct
with its own `epsilon` field made the read return 0.0 instead of c.
