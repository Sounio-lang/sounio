Madaros no longer segfaults (rc 139, no output) on a `main` or any other
function that calls an `f64` helper taking `&Struct` with the address of an
owned local, `f(&rec)` or `f(&!rec)`. The variance inline re-lowering bound
the helper's reference parameter to the register that holds `rec` itself, not
to its address, so `(*d).field[i]` indexed the struct's object handle as a raw
pointer; for a struct kept behind a small handle number that read address
3 + 8 i. It now binds the parameter to the address the call passes, as a
reference, and aliases its field variance to `rec` as `let r = &rec` does.
A raw-pointer parameter given `&rec` is no longer re-lowered. Typical victims:
`circ28_drug_value(&rec, i)` or `circ28_drug_params(&rec)` called directly in
`main` of a circ28 drug test, which worked only when moved into a helper.
