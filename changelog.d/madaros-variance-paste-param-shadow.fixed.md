Madaros no longer segfaults (SIGSEGV, stdout sometimes lost) on a function
that passes a `&Struct` parameter on to a helper whose own parameter has the
same name. The variance inline re-lowering bound the helper's parameter
first and then looked up the caller's argument by that same name, so it
found the fresh, untyped binding. It then lowered `(*d).field[i]` as a raw
read of the caller's stack slot, whose content depended on which `f64`
values `main` had computed earlier. It now reads the caller local's struct
type and reference-ness before binding anything. Typical victims: record
accessors such as `circ28_drug_theta(d)` calling `circ28_drug_value(d, i)`.
lean_single was never affected.
