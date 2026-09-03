<!-- docs:meta
topic_id: repo.docs.audit.madaros-root2-enum-fncount-2026-06-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-root2-enum-fncount-2026-06-20
-->

# Madaros Root 2 — enum-variant use crash = corrupted module fn_count (2026-06-20)

Deep core-dump forensics on the Root-2 null-deref (the crash that `ulimit -s unlimited`
does **not** fix). Cleanest repro:
```
enum E { A, B }
fn main() -> i64 { let e = E::A   5 }     # SIGSEGV even under unlimited stack
```

## Mechanism (read from the core, not inferred)
Crash at RIP `0x3ebe4f2`: a loop `for counter in 0..len { el = data[counter]; deref el }`.
From the core at the fault:
- `data` (`rcx`) = `0xf55523f4000` → an array of pointers: **`data[0]` → bytes `"main"`**,
  `data[1..]` → pointers to **zero-filled** 8 KB buffers (uninitialized function entries).
  This is **`module.functions`**.
- The loop's length bound is read via double-indirection `(*(*param1).data).field@0x10`
  (i.e. `(*self.module).fn_count`, since `module` is a `Box`). Its value in the core is
  **`16859291983880`** — a heap address, i.e. **garbage**, not a small count.
- `counter` (`[rbp-0x20]`) reached **2048 = `IR_MAX_FUNCS`**: with a garbage (huge) length
  the loop never terminates via the bound, so it scans all 2048 slots, hits an
  uninitialized slot holding the int `1`, uses it as a pointer, and `mov 0x0(%rdx)` faults.

**Root: `module.fn_count` (or the length field read by the scan) is corrupted to a garbage
value** on the enum-variant-use path, so a `module.functions` scan
(`find_or_add_fn_id` / `lookup_fn_by_name`, both iterate `self.module.fn_count`) runs off
the valid entries into uninitialized memory.

## Why enum *use* (not declaration)
`E::A` lowers as `ExprPath` (`lower.sio:6939`): if `lookup_variant_discriminant("A")` does
**not** resolve, the code falls through to `lookup_local` → `lookup_fn_by_name`, which scans
`module.functions`. With a corrupted `fn_count`, that scan crashes. (Enum *declaration* never
triggers a function-list scan, so it is fine.) Whether the corruption is (a) the variant
table not being seeded into the body lowerer so the fall-through fires, or (b) a by-value
`Lowerer`/`IrModule` copy corrupting `fn_count`, or (c) a wrong field-offset read of the
count, is the remaining unknown — all three are documented lean_single miscompile classes.

## Status / what's needed to close it
The exact corrupting source statement cannot be named from the stripped binary, and the
built-in `SOUNIO_LOWER_BODY_TRACE` diagnostics are **dead in the prebuilt** because `read_env`
returns `""` (the procfs `file_size`=0 bug; fixed only on `m2/effect-firewall` @ `8c34a11a8`,
not in this binary).

**Recommended path to the fix:** rebuild madaros with the m2 `read_env` fix merged (re-enables
`SOUNIO_LOWER_*_TRACE`), then run the enum repro with `SOUNIO_LOWER_BODY_TRACE=1` to print the
exact lowering step before the fault → names the corrupting statement. Candidate fixes once
located: seed `enum_variants` into the body lowerer so the function-list fall-through never
fires; and/or apply the build-in-a-local workaround to the by-value copy that corrupts
`fn_count`. (Same rebuild can batch the already-landed int-println + for-loop fixes.)

## Relation to the other crashes
- Box::new crashes at the **same** RIP `0x3ebe4f2` (same garbage-length function scan).
- method-call crashes at `0x5d9f82b` (a different scan, same "garbage length → run off list"
  family — also derefs a stray value).
- All three are **count/length-corruption** miscompiles, distinct from Root 1 (stack overflow).
