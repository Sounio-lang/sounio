1. [BLOCKER] Monkey-patches `base.recondition` and `chain.project_upward_cover` with closures that mutate nonlocal lists and restore via try/finally.
   main:92 (capture_accepted_raw_projection) and transport_next_return:140
   Any uncaught exception, generator use, or concurrent import leaves the imported modules in a permanently altered state, violating the implicit contract that those workers are pure.
   Replace the global assignments with explicit reconditioner and cover parameters passed down the call chain.

2. [BLOCKER] `prior.state_metrics` and `centered.variable_weights` are called on objects whose definitions live in opaque worker modules never supplied.
   main:178 and 115
   The script cannot be executed or audited from the given artifact alone, making the reproducibility claim false.
   Either vendor the missing modules or replace the calls with locally defined, self-contained equivalents.

3. [MAJOR] The `while pending` splitting loop can exceed stated resource limits because `split_nodes` is only incremented on refusal paths and `MAX_SPLIT_NODES` is a soft cap that is never enforced before recursion.
   transport_next_return:155
   A deep or bushy symbolic domain produces unbounded memory and time before the guard at line 160 is reached.
   Move the node-limit test before the recursive `pending.extend` and make it a hard abort that still produces a defined result.

4. [MAJOR] `lineage_preserving_recondition` constructs a fresh `TM2R` by copying coefficient dicts and remainder, yet `same_component` later assumes that equal interval endpoints imply identical symbolic meaning.
   lineage_preserving_recondition:78 and same_component:64
   Under interval arithmetic with outward rounding this identity can fail for mathematically identical carriers, causing the “exact identity” check to spuriously reject valid runs.
   Replace structural equality with the library’s own equality or a certified Hausdorff-distance test.

5. [MINOR] `hull` performs repeated `union` without tracking precision loss; the resulting interval is then emitted as the authoritative event time.
   transport_next_return:105
   Accumulated widening can make the reported `[event_time, event_derivative, event_normal]` intervals strictly larger than any concrete trajectory, silently weakening downstream clinical claims.
   Emit both the hull and the maximum widening factor, or use a certified range-union primitive.

6. [NIT] Hard-coded `EXPECTED_PRERECOND_RECEIPT_SHA256` and `SCHEMA` literal are compared against files whose paths are derived from `__file__`, yet no check ensures the receipt directory layout matches the claimed tile.
   main:200 and 48
   A simple directory rename or checkout on a case-insensitive filesystem silently selects the wrong receipt.
   Replace the constant with a manifest file that also records the expected directory tree.
