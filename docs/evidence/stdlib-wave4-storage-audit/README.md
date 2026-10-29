# Independent audit of the closed Storage foundation

Audited commits: `ab55d0c6f9f6cbbce2e6813594aa020065f13d24` and `584eedfe7dfc0fbb23c91fc17eafdf0ca5a2feb7`, against `docs/design/typed-storage-authority.md` and the foundation delta. Root has not adopted or opened this foundation. This audit changes no production source, public recognizer, type kind, compatibility rule, minter, facade, runtime, backend or lowerer.

The initial paired component passed Slurm19781, COMPLETED0:0 in34 seconds (30 rows); Slurm19786 then passed the extended32-row paired component, also COMPLETED0:0 in34 seconds. The modern compiler checked the original and candidate production helper graphs; legacy9cf compiled and executed the injected private probes. C++23 independently required every recorded row and the complete marker. The compatibility dependency helpers were extracted literally from production epistemic.sio, with per-declaration hashes, instead of replacing them with stubs. This is not a loader/BFS, complete-checker or source admission gate.

## Confirmed defect: nominal source-record budget is not enforced

`check/storage_types.sio:80` places the4352 guard inside a duplicate-table-match branch that returns unconditionally anyway. The common append at96 has no guard. A checked-shape ItemList containing repeated copies of the same physically stamped declaration therefore appends more records than the documented4352 limit. The component's table has one matching concrete record; it does not fabricate4353 table indices.

| Input | Original | Isolated candidate |
| --- | --- | --- |
| 4352 repeated declaration items | context live,4352 records | context live,4352 records |
| 4353 repeated declaration items | context live,4353 records | context invalid,4352 records |
| Scalar query after4353 items | Complete | Unknown |
| Duplicated nominal query after4353 items | Unknown | Unknown |

This is a compiler resource-budget defect. It is not an array overrun, an unbounded allocation claim or observed authority mint. The outer source-item walk still stops at65536 visits, and duplicate nominal identities remain Unknown. The broken narrower bound nevertheless permits unnecessary source-record allocation and contradicts the documented budget.

`source-count-proposed.patch` moves only that guard to the common append, covering records and enums before allocation/increment. It was applied only to the qualification snapshot. The original source and its custody remain unchanged. Root owns any decision to apply it to the foundation.

## Compatibility closure is direct, not recursive across all carriers

`check/compat.sio:120` rejects a direct TyStorage on either side before Unknown/Error/Never. Array and tuple pairwise recursion also reaches that guard when it actually compares Storage elements. The existing general wildcards precede structural recursion, however; ordinary named compatibility at261 compares bare names and usually ignores inner metadata.

The actual unchanged function produced:

| Comparison | Compatible |
| --- | --- |
| Storage<Unit> with itself | false |
| Tuple<Storage<Unit>> with itself | false |
| Tuple<Storage<Unit>> with Unknown or Error | true |
| Array<Storage<Unit>> with Unknown or Error | true |
| Named Carrier with inner Storage<Unit>, against Unknown or Error | true |
| That same named carrier with itself | true |
| Same named carrier with inner Storage<Unit> versus inner i64 | true |

These are visible TypeEntry-wrapper controls, not claims about hidden nominal fields. The general function has no Defs tables with which to inspect those fields. The existing receipt's `general_compat=closed` must be read as its direct Storage comparisons, not an exhaustive recursive closure.

Before opening source Storage, root must choose an exhaustive shutter and compatibility/authority-preservation policy. A structural contains-Storage scan would need explicit budgets and Unknown handling; hidden nominal fields require resolved definition context, not guessed bare-name lookup. This audit does not implement that architecture or change existing wildcard semantics.

## Epoch and nominal snapshots: what is actually revalidated

`storage_context_live` checks epoch, validity, exact table addresses and counts. Nominal resolution requires uniqueness across both namespaces and a concrete stamped declaration with owner/span. Old wrappers are rejected after begin/reset, including reuse of the same table addresses and indices; epoch exhaustion fails closed before wraparound. Those are useful component invariants.

The context does not freeze or fingerprint table contents. The private component captures Storage<ScalarOwner> while its private field is i64, changes that field to f64 without changing the table address/count/owner, then observes:

- The classifier remains Complete because the current f64 field is admissible.
- Equality of the old wrapper with itself remains true.
- A newly captured wrapper compares equal with the old wrapper.
- Changing the same field to Ref<i64> yields Blocked and makes old-wrapper self-equality false. Equality therefore does reclassify the current payload; it does not enforce layout stability between two admissible payloads.

The identity snapshot records the nominal declaration kind/index/owner/span, not its recursive field graph. Equality reruns the current classifier and checks that declaration identity; it cannot detect that this field changed between captures. The original AST field contents also are not compared with table fields. A checked, immutable and consistently stamped graph is a caller precondition today, not enforced immutability or public authority. Integration must define freezing, epoch invalidation or stronger retained evidence before depending on that guarantee. No such policy is silently added here.

## Type classification and budgets

By source inspection, references/slices/raw pointers, functions, Worker capabilities, nested Storage, wide types and Knowledge are Blocked. Missing/ambiguous/phantom generic declarations, cycles, count mismatches and exhausted walks remain Unknown. Private fields and enum payloads are traversed; a zero-length array does not conceal a borrowed element. Complete remains type information rather than target layout or a storage-operation certificate.

The limits are distinct:

| Bound | Actual role |
| --- | --- |
|64 recursive levels | Classifier and identity recursion guard |
|4096 type nodes per query | Fresh classifier/equality fuel; does not certify unlimited work |
|65536 source items | Census termination bound, still active despite the4352 bug |
|4352 source declaration records | Documented narrower cap; broken in original, enforced by proposed snapshot guard |
|65536 identity/list nodes per context | Monotonic construction counter; exhaustion returns None |

The private boundary control sets the identity counter to65535: one scalar identity succeeds and reaches65536, then the next resolve returns None without increasing it. A tuple begun at65534 exhausts midway, returns None and publishes no wrapper; counter stays65536. These are explicit white-box counter boundary controls, not a claim of65536 complete resolve calls or a physical-memory bound. Wrapper boxes, semantic TypeEntry boxes and source-record allocations are not all the same counter. Runtime Storage data/bitmap/control/transaction-byte accounting is absent.

Two additional shape observations need root policy review rather than being merged into the budget fix: a concrete nongeneric nominal with an extra admissible inner is Complete; a tuple's irrelevant array_size can change while Complete/equality stay unchanged. The first retains inner identity, so it does not collapse with the bare nominal. Neither observation mints authority. Canonical shape validation and legitimate specialized-wrapper representation must be specified before admission is opened.

## Linearity and scope still closed

`check/check.sio:4425` recognizes TyStorage before inspecting its T, so direct Storage<Unit>, Storage<Copy> and Storage<Unknown> are linear. The existing owner walk recurses into inhabited arrays, tuples and resolved record/enum fields. References/raw pointers remain nonowning, and an empty array has no Storage instance to own. This does not establish arbitrary imported nominal carrier identity or a complete ownership/loan graph; the older nominal owner walk still has its own resolution contracts.

There is no source Storage recognizer, authority mint, managed runtime owner, allocation/bitmap state machine, target descriptor admission, final-IR provenance proof or HeapVec/HeapDeque integration in these commits. The exhaustive frontend shutter in the foundation document remains a specification. Cast/deref/manual construction/serialized-word forgery and recursive carrier transport are future required gates, not passed tests. Private compiler records are not a sandbox against raw operations inside compiler components.

The only proposed source change here is the narrow SourceCount guard move. The identity/compatibility findings describe the current component and its caller preconditions; root retains architecture and integration decisions.
