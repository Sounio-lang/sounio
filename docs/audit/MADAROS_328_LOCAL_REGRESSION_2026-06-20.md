<!-- docs:meta
topic_id: repo.docs.audit.madaros-328-local-regression-2026-06-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-328-local-regression-2026-06-20
-->

# Madaros: #328 regressed the prebuilt — crashes on EVERY program in the pod (2026-06-20)

Recovered during the 2026-06-28 branch consolidation sweep from the untracked
WIP state in `/workspace/sounio-integ` (`fix/root2-enum-inplace`) before that
already-absorbed worktree was archived and removed.

## Symptom
The current-main prebuilt `bin/madaros-linux-x86_64` (post-PR #328, "fix(madaros): make
Box new safe in native prebuilt") **SIGSEGVs at codegen on every program**, including the
trivial:
```
fn main() -> i64 { 5 }      # NEW (post-#328): rc 139 at all stack limits
                            # OLD (main@659492156): rc 0, runs, returns 5
```
Verified same-environment (this workspace pod), same instant:
- `git show 659492156:bin/madaros-linux-x86_64` → builds+runs `main(){5}` ✅
- current `bin/madaros-linux-x86_64` (post-#328) → SIGSEGV ❌
- a fresh CI build from current-main source → also SIGSEGV in the pod ❌

So it is the **binary/source change in #328**, not the environment.

## Environment-sensitive
The #328 prebuilt-refresh **passed the CI gate** (which builds+runs `fn main()->i64{0}` on
`ubuntu-24.04`). So #328's madaros works in CI but crashes in the pod. That fingerprints an
**uninitialised-memory / layout-dependent bug**: the same code faults under the pod's mmap/
ASLR layout but not CI's (the crash reads a list element that points to valid memory in one
layout, garbage in the other).

## Crash
RIP `0x3f28ea7`:
```
mov (%rcx,%rdx,8),%rax     ; el = list[index]
mov (%rax),%rax            ; deref el  <-- SIGSEGV (el points to bad memory)
```
A list-iteration deref — same family as the other count/length-corruption crashes.

## Likely cause in #328
#328 changed, in `lowerer_lower_fn_item_mut` (`lower.sio` ~1566, ~1666):
```
- module.functions[(*lo).current_fn as usize] = *(*lo).current_func     # write to a LOCAL copy
+ (*(*lo).module).functions[(*lo).current_fn as usize] = *(*lo).current_func   # write in-place via Box
```
Switching the function-table write from a local `module` to the in-place `(*lo).module` Box.
If `(*lo).module` and the local `module` were not the same object (or the in-place write
races the by-value summary copy elsewhere), the function table ends up with entries whose
pointers are valid in CI's layout but dangling in the pod's → the iterate-and-deref crash.

## Impact
- **Current main's madaros is unusable in this pod** (crashes on all programs). The committed
  prebuilt should be treated as broken locally until this is fixed or reverted.
- The CI gate does **not** catch it (passes on `ubuntu-24.04`); the gate needs a layout-varied
  or sanitiser run, or the bug needs fixing at the source.

## Workaround used here
Integration/trace work is based off **pre-#328 main `659492156`** (which builds locally),
plus the read_env + print-int + for-loop fixes — so a locally-runnable, trace-enabled madaros
can be built to localize Root 2.
