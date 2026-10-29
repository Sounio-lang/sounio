# Variance-shadow qualification draft

This package prepares independent controls for pending patch SHA-256
127b070e8a0e3e9fae8891383797a6cd25cd13094429f5f19b42d8dbb73f74a6.
It never applies that patch, edits lower.sio, or modifies the candidate worktree.
The fixture worktree starts at 237ff2d674eb8e8f350d948c4f6f6393706cd67d.

The compiler defect remains the confirmed root19737 result: value computation
4/16 succeeds, while a variance zero defined in the conditional arm has three
non-dominated later uses. Native value agreement alone cannot qualify this fix.

## Prepared components

- closed_integer.sio: seven seeds, original recurrence, alias arithmetic, nested
  branches and a loop; C++23 checks all four columns and complete row order.
- assertion_mutant.sio: check/compile must succeed and native assertion must fail.
- Six semantic-refusal drafts: borrow, method, closure and cast fences, call/field
  loopback declassification. Checker success is required before the exact lower
  refusal; parse errors, timeouts, crashes and unrelated diagnostics fail.
- Eight profile-only sources: i128/u128 parameters and casts; word, float and wide
  suffix representations. These are not counted as compiler rejection tests.
- Six immutable canonical originals, including GUM runtime branches, loop
  accumulation, helper propagation, direct Knowledge and branded covariance.
  Their original assertions and source/oracle hashes are retained in manifest.json.

MCP sounio_verificado was called first for the small seed recurrence and reported
one syntax compilation. That is neither compilation of these complete fixtures
nor qualification against the proposed compiler.

## Entry points and boundaries

Run scripts/stdlib/wave4_variance_shadow_gate.sh --inventory for metadata only.
Under Slurm, --oracle-controls FRESH_RESULTS builds the C++23 reference with an
explicit CXX/CXX_SHA256 and rejects altered, dropped, extra and truncated outputs.
No Sounio compiler runs in that mode.

--language-components additionally requires SHADOW_AUTHORIZATION JSON with
authorized_by=root, raw/source paths, raw_ready path and hash, raw compiler hash,
patch hash and exact candidate lower hash. This is an execution record of root
authorization, not a mechanism for granting it. The harness never applies patches.
The anticipated candidate lower hash is 291d815f4fec96b9c149df0ed8606bc2ae70228807ea80bc43b4f2fe34b185af;
a different reviewed candidate requires a visible manifest revision.

The language stage records each check, compile and native status separately in
both LOOP_REGIONS modes and default/-O configurations. It retains all stdout,
stderr and ELFs in a fresh directory, with process-group timeouts. Successful
component execution writes LANGUAGE_COMPONENTS_ONLY. It does not write a full
qualification marker.

## Mandatory gates still outside this draft harness

The source/full-IR must-def audit, private Lowerer component, actual wide AST/type
inspection, unchanged checker/loans51/21 and WIDE14 numerical gate remain required.
See PRIVATE-COMPONENT.md and CFG-PROTOCOL.md. No receipt from these gates exists
for the candidate at package creation. Wide suffix parse refusal remains an
unsupported producer contract, not proof of semantic classification.

The canonical method/Knowledge originals may expose intended conservative
rejections or pre-existing numerical failures. Preserve those failures. Do not
relabel their acceptance contract or remove them to obtain a green package.
