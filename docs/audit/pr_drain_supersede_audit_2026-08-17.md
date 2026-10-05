<!-- docs:meta
topic_id: repo.docs.audit.pr-drain-supersede-audit-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: claude (independent audit lane)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pr-drain-supersede-audit-2026-08-17
-->

# Independent audit — PR drain SUPERSEDED verdicts (2026-08-17)

> **Salvaged 2026-08-26 — read the Status section before trusting this file.**
> Recovered from an abandoned worktree (`.claude/worktrees/witness-matrix-20260814`,
> last touched 2026-08-17) during its removal. It had never been committed and
> existed on no branch; content below is preserved exactly as found.
>
> Its monitoring was **never carried out**: it began 2026-08-17 and no SUPERSEDED
> closure was ever recorded, so the Findings table is empty because the audit
> stopped — **not** because the drain was checked and found clean. The durable
> content here is the Method and the #1787 / #1790 calibration; the empty table is
> not evidence of anything.

## Why this exists

On 2026-08-17, 13 agent lanes (`pr-drain-lot`, orchestrated by
`fleet-orchestrator-drain` and `epistemic-spec-divergence`) began draining
the ~74 open PRs on `Sounio-lang/sounio` in disjoint batches, per
`PR_DRAIN_PROTOCOL.md`. The protocol requires symbol-by-symbol verification
before closing a PR as **SUPERSEDED** (work claimed to have reached `main`
by another route), citing the August precedent: 19 PRs correctly closed
this way, but 2 of them (#887, #889) carried work that never actually
landed — now tracked as issue #1702.

The protocol's verification step is self-attested: the same lane that
decides to close is the same lane that verifies. This file is the
independent second check the protocol does not otherwise have. Every PR
closed as SUPERSEDED by any `pr-drain-lot` lane gets re-verified here,
from scratch, against current `main` — not by trusting the closing lane's
stated evidence.

**Scope**: SUPERSEDED verdicts only. REBASE / DIFFERENT ERA / AUTHOR WORK
closures are the protocol's own responsibility and are not re-audited here
unless something incidentally surfaces (e.g. a DIFFERENT ERA closure that
looks like it might hide unique work, per the protocol's own #816 warning).
No PR is closed, reopened, or commented on by this audit — read-only
verification and reporting only.

## Method (applied identically to every row below)

1. Reconstruct what the PR actually carried: `gh pr diff <N>`, `gh pr view
   <N> --json files,title,body`.
2. Extract the closer's claim: the close comment / stated superseding
   commit or PR.
3. Independently re-derive — grep / `git log -S` on current `main` for
   every symbol, file, gate script, or error code the PR introduced. Not
   the closer's list; my own search.
4. If the claim rests on a CI gate passing, confirm the gate has fixtures
   that actually exercise the condition (not just that the script exists
   — this is exactly what made #887's closure risky: the gate script
   existed, but its fixtures didn't).
5. Verdict: **AGREE** (evidence quoted) or **DISAGREE** (name exactly what
   is missing from `main` — file, symbol, gate, or fixture, with path and
   line where possible).

## Live calibration context: issue #1787 / PR #1790

While reconciling a claim in this session's earlier discussion ("E137 is
undeclared-variable, not ambiguity"), I traced it — it is **not** in #1702.
It is in **issue #1787** ("CI rerun: confirm/refute the two findings from
#1702 audit against current-source Madaros", opened 2026-08-17T15:42Z) and,
specifically, in the founder's own comment on it (2026-08-17T16:57Z).

#1702's original investigation (the August sweep that closed #887 and #889
as SUPERSEDED) used a prebuilt `bin/madaros-linux-x86_64` whose source
commit (`3d1f143e7a`) is a month older than `origin/main` (`db750980b4`) —
a prior rerun attempt (#1689) was itself derailed by the same staleness.
#1787 requests a CI rerun on a Madaros built fresh from current `main`,
against two witnesses in the audit corpus (`docs/audit/pr1702_ref_deref_verification/reference/`,
also carried by **PR #1790**):

- **Witness 1 / Finding A** (`collision_direct.sio`): does the prebuilt's
  `explicit_ref_deref` store/read divergence (PR #889's target) still
  reproduce on a compiler built from current `main`? PASS (bug gone) →
  #889 stays superseded, but the static gate
  `scripts/ci/ref_field_autoderef_static_gate.sh` (a `grep` for
  `explicit_ref_kind == 2`) is called out as insufficient regardless,
  because the bug is state-dependent and a string-grep would still match a
  broken lowerer. FAIL (bug still there) → #889's port is justified after
  all.
- **Witness 2 / Finding B** (`ambiguous/strong5.sio`): does `souc check`
  actually raise `E014` (`ambiguous_name`) at name-resolution when two
  imports collide — the condition `madaros_visibility_context_gate.sh` is
  supposed to guard? The issue's own trace of current `main`'s
  `report_error_at(.., 14, ..)` callsites finds none at name-resolution —
  both live `E014` sites are in `check_index_expr`, for non-integer array
  indices, unrelated to ambiguity. The negative-control fixture
  (`ambiguous/no_use.sio`) is expected to still raise **E137**
  (undeclared-variable) — the generic path, not a real ambiguity
  diagnostic. That is the exact claim: what fires in this area today is
  E137, not a genuine ambiguity check, which is what would make the gate
  pass vacuously.

**Status of #1787 as of this check: open, unresolved.** The CI rerun this
issue asks for has not yet posted results. #887 and #889 themselves are
already closed (part of the August sweep, not today's `pr-drain-lot`
batch) — no action needed on them from this audit directly.

**Why this matters here**: if today's drain touches PR #1790 itself, or
any PR whose SUPERSEDED claim leans on `madaros_visibility_context_gate.sh`
or on E014/E137 ambiguity behavior, that claim gets the same scrutiny as
everything else in this file — Finding B is a live open question right
now, not a settled one, and a SUPERSEDED close that assumes it's settled
in either direction would be exactly the premature-closure risk this audit
exists to catch. This does not expand this audit's scope beyond SUPERSEDED
verdicts from today's drain; it's calibration for reading those verdicts
when they touch this area.

## Status

Monitoring started 2026-08-17T21:4x UTC. As of the last check, 0 PRs have
been closed with a SUPERSEDED verdict by today's `pr-drain-lot` lanes (all
activity so far has been REBASE verdicts — see coordination-bus messages
from `cursor-1`/`cursor-2` in the `pr-drain-lot` lane). This file will be
updated as SUPERSEDED closures appear.

## Findings

| PR | Claim (from closer) | What I found | Verdict |
|----|---------------------|---------------|---------|
| _(none yet — 0 SUPERSEDED closures observed as of this check)_ | | | |
