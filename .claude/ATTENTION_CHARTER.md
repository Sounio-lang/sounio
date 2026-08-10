# Attention Charter — `5 = 1 + 2`

Status: **binding** for Claude Code, Codex, Cursor, subagents, and any lane that
writes under a Sounio worktree.  
Owner: human integration shepherd (founder).  
Created: 2026-08-03.  
Equation: governance of attention (**5**) *is* compiler sovereignty (**1**) plus
epistemic honesty (**2**). Talking without closing 1 or 2 is not progress.

This charter does not replace:

- `.claude/PARALLEL_BLOCKER_CONTRACT.md` (blocker typing / merge)
- `docs/internal/coordination/COMPILER_LANE_CONTRACT.md` (compiler lane states)
- `.claude/AGENT_OFFLOAD_POLICY.md` (mandatory offload)
- `docs/governance/BRANCH_POLICY.md` (branch sprawl)
- `bin/sounio-coord` (live claims + agent inbox)

It **ranks** them: only work that closes **1** or **2** with a named gate ≥ E3
may hold P0 attention.

---

## 1. Equation

```text
5 = 1 + 2
```

| Symbol | Meaning | Done means |
|--------|---------|------------|
| **1** | Compiler sovereignty | Madaros residual closed or waived with gate; native path does not silently miscompile |
| **2** | Epistemic honesty | Named trust/receipt gate green; claim label exact; no overclaim |
| **5** | Attention governance | ≤1 active P0; write-sets disjoint; inbox used; halt is a deliverable |

Science lanes (ZD-fiber, FPGA, ontology, O-SSM, clinical) enter P0 **only** when
they *are* a test of **2** (or unblock **1**). Otherwise they are Garden / B4.

---

## 2. P0 queue (machine file)

Authoritative machine snapshot:

```text
.claude/attention_p0.v1.json
```

Human-readable slots (keep ≤5; prefer ≤3):

| ID | Horizon | Done | Primary gate |
|----|---------|------|----------------|
| A | **1** | D3 exclusive-ref / memory-wall class: repro + fix or explicit waiver | named Madaros multi-mod gate + `compiler_lane_status.sh` |
| B | **1** | One bounded native-v2 zero-event → green (not "generalize all") | native-v2 lane gate |
| C | **2** | One trust-map red → `TRUSTWORTHY` under native import | `scripts/epistemic_trust_gate.sh` |
| D | **2** | Claim registry / docs aligned for the touched path | `public-claim-registry` + no `doc-claim` overclaim |
| E | **5** | Hygiene: prune STALE claims; ≤2 compiler writers; control worktree not a dirty bench | `bin/sounio-coord prune` + `attention_brief` |

If A and B share write surface, **A wins**.

---

## 3. Surfaces

| Surface | Role |
|---------|------|
| `/workspace/sounio` | Control: shepherd, brief, merge, inbox. **Not** a heavy compile bench. |
| Dedicated worktree (`/tmp/sounio-*` or `.claude/worktrees/...`) | Implementation benches with exclusive write claims |
| `bin/sounio-coord` | Claims, heartbeat, release, **send / inbox / ack** |
| MCP `sounio-coord` (`scripts/mcp/sounio_coord_mcp.py`) | Same bus over MCP tools for Cursor / Claude |
| `.claude/attention_p0.v1.json` | Current P0 ids + owner + status |

Ceiling: **≤2 agents doing compiler work** at once on this pod (see `CLAUDE.md` §4).

---

## 4. Dispatch contract (before any write-bearing agent)

Fill this or do not start:

```text
Lane:
Owner:
Closes: A | B | C | D | E | none(Garden)
Worktree:   # must NOT be /workspace/sounio for heavy 1/2 work
Branch:
Write-Set:  # exact paths; claim via sounio-coord or MCP coord_claim
Required-Gates:
Merge-Target:
Known-Blockers:  # PARALLEL_BLOCKER_CONTRACT shape or "none"
```

If `Closes: none(Garden)`, the agent is **read-only** unless the shepherd
explicitly opens a B4 sandbox with TTL.

---

## 5. Agent messaging (the bus you already built)

Agents talk through `bin/sounio-coord` (also MCP):

```bash
bin/sounio-coord send --agent <me> --lane <my-lane> \
  --to-agent <them> --to-lane <their-lane> \
  --kind request --message "…"

bin/sounio-coord inbox --agent <me> --lane <my-lane>
bin/sounio-coord ack   --agent <me> --lane <my-lane> --message <id>
```

Kinds: `info` | `request` | `reply` | `blocker` | `handoff`.

Rules:

1. Ownership conflict → `kind=request` to current owner, then wait or open
   `ownership-conflict` blocker. Do not overwrite.
2. Halt / handoff → `kind=handoff` with `Next-Command` in the text.
3. Broadcast freeze → omit `--to-agent` / `--to-lane` (message visible to all).
4. Unread inbox is part of SessionStart / PostToolUse via
   `scripts/dev/sounio_coord_agent_hook.py` — **ack after acting**.

Durable truth still lives in Git + blocker records. The bus is presence, not
archive.

---

## 6. Shepherd ritual (founder)

**Daily (~10 min):**

```bash
bash scripts/dev/attention_brief.sh
```

Act on: one P0 owner, STALE claim count, conflicts, freeze need.

**Before dispatching an agent:** force the dispatch contract (§4); refuse vague
"explore and fix" with write access.

**Weekly:** at most one promotion to `main` per serialized surface
(`check.sio`, `bin/souc`, harness). Else leave `REVIEW_READY`.

**Freeze protocol** (when control worktree is dirty with alien lanes):

1. Broadcast `kind=info` freeze via coord (see `attention_brief.sh --freeze`).
2. No new write claims on `/workspace/sounio` except shepherd + declared P0.
3. Implementation continues only in claimed dedicated worktrees.

---

## 7. Merge / halt

Merge eligibility = `.claude/PARALLEL_BLOCKER_CONTRACT.md` Merge Contract
**and** the change closed a P0 slot or was explicitly B4.

Halt is a deliverable: stop with Blocker-ID or handoff containing
`Next-Command`. Exploring another file without a gate is drift.

---

## 8. Precedence

If this charter conflicts with a convenience prompt, **this charter wins**.
If it conflicts with executable gate evidence, **evidence wins** and the
charter or `attention_p0.v1.json` must be updated in the same change-set.
