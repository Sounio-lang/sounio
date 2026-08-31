<!-- docs:meta
topic_id: repo.docs.audit.token-table-ceiling-refusal-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.token-table-ceiling-refusal-2026-08-17
-->

# Token-table ceiling — fail-closed refusal (E229)

**Date:** 2026-08-17  
**Lane:** grok-cli5 / `token-ceiling-refusal`  
**Code:** `error[E229]`  
**Precedent:** E219 (`extern "C"` unimplemented → refuse, do not fabricate)

---

## Defect

Madaros lexer/parser token storage is fixed at **2097152** slots
(`PARSER_TOKENS` / `PT_*` in `self-hosted/parser/parser.sio`, emits in
`self-hosted/lexer/mod.sio`). When `tc` reached the wall:

1. `parser_set_token_flat` / `parser_set_token_at` **no-op** past the index.  
2. The lexer **kept walking** source and could still return a large `token_count`.  
3. Out-of-range reads synthesized **Eof**.  
4. The parser then failed on a **truncated** stream and blamed a line that was
   never the cause.

Separately, `lex_file_to_globals` / `lex_source_to_globals` **clipped** source
length to 2097152 bytes with the same silence.

A compiler that discards its own input and misattributes the error is exactly
the honesty failure this language exists to make impossible.

Dispatch stated **41 run-pass tests** blocked by this class of failure. This
note does not re-census those 41 under a fresh Madaros build (heavy); the
witness below is the structural reproduction.

> **Re-measured on rebase (2026-08-27, base `origin/main@055825a3f9`).** Two of
> this note's premises moved and are corrected here rather than republished:
>
> 1. **The "41" is stale.** `docs/audit/TOKEN_CEILING_BLOCKED_RUNPASS_CENSUS_2026-08-17.md`
>    (now on main) re-counted the blocked set as **169** imported run-pass tests
>    (`tests/run-pass/**/*.sio` importing `theorem::portfolio`), all already
>    `//@ known-failure`. Treat **41 → 169**; "41" was an un-recounted dispatch figure.
> 2. **The blocking module has since been split (path C landed).**
>    `stdlib/theorem/portfolio.sio` measured 2 109 065 bytes when this note was
>    written; on the rebase base it is a **1 130-byte façade**, and
>    `scripts/ci/stdlib_source_byte_ceiling_gate.sh` (new on main) passes:
>    `PASS no stdlib .sio over hard cap 2097152; portfolio façade=1130B lorenz façade=1127B`.
>
> Neither correction weakens the fix: the **refusal itself is still absent from
> main** (`git grep E229 origin/main -- self-hosted` → no hits; the byte-ceiling
> gate on main is annotated "E229 after refusal lands"). The split makes the
> blocked tests runnable; E229 makes the wall honest for the *next* input past it.
> What changes is the motivation: E229 is no longer needed to unblock 41/169
> tests — it is needed because a silent clip is still reachable on any input.

---

## Fix (not “bigger table”)

| Change | Role |
|--------|------|
| `lexer_push_flat` | Emit only if `tc < 2097152`; else mark overflow and return −1 |
| Loop break + `return -1` | Never hand the parser a truncated stream |
| Source length `> 2097152` | Refuse (kind=2), do not clip |
| `lex_file_parse_items` | Print **E229**, `parser_set_last_errors(true, 1)`, return `None` |
| `scripts/ci/token_table_ceiling_gate.sh` | Generates 2097152-comma witness; requires E229 + nonzero rc |

**Raising the ceiling without refusal is explicitly rejected** even if it would
green the 41 tests: the next input past the new wall would lie again.

---

## Should the cap also move? (measurement, not convenience)

| Fact | Implication |
|------|-------------|
| Table is **2²⁰** slots × several parallel arrays (kind, start, end, line, col, …) | Memory is multi‑MB **per process** already |
| Source buffer is the **same** 2097152-byte wall | Byte clip and token wall interact |
| Product modules that need >2M tokens are a **modularity** smell | Prefer split compilation units |
| Historical raise 1M→2M without rebuild left a **stale binary** lying (PL adoption audit) | Ceiling changes without rebuild/refusal are hazardous |

**Recommendation:** **Keep 2097152 for now.** Ship **E229 first**. Revisit a
raise only with: (1) measured peak `token_count` on the blocked 41 and on
`self-hosted/compiler/main.sio` multi-module loads, (2) memory budget on the
pod/CI, (3) **E229 retained** at the new bound. Do not raise “so the tests
pass” alone.

---

## Witness (measured)

### Baseline (prebuilt Madaros, before this fix)

```bash
TOKEN_CEILING_EXPECT=baseline_silent bash scripts/ci/token_table_ceiling_gate.sh
```

| Witness | Measured on prebuilt `bin/souc` |
|---------|--------------------------------|
| **W1** valid `main` + pad to 2097152 bytes + trailing `fn should_not_be_dropped` | **`check: OK` rc=0** — trailing source past the byte wall was **silently clipped** |
| **W2** 2097152 commas | rc=1, floods `parse error: unrecognised item start`, **no E229** |

W1 is the smoking gun: the compiler discarded input and reported success.

### After rebuild from this source

```bash
bash scripts/ci/build_modular_madaros.sh "$PWD/artifacts/self-hosted/madaros"
MADAROS_RAW_BIN=$PWD/artifacts/self-hosted/madaros bash scripts/ci/token_table_ceiling_gate.sh
```

| Witness | Measured after fix |
|---------|-------------------|
| W1 | **error[E229]** source exceeds lexer byte buffer, rc≠0 |
| W2 | **error[E229]** lexer token table full, rc≠0 |

Gate result: `TOKEN_TABLE_CEILING_GATE_OK`.

### Re-measured on rebase (2026-08-27, base `origin/main@055825a3f9`)

Both columns above were re-run, not republished. The original "prebuilt Madaros"
binary no longer exists at this base (`artifacts/self-hosted/madaros` is not
checked in), so the **before** column is re-scoped to the engine the tree actually
ships — `bin/souc-lean-single-x86_64`, which carries the identical 2097152 walls
(`self-hosted/compiler/lean_single.sio:33-42`, `token_capacity()` at :1041) and no E229:

```bash
SOUNIO_SOUC_ENGINE=lean_single TOKEN_CEILING_EXPECT=baseline_silent \
  bash scripts/ci/token_table_ceiling_gate.sh
# W1 source-clip: rc=0
# W1 baseline CONFIRMED: rc=0 while file contains bytes past 2097152 (silent clip)
# W2 token-table: rc=1 ... W2 baseline OK: no E229 (rc=1)
# TOKEN_TABLE_CEILING_GATE_OK: baseline: silent clip / non-E229 failure still present (W1 rc=0 W2 rc=1)
```

After building Madaros from **this rebased source** and re-running in refusal mode:

```bash
bash scripts/ci/build_modular_madaros.sh "$PWD/artifacts/self-hosted/madaros"
MADAROS_RAW_BIN=$PWD/artifacts/self-hosted/madaros bash scripts/ci/token_table_ceiling_gate.sh
#   W1 source-clip: rc=1 ...  W1 OK: E229 on source past byte wall
#   W2 token-table: rc=1 ...  W2 OK: E229 on token table full
# TOKEN_TABLE_CEILING_GATE_OK: E229 refusal on source-byte and token-table ceilings
```

### Re-measured again on the second rebase (2026-08-29, base `origin/main@64db7167f8`)

Every line above was re-run, not republished, on a Madaros built from **this**
tree (`scripts/ci/build_modular_madaros.sh`, `chmod +x`, `MADAROS_RAW_BIN=`).

Baseline, both engines the tree actually ships:

```
# env -u SOUC_BIN TOKEN_CEILING_EXPECT=baseline_silent bash scripts/ci/token_table_ceiling_gate.sh
#   (committed bin/souc -> Madaros ELF, which predates the refusal)
  W1 source-clip: rc=0
  W1 baseline CONFIRMED: rc=0 while file contains bytes past 2097152 (silent clip)
  W2 token-table: rc=139
TOKEN_TABLE_CEILING_GATE_OK: baseline: silent clip / non-E229 failure still present (W1 rc=0 W2 rc=139)

# SOUNIO_SOUC_ENGINE=lean_single, same command
  W1 source-clip: rc=0
  W1 baseline CONFIRMED: rc=0 while file contains bytes past 2097152 (silent clip)
  W2 token-table: rc=1
TOKEN_TABLE_CEILING_GATE_OK: baseline: silent clip / non-E229 failure still present (W1 rc=0 W2 rc=1)
```

W1 rc=0 reproduces on both. **W2 on the committed Madaros is rc=139, a
segmentation fault, not the rc=1 parse-error flood recorded on 2026-08-17.** A
crash is not a refusal either, so the baseline claim is unaffected — but the
2097152-comma input SEGVs the shipped compiler, which the earlier note did not
say and a reader would not guess.

Two invocation traps, both hit while producing the table above:

- The baseline must be taken through `bin/souc` (with `SOUNIO_SOUC_ENGINE=lean_single`
  where the seed is the subject), **not** by pointing `SOUC_BIN` at
  `bin/souc-lean-single-x86_64` directly. The raw seed ELF returns rc=1 on W1
  where the wrapper returns rc=0; only the wrapper path is the compiler a user
  invokes, and only it shows the silent clip.
- `MADAROS_RAW_BIN` pointing at a non-executable ELF used to fall through to the
  committed compiler in silence. It aborts loudly now; `chmod +x` anyway.

After, on the Madaros built from this rebased source:

```
  W1 source-clip: rc=1  ...  W1 OK: E229 on source past byte wall
  W2 token-table: rc=1  ...  W2 OK: E229 on token table full
TOKEN_TABLE_CEILING_GATE_OK: E229 refusal on source-byte and token-table ceilings
```

The identifier slot was the one merge decision in this rebase: `main` moved it to
`parser_set_token_scalar` + `keyword_lookup` (#2229, so a new keyword needs no arm
in `parser_set_token_flat`'s dispatch). That is kept, with the fail-closed guard
inlined around it — the guard is about the token slot, not about which setter
writes it. The refusal above is measured through that resolution.


**Scope limit, stated plainly.** This patch changes `self-hosted/lexer/mod.sio`
only, i.e. **Madaros**. `lean_single` — still the bootstrap seed, still reachable
via `SOUNIO_SOUC_ENGINE=lean_single`, and the engine several dissertation gates
force — retains the silent clip (the baseline run above *is* that engine, at this
base). E229 is not yet a whole-toolchain guarantee; closing lean_single is
follow-up work, not something this note may claim.

### Resolved: `diagnostic_identity_gate.sh` — orphan gone, collision found and fixed

The version of this note written on 2026-08-17 recorded an **open** finding and
left a gate red on purpose: `diagnostic_identity_gate.sh` set
`CHECK="self-hosted/check/check.sio"` and surveyed that **one file** for
emitters, so E229 — emitted from `self-hosted/lexer/mod.sio`
(`lexer_report_token_overflow_if_any`) — read as a catalogue row with no
emitter, `orphaned=22` against a ceiling of 21. The note argued the correct
repair was to widen the survey, in its own change with its own measurement, and
explicitly refused to raise `SOUNIO_DIAG_ORPHANED_CEILING` to absorb it.

**That is exactly what happened, upstream and independently.** #2260 widened the
aperture to every `self-hosted/**/*.sio` and re-derived all three populations
from the wider scan:

| population | narrow (one file) | wide (`self-hosted/**/*.sio`) |
|---|---:|---:|
| collisions | 25 | **34** |
| undocumented | 140 | **141** |
| orphaned | 21 | **14** |

Orphaned *fell* by seven: seven of the twenty-one "orphans" were emitted all
along, from `lean_single.sio`, `parser/types.sio`, `parser/stmts.sio`,
`parser/exprs.sio` and `lexer/mod.sio`. E229 is not an orphan under the wide
aperture, and the section this replaces no longer describes any state of the
repository.

**One real defect surfaced when the aperture widened, and it is fixed here.**
Under the wide scan E229 read as a **collision**, not an orphan: the catalogue
row said `token table / source byte buffer full (fail-closed)` while
`lexer/mod.sio` prints `source exceeds lexer byte buffer (capacity 2097152
bytes)`. Two texts, one number — precisely what the gate exists to catch. The
row now reads `source exceeds lexer byte buffer, or token table full
(fail-closed)`, which is what the emitter says and covers both walls.

Measured on this tree, and identical to `origin/main` at the rebase point:

```
diagnostic_identity: status=pass collisions=34 (ceiling 34) undocumented=141 (ceiling 141) orphaned=14 (ceiling 14)
DIAGNOSTIC_IDENTITY_OK: no ratchet moved
```

No ceiling was raised for E229. All three ratchets are where `main` left them.

Witnesses avoid brace-bisect traps: W1 is a complete valid prefix plus pad plus a
complete trailing item past the wall; W2 is only commas (complete punctuation
tokens), not unbalanced items.

---

## Bisect warning (from dispatch)

If you must bisect a large real module to study this class of failure: balance
`{}` `()` `[]` and delete **complete** items only. Unbalanced deletion produces
a **different** bug and has wasted time here before.

---

## Files

- `self-hosted/lexer/mod.sio` — overflow state, `lexer_push_flat`, report, parse gate  
- `docs/llm-guide/explanations/E229.md`  
- `scripts/ci/token_table_ceiling_gate.sh`  
- this note
