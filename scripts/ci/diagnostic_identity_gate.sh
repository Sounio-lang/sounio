#!/usr/bin/env bash
# A diagnostic code is a global semantic identity. It may name one thing.
#
# Why this exists
# ---------------
# Error codes are keys: fixtures grep them, receipts record them, docs explain
# them, tooling routes on them. A number that names two diagnostics silently
# routes half its traffic wrong, and nothing in this repository noticed until a
# code was allocated twice by hand, weeks apart, by two different people.
#
# The census (#2170) found SEVENTEEN, six with both sides emitted, two carrying
# three identities each. The systematic cause is worth stating because it is not
# carelessness:
#
#   lean_single.sio prints the message text of the E200-E228 family WITHOUT the
#   E<N> tag. The number is assigned in the catalogue and in explanations/E<N>.md,
#   and the engine that owns it never pronounces it. So a survey from inside
#   check.sio sees the number free -- and free is exactly what nothing looks like.
#
# That is the same shape as the rest of this week's findings: an identity that
# exists but is never asserted reads as absent.
#
# What this gate measures
# -----------------------
#   collisions   one code claimed by two DIFFERENT identities
#   undocumented a code the compiler emits with no catalogue row
#   orphaned     a catalogue row no emitter ever produces
#
# All three are ratchets, frozen at today's measurement and lowered only by
# editing the line -- which puts each repair in a diff next to the work that did
# it. The last two are not cosmetic: they are the raw material of the next
# collision, because each makes a number look taken to one side and free to the
# other.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "diagnostic_identity"

# Ceilings are THIS gate's own measurement, not the census's.
#
# The #2170 census, by a different method (call-site extraction, hand-resolved
# indirect emitters, binary-file purging), reported 17 collisions, 132
# undocumented and 10 orphaned. This gate reports 26, 140 and 20 -- and the six
# hand-verified collisions (E208, E210, E217, E218, E219, E220) appear in both
# with the same two identities each, so neither method is simply broken.
#
# The difference is NOT reconciled and is deliberately not averaged away. Two
# independent routes disagreeing by nine is a measurement about the namespace,
# not a defect in one of the counts, and forcing them to agree would destroy the
# only evidence of where the boundary of "one identity" is unclear. Whoever
# reconciles them should record which route was right per code, and lower these
# lines accordingly.
# 34 -> 30. #2170 keeps the published lean_single identities E208/E217/E218/E219
# and moves the unrelated Madaros diagnostics that had been re-issued on those
# same numbers to E247/E248/E249/E250. Four codes, each of which named two
# things, now name one. Measured against origin/main on the same command, and
# the delta is exactly those four with nothing else moving.
COLLISION_CEILING="${SOUNIO_DIAG_COLLISION_CEILING:-30}"
UNDOCUMENTED_CEILING="${SOUNIO_DIAG_UNDOCUMENTED_CEILING:-141}"
# 20 -> 21 -> 14 -> 1. The comment that used to stand here said:
#
#   "every collision fixed by moving an emitter will push this number up by one
#    until the untagged lean_single prints are tagged, at which point it falls by
#    all of them at once."
#
ORPHANED_CEILING="${SOUNIO_DIAG_ORPHANED_CEILING:-2}"

ART_DIR="$ROOT_DIR/artifacts/gates"; mkdir -p "$ART_DIR"
ART="$ART_DIR/diagnostic_identity.json"

CHECK="self-hosted/check/check.sio"
CAT="docs/llm-guide/error-catalog.md"
for f in "$CHECK" "$CAT"; do
  [[ -r "$f" ]] || gate_fail "cannot read $f -- this gate's entire input"
done

python3 - "$CHECK" "$CAT" "$ART" "$COLLISION_CEILING" "$UNDOCUMENTED_CEILING" "$ORPHANED_CEILING" <<'PY'
import re, sys, json, os
check, cat, art, c_ceil, u_ceil, o_ceil = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])

# APERTURE. This gate read ONE file until 2026-08-28: self-hosted/check/check.sio.
# It nevertheless reported "a catalogue row no emitter ever produces" -- a claim
# about the whole compiler drawn from a survey of one file. Measured on main the
# day this changed: 20 distinct codes are emitted from OUTSIDE check.sio, in
# lean_single.sio, parser/types.sio, parser/stmts.sio, parser/exprs.sio and
# lexer/mod.sio. Seven of the twenty-one "orphans" were among them.
#
# It now reads every self-hosted/**/*.sio. The gate's own warning applies to the
# gate: "a survey of one mechanism reports the other as free".
import glob as _glob
_files = [check] + sorted(f for f in _glob.glob('self-hosted/**/*.sio', recursive=True)
                          if os.path.abspath(f) != os.path.abspath(check))
src = '\n'.join(open(f, errors='replace').read() for f in _files)

# lean_single is the OTHER owner of this namespace and needs its own control
# floor (below). Until #2212 it printed 42 of its diagnostics as a bare
# `error: <text>`: the number lived only in the catalogue and in
# explanations/E<N>.md, so every code lean_single owned read as FREE from
# inside check.sio, and ten of them were re-allocated by hand. It now
# pronounces them. A survey that silently stops seeing this file would report
# a namespace that got tidier; it did not, the gate went half-blind again.
_lean = 'self-hosted/compiler/lean_single.sio'
lean_src = open(_lean, errors='replace').read() if os.path.exists(_lean) else ''

# The message row is the identity: `else if code == N { print("...") }`.
# Taken from the printer rather than from call sites, because a call site says
# only that a number is used, never what it means.
# There are TWO emission mechanisms and a survey of one reports the other as
# free. That is not hypothetical: the first version of this gate saw only the
# message-row table and missed E210 entirely, because E210 is printed by a
# bespoke `checker_emit_e210` that writes `error[E210]` directly. Its own
# positive control refused the result rather than reporting a clean zero.
emitted = {}

#   (1) the message-row table:  else if code == N { print("...") }
#
#       A diagnostic usually has THREE arms for one code -- the message, a
#       `help:` line and a `note:` line -- and treating those as separate
#       identities reported 119 collisions where the census, by another route,
#       measured 17. Continuation arms are scaffolding; only the first arm is
#       the identity.
def _is_continuation(t):
    t = t.strip()
    return (t.startswith('|') or t.startswith('=') or t.startswith('help')
            or t.startswith('note') or t.startswith('\\n') or len(t) < 8)

_first = {}
for m in re.finditer(r'code\s*==\s*(\d+)\s*\{\s*print\("([^"]{4,200})"', src):
    n, msg = int(m.group(1)), m.group(2)
    if _is_continuation(msg):
        continue
    if n not in _first:
        _first[n] = msg.strip()
for n, msg in _first.items():
    emitted.setdefault(n, set()).add(msg)

#   (2) bespoke emitters:  print("error[EN]") inside a dedicated function.
#       The identity is the first message-looking string printed after the tag;
#       a bare punctuation print (" in ", "::", ": ") is scaffolding, not identity.
#       The tag comes in TWO shapes and a survey of one shape misses the other,
#       which is how this gate came to read a single engine. check.sio splits the
#       tag from its text (`print("error[E")` ... `print_error_message(code)`),
#       while lean_single writes both in one literal
#       (`print("error[E040]: Sounio uses 'var' ...")`). Take the inline text as
#       the identity when there is one; otherwise fall through to the next
#       message-looking print, as before. Without this arm the forward scan walks
#       past the end of one emitter into the NEXT one and files its text under
#       the wrong number -- measured before this arm existed: E220 reported as
#       "error e221 no main", E225 as "error e226 import path table full".
lines = src.split('\n')
for i, line in enumerate(lines):
    m = re.search(r'print\("error\[E(\d+)\](.*?)"', line)
    if not m:
        continue
    n = int(m.group(1))
    inline = m.group(2).lstrip(': ').strip()
    if len(inline) >= 5 and not _is_continuation(inline):
        emitted.setdefault(n, set()).add(inline)
        continue
    for j in range(i + 1, min(i + 40, len(lines))):
        t = re.search(r'print\("([^"]{5,200})"', lines[j])
        if not t:
            continue
        cand = t.group(1)
        if re.fullmatch(r'[\s:.,;\-\[\]()]*', cand) or cand.strip() in ('in', '::', 'error'):
            continue
        if _is_continuation(cand):
            continue
        if re.match(r'^[\s:]*$', cand):
            continue
        emitted.setdefault(n, set()).add(cand.strip())
        break

documented = {}
for line in open(cat, errors='replace'):
    m = re.match(r'^\|\s*E(\d+)\s*\|[^|]*\|[^|]*\|\s*([^|]+?)\s*\|', line)
    if m:
        documented.setdefault(int(m.group(1)), set()).add(m.group(2).strip())

# Positive control: the extraction must demonstrably see BOTH emission
# mechanisms, or a zero means the pattern and not the tree.
#
# The first version pinned E220 and E210 -- the two collisions found by hand --
# and broke the moment they were repaired. A control that names the specific
# defect it watches stops working exactly when the work succeeds, which is the
# opposite of what a control is for. It now asserts the mechanisms instead: at
# least one code from the message-row table, at least one from a bespoke
# error[E<N>] emitter, and a floor on the total.
mech1 = len(_first)
mech2 = len(emitted) - len(set(_first) - set())
tagged = len(re.findall(r'print\("error\[E\d+\]', src))
# The third mechanism -- the other engine -- needs its own floor for the same
# reason the second one does. Verified in both directions when it was added:
# with lean_single restored to its untagged state the gate REFUSES to report
# (rc=3, lean_tagged=18); intact it passes.
lean_tagged = len(re.findall(r'print\("error\[E\d+\]', lean_src))
if mech1 < 50 or tagged < 20 or lean_tagged < 40 or len(emitted) < 100:
    print(f"  CONTROL-FAIL  extraction saw message_rows={mech1} tagged_prints={tagged} "
          f"lean_tagged={lean_tagged} total={len(emitted)}")
    print("                the two engines are known to emit well over 100 codes by three")
    print("                mechanisms; a low count here is the pattern, not the tree")
    sys.exit(3)

def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()[:60]

collisions = {}
for n in sorted(set(emitted) | set(documented)):
    ids = set()
    for s in emitted.get(n, ()):  ids.add(norm(s))
    for s in documented.get(n, ()): ids.add(norm(s))
    # Two identities that share a prefix are the same diagnostic documented and
    # emitted, which is correct. Only genuinely different texts collide.
    distinct = []
    for i in sorted(ids):
        if not any(i.startswith(d[:24]) or d.startswith(i[:24]) for d in distinct):
            distinct.append(i)
    if len(distinct) > 1:
        collisions[n] = distinct

undocumented = sorted(set(emitted) - set(documented))
orphaned     = sorted(set(documented) - set(emitted))

print(f"  emitted={len(emitted)} documented={len(documented)}")
for n, ids in sorted(collisions.items()):
    print(f"  COLLISION  E{n}: " + " || ".join(x[:44] for x in ids))
print(f"  undocumented={len(undocumented)} orphaned={len(orphaned)}")

status = "pass"
fails = []
if len(collisions) > c_ceil:
    fails.append(f"collisions rose {c_ceil} -> {len(collisions)}")
if len(undocumented) > u_ceil:
    fails.append(f"undocumented rose {u_ceil} -> {len(undocumented)}")
if len(orphaned) > o_ceil:
    fails.append(f"orphaned catalogue rows rose {o_ceil} -> {len(orphaned)}")
if fails: status = "fail"

_payload = {"status": status,
           "metrics": {"total": len(set(emitted) | set(documented)),
                       "passed": len(set(emitted) | set(documented)) - len(collisions),
                       "failed": len(collisions), "not_run": 0},
           "collisions": {str(k): v for k, v in collisions.items()},
           "undocumented": undocumented, "orphaned": orphaned,
           "ceilings": {"collisions": c_ceil, "undocumented": u_ceil, "orphaned": o_ceil}}
# Write only when the bytes differ: a ratchet artefact should dirty the tree
# exactly when its number moves. Running a gate must not block `git checkout`
# (scripts/lib/gate_assert.sh: gate_write_artifact, same rule in bash).
_new = json.dumps(_payload)
if not os.path.exists(art) or open(art, errors='replace').read() != _new:
    open(art, 'w').write(_new)

print(f"diagnostic_identity: status={status} collisions={len(collisions)} (ceiling {c_ceil}) undocumented={len(undocumented)} (ceiling {u_ceil}) orphaned={len(orphaned)} (ceiling {o_ceil})")
sys.exit(1 if fails else 0)
PY
rc=$?
if [[ $rc -eq 3 ]]; then
  gate_fail "the extraction failed its own positive control"
fi
if [[ $rc -ne 0 ]]; then
  gate_fail "a diagnostic-identity ratchet moved the wrong way (see the lines above)"
fi
gate_pass "no ratchet moved; artifact at $ART"
exit 0
