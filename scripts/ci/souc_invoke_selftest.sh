#!/usr/bin/env bash
# Two-armed selftest for scripts/lib/souc_invoke.sh.
#
# souc_invoke exists to stop one specific silent failure: Madaros accepts
# lean_single's argv (`<bin> <src> <out>`) without complaint, treats every bare
# positional as `input_file`, keeps the last, and writes `a.out`. Exit 0. A
# helper that only ever confirms "yes, this is Madaros" has not been shown to
# discriminate, so this test drives both arms with fakes and needs no compiler
# at all — which means it runs on every PR, in under a second.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$ROOT_DIR/scripts/lib/souc_invoke.sh"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/souc-invoke-selftest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "SOUC_INVOKE_SELFTEST_FAIL: $*" >&2; exit 1; }

# ── fake Madaros: the real three-line banner, and a CLI that requires `build` ──
cat >"$WORK/fake-madaros" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  echo "Madaros v0.80.0 -- the Sounio self-hosted compiler"
  echo "the bare highland that does not negotiate with ill-formed code -- Sfakia, Crete"
  echo "Horizon 3: self-hosted primary compiler."
  exit 0
fi
if [[ "${1:-}" == "build" ]]; then echo "$2 -> $3" > "$3"; exit 0; fi
# Anything else: swallow the positionals and write a.out, exactly like the real
# parse_options does. Exit 0 — this is the whole trap.
echo "wrong-argv" > "$(dirname "${2:-.}")/a.out"
exit 0
EOF

# ── fake lean_single: the usage line it actually prints, positional argv ──────
cat >"$WORK/fake-lean" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" || "$#" -eq 0 ]]; then
  echo "Usage: mini_native <source.sio> <output> [--show-ast] [--show-types] [--r15-monitor] [--target x86_64-windows]"
  exit 0
fi
echo "$1 -> $2" > "$2"
exit 0
EOF

cat >"$WORK/fake-stranger" <<'EOF'
#!/usr/bin/env bash
echo "some other program entirely"
exit 0
EOF

chmod +x "$WORK/fake-madaros" "$WORK/fake-lean" "$WORK/fake-stranger"
echo 'fn main() {}' >"$WORK/in.sio"

# ── arm 1: it recognises each engine ─────────────────────────────────────────
[[ "$(souc_banner "$WORK/fake-madaros")" == "madaros" ]] \
  || fail "did not recognise a Madaros banner"
[[ "$(souc_banner "$WORK/fake-lean")" == "lean_single" ]] \
  || fail "did not recognise a lean_single usage line"

# ── arm 2: it REFUSES what it cannot identify ────────────────────────────────
# The failure mode this guards is guessing. A helper that returns 'madaros' by
# default would hand the Madaros argv to anything.
[[ "$(souc_banner "$WORK/fake-stranger")" == "unknown" ]] \
  || fail "claimed to identify a program that announces neither engine — it is guessing"
[[ "$(souc_banner "$WORK/does-not-exist")" == "unknown" ]] \
  || fail "claimed to identify a binary that is not there"

set +e
souc_compile "$WORK/fake-stranger" "$WORK/in.sio" "$WORK/out.unknown" >/dev/null 2>&1
rc=$?
set -e
[[ "$rc" -eq 78 ]] || fail "compiled with an unidentified binary (rc=$rc) instead of refusing"

# ── arm 3: each engine gets ITS OWN argv ─────────────────────────────────────
souc_compile "$WORK/fake-madaros" "$WORK/in.sio" "$WORK/out.madaros" >/dev/null 2>&1 \
  || fail "souc_compile failed against the Madaros fake"
[[ -s "$WORK/out.madaros" ]] \
  || fail "Madaros arm produced no output at the requested path — this is the a.out trap"
[[ ! -e "$WORK/a.out" ]] \
  || fail "Madaros arm wrote a.out, i.e. it used the lean_single argv"

souc_compile "$WORK/fake-lean" "$WORK/in.sio" "$WORK/out.lean" >/dev/null 2>&1 \
  || fail "souc_compile failed against the lean_single fake"
[[ -s "$WORK/out.lean" ]] \
  || fail "lean_single arm produced no output at the requested path"

echo "SOUC_INVOKE_SELFTEST_OK: both engines identified, unknown binaries refused, each argv kept apart"
