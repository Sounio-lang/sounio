#!/usr/bin/env bash
# gate_write_artifact <path>   -- write stdin only when the bytes differ.
#
# Deliberately its own file, with NO side effects: no `set`, no traps, no
# globals. Seven of the eleven gates that needed this do not source
# gate_assert.sh, and adding that source to make one helper available would
# have imported whatever else the library does into scripts that never asked
# for it.
#
# Why any of this exists
# ----------------------
# Fourteen gates write into artifacts/gates/, twelve of those files are tracked,
# and each rewrote its artefact unconditionally. Merely RUNNING a gate left the
# tree dirty, so git refused to move:
#
#   error: Your local changes to the following files would be overwritten by
#          checkout: artifacts/gates/diagnostic_identity.json
#
# Measured cost on 2026-08-26: a verification worktree stayed pinned three
# merges behind because the checkout was refused, and the gate run afterwards
# reported confidently about the stale tree. The refusal was printed directly
# above the number that was read instead.
#
# It is also the governance rule this repository already had -- a gate must not
# dirty the tree the next gate inspects -- broken by the gates themselves.
#
# A ratchet artefact should dirty the tree exactly when the number it records
# has moved, and never when it has not. The file stays tracked, so a moved
# number still shows up in a diff.
gate_write_artifact() {
  local dest="$1" tmp
  [ -n "$dest" ] || { echo "gate_write_artifact: no destination" >&2; return 2; }
  mkdir -p "$(dirname "$dest")"
  tmp="$(mktemp "${TMPDIR:-/tmp}/gate-artifact.XXXXXX")" || return 2
  cat > "$tmp"
  if [ -f "$dest" ] && cmp -s "$tmp" "$dest"; then
    rm -f "$tmp"
    return 0
  fi
  mv -f "$tmp" "$dest"
}
