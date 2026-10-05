#!/usr/bin/env bash
# Refuse early, say which verb is missing, and name a binary that has it.
#
# Seven gates invoke `souc bootstrap …` or `souc opt policy …`. Against the
# default `bin/souc` (Madaros) both families fall through to the file-input path:
#
#   $ souc bootstrap verify --bundle bootstrap
#   error: at bootstrap:0:0 - could not read input file
#
# A subcommand is missing and the diagnostic reports a missing FILE. Someone
# debugging a red gate goes looking for `bootstrap` on disk -- and finds it,
# because `bootstrap/` is a real directory, which makes the false trail longer.
#
# The verbs are NOT gone from the repository. They live in the checked artifact
# `artifacts/omega/souc-bin/souc-linux-x86_64-gpu`, where they work:
#
#   $ artifacts/omega/souc-bin/souc-linux-x86_64-gpu bootstrap verify --bundle bootstrap
#   bootstrap verify ok: manifest=… artifacts=1 target=linux-x86_64 …   (rc=0)
#
# So these gates are not dead code. They are pointed at the wrong binary. That is
# worth stating precisely, because the obvious reading -- "the Rust crate went, so
# the capability went" -- is wrong, and acting on it would delete working gates.
#
# ── Why the probe is differential ────────────────────────────────────────────
#
# The first version of this guard tested for the literal string
# "could not read input file". That is Madaros's fall-through signature. Pointed
# at the checked artifact, which is clap-based and answers an unknown verb with a
# usage error, the probe could no longer tell present from absent -- and its own
# negative control caught that and refused, rather than reporting that a verb the
# binary HAS was missing.
#
# So: ask the binary what it does with a verb that certainly does not exist, ask
# what it does with the verb in question, and compare. Two engines, one method,
# no engine-specific string.

_souc_verb_response() {
  local souc="$1" verb="$2" out
  out="$("$souc" "$verb" 2>&1)" || true
  # Blank the verb itself so two absent verbs compare equal despite their names
  # appearing in the message.
  printf '%s' "${out//$verb/<VERB>}"
}

_souc_verb_absent() {
  local souc="$1" verb="$2"
  [[ "$(_souc_verb_response "$souc" "$verb")" == "$(_souc_verb_response "$souc" __sounio_absent_probe__)" ]]
}

# require_souc_verb <souc-binary> <verb> [what the gate needs it for]
#
# Controlled in both directions before it is believed: `info` exists on every
# souc and must not look absent; a second nonsense verb does not exist and must
# look absent. A probe that cannot tell those two apart cannot answer about
# anything, and must refuse rather than return a confident verdict.
require_souc_verb() {
  local souc="$1" verb="$2" purpose="${3:-}"

  if _souc_verb_absent "$souc" info; then
    echo "error: verb probe failed its positive control -- \`$souc info\` looks absent." >&2
    echo "       The probe cannot distinguish a live verb from a dead one, so its" >&2
    echo "       answer about \`$verb\` would mean nothing. Refusing to report." >&2
    exit 3
  fi
  if ! _souc_verb_absent "$souc" __sounio_second_absent_probe__; then
    echo "error: verb probe failed its negative control -- a verb that cannot exist" >&2
    echo "       does not look absent. Fix this guard before trusting it." >&2
    exit 3
  fi

  _souc_verb_absent "$souc" "$verb" || return 0

  local fallback="" root
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  local checked="$root/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
  # Only recommend it if it actually answers for this verb. A suggestion that
  # does not work is the same defect one level along.
  if [[ -x "$checked" ]] && ! _souc_verb_absent "$checked" "$verb"; then
    fallback="$checked"
  fi

  echo "error: \`$souc $verb\` is not a subcommand of this binary." >&2
  [[ -n "$purpose" ]] && echo "       This gate needs it for: $purpose" >&2
  if [[ -n "$fallback" ]]; then
    cat >&2 <<EOF
       It is not missing from the repository -- it is missing from THIS binary.
       The checked artifact has it and answers for it:

           SOUC_BIN=$fallback \\
             $0 ...

       The default \`bin/souc\` is Madaros, the self-hosted compiler, and the
       \`bootstrap\` and \`opt\` verb families were never ported to it. They came
       from the Rust \`souc\` crate removed on 2026-02-26 by 79acc192e1, and the
       checked artifact predates that removal.

       Without this guard you would have seen

           error: at bootstrap:0:0 - could not read input file

       which names a missing FILE for a missing SUBCOMMAND and sends you looking
       for \`bootstrap/\`, a directory that does exist.
EOF
  else
    cat >&2 <<EOF
       Neither this binary nor the checked artifact at
       artifacts/omega/souc-bin/souc-linux-x86_64-gpu answers for it, so the
       capability is genuinely unavailable here rather than merely on another
       binary. The signed data it operated on is still in the tree
       (bootstrap/artifacts/manifest.v2.json, bootstrap/policies/policy.v1.json).
EOF
  fi
  exit 1
}
