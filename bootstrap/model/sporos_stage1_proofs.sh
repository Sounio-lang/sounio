#!/usr/bin/env bash
# Kernel-checked single-job (`-j 1`) build and axiom verification gate for the
# Sporos stage-1 final correctness theorems (`BootstrapSporosStage1`).
#
# Why separate from PR CI (`lean_lib Bootstrap`):
#   `SporosX1FinalRefuse.lean` (68k lines) measured ~8.4 GiB max RSS and ~26 min
#   on its own (37+ min across the 6 `SporosX1Final*` modules at `-j 1`), which
#   exceeds the 7 GB / 35 min pull-request runner on a cold cache miss. Keeping
#   it in `BootstrapSporosStage1` mirrors `BootstrapSeeds`.
#
# Usage:
#   bootstrap/model/sporos_stage1_proofs.sh [--selftest]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -d "${HOME}/.elan/bin" ]; then
  case ":${PATH}:" in
    *":${HOME}/.elan/bin:"*) ;;
    *) PATH="${HOME}/.elan/bin:${PATH}"; export PATH ;;
  esac
fi

W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

check_axioms_log() {
  local logfile="$1"
  awk '
    BEGIN { sq = sprintf("%c", 39); bad = 0; total = 0 }
    {
      line = $0
      sub(/\r$/, "", line)
      if (index(line, "sorryAx") > 0 || index(line, "ofReduceBool") > 0 || index(line, "uses " sq "sorry" sq) > 0) {
        bad = 1
      }
      if (substr(line, 1, 1) == sq) {
        qpos = index(substr(line, 2), sq)
        if (qpos > 1) {
          name = substr(line, 2, qpos - 1)
          if (name == "SporosX1.stage1_oracle_exists_target" ||
              name == "SporosX1.stage1_final_target" ||
              name == "SporosX1.stage1_final_exit2_target") {
            tail = substr(line, qpos + 2)
            if (tail == " does not depend on any axioms") {
              seen[name]++
              total++
            } else if (substr(tail, 1, 21) == " depends on axioms: [") {
              lb = substr(tail, 22)
              if (substr(lb, length(lb), 1) != "]") {
                bad = 1
              } else {
                inner = substr(lb, 1, length(lb) - 1)
                n = split(inner, toks, ",")
                if (n < 1) {
                  bad = 1
                } else {
                  for (i = 1; i <= n; i++) {
                    t = toks[i]
                    gsub(/^[ \t]+|[ \t]+$/, "", t)
                    if (t != "propext" && t != "Classical.choice" && t != "Quot.sound") {
                      bad = 1
                    }
                  }
                }
                seen[name]++
                total++
              }
            } else {
              bad = 1
            }
          }
        }
      }
    }
    END {
      if (bad) exit 1
      if (seen["SporosX1.stage1_oracle_exists_target"] != 1) exit 1
      if (seen["SporosX1.stage1_final_target"] != 1) exit 1
      if (seen["SporosX1.stage1_final_exit2_target"] != 1) exit 1
      if (total != 3) exit 1
    }
  ' "$logfile"
}

selftest() {
  cat <<'EOF' > "$W/st_ok.log"
'SporosX1.stage1_oracle_exists_target' depends on axioms: [propext, Classical.choice, Quot.sound]
'SporosX1.stage1_final_target' depends on axioms: [propext, Classical.choice, Quot.sound]
'SporosX1.stage1_final_exit2_target' depends on axioms: [propext, Classical.choice, Quot.sound]
EOF
  check_axioms_log "$W/st_ok.log" || { echo "FAIL: positive control rejected" >&2; return 1; }

  cat <<'EOF' > "$W/st_sorry.log"
'SporosX1.stage1_oracle_exists_target' depends on axioms: [propext, Classical.choice, Quot.sound]
'SporosX1.stage1_final_target' depends on axioms: [propext, sorryAx]
'SporosX1.stage1_final_exit2_target' depends on axioms: [propext, Classical.choice, Quot.sound]
EOF
  if check_axioms_log "$W/st_sorry.log"; then
    echo "FAIL: sorryAx negative control passed" >&2; return 1
  fi

  cat <<'EOF' > "$W/st_nd.log"
'SporosX1.stage1_oracle_exists_target' depends on axioms: [propext, Lean.ofReduceBool]
'SporosX1.stage1_final_target' depends on axioms: [propext, Classical.choice, Quot.sound]
'SporosX1.stage1_final_exit2_target' depends on axioms: [propext, Classical.choice, Quot.sound]
EOF
  if check_axioms_log "$W/st_nd.log"; then
    echo "FAIL: Lean.ofReduceBool negative control passed" >&2; return 1
  fi

  cat <<'EOF' > "$W/st_custom.log"
'SporosX1.stage1_oracle_exists_target' depends on axioms: [propext, Classical.choice, Quot.sound]
'SporosX1.stage1_final_target' depends on axioms: [propext, Foo.ax]
'SporosX1.stage1_final_exit2_target' depends on axioms: [propext, Classical.choice, Quot.sound]
EOF
  if check_axioms_log "$W/st_custom.log"; then
    echo "FAIL: custom axiom negative control passed" >&2; return 1
  fi

  cat <<'EOF' > "$W/st_missing.log"
'SporosX1.stage1_oracle_exists_target' depends on axioms: [propext, Classical.choice, Quot.sound]
'SporosX1.stage1_final_target' depends on axioms: [propext, Classical.choice, Quot.sound]
EOF
  if check_axioms_log "$W/st_missing.log"; then
    echo "FAIL: missing theorem negative control passed" >&2; return 1
  fi

  echo "PASS: sporos_stage1_proofs selftest"
}

case "${1:-}" in
  --selftest)
    selftest
    exit $?
    ;;
  "")
    ;;
  *)
    echo "usage: $0 [--selftest]" >&2
    exit 2
    ;;
esac

selftest >/dev/null

command -v lake >/dev/null 2>&1 || { echo "FAIL: lake not found" >&2; exit 1; }

( cd "$ROOT/formal" &&
  for target in \
    Bootstrap.SporosX1Image \
    Bootstrap.SporosX1FinalDefs \
    Bootstrap.SporosX1FinalLoader \
    Bootstrap.SporosX1FinalParser \
    Bootstrap.SporosX1FinalRefuse \
    Bootstrap.SporosX1FinalChkRefuse \
    Bootstrap.SporosX1FinalWalkRefuse \
    Bootstrap.SporosX1FinalCompose \
    Bootstrap.SporosX1Final \
    BootstrapSporosStage1; do
    lake build "$target"
  done
)

cat <<'EOF' > "$W/axioms.lean"
import Bootstrap.SporosX1Final
#print axioms SporosX1.stage1_oracle_exists_target
#print axioms SporosX1.stage1_final_target
#print axioms SporosX1.stage1_final_exit2_target
EOF

( cd "$ROOT/formal" && lake env lean "$W/axioms.lean" ) > "$W/axioms.log" 2>&1 || {
  cat "$W/axioms.log" >&2
  echo "FAIL: lake env lean axioms check failed" >&2
  exit 1
}

cat "$W/axioms.log"
check_axioms_log "$W/axioms.log" || {
  echo "FAIL: unexpected or missing axioms in SporosX1 final theorems" >&2
  exit 1
}

echo "SPOROS_STAGE1_PROOFS_OK"
