#!/usr/bin/env bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/gate_artifact.sh"
# fn_type_effect_ratchet_gate.sh — freeze the number of function types that
# carry no effect clause, so the gap to SOUNIO-SPEC-06 §6.0 cannot widen.
#
# §6.0 (founder ruling, 2026-08-19): "A function type carries the effects of the
# function." At the time of the ruling, 559 function types occurred in live .sio
# source and NOT ONE declared an effect. This gate does not implement the
# ruling. It stops the distance from growing while the ruling is unimplemented:
# a new bare function type fails; converting one to carry effects passes and
# lowers the frozen count.
#
# Why a ratchet and not a check: refusing all 559 today would refuse the whole
# repository. Refusing the 560th costs nothing and is the only thing that can be
# true right now.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
# Shared assertions rather than hand-rolled ones: gate_vacuity_gate.sh requires
# them, and it is right to. count_matches separates "no match" from "the tool
# broke", which `grep -c ... || true` collapses into the same 0 — the exact error
# that voided a measurement of mine earlier today.
. "scripts/lib/gate_assert.sh"
gate_name "fn_type_effect_ratchet"

REF="scripts/ci/fn_type_effect_ratchet.frozen"
REF_LIST="scripts/ci/fn_type_effect_ratchet.frozen.list"
OUT="${GATE_ARTIFACT:-artifacts/gates/fn_type_effect_ratchet.json}"

# A function TYPE is `fn(` — no name between `fn` and `(`. A function
# DECLARATION is `fn name(`. The distinction is the whole instrument.
# Only PARAMETER position. A function type in return position ("-> fn(i64) ->
# i64 with Mut") sits on the same line as the enclosing declaration's own `with`
# clause, and no line-local pattern can tell the two apart — an earlier revision
# of this gate counted the outer clause as the type's and undercounted bare
# types. Parameter position is anchored by the `:` that introduces the
# annotation, so the capture cannot reach the outer clause.
PAT_TYPE=':[[:space:]]*fn\([^)]*\)[[:space:]]*->'

strip_noise() {
  # drop // line comments and "..." string literals before matching, so a
  # function type written inside prose or a message is not counted.
  sed -e 's|//.*$||' -e 's/"[^"]*"//g' "$1"
}

enumerate() {
  git ls-files -z '*.sio' \
    | tr '\0' '\n' \
    | grep -vE '^(archive|bootstrap)/' \
    | grep -vE '\.sio\.old$' \
    | while IFS= read -r f; do
        [ -f "$f" ] || continue
        strip_noise "$f" | grep -oE "${PAT_TYPE}[^,)]*" | while IFS= read -r hit; do
          # a bare function type is one whose text carries no `with` clause
          printf '%s' "$hit" | grep -qE '\bwith[[:space:]]+[A-Za-z]' || printf '%s\t%s\n' "$f" "$hit"
        done
      done
}

selftest() {
  local tmp rc=0
  tmp="$(mktemp -d)"
  # POSITIVE control: a bare function type must be seen.
  printf 'fn deriv(f: fn(f64) -> f64, x: f64) -> f64 with Div { 0.0 }\n' > "$tmp/pos.sio"
  if strip_noise "$tmp/pos.sio" | grep -oE "${PAT_TYPE}[^,)]*" | grep -q .; then
    echo "  ok   POSITIVO: tipo-funcao nu e detectado"
  else echo "  FALHA POSITIVO: nao detectou um tipo-funcao nu"; rc=1; fi
  # NEGATIVE control 1: a function DECLARATION must not be counted as a type.
  printf 'fn soma(a: i64, b: i64) -> i64 { a + b }\n' > "$tmp/neg1.sio"
  if strip_noise "$tmp/neg1.sio" | grep -oE "${PAT_TYPE}" | grep -q .; then
    echo "  FALHA NEGATIVO 1: contou uma DECLARACAO como tipo-funcao"; rc=1
  else echo "  ok   NEGATIVO 1: declaracao nao conta como tipo"; fi
  # NEGATIVE control 2: a function type inside a comment must not be counted.
  printf '// takes fn(f64) -> f64 as the kernel\nfn k(x: i64) -> i64 { x }\n' > "$tmp/neg2.sio"
  if strip_noise "$tmp/neg2.sio" | grep -oE "${PAT_TYPE}" | grep -q .; then
    echo "  FALHA NEGATIVO 2: contou um tipo-funcao dentro de comentario"; rc=1
  else echo "  ok   NEGATIVO 2: comentario nao conta"; fi
  # NEGATIVE control 3: a function type that DOES carry effects must not be
  # reported as bare — otherwise the ratchet can never be lowered.
  printf 'fn m(f: fn(f64) -> f64 with Div, x: f64) -> f64 { 0.0 }\n' > "$tmp/neg3.sio"
  if strip_noise "$tmp/neg3.sio" | grep -oE "${PAT_TYPE}[^,)]*" \
       | grep -qvE '\bwith[[:space:]]+[A-Za-z]'; then
    echo "  FALHA NEGATIVO 3: tipo-funcao COM efeitos contado como nu"; rc=1
  else echo "  ok   NEGATIVO 3: tipo com efeitos nao conta como nu"; fi
  # NEGATIVE control 4: a function type in RETURN position must not be counted,
  # because the `with` on that line belongs to the enclosing declaration.
  printf 'fn select_op(w: i64) -> fn(i64) -> i64 with Mut, Panic { f }\n' > "$tmp/neg4.sio"
  if strip_noise "$tmp/neg4.sio" | grep -oE "${PAT_TYPE}" | grep -q .; then
    echo "  FALHA NEGATIVO 4: contou um tipo-funcao em posicao de RETORNO"; rc=1
  else echo "  ok   NEGATIVO 4: posicao de retorno nao conta"; fi
  rm -rf "$tmp"
  echo "falhas: $rc"
  return $rc
}

[ "${1:-}" = "--selftest" ] && { selftest; exit $?; }

selftest >/dev/null 2>&1 || {
  echo "ABORT: the gate's own controls fail — its number would be noise, not evidence."
  selftest
  exit 2
}

# Anti-vacuity: the sweep must see the corpus at all. If enumerate returns
# nothing because the pattern rotted or the file list came back empty, that is a
# broken instrument, not a repository with zero bare function types.
ficheiros=$(git ls-files '*.sio' | grep -vE '^(archive|bootstrap)/' | wc -l | tr -d ' ')
require_nonempty "$ficheiros" "the .sio file list came back empty"
require_min_count "$ficheiros" 500 "live .sio files"

[ "${1:-}" = "--list" ] && { enumerate | sort; exit 0; }

atual=$(enumerate | wc -l | tr -d ' ')
require_nonempty "$atual" "the bare-function-type count came back empty"
[ -f "$REF" ] || printf '%s\n' "$atual" | gate_write_artifact "$REF"
congelado=$(head -1 "$REF" | tr -d ' ')

# The count and the list must describe the same corpus. The sibling ratchet
# carried .frozen=472 against a 474-line .frozen.list for weeks, which is how a
# count gets lowered without regenerating what it summarises.
if [ -f "$REF_LIST" ]; then
  lista_n=$(wc -l < "$REF_LIST" | tr -d ' ')
  if [ "$lista_n" != "$congelado" ]; then
    echo "REFUSE: ${REF} says ${congelado} but ${REF_LIST} holds ${lista_n} lines." >&2
    echo "  They must describe the same corpus. Regenerate the list:" >&2
    echo "    bash $0 --list > ${REF_LIST}" >&2
    exit 1
  fi
fi

mkdir -p "$(dirname "$OUT")"
estado=pass; falhou=0
if [ "$atual" -gt "$congelado" ]; then
  estado=fail; falhou=1
  echo "REFUSE: bare function types rose ${congelado} -> ${atual}."
  echo "SOUNIO-SPEC-06 §6.0 rules that a function type carries the function's effects."
  echo "A new function type without a 'with' clause widens the gap to that ruling."
  # Name the sites by SET DIFFERENCE, never by `tail -n <delta>`.
  #
  # This used to print `enumerate | tail -n $(( atual - congelado ))` -- the last
  # N lines of the whole corpus in git ls-files order, which has nothing to do
  # with which sites are new. Measured 2026-08-30 on #2225: it named six sites in
  # tools/test-framework/src/lib.sio, a file that PR does not touch, while the
  # six real ones were in examples/higher_order.sio and examples/newton_root.sio.
  # A gate that reports a violation and points at the wrong file sends the reader
  # to edit innocent code; that is worse than reporting a bare count.
  if [ -f "$REF_LIST" ]; then
    echo "New sites (present now, absent from ${REF_LIST}):"
    comm -13 "$REF_LIST" <(enumerate | sort) | sed 's/^/  /'
  else
    echo "New sites: cannot say -- ${REF_LIST} is missing, so there is nothing to"
    echo "  diff against. Regenerate it with:"
    echo "    bash $0 --list > ${REF_LIST}"
    echo "  Reporting the count only, rather than guessing at filenames."
  fi
elif [ "$atual" -lt "$congelado" ]; then
  echo "OK: bare function types fell ${congelado} -> ${atual}. Lower the frozen count:"
  echo "  printf '%s\\n' ${atual} > ${REF}"
else
  echo "OK: bare function types hold at ${congelado}."
fi

cat <<JSON | gate_write_artifact "$OUT"
{
  "gate": "fn_type_effect_ratchet",
  "status": "${estado}",
  "spec_section": "SOUNIO-SPEC-06",
  "frozen": ${congelado},
  "measured": ${atual},
  "metrics": { "total": ${atual}, "passed": $(( atual - falhou )), "failed": ${falhou}, "not_run": 0 }
}
JSON
exit "${falhou}"
