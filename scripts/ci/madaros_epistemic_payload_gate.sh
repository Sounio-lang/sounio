#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WITNESS_GLOB="${SOUNIO_WITNESS_GLOB:-tests/compiler/epistemic_payload_gate/*.sio}"
WORK="$(mktemp -d /tmp/sounio-epistemic-payload.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

shopt -s nullglob
witnesses=( $WITNESS_GLOB )
shopt -u nullglob
if [[ ${#witnesses[@]} -ne 4 ]]; then
  echo "epistemic-payload-gate: FAIL expected 4 witnesses from $WITNESS_GLOB, got ${#witnesses[@]}" >&2
  exit 1
fi

normalized=""
for src in "${witnesses[@]}"; do
  current="$(sed -E 's/Epistemic\([0-9]+\)/Epistemic(N)/' "$src")"
  if [[ -z "$normalized" ]]; then
    normalized="$current"
  elif [[ "$current" != "$normalized" ]]; then
    echo "epistemic-payload-gate: FAIL witnesses differ outside Epistemic(N): $src" >&2
    exit 1
  fi
done

compile_one() {
  local engine="$1" src="$2" out="$3" log="$4"
  if [[ "$engine" == "madaros" ]]; then
    MADAROS_RAW_BIN="${MADAROS_RAW_BIN:?current-source Madaros ELF required}" \
      "$ROOT_DIR/bin/souc" compile "$src" -o "$out" >"$log" 2>&1
  else
    SOUNIO_SOUC_ENGINE=lean_single \
      "$ROOT_DIR/bin/souc" compile "$src" -o "$out" >"$log" 2>&1
  fi
}

fails=0
for engine in madaros lean_single; do
  echo "ENGINE=$engine"
  for src in "${witnesses[@]}"; do
    base="$(basename "$src" .sio)"
    n="${base#n}"
    out="$WORK/${engine}_${base}.elf"
    log="$WORK/${engine}_${base}.log"
    rm -f "$out"
    set +e
    compile_one "$engine" "$src" "$out" "$log"
    rc=$?
    set -e
    if [[ "$n" == "400" ]]; then
      if [[ $rc -eq 0 && -s "$out" ]]; then
        verdict="PASS"
      else
        verdict="FAIL"
        fails=$((fails + 1))
      fi
      echo "N=$n expected=accept rc=$rc elf=$([[ -s "$out" ]] && echo yes || echo no) verdict=$verdict"
    else
      diag=no
      grep -q "EpistemicComplete violation" "$log" && diag=yes
      if [[ $rc -ne 0 && ! -e "$out" && "$diag" == yes ]]; then
        verdict="PASS"
      else
        verdict="FAIL"
        fails=$((fails + 1))
      fi
      echo "N=$n expected=reject rc=$rc elf=$([[ -e "$out" ]] && echo yes || echo no) diagnostic=$diag verdict=$verdict"
    fi
    if [[ "$verdict" == "FAIL" ]]; then
      sed -n '1,20p' "$log" | sed 's/^/  | /'
    fi
  done
done

if [[ $fails -ne 0 ]]; then
  echo "epistemic-payload-gate: FAIL ($fails)"
  exit 1
fi
echo "epistemic-payload-gate: PASS"
