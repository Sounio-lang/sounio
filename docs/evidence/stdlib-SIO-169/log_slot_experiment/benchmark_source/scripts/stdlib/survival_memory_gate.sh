#!/usr/bin/env bash
# Paired standalone memory measurement, executed inside allocated Slurm only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BEFORE="${1:?usage: survival_memory_gate.sh BEFORE_SOURCE_ROOT FRESH_OUT}"
OUT="${2:?fresh output directory required}"
: "${MADAROS_RAW_BIN:?explicit source-built compiler required}"
[[ -z "${SOUNIO_SOUC_BIN:-}" ]] || exit 64
mkdir -p "$(dirname "$OUT")"
if ! mkdir "$OUT";then echo 'fresh output directory required' >&2;exit 65;fi
OUT="$(cd "$OUT" && pwd)";BEFORE="$(cd "$BEFORE" && pwd)"
ulimit -s 524288
cd "$ROOT"
id > "$OUT/identity.txt"
sha256sum "$MADAROS_RAW_BIN" bin/souc-lean-single-x86_64 "$BEFORE/bin/souc-lean-single-x86_64" \
 "$BEFORE/stdlib/stats/checked_survival.sio" stdlib/stats/checked_survival.sio \
 tests/reference/survival_memory_reference.cpp scripts/stdlib/survival_memory_gate.sh \
 stdlib/math/{pure,logarithmic}.sio > "$OUT/inputs.sha256"
g++ --version > "$OUT/reference.compiler.txt"
g++ -std=c++23 -O2 -Wall -Wextra tests/reference/survival_memory_reference.cpp -o "$OUT/reference" > "$OUT/reference.compile.stdout" 2> "$OUT/reference.compile.stderr"
sha256sum "$OUT/reference" > "$OUT/reference.sha256"
for n in 100000 1000000;do
 "$OUT/reference" --generate "$n" > "$OUT/n$n.generated.sio"
 sha256sum "$OUT/n$n.generated.sio" >> "$OUT/fixtures.sha256"
 for engine in lean_single madaros;do
  for version in before after;do
   if [[ "$version" == before ]];then cd "$BEFORE";else cd "$ROOT";fi
   export SOUNIO_STDLIB_PATH="$PWD/stdlib"
   env SOUNIO_SOUC_ENGINE="$engine" bin/souc check "$OUT/n$n.generated.sio" > "$OUT/$version.n$n.$engine.check.stdout" 2> "$OUT/$version.n$n.$engine.check.stderr"
   if grep -Eq '^error:|error\[E' "$OUT/$version.n$n.$engine.check.stdout" "$OUT/$version.n$n.$engine.check.stderr";then exit 1;fi
   /usr/bin/time -v env SOUNIO_SOUC_ENGINE="$engine" bin/souc compile "$OUT/n$n.generated.sio" -o "$OUT/$version.n$n.$engine.elf" > "$OUT/$version.n$n.$engine.compile.stdout" 2> "$OUT/$version.n$n.$engine.compile.stderr"
   if grep -Eq '^error:|error\[E' "$OUT/$version.n$n.$engine.compile.stdout" "$OUT/$version.n$n.$engine.compile.stderr";then exit 1;fi
   chmod +x "$OUT/$version.n$n.$engine.elf";sha256sum "$OUT/$version.n$n.$engine.elf" > "$OUT/$version.n$n.$engine.elf.sha256"
   set +e
   /usr/bin/time -v timeout 900 "$OUT/$version.n$n.$engine.elf" > "$OUT/$version.n$n.$engine.native.stdout" 2> "$OUT/$version.n$n.$engine.native.stderr"
   code=$?;echo "$code" > "$OUT/$version.n$n.$engine.native.exit"
   set -e
   if [[ "$code" == 0 ]];then
    "$OUT/reference" "$OUT/$version.n$n.$engine.native.stdout" > "$OUT/$version.n$n.$engine.verify.stdout" 2> "$OUT/$version.n$n.$engine.verify.stderr"
    sha256sum "$OUT/$version.n$n.$engine.native.stdout" > "$OUT/$version.n$n.$engine.native.sha256"
   elif [[ "$version" == after || "$n" == 100000 ]];then
    echo "required completed memory witness failed: $version n=$n engine=$engine exit=$code" >&2;exit 1
   else
    printf 'BASELINE_UNAVAILABLE n=%s engine=%s exit=%s no_ratio_claim\n' "$n" "$engine" "$code" >> "$OUT/baseline_unavailable.txt"
   fi
  done
  if [[ "$(cat "$OUT/before.n$n.$engine.native.exit")" == 0 ]];then
   cmp "$OUT/before.n$n.$engine.native.stdout" "$OUT/after.n$n.$engine.native.stdout"
   printf 'MATCHED_ALL_FIELD_DIGESTS_AND_PUBLIC_OUTPUT n=%s engine=%s\n' "$n" "$engine" >> "$OUT/comparisons.txt"
  fi
 done
done
cd "$ROOT"
sha256sum -c "$OUT/inputs.sha256" > "$OUT/inputs.verify.stdout"
sha256sum -c "$OUT/fixtures.sha256" > "$OUT/fixtures.verify.stdout"
sha256sum -c "$OUT/reference.sha256" > "$OUT/reference.verify.stdout"
printf 'SURVIVAL_MEMORY_GATE_PASS\n' | tee "$OUT/gate.result"
