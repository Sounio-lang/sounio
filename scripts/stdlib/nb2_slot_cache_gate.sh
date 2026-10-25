#!/usr/bin/env bash
# Run on an allocated Slurm CPU, never the workstation.
# Requires independently qualified frozen before/after roots from the same
# compiler pins. This gate measures real 512-row profiles; it does not rebuild
# source, silently choose compilers, alter quota, or publish anything.
set -euo pipefail
: "${NB2_CACHE_BEFORE_ROOT:?frozen uncached source/out receipt root}"
: "${NB2_CACHE_AFTER_ROOT:?frozen cached source/out receipt root}"
: "${NB2_CACHE_OUTPUT:?fresh absolute output directory}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ "$NB2_CACHE_OUTPUT" == /* ]] || { echo "output must be absolute" >&2;exit 2; }
[[ "$(cat "$NB2_CACHE_BEFORE_ROOT/STATUS")" == 0 ]]
[[ "$(cat "$NB2_CACHE_AFTER_ROOT/STATUS")" == 0 ]]
mkdir "$NB2_CACHE_OUTPUT"
out="$NB2_CACHE_OUTPUT"
trap 's=$?;echo "$s">"$out/STATUS"' EXIT
ulimit -s 524288
ulimit -c 0
sha256sum "$ROOT/tests/reference/nb2_slot_cache_reference.cpp" "$0" \
 "$NB2_CACHE_BEFORE_ROOT/source/stdlib/stats/regression/nb2_likelihood.sio" \
 "$NB2_CACHE_AFTER_ROOT/source/stdlib/stats/regression/nb2_likelihood.sio" > "$out/SOURCE.sha256"
id>"$out/identity.txt"
g++ --version>"$out/compiler.txt"
g++ -std=c++23 -O2 -Wall -Wextra -Werror "$ROOT/tests/reference/nb2_slot_cache_reference.cpp" -lquadmath -o "$out/reference" >"$out/build.stdout" 2>"$out/build.stderr"
for engine in madaros lean_single;do
 "$out/reference" "$NB2_CACHE_BEFORE_ROOT/out/$engine.stdlib_nb2_slot_cache.native.stdout" "$NB2_CACHE_AFTER_ROOT/out/$engine.stdlib_nb2_slot_cache.native.stdout" >"$out/$engine.scalar-reference.stdout" 2>"$out/$engine.scalar-reference.stderr"
 for scenario in profile_main control_main;do
  for role in A B;do
   base="$NB2_CACHE_BEFORE_ROOT"
   if [[ "$role" == B ]];then base="$NB2_CACHE_AFTER_ROOT";fi
   sha256sum "$base/out/$engine.$scenario.elf">>"$out/ELFS.sha256"
  done
  pair=0
  for order in AB BA AB;do
   pair=$((pair+1))
   roles="A B";if [[ "$order" == BA ]];then roles="B A";fi
   for role in $roles;do
    base="$NB2_CACHE_BEFORE_ROOT"
    if [[ "$role" == B ]];then base="$NB2_CACHE_AFTER_ROOT";fi
    name="$out/$engine.$scenario.$pair.$role"
    /usr/bin/time -v "$base/out/$engine.$scenario.elf">"$name.stdout" 2>"$name.time.txt"
    cmp "$NB2_CACHE_BEFORE_ROOT/out/$engine.$scenario.native.stdout" "$name.stdout"
    printf '%s\t%s\t%s\t%s\t%s\n' "$engine" "$scenario" "$pair" "$role" "$name.time.txt">>"$out/samples.tsv"
   done
  done
 done
done
"$out/reference" --perf "$out/samples.tsv">"$out/performance.stdout" 2>"$out/performance.stderr"
sha256sum "$out/reference" "$out"/*.stdout "$out"/*.time.txt "$out/samples.tsv">"$out/ARTIFACTS.sha256"
echo NB2_SLOT_CACHE_PAIRED_GATE_PASS
