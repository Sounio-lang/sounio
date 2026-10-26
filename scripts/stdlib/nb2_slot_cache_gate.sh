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
kernel=stdlib/stats/regression/nb2_likelihood.sio
cmp "$NB2_CACHE_AFTER_ROOT/source/$kernel" "$ROOT/$kernel"
printf '%s  %s\n' 26b0d7332c7f9c25d37e4c7993ddf0e43f9491c72caadfa9ee896ef2656f7d56 "$NB2_CACHE_BEFORE_ROOT/source/$kernel" | sha256sum -c - >"$out/BASELINE_KERNEL.verify.txt"
cmp "$NB2_CACHE_BEFORE_ROOT/out/COMPILERS.sha256" "$NB2_CACHE_AFTER_ROOT/out/COMPILERS.sha256"
for base in "$NB2_CACHE_BEFORE_ROOT" "$NB2_CACHE_AFTER_ROOT";do
 mapfile -t pins < "$base/out/COMPILERS.sha256"
 [[ ${#pins[@]} == 2 ]]
 [[ "${pins[0]%% *}" == a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3 ]]
 [[ "${pins[1]%% *}" == 8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc ]]
 (cd "$base/source";sha256sum -c "$base/out/COMPILERS.sha256") >>"$out/COMPILER_PINS.verify.initial.txt"
done
# The corrected stdout annotation is the sole difference from frozen callers.
for scenario in profile_main control_main;do
 caller="tests/stdlib/nb2_slot_cache/$scenario.sio"
 cmp "$NB2_CACHE_BEFORE_ROOT/source/$caller" "$NB2_CACHE_AFTER_ROOT/source/$caller"
 cmp <(tail -n +4 "$ROOT/$caller") <(tail -n +4 "$NB2_CACHE_AFTER_ROOT/source/$caller")
done
sha256sum "$ROOT/tests/reference/nb2_slot_cache_reference.cpp" "$0" \
 "$NB2_CACHE_BEFORE_ROOT/source/stdlib/stats/regression/nb2_likelihood.sio" \
 "$NB2_CACHE_AFTER_ROOT/source/stdlib/stats/regression/nb2_likelihood.sio" \
 "$ROOT/$kernel" "$ROOT/tests/stdlib/nb2_slot_cache/"{profile_main,control_main}.sio \
 "$NB2_CACHE_BEFORE_ROOT/out/COMPILERS.sha256" "$NB2_CACHE_AFTER_ROOT/out/COMPILERS.sha256" > "$out/SOURCE.sha256"
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
sha256sum -c "$out/SOURCE.sha256">"$out/SOURCE.verify.final.txt"
sha256sum -c "$out/ELFS.sha256">"$out/ELFS.verify.final.txt"
for base in "$NB2_CACHE_BEFORE_ROOT" "$NB2_CACHE_AFTER_ROOT";do
 (cd "$base/source";sha256sum -c "$base/out/COMPILERS.sha256") >>"$out/COMPILER_PINS.verify.final.txt"
done
sha256sum "$out/reference" "$out"/*.stdout "$out"/*.time.txt "$out/samples.tsv">"$out/ARTIFACTS.sha256"
echo NB2_SLOT_CACHE_PAIRED_GATE_PASS
