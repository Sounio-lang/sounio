#!/usr/bin/env bash
# Linux Slurm worker, UID1000. This gate never rebuilds or copies the compiler.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out=${PHYSICS_UNCERTAINTY_OUT:-${1:-}}
[[ $(id -u) == 1000 && $(uname -s) == Linux ]] || { echo 'requires Linux worker UID1000' >&2; exit 64; }
[[ $out == /* && -x ${MADAROS_RAW_BIN:?set explicit source-built compiler} ]] || exit 64
mkdir -p "$out"
[[ -z $(find "$out" -mindepth 1 -maxdepth 1 -print -quit) ]] || { echo 'refusing nonempty output directory' >&2; exit 65; }
cd "$repo"
export SOUNIO_STDLIB_PATH="$repo/stdlib" SOUNIO_SOUC_ENGINE=madaros
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288
sha256sum "$MADAROS_RAW_BIN" > "$out/COMPILER.sha256"
find stdlib/physics stdlib/units stdlib/math stdlib/epistemic stdlib/constants -type f -print0 | sort -z | xargs -0 sha256sum > "$out/INPUTS.sha256"
sha256sum tests/run-pass/physics_uncertainty_{checked,legacy,factor}.sio tests/reference/physics_uncertainty_reference.cpp scripts/stdlib/physics_uncertainty_gate.sh > "$out/GATE.sha256"
g++ --version > "$out/CXX_TOOLCHAIN.txt"
printf 'phase\tfixture\texit_status\n' > "$out/RESULTS.tsv"
record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$out/RESULTS.tsv"; }
compile() {
 local src=$1 name=$2 rc=0
 timeout 180 bin/souc check "$src" > "$out/$name.check.txt" 2>&1 || rc=$?
 record check "$name" "$rc";[[ $rc == 0 ]] || return "$rc"
 timeout 240 bin/souc compile "$src" -o "$out/$name.elf" > "$out/$name.compile.txt" 2>&1 || rc=$?
 record compile "$name" "$rc";[[ $rc == 0 ]] || return "$rc"
 chmod +x "$out/$name.elf"
 sha256sum "$src" "$out/$name.elf" > "$out/$name.identities"
}
native() {
 local src=$1 name=$2 marker=$3 rc=0
 compile "$src" "$name"
 /usr/bin/time -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' -o "$out/$name.resource" timeout 30 "$out/$name.elf" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?
 record native "$name" "$rc";[[ $rc == 0 ]] || return "$rc"
 grep -Fxq "$marker" "$out/$name.stdout"
 # A bounded acceptance fixture, not a claim of asymptotic improvement.
 awk -F= '$1=="elapsed_seconds" && $2>10 {exit 1} $1=="max_rss_kib" && $2>131072 {exit 1}' "$out/$name.resource"
 record resource-bound "$name" 0
}
native tests/run-pass/physics_uncertainty_checked.sio checked PHYSICS_UNCERTAINTY_CHECKED_PASS
native tests/run-pass/physics_uncertainty_legacy.sio legacy PHYSICS_UNCERTAINTY_LEGACY_PASS
native tests/run-pass/physics_uncertainty_factor.sio factor PHYSICS_COVARIANCE_FACTOR_PASS
native stdlib/physics/classical.sio classical 'CLASSICAL E2E REAL ALL PASS'
native stdlib/physics/sr.sio sr 'SR CHECKED ALL PASS'
native tests/run-pass/physics_sr_compton.sio compton 'PHYSICS_SR_COMPTON_PASS'
timeout 180 g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/physics_uncertainty_reference.cpp -o "$out/reference" > "$out/reference.compile.txt" 2>&1
record cpp-compile reference 0
for mode in checked legacy factor;do
 timeout 30 "$out/reference" "$mode" "$out/$mode.stdout" > "$out/$mode.reference.stdout" 2> "$out/$mode.reference.stderr"
 record concordance "$mode" 0
 # Both a plausible corrupted result and missing rows must be rejected.
 awk 'BEGIN{done=0} $1=="PHYSICS_SCALAR" && $2==0 && !done {$4="4607182418800017408";done=1} $1=="PHYSICS_LEGACY" && $2==0 && !done {$3="4607182418800017408";done=1} {print}' "$out/$mode.stdout" > "$out/$mode.corrupt.stdout"
 sed '/^PHYSICS_\(SCALAR\|LEGACY\) 0 /d' "$out/$mode.stdout" > "$out/$mode.missing.stdout"
 for control in corrupt missing;do
  rc=0;timeout 30 "$out/reference" "$mode" "$out/$mode.$control.stdout" > "$out/$mode.$control.reference.stdout" 2> "$out/$mode.$control.reference.stderr" || rc=$?
  [[ $rc == 1 ]] || { echo "reference accepted $mode $control control" >&2;exit 1; }
  record oracle-rejection "$mode:$control" 0
 done
done
# Real public legacy calls must reject unavailable results before a return marker.
# Keep each process independent so a panic cannot skip other invalid scenarios.
for kind in gamma negative_u lightcone units boost;do
 case "$kind" in
 gamma) call='let r=gamma(299792458.0,0.0)' ;;
 negative_u) call='let r=time_dilation(1.0,-1.0,0.0,0.0)' ;;
 lightcone) call='let r=invariant_mass(299792458.0,0.0,1.0,0.0)' ;;
 units) call='let m=quantity_new(1.0,0.0,dim_time());let a=quantity_new(1.0,0.0,dim_acceleration());let r=classical_force(&m,&a)' ;;
 boost) call='let r=boost_e_p(1.0,0.0,0.0,0.0,299792458.0,0.0)' ;;
 esac
 cat > "$out/reject_$kind.sio" <<SIO
use physics::sr::*
use physics::classical::{classical_force}
use units::lib::*
fn main()->i32 with IO,Mut,Div,Panic {
 println("PHYSICS_REJECTION_STARTED")
 $call;
 println("PHYSICS_REJECTION_RETURNED")
 0
}
SIO
 compile "$out/reject_$kind.sio" "reject_$kind"
 rc=0;timeout 30 "$out/reject_$kind.elf" > "$out/reject_$kind.stdout" 2> "$out/reject_$kind.stderr" || rc=$?
 [[ $rc == 1 ]] && grep -Fxq PHYSICS_REJECTION_STARTED "$out/reject_$kind.stdout" && ! grep -Fxq PHYSICS_REJECTION_RETURNED "$out/reject_$kind.stdout"
 record legacy-rejection "$kind" 0
done
# Detect source/ELF replacement during validation.
sha256sum -c "$out/COMPILER.sha256" > "$out/COMPILER.verify.txt"
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.verify.txt"
sha256sum -c "$out/GATE.sha256" > "$out/GATE.verify.txt"
printf 'PHYSICS_UNCERTAINTY_GATE_PASS\n' | tee "$out/FINAL.txt"
find "$out" -type f ! -name ARTIFACTS.sha256 -print0 | sort -z | xargs -0 sha256sum > "$out/ARTIFACTS.sha256"
