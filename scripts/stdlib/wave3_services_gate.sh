#!/usr/bin/env bash
# Mandatory focused service acceptance. Run inside a scheduled Slurm job.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
# Metadata mode evaluates only execution authority; it cannot accept any fixture.
if [[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]]; then
  AUTHORITY_KIND=slurm; AUTHORITY_RUN="$SLURM_JOB_ID"; AUTHORITY_JOB=scheduled
elif [[ "${GITHUB_ACTIONS:-}" == true && "${GITHUB_RUN_ID:-}" =~ ^[1-9][0-9]*$ && "${GITHUB_JOB:-}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  AUTHORITY_KIND=github-actions; AUTHORITY_RUN="$GITHUB_RUN_ID"; AUTHORITY_JOB="$GITHUB_JOB"
else
  echo "requires scheduled Slurm or identified official GitHub Actions context" >&2; exit 2
fi
authority_json() { printf '{"kind":"%s","run_id":"%s","job":"%s"}\n' "$AUTHORITY_KIND" "$AUTHORITY_RUN" "$AUTHORITY_JOB"; }
if [[ $# == 1 && "$1" == --execution-context ]]; then authority_json; exit 0; fi
[[ $# == 1 && "$1" == /* && "$1" =~ ^/[A-Za-z0-9_./-]+$ ]] || { echo "usage: $0 FRESH_ABSOLUTE_OUTPUT" >&2; exit 2; }
OUT="$(realpath -m -- "$1")"
[[ ! -e "$OUT" ]] || { echo "refusing existing output: $OUT" >&2; exit 2; }
: "${MADAROS_RAW_BIN:?absolute qualified raw compiler required}"
: "${STDLIB_PLATFORM_LIBRARY:?absolute qualified platform library required}"
NODE="${NODE:-node}"
CXX="${CXX:-g++}"
RAW="$(realpath -e -- "$MADAROS_RAW_BIN")"
PLATFORM="$(realpath -e -- "$STDLIB_PLATFORM_LIBRARY")"
[[ "$MADAROS_RAW_BIN" == /* && "$STDLIB_PLATFORM_LIBRARY" == /* && -x "$RAW" && -s "$PLATFORM" && "$(basename "$PLATFORM")" == libstdlib_platform.so ]] || exit 2
WRAPPER="$ROOT/scripts/ci/souc-native-wrapper.sh"
BOUNDER="$ROOT/scripts/stdlib/bounded_command.mjs"
[[ -f "$WRAPPER" && -f "$BOUNDER" && -f "$ROOT/tests/reference/stdlib_pollset_reference.cpp" && -f "$ROOT/runtime/stdlib_platform.cpp" ]] || { echo "missing mandatory gate dependency" >&2; exit 2; }
mkdir -p -- "$(dirname "$OUT")"
mkdir -- "$OUT"
mkdir -- "$OUT/inputs" "$OUT/runs"
PRIVATE="$(mktemp -d /tmp/sw3-services.XXXXXXXX)"
printf '%s\n' "$PRIVATE" > "$OUT/PRIVATE_PATH_DIRECTORY"
authority_json > "$OUT/EXECUTION_AUTHORITY.json"
date -u +%FT%TZ > "$OUT/STARTED_UTC"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_SOUC_BIN="$RAW" SOUC_NATIVE_BIN="$RAW" SOUNIO_SOUC_RAW_MODE=modular
export MADAROS_RAW_BIN="$RAW" SOUNIO_SOUC_ENGINE=madaros SOUNIO_REQUIRE_COMMITTED_MADAROS=0
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export LD_LIBRARY_PATH="$(dirname "$PLATFORM")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ulimit -s 524288
ulimit -c 0
finish() {
  local status=$? changed=0
  trap - EXIT
  if [[ -d "$PRIVATE" && ! -e "$OUT/private-files" ]]; then cp -a -- "$PRIVATE" "$OUT/private-files" || changed=1; fi
  find "$OUT/runs" "$OUT/private-files" -type f -print0 | sort -z | xargs -0 -r sha256sum > "$OUT/ARTIFACTS.sha256" || changed=1
  for manifest in SOURCES TOOLS DERIVED EXECUTABLES ARTIFACTS; do
    if [[ -f "$OUT/$manifest.sha256" ]]; then
      [[ -s "$OUT/$manifest.sha256" ]] || continue
      sha256sum -c "$OUT/$manifest.sha256" > "$OUT/$manifest.final.verify" 2>&1 || changed=1
    fi
  done
  [[ "$changed" == 0 ]] || status=1
  date -u +%FT%TZ > "$OUT/ENDED_UTC"
  printf '%s\n' "$status" > "$OUT/STATUS"
  if [[ "$status" == 0 ]]; then echo WAVE3_SERVICES_GATE_OK; fi
  exit "$status"
}
trap finish EXIT
"$NODE" --input-type=module - "$RAW" "$PLATFORM" <<'JS'
import fs from 'node:fs';
for(const p of process.argv.slice(2)){
 const f=fs.openSync(p,'r'),b=Buffer.alloc(4);fs.readSync(f,b,0,4,0);fs.closeSync(f);
 if(!b.equals(Buffer.from([127,69,76,70])))throw Error('expected real ELF: '+p);
}
JS
cat > "$OUT/PLAN.tsv" <<'PLAN'
tests/stdlib/massive/test_csv_checked_numeric.sio	CSV_CHECKED_NUMERIC_OK
tests/stdlib/massive/test_csv_strict.sio	CSV_STRICT_OK
tests/stdlib/massive/test_csv_strict_offsets.sio	CSV_STRICT_OFFSETS_OK
tests/stdlib/massive/test_native_csv.sio	NATIVE_CSV_OK
tests/stdlib/massive/test_native_file.sio	NATIVE_FILE_OK
tests/stdlib/massive/test_csv_stream.sio	CSV_STREAM_OK
tests/stdlib/massive/test_json_dom.sio	JSON_DOM_OK
tests/stdlib/massive/test_json_writer.sio	JSON_WRITER_OK
tests/stdlib/massive/test_poll_executor.sio	POLL_EXECUTOR_OK
tests/stdlib/massive/test_poll_socket.sio	POLL_SOCKET_OK
tests/stdlib/massive/test_poll_set.sio	POLL_SET_OK
tests/stdlib/massive/test_poll_reap.sio	POLL_REAP_OK
tests/stdlib/massive/test_deadline_socket.sio	DEADLINE_SOCKET_OK
tests/stdlib/massive/test_dynamic_fasta.sio	DYNAMIC_FASTA_OK
tests/stdlib/massive/test_dynamic_gff.sio	DYNAMIC_GFF_OK
tests/stdlib/massive/test_vcf.sio	VCF_CORE_OK
tests/stdlib/massive/test_vcf_header.sio	VCF_HEADER_OK
tests/stdlib/massive/test_gff_graph.sio	GFF_GRAPH_OK
tests/stdlib/massive/test_vcf_document.sio	VCF_DOCUMENT_OK
tests/stdlib/data/test_csv_e2e.sio	DATA_CSV_LOADER_OK
tests/stdlib/data/test_frame_e2e.sio	DATA_FRAME_OK
tests/stdlib/massive/test_frame_nonempty.sio	FRAME_NONEMPTY_OK
tests/stdlib/massive/test_csv_loader_nonempty.sio	CSV_LOADER_NONEMPTY_OK
tests/stdlib/massive/test_series_nonempty.sio	SERIES_NONEMPTY_OK
tests/stdlib/massive/test_argparse_nonempty.sio	ARGPARSE_NONEMPTY_OK
tests/stdlib/massive/test_byte_buffer_budget.sio	BYTE_BUFFER_BUDGET_OK
tests/stdlib/massive/test_service_root_exports.sio	SERVICE_ROOT_EXPORTS_OK
tests/stdlib/massive/test_active_log_concordance.sio	ACTIVE_LOG_CONCORDANCE_OK
PLAN
(
  cd "$ROOT"
  find stdlib -type f -print0
  while IFS=$'\t' read -r source marker; do printf '%s\0' "$source"; done < "$OUT/PLAN.tsv"
  printf '%s\0' scripts/stdlib/wave3_services_gate.sh scripts/stdlib/bounded_command.mjs scripts/ci/souc-native-wrapper.sh tests/reference/stdlib_pollset_reference.cpp tests/reference/stdlib_active_log_reference.cpp runtime/stdlib_platform.cpp
) | sort -zu | while IFS= read -r -d '' path; do sha256sum "$ROOT/$path"; done > "$OUT/SOURCES.sha256"
"$CXX" --version > "$OUT/CXX_VERSION"
sha256sum "$RAW" "$PLATFORM" "$(command -v "$NODE")" "$(command -v "$CXX")" > "$OUT/TOOLS.sha256"
"$NODE" --input-type=module - "$ROOT" "$OUT" "$PRIVATE" <<'JS'
import fs from 'node:fs';import path from 'node:path';import crypto from 'node:crypto';import assert from 'node:assert/strict';
const [root,out,privateDir]=process.argv.slice(2),hash=s=>crypto.createHash('sha256').update(s).digest('hex');
const lines=fs.readFileSync(out+'/PLAN.tsv','utf8').trimEnd().split('\n'),names=new Set(),destinations=new Set(),ledger=[];
for(let index=0;index<lines.length;index++){
 const [source,marker,...extra]=lines[index].split('\t');assert(source&&marker&&!extra.length);
 const name=path.basename(source,'.sio');assert(!names.has(name),'fixture basename collision');names.add(name);
 const original=fs.readFileSync(path.join(root,source)),text=original.toString('utf8');assert(Buffer.from(text).equals(original),'non UTF8 fixture');
 const literals=[...text.matchAll(/"\/tmp\/[^"\r\n]*"/g)].map(m=>m[0]);
 const keys=[...new Set(literals)];let derived=text;const substitutions=[];
 for(let n=0;n<keys.length;n++){
  const from=keys[n],oldPath=JSON.parse(from);assert(/^\/tmp\/[A-Za-z0-9_.-]+$/.test(oldPath),'ambiguous /tmp literal');
  const newPath=privateDir+'/'+index+'-'+n+'-'+path.basename(oldPath),to=JSON.stringify(newPath);
  assert(!text.includes(newPath)&&!destinations.has(newPath)&&!fs.existsSync(newPath),'relocation collision');
  destinations.add(newPath);assert(!keys.some(k=>k!==from&&(k.includes(from)||from.includes(k))),'overlapping source paths');
  const occurrences=derived.split(from).length-1;assert(occurrences>0);
  derived=derived.split(from).join(to);substitutions.push({from,to,occurrences});
 }
 assert(keys.every(k=>!derived.includes(k)),'unrelocated original /tmp literal');
 let reversed=derived;for(const {from,to,occurrences} of [...substitutions].reverse()){
  assert.equal(reversed.split(to).length-1,occurrences);reversed=reversed.split(to).join(from);
 }
 assert(Buffer.from(reversed).equals(original),'reverse byte inequality');
 const destination=out+'/inputs/'+name+'.sio';fs.writeFileSync(destination,derived,{flag:'wx'});
 fs.writeFileSync(out+'/inputs/'+name+'.original.sio',original,{flag:'wx'});
 ledger.push({source,derived:destination,marker,original_sha256:hash(original),derived_sha256:hash(derived),reverse_byte_equal:true,substitutions});
}
fs.writeFileSync(out+'/RELOCATIONS.json',JSON.stringify(ledger,null,2)+'\n',{flag:'wx'});
JS
find "$OUT/inputs" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/DERIVED.sha256"
sha256sum "$OUT/PLAN.tsv" "$OUT/RELOCATIONS.json" >> "$OUT/DERIVED.sha256"
printf 'fixture\tphase\tstatus\n' > "$OUT/RESULTS.tsv"
: > "$OUT/EXECUTABLES.sha256"
failed=0
phase() {
  local name="$1" kind="$2" seconds="$3" bytes="$4" status=0
  shift 4
  "$NODE" "$BOUNDER" "$seconds" "$bytes" -- "$@" > "$OUT/$name.$kind.log" 2>&1 || status=$?
  printf '%s\t%s\t%s\n' "$name" "$kind" "$status" >> "$OUT/RESULTS.tsv"
  [[ "$status" == 0 ]]
}
while IFS=$'\t' read -r source marker; do
  name="$(basename "$source" .sio)"
  derived="$OUT/inputs/$name.sio"
  if ! phase "$name" check 300 4194304 bash "$WRAPPER" check "$derived"; then failed=1; continue; fi
  if ! grep -Fxq "check: OK" "$OUT/$name.check.log"; then printf '%s\tcheck-marker\t1\n' "$name" >> "$OUT/RESULTS.tsv"; failed=1; continue; fi
  if ! phase "$name" compile 300 4194304 bash "$WRAPPER" compile "$derived" -o "$OUT/$name.elf"; then failed=1; continue; fi
  if [[ ! -s "$OUT/$name.elf" || "$(od -An -tx1 -N4 "$OUT/$name.elf" | tr -d ' \n')" != 7f454c46 ]]; then printf '%s\telf\t1\n' "$name" >> "$OUT/RESULTS.tsv"; failed=1; continue; fi
  chmod +x "$OUT/$name.elf"
  sha256sum "$OUT/$name.elf" >> "$OUT/EXECUTABLES.sha256"
  mkdir "$OUT/runs/$name"
  if ! (cd "$OUT/runs/$name"; phase "$name" native 120 1048576 "$OUT/$name.elf"); then failed=1; continue; fi
  count="$(grep -Fxc -- "$marker" "$OUT/$name.native.log" || true)"
  if [[ "$count" != 1 ]]; then printf '%s\tmarker\t1\n' "$name" >> "$OUT/RESULTS.tsv"; failed=1; else printf '%s\tmarker\t0\n' "$name" >> "$OUT/RESULTS.tsv"; fi
done < "$OUT/PLAN.tsv"
LIBDIR="$(dirname "$PLATFORM")"
for mode in normal sanitized; do
  flags=(-O2)
  if [[ "$mode" == sanitized ]]; then flags=(-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined); fi
  bindings=("$PLATFORM" "-Wl,-rpath,$LIBDIR")
  if [[ "$mode" == sanitized ]]; then bindings=("$ROOT/runtime/stdlib_platform.cpp" -ldl); fi
  if ! phase "pollset-reference-$mode" compile 180 1048576 "$CXX" -std=c++23 -Wall -Wextra -Werror -pthread "${flags[@]}" "$ROOT/tests/reference/stdlib_pollset_reference.cpp" "${bindings[@]}" -o "$OUT/pollset-reference-$mode"; then failed=1; continue; fi
  sha256sum "$OUT/pollset-reference-$mode" >> "$OUT/EXECUTABLES.sha256"
  ldd "$OUT/pollset-reference-$mode" > "$OUT/pollset-reference-$mode.bound-libraries.txt"
  if [[ "$mode" == normal ]] && ! grep -Fq -- "$PLATFORM" "$OUT/pollset-reference-$mode.bound-libraries.txt"; then failed=1; continue; fi
  if ! phase "pollset-reference-$mode" native 120 1048576 env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 "$OUT/pollset-reference-$mode"; then failed=1; continue; fi
  if [[ "$(grep -Fxc STDLIB_POLLSET_UDP_ABSOLUTE_DEADLINE_OK "$OUT/pollset-reference-$mode.native.log" || true)" != 1 ]]; then failed=1; fi
done
if ! phase active-log-reference compile 180 1048576 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror "$ROOT/tests/reference/stdlib_active_log_reference.cpp" -o "$OUT/active-log-reference"; then
  failed=1
else
  sha256sum "$OUT/active-log-reference" >> "$OUT/EXECUTABLES.sha256"
  if ! phase active-log-reference native 60 1048576 "$OUT/active-log-reference" "$OUT/test_active_log_concordance.native.log"; then
    failed=1
  elif ! grep -Fxq "ACTIVE_LOG_CPP_CONCORDANCE_OK finite_rows=30 domain_rows=4 absolute_tolerance=1e-11" "$OUT/active-log-reference.native.log"; then
    failed=1
  fi
fi
expected="$(wc -l < "$OUT/PLAN.tsv")"
actual="$(awk -F '\t' 'NR>1 && $2=="marker" && $3==0 {n++} END{print n+0}' "$OUT/RESULTS.tsv")"
[[ "$actual" == "$expected" ]] || failed=1
sha256sum "$OUT/RESULTS.tsv" "$OUT/"*.bound-libraries.txt >> "$OUT/DERIVED.sha256"
[[ "$failed" == 0 ]]

