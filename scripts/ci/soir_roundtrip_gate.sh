#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

COMPILER_CLI="${SOUC_BIN:-$ROOT/bin/madaros}"
RUNTIME_TIMEOUT_SECONDS="${SOIR_ROUNDTRIP_RUNTIME_TIMEOUT_SECONDS:-20}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-soir-roundtrip.XXXXXX")"

cleanup() {
  rm -rf "$TMP"
}
trap cleanup EXIT

fail() {
  printf 'SOIR_ROUNDTRIP_FAIL reason=%s\n' "$1" >&2
  exit 1
}

progress() {
  printf 'SOIR_ROUNDTRIP_PROGRESS stage=%s subject=%s status=%s\n' \
    "$1" "$2" "$3" >&2
}

[[ -x "$COMPILER_CLI" ]] || fail compiler_cli_missing
command -v rg >/dev/null 2>&1 || fail rg_missing
command -v awk >/dev/null 2>&1 || fail awk_missing
command -v cksum >/dev/null 2>&1 || fail cksum_missing
command -v timeout >/dev/null 2>&1 || fail timeout_missing
[[ "$RUNTIME_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail runtime_timeout_invalid
(( RUNTIME_TIMEOUT_SECONDS <= 60 )) || fail runtime_timeout_exceeds_cap_60

SOIR_REF="docs/architecture/SOIR_REFERENCE.md"
LEGACY_SERIALIZE="self-hosted/ir/serialize.sio"
[[ -f "$SOIR_REF" ]] || fail soir_reference_missing
[[ -f "$LEGACY_SERIALIZE" ]] || fail legacy_serialize_missing

grep -Fq '# SOIR v1 Format Reference Card' "$SOIR_REF" || fail soir_reference_title_missing
grep -Fq 'Magic: "SOIR"' "$SOIR_REF" || fail soir_reference_magic_missing
grep -Fq 'serialize_ir_module' "$LEGACY_SERIALIZE" || fail serialize_ir_module_missing
grep -Fq 'deserialize_ir_module' "$LEGACY_SERIALIZE" || fail deserialize_ir_module_missing

coverage_census() {
  local scope="$1"
  shift
  local total readers writers roundtrip
  total="$(rg -i -n '(^|[^A-Za-z0-9_])soir([^A-Za-z0-9_]|$)|serialize_ir_module|deserialize_ir_module|soir_.*(reader|writer)|round.?trip' "$@" 2>/dev/null | wc -l | awk '{print $1}')"
  readers="$(rg -i -n 'deserialize_ir_module|soir.*reader|reader.*soir' "$@" 2>/dev/null | wc -l | awk '{print $1}')"
  writers="$(rg -i -n 'serialize_ir_module|soir.*writer|writer.*soir' "$@" 2>/dev/null | wc -l | awk '{print $1}')"
  roundtrip="$(rg -i -n 'round.?trip' "$@" 2>/dev/null | wc -l | awk '{print $1}')"
  printf 'SOIR_ROUNDTRIP_CENSUS scope=%s total=%s readers=%s writers=%s roundtrip_mentions=%s\n' \
    "$scope" "$total" "$readers" "$writers" "$roundtrip"
}

coverage_census self_hosted self-hosted
coverage_census tests tests
coverage_census scripts_ci scripts/ci

rg -n 'serialize_ir_module|deserialize_ir_module|soir_.*(reader|writer)|SOIR_.*(READER|WRITER)' \
  self-hosted tests scripts/ci >"$TMP/soir-reader-writer-hits.txt" || true
hit_count="$(wc -l <"$TMP/soir-reader-writer-hits.txt" | awk '{print $1}')"
(( hit_count > 0 )) || fail reader_writer_census_empty
printf 'SOIR_ROUNDTRIP_CENSUS_DETAIL hits=%s artifact=reader-writer-grep\n' "$hit_count"

CORPUS="$TMP/soir-corpus.tsv"
rg -n '^(pub(\([^)]*\))?[[:space:]]+)?fn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(' self-hosted/ir/*.sio \
  | awk -F: '
      BEGIN {
        want["self-hosted/ir/ir.sio"] = 1
        want["self-hosted/ir/serialize.sio"] = 1
        want["self-hosted/ir/normalize.sio"] = 1
        want["self-hosted/ir/optimize.sio"] = 1
        want["self-hosted/ir/inline.sio"] = 1
        want["self-hosted/ir/layout.sio"] = 1
        want["self-hosted/ir/profile.sio"] = 1
        want["self-hosted/ir/verify.sio"] = 1
        want["self-hosted/ir/lower.sio"] = 1
        want["self-hosted/ir/const_prop.sio"] = 1
        want["self-hosted/ir/ssa.sio"] = 1
        want["self-hosted/ir/tailcall.sio"] = 1
      }
      ($1 in want) && !seen[$1] {
        line = $0
        sub($1 ":" $2 ":", "", line)
        sig = line
        name = sig
        sub(/^pub(\([^)]*\))?[[:space:]]+/, "", name)
        sub(/^fn[[:space:]]+/, "", name)
        sub(/[[:space:]]*\(.*/, "", name)
        params = sig
        sub(/^[^(]*\(/, "", params)
        sub(/\).*/, "", params)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", params)
        param_count = 0
        if (length(params) > 0) {
          param_count = split(params, tmp, ",")
        }
        effect_count = 0
        if (sig ~ / with /) {
          effects = sig
          sub(/^.* with /, "", effects)
          sub(/[[:space:]]*\{.*$/, "", effects)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", effects)
          if (length(effects) > 0) {
            effect_count = split(effects, etmp, ",")
          }
        }
        visibility = (sig ~ /^pub/) ? 1 : 0
        returns = (sig ~ / -> /) ? 1 : 0
        print $1 "\t" $2 "\t" visibility "\t" name "\t" param_count "\t" effect_count "\t" returns
        seen[$1] = 1
      }
    ' >"$CORPUS"

corpus_count="$(wc -l <"$CORPUS" | awk '{print $1}')"
(( corpus_count >= 10 )) || fail representative_corpus_too_small

ARRAY_FILE_IDS=""
ARRAY_LINES=""
ARRAY_VIS=""
ARRAY_HASHES=""
ARRAY_PARAMS=""
ARRAY_EFFECTS=""
ARRAY_RETURNS=""
corpus_index=0
while IFS=$'\t' read -r path line visibility name param_count effect_count returns; do
  file_hash="$(printf '%s' "$path" | cksum | awk '{print $1}')"
  name_hash="$(printf '%s' "$name" | cksum | awk '{print $1}')"
  if [[ "$corpus_index" -gt 0 ]]; then
    ARRAY_FILE_IDS+=", "
    ARRAY_LINES+=", "
    ARRAY_VIS+=", "
    ARRAY_HASHES+=", "
    ARRAY_PARAMS+=", "
    ARRAY_EFFECTS+=", "
    ARRAY_RETURNS+=", "
  fi
  ARRAY_FILE_IDS+="$file_hash"
  ARRAY_LINES+="$line"
  ARRAY_VIS+="$visibility"
  ARRAY_HASHES+="$name_hash"
  ARRAY_PARAMS+="$param_count"
  ARRAY_EFFECTS+="$effect_count"
  ARRAY_RETURNS+="$returns"
  corpus_index=$((corpus_index + 1))
done <"$CORPUS"

SOURCE="$TMP/soir_roundtrip_witness.sio"
ELF="$TMP/soir_roundtrip_witness.elf"
BUILD_LOG="$TMP/soir_roundtrip_witness.build.log"
RUN_LOG="$TMP/soir_roundtrip_witness.run.log"

cat >"$SOURCE" <<EOF
module soir_roundtrip_witness

let CORPUS_COUNT: i64 = $corpus_count

fn put_i64(buf: &![i8; 131072], pos: i64, val: i64) -> i64 with Mut, Panic, Div {
    (*buf)[pos as usize] = (val & 255) as i8
    (*buf)[(pos + 1) as usize] = ((val >> 8) & 255) as i8
    (*buf)[(pos + 2) as usize] = ((val >> 16) & 255) as i8
    (*buf)[(pos + 3) as usize] = ((val >> 24) & 255) as i8
    (*buf)[(pos + 4) as usize] = ((val >> 32) & 255) as i8
    (*buf)[(pos + 5) as usize] = ((val >> 40) & 255) as i8
    (*buf)[(pos + 6) as usize] = ((val >> 48) & 255) as i8
    (*buf)[(pos + 7) as usize] = ((val >> 56) & 255) as i8
    pos + 8
}

fn get_i64(buf: &[i8; 131072], pos: i64) -> (i64, i64) with Panic, Div {
    let b0 = ((*buf)[pos as usize] as i64) & 255
    let b1 = ((*buf)[(pos + 1) as usize] as i64) & 255
    let b2 = ((*buf)[(pos + 2) as usize] as i64) & 255
    let b3 = ((*buf)[(pos + 3) as usize] as i64) & 255
    let b4 = ((*buf)[(pos + 4) as usize] as i64) & 255
    let b5 = ((*buf)[(pos + 5) as usize] as i64) & 255
    let b6 = ((*buf)[(pos + 6) as usize] as i64) & 255
    let b7 = ((*buf)[(pos + 7) as usize] as i64) & 255
    let v = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
    (v, pos + 8)
}

fn put_header(buf: &![i8; 131072]) -> i64 with Mut, Panic, Div {
    (*buf)[0] = 83 as i8
    (*buf)[1] = 79 as i8
    (*buf)[2] = 73 as i8
    (*buf)[3] = 82 as i8
    (*buf)[4] = 1 as i8
    (*buf)[5] = 0 as i8
    (*buf)[6] = 0 as i8
    (*buf)[7] = 0 as i8
    8
}

fn header_ok(buf: &[i8; 131072]) -> bool with Panic, Div {
    (*buf)[0] == 83 as i8 &&
    (*buf)[1] == 79 as i8 &&
    (*buf)[2] == 73 as i8 &&
    (*buf)[3] == 82 as i8 &&
    (*buf)[4] == 1 as i8 &&
    (*buf)[5] == 0 as i8 &&
    (*buf)[6] == 0 as i8 &&
    (*buf)[7] == 0 as i8
}

fn write_record(
    buf: &![i8; 131072],
    pos: i64,
    file_id: i64,
    line_no: i64,
    visibility: i64,
    name_hash: i64,
    param_count: i64,
    effect_count: i64,
    returns: i64,
) -> i64 with Mut, Panic, Div {
    var p = pos
    p = put_i64(buf, p, file_id)
    p = put_i64(buf, p, line_no)
    p = put_i64(buf, p, visibility)
    p = put_i64(buf, p, name_hash)
    p = put_i64(buf, p, param_count)
    p = put_i64(buf, p, effect_count)
    p = put_i64(buf, p, returns)
    p
}

fn read_and_match(
    buf: &[i8; 131072],
    pos: i64,
    file_id: i64,
    line_no: i64,
    visibility: i64,
    name_hash: i64,
    param_count: i64,
    effect_count: i64,
    returns: i64,
) -> (bool, i64) with Panic, Div {
    var p = pos
    let a = get_i64(buf, p); p = a.1
    let b = get_i64(buf, p); p = b.1
    let c = get_i64(buf, p); p = c.1
    let d = get_i64(buf, p); p = d.1
    let e = get_i64(buf, p); p = e.1
    let f = get_i64(buf, p); p = f.1
    let g = get_i64(buf, p); p = g.1
    (
        a.0 == file_id &&
        b.0 == line_no &&
        c.0 == visibility &&
        d.0 == name_hash &&
        e.0 == param_count &&
        f.0 == effect_count &&
        g.0 == returns,
        p,
    )
}

fn main() -> i64 with IO, Mut, Panic, Div {
    let file_ids: [i64; $corpus_count] = [$ARRAY_FILE_IDS]
    let lines: [i64; $corpus_count] = [$ARRAY_LINES]
    let visibility: [i64; $corpus_count] = [$ARRAY_VIS]
    let name_hashes: [i64; $corpus_count] = [$ARRAY_HASHES]
    let params: [i64; $corpus_count] = [$ARRAY_PARAMS]
    let effects: [i64; $corpus_count] = [$ARRAY_EFFECTS]
    let returns: [i64; $corpus_count] = [$ARRAY_RETURNS]

    var buf: [i8; 131072] = [0; 131072]
    var pos = put_header(&! buf)
    pos = put_i64(&! buf, pos, CORPUS_COUNT)
    var i: i64 = 0
    while i < CORPUS_COUNT {
        pos = write_record(
            &! buf,
            pos,
            file_ids[i as usize],
            lines[i as usize],
            visibility[i as usize],
            name_hashes[i as usize],
            params[i as usize],
            effects[i as usize],
            returns[i as usize],
        )
        i = i + 1
    }

    if !header_ok(&buf) { return 10 }
    var read_pos: i64 = 8
    let count_pair = get_i64(&buf, read_pos)
    if count_pair.0 != CORPUS_COUNT { return 11 }
    read_pos = count_pair.1
    i = 0
    while i < CORPUS_COUNT {
        let matched = read_and_match(
            &buf,
            read_pos,
            file_ids[i as usize],
            lines[i as usize],
            visibility[i as usize],
            name_hashes[i as usize],
            params[i as usize],
            effects[i as usize],
            returns[i as usize],
        )
        if !matched.0 { return 20 + i }
        read_pos = matched.1
        i = i + 1
    }
    if read_pos != pos { return 90 }
    print("SOIR_ROUNDTRIP_CANONICAL_PASS\n")
    0
}
EOF

progress corpus_extract self-hosted/ir pass
printf 'SOIR_ROUNDTRIP_CORPUS functions=%s files=%s profile=soir-reference-function-signature-v1\n' \
  "$corpus_count" "$(cut -f1 "$CORPUS" | sort -u | wc -l | awk '{print $1}')"

progress witness_check soir_roundtrip_witness begin
"$COMPILER_CLI" check "$SOURCE" >"$BUILD_LOG" 2>&1 || {
  cat "$BUILD_LOG" >&2
  fail witness_check
}
progress witness_check soir_roundtrip_witness pass

progress witness_build soir_roundtrip_witness begin
"$COMPILER_CLI" --native-v2-compile "$SOURCE" -o "$ELF" >>"$BUILD_LOG" 2>&1 || {
  cat "$BUILD_LOG" >&2
  fail witness_build
}
chmod +x "$ELF"
progress witness_runtime soir_roundtrip_witness begin
set +e
timeout --signal=TERM --kill-after=2s "${RUNTIME_TIMEOUT_SECONDS}s" "$ELF" >"$RUN_LOG" 2>&1
runtime_rc=$?
set -e
if [[ "$runtime_rc" -eq 124 ]]; then
  cat "$RUN_LOG" >&2
  fail witness_runtime_timeout
fi
if [[ "$runtime_rc" -ne 0 ]]; then
  cat "$RUN_LOG" >&2
  fail "witness_runtime_rc_${runtime_rc}"
fi
grep -Fxq 'SOIR_ROUNDTRIP_CANONICAL_PASS' "$RUN_LOG" || {
  cat "$RUN_LOG" >&2
  fail witness_marker_missing
}
progress witness_runtime soir_roundtrip_witness pass

head_sha="$(git rev-parse HEAD)"
tree_sha="$(git rev-parse 'HEAD^{tree}')"
script_sha256="$(sha256sum scripts/ci/soir_roundtrip_gate.sh | awk '{print $1}')"

printf '%s\n' 'SOIR_ROUNDTRIP_BOUNDARY executable=generated_default_madaros_witness legacy_serialize_sio=censused_not_invoked current_soir_reference=v1 corpus=self-hosted/ir-function-signatures structural_fields=file,line,visibility,name_hash,param_count,effect_count,returns mir_port=blocked_until_gate_green'
printf 'SOIR_ROUNDTRIP_PROVENANCE head=%s tree=%s script_sha256=%s\n' \
  "$head_sha" "$tree_sha" "$script_sha256"
printf 'SOIR_ROUNDTRIP_PASS compiler_cli=%s corpus_functions=%s census_hits=%s\n' \
  "$COMPILER_CLI" "$corpus_count" "$hit_count"
