#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

BASE_HEAD="4bc8c609d7d342d9dcf5fb8358d229fb99b70a24"
PLACE_HEAD="bfc6a7ca2436451946ff239164d7f6454824ac1f"
PLACE_PATH="self-hosted/ir/place_v0.sio"
PLACE_BLOB="7b00b7bd7b838856f6b297a47b0d0496b16512cb"
PLACE_SHA256="b4613b4eb40afef8ca03ff9b02ec08fab59e82a16bd72096f006e6daf48e7e91"
ARENA_HEAD="e226d70ce23f513a8e1fef527171624cf5653301"
ARENA_PATH="self-hosted/ir/arena_v2_shadow.sio"
ARENA_BLOB="17b92116da84e0c2ec4d2ef3860cc3b0378de4dc"
ARENA_SHA256="8ac4b0c4e9b9441fc21072ff6258d44afd2d9d094659d2aecb9839f25ccf6e23"
WRITER_HEAD="02f876b48d4656eb5f68695d92ea20eeb29d4ef6"
WRITER_PATH="self-hosted/ir/soir_writer.sio"
WRITER_BLOB="bb8634991af5e26d1c74e570bcb09fca292e8a2b"
WRITER_SHA256="1b9b683158f6ff50783617d66a03d35186ca8206ee39cb308d1cd29b53655bf2"
COMPILER="/tmp/sounio-d1-compiler-899/souc-stage2"
COMPILER_SHA256="204dc3665af5bb1cc4dff298bcfffe15f5331d7d4604cbd3d49648724c2b9476"
WITNESS="tests/native-v2/place_arena_soir_d1_witness.sio"
GATE="scripts/dev/place_arena_soir_d1_shadow_gate.sh"
RECEIPT_JSON="/tmp/sounio-place-arena-soir-d1-receipt.json"

usage() {
  echo "usage: $0 [--compiler-elf /absolute/path] [--receipt-json /absolute/path]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compiler-elf)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      COMPILER="$2"
      shift 2
      ;;
    --receipt-json)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      RECEIPT_JSON="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

[[ "$RECEIPT_JSON" = /* ]] || { echo "receipt path must be absolute" >&2; exit 2; }
[[ "$COMPILER" = /* ]] || { echo "compiler ELF path must be absolute" >&2; exit 2; }
[[ -x "$COMPILER" ]] || { echo "missing Stage2 compiler: $COMPILER" >&2; exit 1; }
git merge-base --is-ancestor "$BASE_HEAD" HEAD || {
  echo "exact #899 head $BASE_HEAD is not an ancestor of $(git rev-parse HEAD)" >&2
  exit 1
}

BASE_SNAPSHOT_EQUALITY=false
if [[ "$(git rev-parse HEAD)" == "$BASE_HEAD" ]]; then
  BASE_SNAPSHOT_EQUALITY=true
  mapfile -t precommit_paths < <(git status --short | sed 's/^...//' | sort)
  [[ "${precommit_paths[*]}" == "$GATE $WITNESS" ]] || {
    echo "pre-commit delta is not exactly the two authorized files" >&2
    printf '%s\n' "${precommit_paths[@]}" >&2
    exit 1
  }
else
  mapfile -t committed_paths < <(git diff --name-only "$BASE_HEAD"..HEAD | sort)
  [[ "${committed_paths[*]}" == "$GATE $WITNESS" ]] || {
    echo "committed delta from exact #899 is not exactly the two authorized files" >&2
    printf '%s\n' "${committed_paths[@]}" >&2
    exit 1
  }
  [[ -z "$(git status --short)" ]] || {
    echo "post-commit gate requires a clean worktree" >&2
    exit 1
  }
fi

TMP="$(mktemp -d /tmp/sounio-place-arena-soir-d1.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
PLACE_SNAPSHOT="$TMP/place_v0.sio"
ARENA_SNAPSHOT="$TMP/arena_v2_shadow.sio"
WRITER_SNAPSHOT="$TMP/soir_writer.sio"
COMPOSITE="$TMP/place_arena_soir_d1_composite.sio"
ELF="$TMP/place_arena_soir_d1"
OUTPUT="$TMP/output.txt"

verify_snapshot() {
  local head="$1" path="$2" blob="$3" content_sha="$4" output="$5"
  [[ "$(git rev-parse "$head:$path")" == "$blob" ]] || {
    echo "object mismatch for $head:$path" >&2
    exit 1
  }
  git show "$head:$path" > "$output"
  [[ "$(sha256sum "$output" | awk '{print $1}')" == "$content_sha" ]] || {
    echo "content mismatch for $head:$path" >&2
    exit 1
  }
}

verify_snapshot "$PLACE_HEAD" "$PLACE_PATH" "$PLACE_BLOB" "$PLACE_SHA256" "$PLACE_SNAPSHOT"
verify_snapshot "$ARENA_HEAD" "$ARENA_PATH" "$ARENA_BLOB" "$ARENA_SHA256" "$ARENA_SNAPSHOT"
verify_snapshot "$WRITER_HEAD" "$WRITER_PATH" "$WRITER_BLOB" "$WRITER_SHA256" "$WRITER_SNAPSHOT"
[[ "$(sha256sum "$COMPILER" | awk '{print $1}')" == "$COMPILER_SHA256" ]] || {
  echo "Stage2 compiler hash mismatch" >&2
  exit 1
}

# D1 remains absent from all default imports and compiler/bootstrap entrypoints.
if rg -n 'place_arena_soir_d1' self-hosted/ir/mod.sio self-hosted/compiler/main.sio scripts/bootstrap/bootstrap_concat.sh >/dev/null; then
  echo "D1 unexpectedly appears in a default pipeline surface" >&2
  exit 1
fi

sed '/^module ir::arena_v2_shadow$/d' "$ARENA_SNAPSHOT" > "$COMPOSITE"
printf '\n' >> "$COMPOSITE"
cat "$PLACE_SNAPSHOT" >> "$COMPOSITE"
printf '\n' >> "$COMPOSITE"
cat "$WITNESS" >> "$COMPOSITE"

COMPOSITE_SHA256="$(sha256sum "$COMPOSITE" | awk '{print $1}')"
WITNESS_SHA256="$(sha256sum "$WITNESS" | awk '{print $1}')"
GATE_SHA256="$(sha256sum "$GATE" | awk '{print $1}')"

# Exactly one composite compile and one emitted-ELF execution prove both lanes.
timeout 300 "$COMPILER" "$COMPOSITE" "$ELF"
[[ -f "$ELF" ]] || { echo "compiler did not emit an ELF" >&2; exit 1; }
chmod +x "$ELF"
ELF_SHA256="$(sha256sum "$ELF" | awk '{print $1}')"
timeout 60 "$ELF" | tee "$OUTPUT"

grep -Fx 'D1_RAW_CONTROL original_value=42 n1_value=42 collision=true value_preserved=true path_order_preserved=false status=information_loss code=301' "$OUTPUT" >/dev/null
grep -Fx 'D1_STRUCTURED path=Deref/Field/Index value=42 logical_id=7001 source_arena=101 fresh_arena=202 rekeyed=true status=pass' "$OUTPUT" >/dev/null
grep -Fx 'D1_NEGATIVE name=projection_type_layout_mismatch status=pass code=201' "$OUTPUT" >/dev/null
grep -Fx 'D1_NEGATIVE name=write_to_shared status=pass code=202 place_code=110' "$OUTPUT" >/dev/null
grep -Fx 'D1_NEGATIVE name=cross_module_identity status=pass code=203' "$OUTPUT" >/dev/null
grep -Fx 'D1_NEGATIVE name=stale_runtime_handle status=pass code=204' "$OUTPUT" >/dev/null
grep -Fx 'PLACE_ARENA_SOIR_D1_PASS same_build=true default_pipeline=false legacy_kept=true promotion_ready=false writer_contract_only=true actual_soir_version=none' "$OUTPUT" >/dev/null

mkdir -p "$(dirname "$RECEIPT_JSON")"
jq -n \
  --arg base_head "$BASE_HEAD" \
  --argjson base_snapshot_equality "$BASE_SNAPSHOT_EQUALITY" \
  --arg place_head "$PLACE_HEAD" \
  --arg place_blob "$PLACE_BLOB" \
  --arg place_sha256 "$PLACE_SHA256" \
  --arg arena_head "$ARENA_HEAD" \
  --arg arena_blob "$ARENA_BLOB" \
  --arg arena_sha256 "$ARENA_SHA256" \
  --arg writer_head "$WRITER_HEAD" \
  --arg writer_blob "$WRITER_BLOB" \
  --arg writer_sha256 "$WRITER_SHA256" \
  --arg compiler_path "$COMPILER" \
  --arg compiler_sha256 "$COMPILER_SHA256" \
  --arg witness_sha256 "$WITNESS_SHA256" \
  --arg gate_sha256 "$GATE_SHA256" \
  --arg composite_sha256 "$COMPOSITE_SHA256" \
  --arg elf_sha256 "$ELF_SHA256" \
  '{
    schema: "sounio.place-arena-soir-d1-shadow.v1",
    status: "pass",
    base: {head: $base_head, ancestor: true, snapshot_equality: $base_snapshot_equality},
    place: {head: $place_head, object: $place_blob, content_sha256: $place_sha256},
    arena: {head: $arena_head, object: $arena_blob, content_sha256: $arena_sha256},
    writer_contract: {head: $writer_head, object: $writer_blob, content_sha256: $writer_sha256},
    stage2_compiler: {path: $compiler_path, sha256: $compiler_sha256, compile_count: 1},
    witness: {sha256: $witness_sha256},
    gate: {sha256: $gate_sha256},
    composite: {sha256: $composite_sha256},
    emitted_elf: {sha256: $elf_sha256, run_count: 1},
    raw_control: {original_value: 42, n1_value: 42, collision: true, value_preserved: true, path_order_preserved: false, classification: "information_loss"},
    structured: {path: ["Deref", "Field", "Index"], rekeyed: true},
    negatives: [
      {name: "projection_type_layout_mismatch", code: 201, status: "pass"},
      {name: "write_to_shared", code: 202, place_code: 110, status: "pass"},
      {name: "cross_module_identity", code: 203, status: "pass"},
      {name: "stale_runtime_handle", code: 204, status: "pass"}
    ],
    same_build: true,
    default_pipeline: false,
    legacy_kept: true,
    promotion_ready: false,
    writer_contract_only: true,
    actual_soir_version: null,
    compiler_selftest_claimed: false
  }' > "$RECEIPT_JSON"

RECEIPT_SHA256="$(sha256sum "$RECEIPT_JSON" | awk '{print $1}')"
echo "PLACE_ARENA_SOIR_D1_GATE_PASS"
echo "base_head=$BASE_HEAD"
echo "place_head=$PLACE_HEAD place_blob=$PLACE_BLOB place_sha256=$PLACE_SHA256"
echo "arena_head=$ARENA_HEAD arena_blob=$ARENA_BLOB arena_sha256=$ARENA_SHA256"
echo "writer_head=$WRITER_HEAD writer_blob=$WRITER_BLOB writer_sha256=$WRITER_SHA256"
echo "compiler_sha256=$COMPILER_SHA256 composite_sha256=$COMPOSITE_SHA256 elf_sha256=$ELF_SHA256"
echo "receipt=$RECEIPT_JSON receipt_sha256=$RECEIPT_SHA256"
