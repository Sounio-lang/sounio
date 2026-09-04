#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOOM="$ROOT/tools/loom/_build/default/src/loom.exe"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-activation-epoch-ocaml.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
fail(){ printf 'sounio-loom-activation-epoch-ocaml-selftest: FAIL: %s\n' "$*" >&2; exit 1; }
flock -x "$ROOT/tools/loom/_build/.dune-build.lock" dune build --root "$ROOT/tools/loom" src/loom.exe >/dev/null
bash "$ROOT/scripts/dev/build_sounio_loom_activation_epoch.sh" >/dev/null
R="$WORK/runtime"; S="$WORK/state"; P="$S/generation-runtime-pins"
mkdir -p "$R/versions" "$P"
make_runtime(){
  local id="$1" bridge="${2:-false}" d="$R/versions/$1"
  mkdir -p "$d/bin" "$d/hooks"
  cp "$LOOM" "$d/bin/sounio-loom-runtime"; cp "$LOOM" "$d/bin/sounio-coord-runtime"
  if [[ "$bridge" == true ]]; then printf 'forbidden\n' >"$d/hooks/sounio_coord_agent_hook_runtime.py"; fi
  printf '%s\n' "runtime_id=$id" \
    "loom_runtime_sha256=$(sha256sum "$d/bin/sounio-loom-runtime"|cut -d' ' -f1)" \
    "coord_runtime_sha256=$(sha256sum "$d/bin/sounio-coord-runtime"|cut -d' ' -f1)" \
    'source_sha=test' 'capability=loom-generation-pinned-cutover-v1' \
    'capability=loom-activation-epoch-v1' >"$d/manifest"
  chmod 600 "$d/manifest"; chmod 555 "$d/bin/"*
}
make_runtime runtime-a; make_runtime runtime-b; make_runtime runtime-c; make_runtime runtime-python true
ln -s versions/runtime-a "$R/current"
printf '%s\n' 'schema=loom-generation-pin-set-v1' 'state=SEALED' \
  'old_runtime_id=runtime-old' 'candidate_runtime_id=runtime-a' \
  'inventory_sha256=initial' 'pin_count=1' 'semantic_authority=Sounio' \
  'action=9048' 'semantics_sha256=9a323d98a6c732e0a7f70a6d50cf684e5039eb2af211e5f891fd0c9761351549' \
  'freeze_sha256=0765d7e941a5def05e8ae7d08a90c7826491c86b4c1efc8679b40a6a728de29d' >"$P/activation.v1"
printf '%s\n' 'schema=loom-generation-runtime-pin-v1' 'runtime_id=runtime-a' >"$P/test.pin"
chmod 600 "$P/activation.v1" "$P/test.pin"
INITIAL_SHA="$(sha256sum "$P/activation.v1"|cut -d' ' -f1)"; PIN_SHA="$(sha256sum "$P/test.pin"|cut -d' ' -f1)"
advance(){ SOUNIO_COORD_RUNTIME_DIR="$R" SOUNIO_COORD_DIR="$S" "$LOOM" hook-activation-epoch-advance --source-root "$ROOT" --git-common "$WORK" --next-runtime "$1"; }
advance runtime-b | grep -q 'action=9049 epoch=1'
[[ "$(sha256sum "$P/activation-epochs/heads/$INITIAL_SHA.activation.v1"|cut -d' ' -f1)" == "$INITIAL_SHA" ]] || fail 'initial receipt not preserved'
rm "$R/current"; ln -s versions/runtime-b "$R/current"
advance runtime-c | grep -q 'action=9049 epoch=2'
[[ "$(find "$P/activation-epochs/epochs" -type f -name '*.epoch.v1'|wc -l)" == 2 ]] || fail 'epoch chain length'
[[ "$(sha256sum "$P/test.pin"|cut -d' ' -f1)" == "$PIN_SHA" ]] || fail 'pin changed'
grep -q '^candidate_runtime_id=runtime-c$' "$P/activation.v1" || fail 'compatibility head not advanced'
set +e; OUT="$(advance runtime-python 2>&1)"; RC=$?; set -e
[[ $RC -ne 0 && "$OUT" == *'next-runtime-python-bridge'* ]] || fail 'Python bridge was not refused pre-execution'
[[ "$(sha256sum "$P/test.pin"|cut -d' ' -f1)" == "$PIN_SHA" ]] || fail 'negative test changed pin'
printf 'sounio-loom-activation-epoch-ocaml-selftest: PASS semantic_authority=Sounio action=9049 projection_language=OCaml role=OPERATIONAL_PARITY epochs=2 initial_receipt_preserved=true pins_immutable=true python_oracle_attempt=PRE_EXEC_REFUSED python_executed=false rust_executed=false\n'
