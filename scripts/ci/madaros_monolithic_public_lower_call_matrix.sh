#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MATRIX_MANIFEST="$ROOT_DIR/tests/compiler/monolithic_public_lower_call_matrix.tsv"
BIN="${SOUNIO_MADAROS_MONO_PUBLIC_LOWER_BIN:-}"

fail() {
  echo "[madaros-mono-public-lower-matrix] FAIL: $*" >&2
  exit 1
}

[[ -n "$BIN" ]] || fail "SOUNIO_MADAROS_MONO_PUBLIC_LOWER_BIN is required"
[[ -x "$BIN" ]] || fail "explicit Madaros ELF is missing or not executable: $BIN"
[[ -f "$MATRIX_MANIFEST" ]] || fail "matrix manifest is missing: $MATRIX_MANIFEST"

EXPECTED_MATRIX=$'empty\ttests/compiler/fixtures/monolithic_public_lower_call/empty.sio\nmain_only\ttests/compiler/fixtures/monolithic_public_lower_call/main_only.sio\nlocal_i64_add\ttests/compiler/fixtures/monolithic_public_lower_call/local_i64_add.sio\nlocal_f64_add\ttests/compiler/fixtures/monolithic_public_lower_call/local_f64_add.sio\nglobal_i64\ttests/compiler/fixtures/monolithic_public_lower_call/global_i64.sio\nglobal_f64\ttests/compiler/fixtures/monolithic_public_lower_call/global_f64.sio\nbss_no_binop\ttests/compiler/fixtures/monolithic_public_lower_call/bss_no_binop.sio\nbss_typed_adds\ttests/compiler/fixtures/monolithic_public_lower_call/bss_typed_adds.sio'
ACTUAL_MATRIX="$(awk -F '\t' '$1 !~ /^#/ && NF >= 2 {print $1 "\t" $2}' "$MATRIX_MANIFEST")"
[[ "$ACTUAL_MATRIX" == "$EXPECTED_MATRIX" ]] \
  || fail "matrix manifest must contain the exact eight ordered label/source rows"

if [[ -n "${SOUNIO_MADAROS_MONO_PUBLIC_LOWER_MATRIX_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_MONO_PUBLIC_LOWER_MATRIX_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing matrix directory: $WORK"
  mkdir "$WORK" || fail "could not create matrix directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-mono-public-lower-matrix.XXXXXX)"
fi

ELF_PATH="$(readlink -f "$BIN")"
ELF_SIZE="$(stat -c %s "$BIN")"
ELF_SHA256="$(sha256sum "$BIN" | cut -d' ' -f1)"

set +e
"$BIN" --version-json >"$WORK/identity.log" 2>&1
IDENTITY_RC=$?
set -e
IDENTITY_SHA256="$(sha256sum "$WORK/identity.log" | cut -d' ' -f1)"

{
  printf 'schema\tsounio.monolithic-public-lower-call.matrix.v1\n'
  printf 'elf_path\t%s\n' "$ELF_PATH"
  printf 'elf_size\t%s\n' "$ELF_SIZE"
  printf 'elf_sha256\t%s\n' "$ELF_SHA256"
  printf 'identity_rc\t%s\n' "$IDENTITY_RC"
  printf 'identity_sha256\t%s\n' "$IDENTITY_SHA256"
} >"$WORK/metadata.tsv"

printf 'elf_sha256\tidentity_sha256\tlabel\tsource\trc\tbegin\tdone\tlast_boundary\n' >"$WORK/receipt.tsv"

matrix_failed=0
if [[ "$IDENTITY_RC" != "0" ]]; then
  matrix_failed=1
fi

while IFS=$'\t' read -r label source; do
  if [[ -z "$label" || "$label" == \#* ]]; then
    continue
  fi
  [[ -n "$source" ]] || fail "manifest row has no source for label: $label"
  [[ -f "$ROOT_DIR/$source" ]] || fail "fixture is missing for $label: $source"

  set +e
  env -u SOUNIO_MADAROS_BIN MADAROS_RAW_BIN="$BIN" \
    "$ROOT_DIR/bin/madaros" --probe-monolithic-global-f64 "$ROOT_DIR/$source" \
    >"$WORK/$label.log" 2>&1
  rc=$?
  set -e

  begin=no
  done_marker=no
  if grep -Fxq 'probe_monolithic_public_lower_call: lower_begin' "$WORK/$label.log" \
    || grep -Fxq 'probe_monolithic_global_f64: lower_begin' "$WORK/$label.log"; then
    begin=yes
  fi
  if grep -Fxq 'probe_monolithic_public_lower_call: lower_done' "$WORK/$label.log" \
    || grep -Fxq 'probe_monolithic_global_f64: lower_done' "$WORK/$label.log"; then
    done_marker=yes
  fi
  last_boundary="$(sed -n 's/^probe_monolithic_public_lower_boundary: //p' "$WORK/$label.log" | tail -n 1)"
  if [[ -z "$last_boundary" ]]; then
    last_boundary=none
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ELF_SHA256" "$IDENTITY_SHA256" "$label" "$source" "$rc" "$begin" "$done_marker" "$last_boundary" \
    >>"$WORK/receipt.tsv"

  if [[ "$rc" != "0" || "$begin" != "yes" || "$done_marker" != "yes" ]]; then
    matrix_failed=1
  fi
done <"$MATRIX_MANIFEST"

cat "$WORK/metadata.tsv"
cat "$WORK/receipt.tsv"

if [[ "$matrix_failed" != "0" ]]; then
  fail "one or more public lower call rows did not complete"
fi

echo "[madaros-mono-public-lower-matrix] PASS: all public lower call rows completed"
