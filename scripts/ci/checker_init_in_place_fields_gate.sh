#!/usr/bin/env bash
# checker_init_in_place must assign every field of `struct Checker`.
#
# The checker entry points heap-allocate a Checker and call
# checker_init_in_place on it instead of moving checker_new()'s ~164 KB value.
# A field the initializer skips keeps whatever the allocation held. Under the
# seed (lean_single) that is a zeroed inline value, which happens to work. In a
# Madaros-built compiler an aggregate field (Name, array, struct) is a
# reference, so a skipped one is null and the first read traps: gen2 died with
# ud2 (rc 132) on `(*c).current_fn_name` for any program that declares a
# function, because five fields were missing (#2966).
#
# Usage: checker_init_in_place_fields_gate.sh [--selftest] [check.sio]
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

missing_fields() {
  local src="$1"
  local fields inited
  fields="$(awk '/^pub struct Checker \{/{p=1; next} p && /^\}/{exit} p' "$src" \
    | sed -E 's://.*$::' \
    | grep -oE '^[[:space:]]+(pub )?[a-z_][a-z_0-9]*[[:space:]]*:' \
    | sed -E 's/^[[:space:]]+(pub )?//; s/[[:space:]]*:$//')"
  inited="$(awk '/^pub fn checker_init_in_place\(/{p=1; next} p && /^\}/{exit} p' "$src" \
    | grep -oE '\(\*raw\)\.[a-z_][a-z_0-9]*[[:space:]]*(\[[^]]*\])?[[:space:]]*=[^=]' \
    | sed -E 's/^\(\*raw\)\.//; s/[[:space:]\[].*$//; s/=.*$//' | sort -u)"
  [[ -n "$fields" ]] || { echo "could not read struct Checker fields from $src" >&2; return 2; }
  [[ -n "$inited" ]] || { echo "could not read checker_init_in_place from $src" >&2; return 2; }
  local f
  while IFS= read -r f; do
    grep -qx "$f" <<<"$inited" || echo "$f"
  done <<<"$fields"
}

if [[ "${1:-}" == "--selftest" ]]; then
  src="${2:-$ROOT_DIR/self-hosted/check/check.sio}"
  tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
  # Drop the current_fn_name assignment: the gate must name exactly that field.
  awk '/^pub fn checker_init_in_place\(/{p=1} p && /\(\*raw\)\.current_fn_name = /{next} {print} p && /^\}/{p=0}' "$src" >"$tmp"
  got="$(missing_fields "$tmp" || true)"
  if [[ "$got" != "current_fn_name" ]]; then
    echo "checker_init_in_place_fields_gate: SELFTEST FAIL (expected current_fn_name, got: ${got:-<none>})" >&2
    exit 1
  fi
  echo "checker_init_in_place_fields_gate: selftest OK (a dropped assignment is named)"
  exit 0
fi

src="${1:-$ROOT_DIR/self-hosted/check/check.sio}"
missing="$(missing_fields "$src")"
if [[ -n "$missing" ]]; then
  echo "checker_init_in_place_fields_gate: FAIL: checker_init_in_place does not assign these Checker fields:" >&2
  sed 's/^/  /' <<<"$missing" >&2
  echo "  Initialise each as checker_new() does; a skipped aggregate field is null in a Madaros-built compiler." >&2
  exit 1
fi
echo "checker_init_in_place_fields_gate: PASS (every Checker field is assigned)"
