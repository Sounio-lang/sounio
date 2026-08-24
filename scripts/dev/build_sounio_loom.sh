#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"

command -v ocamlopt >/dev/null 2>&1 || {
  echo 'error: ocamlopt is required to build Sounio Loom' >&2
  exit 1
}
command -v dune >/dev/null 2>&1 || {
  echo 'error: dune is required to build Sounio Loom' >&2
  exit 1
}
ocamlfind query cryptokit >/dev/null 2>&1 || {
  echo 'error: the OCaml cryptokit package is required to build Sounio Loom' >&2
  exit 1
}
command -v openssl >/dev/null 2>&1 || {
  echo 'error: OpenSSL is required for Loom Ed25519 receipt verification' >&2
  exit 1
}

dune build --root "$ROOT_DIR/tools/loom" src/loom.exe
"$SCRIPT_DIR/build_sounio_loom_continuity_adapter.sh"
printf 'BUILT path=%s ocaml=%s dune=%s\n' \
  "$ROOT_DIR/tools/loom/_build/default/src/loom.exe" \
  "$(ocamlopt -version)" "$(dune --version)"
