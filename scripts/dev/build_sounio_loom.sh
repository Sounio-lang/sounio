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
language_authority_output="$ROOT_DIR/tools/loom/.runtime/sounio-loom-language-authority-runtime"
if [[ -n "${SOUNIO_LOOM_LANGUAGE_AUTHORITY_PREBUILT:-}" ]]; then
  [[ -x "$SOUNIO_LOOM_LANGUAGE_AUTHORITY_PREBUILT" ]] || {
    echo 'error: SOUNIO_LOOM_LANGUAGE_AUTHORITY_PREBUILT is not executable' >&2
    exit 1
  }
  mkdir -p "$(dirname "$language_authority_output")"
  install -m 0755 "$SOUNIO_LOOM_LANGUAGE_AUTHORITY_PREBUILT" \
    "$language_authority_output"
else
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_OUTPUT="$language_authority_output" \
    "$SCRIPT_DIR/build_sounio_loom_language_authority.sh"
fi
"$SCRIPT_DIR/build_sounio_loom_continuity_adapter.sh"
"$SCRIPT_DIR/build_sounio_loom_obligation_adapter.sh"
"$SCRIPT_DIR/build_sounio_loom_epistemic_adapter.sh"
"$SCRIPT_DIR/build_sounio_loom_attention_adapter.sh"
"$SCRIPT_DIR/build_sounio_loom_portfolio_attention_adapter.sh"
"$SCRIPT_DIR/build_sounio_loom_contingent_policy_adapter.sh"
"$SCRIPT_DIR/build_sounio_loom_outcome_authority_adapter.sh"
"$SCRIPT_DIR/build_sounio_loom_witness_mesh_adapter.sh"
"$SCRIPT_DIR/build_sounio_loom_witness_mesh_v1_adapter.sh"
"$SCRIPT_DIR/build_sounio_loom_witness_epoch_handoff_adapter.sh"
"$SCRIPT_DIR/build_sounio_loom_witness_epoch_transparency_adapter.sh"
printf 'BUILT path=%s ocaml=%s dune=%s\n' \
  "$ROOT_DIR/tools/loom/_build/default/src/loom.exe" \
  "$(ocamlopt -version)" "$(dune --version)"
