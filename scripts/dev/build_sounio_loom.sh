#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
LANGUAGE_AUTHORITY_MANIFEST="$ROOT_DIR/tools/loom/language_authority.freeze.v1"
EXECUTION_AUTHORITY_MANIFEST="$ROOT_DIR/tools/loom/execution_authority.freeze.v2"
EXECUTION_OUTCOME_MANIFEST="$ROOT_DIR/tools/loom/execution_outcome.freeze.v1"
LANE_HEALTH_MANIFEST="$ROOT_DIR/tools/loom/lane_health.freeze.v1"
frozen_toolchain_root=''
execution_outcome_toolchain_root=''
lane_health_toolchain_root=''

cleanup() {
  [[ -z "$frozen_toolchain_root" ]] || rm -rf "$frozen_toolchain_root"
  [[ -z "$execution_outcome_toolchain_root" ]] || rm -rf "$execution_outcome_toolchain_root"
  [[ -z "$lane_health_toolchain_root" ]] || rm -rf "$lane_health_toolchain_root"
}

prepare_execution_outcome_toolchain() {
  local executable_commit wrapper_sha compiler_sha actual_wrapper_sha actual_compiler_sha
  [[ -f "$EXECUTION_OUTCOME_MANIFEST" ]] || {
    echo 'error: frozen Sounio execution-outcome manifest is required' >&2
    exit 1
  }
  executable_commit="$(manifest_value "$EXECUTION_OUTCOME_MANIFEST" sounio_executable_commit)"
  wrapper_sha="$(manifest_value "$EXECUTION_OUTCOME_MANIFEST" toolchain_wrapper_sha256)"
  compiler_sha="$(manifest_value "$EXECUTION_OUTCOME_MANIFEST" toolchain_compiler_sha256)"
  git -C "$ROOT_DIR" cat-file -e "${executable_commit}^{commit}" 2>/dev/null || {
    echo "error: frozen execution-outcome Sounio toolchain commit is unavailable: $executable_commit" >&2
    exit 1
  }
  execution_outcome_toolchain_root="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-execution-outcome-toolchain.XXXXXX")"
  git -C "$ROOT_DIR" archive "$executable_commit" \
    bin/souc bin/souc-lean-single-x86_64 | tar -x -C "$execution_outcome_toolchain_root"
  actual_wrapper_sha="$(sha256sum "$execution_outcome_toolchain_root/bin/souc" | awk '{print $1}')"
  actual_compiler_sha="$(sha256sum "$execution_outcome_toolchain_root/bin/souc-lean-single-x86_64" | awk '{print $1}')"
  [[ "$actual_wrapper_sha" == "$wrapper_sha" && "$actual_compiler_sha" == "$compiler_sha" ]] || {
    echo 'error: reconstructed execution-outcome Sounio toolchain failed hash verification' >&2
    exit 1
  }
}

prepare_lane_health_toolchain() {
  local executable_commit wrapper_sha compiler_sha actual_wrapper_sha actual_compiler_sha
  [[ -f "$LANE_HEALTH_MANIFEST" ]] || {
    echo 'error: frozen Sounio lane-health manifest is required' >&2
    exit 1
  }
  executable_commit="$(manifest_value "$LANE_HEALTH_MANIFEST" sounio_executable_commit)"
  wrapper_sha="$(manifest_value "$LANE_HEALTH_MANIFEST" toolchain_wrapper_sha256)"
  compiler_sha="$(manifest_value "$LANE_HEALTH_MANIFEST" toolchain_compiler_sha256)"
  git -C "$ROOT_DIR" cat-file -e "${executable_commit}^{commit}" 2>/dev/null || {
    echo "error: frozen lane-health Sounio toolchain commit is unavailable: $executable_commit" >&2
    exit 1
  }
  lane_health_toolchain_root="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-lane-health-toolchain.XXXXXX")"
  git -C "$ROOT_DIR" archive "$executable_commit" \
    bin/souc bin/souc-lean-single-x86_64 | tar -x -C "$lane_health_toolchain_root"
  actual_wrapper_sha="$(sha256sum "$lane_health_toolchain_root/bin/souc" | awk '{print $1}')"
  actual_compiler_sha="$(sha256sum "$lane_health_toolchain_root/bin/souc-lean-single-x86_64" | awk '{print $1}')"
  [[ "$actual_wrapper_sha" == "$wrapper_sha" && "$actual_compiler_sha" == "$compiler_sha" ]] || {
    echo 'error: reconstructed lane-health Sounio toolchain failed hash verification' >&2
    exit 1
  }
}
trap cleanup EXIT

manifest_value() {
  local manifest="$1" key="$2"
  sed -n "s/^${key}=//p" "$manifest" | head -1
}

prepare_frozen_toolchain() {
  local executable_commit wrapper_sha compiler_sha execution_wrapper_sha
  local execution_compiler_sha actual_wrapper_sha actual_compiler_sha
  [[ -f "$LANGUAGE_AUTHORITY_MANIFEST" && -f "$EXECUTION_AUTHORITY_MANIFEST" ]] || {
    echo 'error: frozen Sounio authority manifests are required' >&2
    exit 1
  }
  executable_commit="$(manifest_value "$LANGUAGE_AUTHORITY_MANIFEST" sounio_executable_commit)"
  wrapper_sha="$(manifest_value "$LANGUAGE_AUTHORITY_MANIFEST" toolchain_wrapper_sha256)"
  compiler_sha="$(manifest_value "$LANGUAGE_AUTHORITY_MANIFEST" toolchain_compiler_sha256)"
  execution_wrapper_sha="$(manifest_value "$EXECUTION_AUTHORITY_MANIFEST" toolchain_wrapper_sha256)"
  execution_compiler_sha="$(manifest_value "$EXECUTION_AUTHORITY_MANIFEST" toolchain_compiler_sha256)"
  [[ "$wrapper_sha" == "$execution_wrapper_sha" && \
    "$compiler_sha" == "$execution_compiler_sha" ]] || {
    echo 'error: frozen Sounio authorities disagree on their toolchain' >&2
    exit 1
  }
  git -C "$ROOT_DIR" cat-file -e "${executable_commit}^{commit}" 2>/dev/null || {
    echo "error: frozen Sounio toolchain commit is unavailable: $executable_commit" >&2
    exit 1
  }
  frozen_toolchain_root="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-frozen-toolchain.XXXXXX")"
  git -C "$ROOT_DIR" archive "$executable_commit" \
    bin/souc bin/souc-lean-single-x86_64 | tar -x -C "$frozen_toolchain_root"
  actual_wrapper_sha="$(sha256sum "$frozen_toolchain_root/bin/souc" | awk '{print $1}')"
  actual_compiler_sha="$(sha256sum "$frozen_toolchain_root/bin/souc-lean-single-x86_64" | awk '{print $1}')"
  [[ "$actual_wrapper_sha" == "$wrapper_sha" && "$actual_compiler_sha" == "$compiler_sha" ]] || {
    echo 'error: reconstructed frozen Sounio toolchain failed hash verification' >&2
    exit 1
  }
}

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
if [[ -z "${SOUNIO_LOOM_LANGUAGE_AUTHORITY_PREBUILT:-}" || \
  -z "${SOUNIO_LOOM_EXECUTION_AUTHORITY_PREBUILT:-}" ]]; then
  prepare_frozen_toolchain
fi
if [[ -z "${SOUNIO_LOOM_EXECUTION_OUTCOME_PREBUILT:-}" ]]; then
  prepare_execution_outcome_toolchain
fi
if [[ -z "${SOUNIO_LOOM_LANE_HEALTH_PREBUILT:-}" || \
  -z "${SOUNIO_LOOM_LANE_HEALTH_PARITY_PREBUILT:-}" ]]; then
  prepare_lane_health_toolchain
fi
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
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_SOUC="$frozen_toolchain_root/bin/souc" \
    SOUNIO_LOOM_LANGUAGE_AUTHORITY_OUTPUT="$language_authority_output" \
    "$SCRIPT_DIR/build_sounio_loom_language_authority.sh"
fi
execution_authority_output="$ROOT_DIR/tools/loom/.runtime/sounio-loom-execution-authority-runtime"
if [[ -n "${SOUNIO_LOOM_EXECUTION_AUTHORITY_PREBUILT:-}" ]]; then
  [[ -x "$SOUNIO_LOOM_EXECUTION_AUTHORITY_PREBUILT" ]] || {
    echo 'error: SOUNIO_LOOM_EXECUTION_AUTHORITY_PREBUILT is not executable' >&2
    exit 1
  }
  mkdir -p "$(dirname "$execution_authority_output")"
  install -m 0755 "$SOUNIO_LOOM_EXECUTION_AUTHORITY_PREBUILT" \
    "$execution_authority_output"
else
  SOUNIO_LOOM_EXECUTION_AUTHORITY_SOUC="$frozen_toolchain_root/bin/souc" \
    SOUNIO_LOOM_EXECUTION_AUTHORITY_OUTPUT="$execution_authority_output" \
    "$SCRIPT_DIR/build_sounio_loom_execution_authority.sh"
fi
execution_outcome_output="$ROOT_DIR/tools/loom/.runtime/sounio-loom-execution-outcome-runtime"
if [[ -n "${SOUNIO_LOOM_EXECUTION_OUTCOME_PREBUILT:-}" ]]; then
  [[ -x "$SOUNIO_LOOM_EXECUTION_OUTCOME_PREBUILT" ]] || {
    echo 'error: SOUNIO_LOOM_EXECUTION_OUTCOME_PREBUILT is not executable' >&2
    exit 1
  }
  mkdir -p "$(dirname "$execution_outcome_output")"
  install -m 0755 "$SOUNIO_LOOM_EXECUTION_OUTCOME_PREBUILT" \
    "$execution_outcome_output"
else
  SOUNIO_LOOM_EXECUTION_OUTCOME_SOUC="$execution_outcome_toolchain_root/bin/souc" \
    SOUNIO_LOOM_EXECUTION_OUTCOME_OUTPUT="$execution_outcome_output" \
    "$SCRIPT_DIR/build_sounio_loom_execution_outcome.sh"
fi
custody_transfer_output="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-custody-transfer-runtime"
if [[ -n "${SOUNIO_LOOM_CUSTODY_TRANSFER_PREBUILT:-}" ]]; then
  [[ -x "$SOUNIO_LOOM_CUSTODY_TRANSFER_PREBUILT" ]] || {
    echo 'error: SOUNIO_LOOM_CUSTODY_TRANSFER_PREBUILT is not executable' >&2
    exit 1
  }
  install -m 0755 "$SOUNIO_LOOM_CUSTODY_TRANSFER_PREBUILT" \
    "$custody_transfer_output"
else
  SOUNIO_LOOM_CUSTODY_TRANSFER_OUTPUT="$custody_transfer_output" \
    "$SCRIPT_DIR/build_sounio_loom_custody_transfer.sh"
fi
lane_health_output="$ROOT_DIR/tools/loom/.runtime/sounio-loom-lane-health-runtime"
if [[ -n "${SOUNIO_LOOM_LANE_HEALTH_PREBUILT:-}" ]]; then
  [[ -x "$SOUNIO_LOOM_LANE_HEALTH_PREBUILT" ]] || {
    echo 'error: SOUNIO_LOOM_LANE_HEALTH_PREBUILT is not executable' >&2
    exit 1
  }
  mkdir -p "$(dirname "$lane_health_output")"
  install -m 0755 "$SOUNIO_LOOM_LANE_HEALTH_PREBUILT" "$lane_health_output"
else
  SOUNIO_LOOM_LANE_HEALTH_SOUC="$lane_health_toolchain_root/bin/souc" \
    SOUNIO_LOOM_LANE_HEALTH_OUTPUT="$lane_health_output" \
    "$SCRIPT_DIR/build_sounio_loom_lane_health.sh"
fi
lane_health_parity_output="$ROOT_DIR/tools/loom/.runtime/sounio-loom-lane-health-parity-runtime"
if [[ -n "${SOUNIO_LOOM_LANE_HEALTH_PARITY_PREBUILT:-}" ]]; then
  [[ -x "$SOUNIO_LOOM_LANE_HEALTH_PARITY_PREBUILT" ]] || {
    echo 'error: SOUNIO_LOOM_LANE_HEALTH_PARITY_PREBUILT is not executable' >&2
    exit 1
  }
  mkdir -p "$(dirname "$lane_health_parity_output")"
  install -m 0755 "$SOUNIO_LOOM_LANE_HEALTH_PARITY_PREBUILT" \
    "$lane_health_parity_output"
else
  SOUNIO_LOOM_LANE_HEALTH_PARITY_SOUC="$lane_health_toolchain_root/bin/souc" \
    SOUNIO_LOOM_LANE_HEALTH_PARITY_OUTPUT="$lane_health_parity_output" \
    "$SCRIPT_DIR/build_sounio_loom_lane_health_parity.sh"
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
