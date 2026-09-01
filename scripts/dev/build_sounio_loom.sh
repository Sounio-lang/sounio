#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
LANGUAGE_AUTHORITY_MANIFEST="$ROOT_DIR/tools/loom/language_authority.freeze.v1"
NATIVE_HOOK_CUTOVER_MANIFEST="$ROOT_DIR/tools/loom/native_hook_cutover.freeze.v1"
EXECUTION_AUTHORITY_MANIFEST="$ROOT_DIR/tools/loom/execution_authority.freeze.v2"
EXECUTION_OUTCOME_MANIFEST="$ROOT_DIR/tools/loom/execution_outcome.freeze.v1"
SUBPROCESS_MEMBRANE_MANIFEST="$ROOT_DIR/tools/loom/subprocess_membrane.freeze.v1"
RESIDENT_MEMBRANE_MANIFEST="$ROOT_DIR/tools/loom/resident_membrane.runtime.v1"
RESIDENT_MEMBRANE_V2_MANIFEST="$ROOT_DIR/tools/loom/resident_membrane.runtime.v2"
RESIDENT_MEMBRANE_V3_MANIFEST="$ROOT_DIR/tools/loom/resident_membrane.runtime.v3"
RESIDENT_MEMBRANE_V5_MANIFEST="$ROOT_DIR/tools/loom/resident_membrane.runtime.v5"
LANE_HEALTH_MANIFEST="$ROOT_DIR/tools/loom/lane_health.freeze.v1"
frozen_toolchain_root=''
execution_outcome_toolchain_root=''
subprocess_membrane_toolchain_root=''
lane_health_toolchain_root=''
native_hook_cutover_toolchain_root=''
resident_membrane_stage_root=''
resident_membrane_v2_stage_root=''
resident_membrane_v3_stage_root=''
resident_membrane_v5_stage_root=''

cleanup() {
  [[ -z "$frozen_toolchain_root" ]] || rm -rf "$frozen_toolchain_root"
  [[ -z "$execution_outcome_toolchain_root" ]] || rm -rf "$execution_outcome_toolchain_root"
  [[ -z "$subprocess_membrane_toolchain_root" ]] || rm -rf "$subprocess_membrane_toolchain_root"
  [[ -z "$lane_health_toolchain_root" ]] || rm -rf "$lane_health_toolchain_root"
  [[ -z "$native_hook_cutover_toolchain_root" ]] || rm -rf "$native_hook_cutover_toolchain_root"
  [[ -z "$resident_membrane_stage_root" ]] || rm -rf "$resident_membrane_stage_root"
  [[ -z "$resident_membrane_v2_stage_root" ]] || rm -rf "$resident_membrane_v2_stage_root"
  [[ -z "$resident_membrane_v3_stage_root" ]] || rm -rf "$resident_membrane_v3_stage_root"
  [[ -z "$resident_membrane_v5_stage_root" ]] || rm -rf "$resident_membrane_v5_stage_root"
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

prepare_subprocess_membrane_toolchain() {
  local executable_commit wrapper_sha compiler_sha actual_wrapper_sha actual_compiler_sha
  [[ -f "$SUBPROCESS_MEMBRANE_MANIFEST" ]] || {
    echo 'error: frozen Sounio subprocess-membrane manifest is required' >&2
    exit 1
  }
  executable_commit="$(manifest_value "$SUBPROCESS_MEMBRANE_MANIFEST" sounio_executable_commit)"
  wrapper_sha="$(manifest_value "$SUBPROCESS_MEMBRANE_MANIFEST" toolchain_wrapper_sha256)"
  compiler_sha="$(manifest_value "$SUBPROCESS_MEMBRANE_MANIFEST" toolchain_compiler_sha256)"
  git -C "$ROOT_DIR" cat-file -e "${executable_commit}^{commit}" 2>/dev/null || {
    echo "error: frozen subprocess-membrane Sounio toolchain commit is unavailable: $executable_commit" >&2
    exit 1
  }
  subprocess_membrane_toolchain_root="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-subprocess-membrane-toolchain.XXXXXX")"
  git -C "$ROOT_DIR" archive "$executable_commit" \
    bin/souc bin/souc-lean-single-x86_64 | tar -x -C "$subprocess_membrane_toolchain_root"
  actual_wrapper_sha="$(sha256sum "$subprocess_membrane_toolchain_root/bin/souc" | awk '{print $1}')"
  actual_compiler_sha="$(sha256sum "$subprocess_membrane_toolchain_root/bin/souc-lean-single-x86_64" | awk '{print $1}')"
  [[ "$actual_wrapper_sha" == "$wrapper_sha" && "$actual_compiler_sha" == "$compiler_sha" ]] || {
    echo 'error: reconstructed subprocess-membrane Sounio toolchain failed hash verification' >&2
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

prepare_native_hook_cutover_toolchain() {
  local executable_commit wrapper_sha compiler_sha actual_wrapper_sha actual_compiler_sha
  [[ -f "$NATIVE_HOOK_CUTOVER_MANIFEST" ]] || {
    echo 'error: frozen Sounio native-hook cutover manifest is required' >&2
    exit 1
  }
  executable_commit="$(manifest_value "$NATIVE_HOOK_CUTOVER_MANIFEST" sounio_executable_commit)"
  wrapper_sha="$(manifest_value "$NATIVE_HOOK_CUTOVER_MANIFEST" toolchain_wrapper_sha256)"
  compiler_sha="$(manifest_value "$NATIVE_HOOK_CUTOVER_MANIFEST" toolchain_compiler_sha256)"
  git -C "$ROOT_DIR" cat-file -e "${executable_commit}^{commit}" 2>/dev/null || {
    echo "error: frozen native-hook cutover Sounio toolchain commit is unavailable: $executable_commit" >&2
    exit 1
  }
  native_hook_cutover_toolchain_root="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-native-hook-cutover-toolchain.XXXXXX")"
  git -C "$ROOT_DIR" archive "$executable_commit" \
    bin/souc bin/souc-lean-single-x86_64 | tar -x -C "$native_hook_cutover_toolchain_root"
  actual_wrapper_sha="$(sha256sum "$native_hook_cutover_toolchain_root/bin/souc" | awk '{print $1}')"
  actual_compiler_sha="$(sha256sum "$native_hook_cutover_toolchain_root/bin/souc-lean-single-x86_64" | awk '{print $1}')"
  [[ "$actual_wrapper_sha" == "$wrapper_sha" && "$actual_compiler_sha" == "$compiler_sha" ]] || {
    echo 'error: reconstructed native-hook cutover Sounio toolchain failed hash verification' >&2
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
SOUNIO_SOURCE_ROOT="$ROOT_DIR" \
  SOUNIO_LOOM_NATIVE_HOOK_GENERATION_DRAIN_OUTPUT="$ROOT_DIR/tools/loom/.runtime/sounio-loom-native-hook-generation-drain" \
  "$SCRIPT_DIR/build_sounio_loom_native_hook_generation_drain.sh"
SOUNIO_SOURCE_ROOT="$ROOT_DIR" \
  SOUNIO_LOOM_NATIVE_HOOK_GENERATION_RECONCILE_OUTPUT="$ROOT_DIR/tools/loom/.runtime/sounio-loom-native-hook-generation-reconcile" \
  "$SCRIPT_DIR/build_sounio_loom_native_hook_generation_reconcile.sh"
SOUNIO_SOURCE_ROOT="$ROOT_DIR" \
  "$SCRIPT_DIR/build_sounio_loom_sovereign_execution_kernel.sh"
SOUNIO_SOURCE_ROOT="$ROOT_DIR" \
  "$SCRIPT_DIR/build_sounio_loom_sovereign_change_kernel.sh"
SOUNIO_SOURCE_ROOT="$ROOT_DIR" \
  "$SCRIPT_DIR/build_sounio_loom_sovereign_material_change.sh"
if [[ -z "${SOUNIO_LOOM_LANGUAGE_AUTHORITY_PREBUILT:-}" || \
  -z "${SOUNIO_LOOM_EXECUTION_AUTHORITY_PREBUILT:-}" ]]; then
  prepare_frozen_toolchain
fi
if [[ -z "${SOUNIO_LOOM_EXECUTION_OUTCOME_PREBUILT:-}" ]]; then
  prepare_execution_outcome_toolchain
fi
if [[ -z "${SOUNIO_LOOM_SUBPROCESS_MEMBRANE_PREBUILT:-}" || \
  -z "${SOUNIO_LOOM_RESIDENT_MEMBRANE_PREBUILT:-}" || \
  -z "${SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_PREBUILT:-}" || \
  -z "${SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_PREBUILT:-}" || \
  -z "${SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_PREBUILT:-}" ]]; then
  prepare_subprocess_membrane_toolchain
fi
if [[ -z "${SOUNIO_LOOM_LANE_HEALTH_PREBUILT:-}" || \
  -z "${SOUNIO_LOOM_LANE_HEALTH_PARITY_PREBUILT:-}" ]]; then
  prepare_lane_health_toolchain
fi
if [[ -z "${SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_PREBUILT:-}" ]]; then
  prepare_native_hook_cutover_toolchain
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
native_hook_cutover_output="$ROOT_DIR/tools/loom/.runtime/sounio-loom-native-hook-cutover"
if [[ -n "${SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_PREBUILT:-}" ]]; then
  [[ -x "$SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_PREBUILT" ]] || {
    echo 'error: SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_PREBUILT is not executable' >&2
    exit 1
  }
  mkdir -p "$(dirname "$native_hook_cutover_output")"
  install -m 0755 "$SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_PREBUILT" \
    "$native_hook_cutover_output"
else
  SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_SOUC="$native_hook_cutover_toolchain_root/bin/souc" \
    SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_OUTPUT="$native_hook_cutover_output" \
    "$SCRIPT_DIR/build_sounio_loom_native_hook_cutover.sh"
fi
native_hook_cutover_expected_sha="$(manifest_value "$NATIVE_HOOK_CUTOVER_MANIFEST" executable_sha256)"
native_hook_cutover_actual_sha="$(sha256sum "$native_hook_cutover_output" | awk '{print $1}')"
[[ "$native_hook_cutover_actual_sha" == "$native_hook_cutover_expected_sha" ]] || {
  echo 'error: rebuilt native-hook cutover runtime failed frozen hash verification' >&2
  exit 1
}
[[ "$(printf '0\n' | "$native_hook_cutover_output")" == \
  'SOUNIO_NATIVE_HOOK_CUTOVER_SELFTEST PASS cases=12' ]] || {
  echo 'error: rebuilt native-hook cutover runtime failed its install probe' >&2
  exit 1
}
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
subprocess_membrane_output="$ROOT_DIR/tools/loom/.runtime/sounio-loom-subprocess-membrane-runtime"
if [[ -n "${SOUNIO_LOOM_SUBPROCESS_MEMBRANE_PREBUILT:-}" ]]; then
  [[ -x "$SOUNIO_LOOM_SUBPROCESS_MEMBRANE_PREBUILT" ]] || {
    echo 'error: SOUNIO_LOOM_SUBPROCESS_MEMBRANE_PREBUILT is not executable' >&2
    exit 1
  }
  mkdir -p "$(dirname "$subprocess_membrane_output")"
  install -m 0755 "$SOUNIO_LOOM_SUBPROCESS_MEMBRANE_PREBUILT" \
    "$subprocess_membrane_output"
else
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_SOUC="$subprocess_membrane_toolchain_root/bin/souc" \
    SOUNIO_LOOM_SUBPROCESS_MEMBRANE_OUTPUT="$subprocess_membrane_output" \
    "$SCRIPT_DIR/build_sounio_loom_subprocess_membrane.sh"
fi
[[ -f "$RESIDENT_MEMBRANE_MANIFEST" && \
  "$(manifest_value "$RESIDENT_MEMBRANE_MANIFEST" schema)" == \
    loom-resident-membrane-runtime-v1 && \
  "$(manifest_value "$RESIDENT_MEMBRANE_MANIFEST" stage)" == \
    SOUNIO_RESIDENT_REALIZATION && \
  "$(manifest_value "$RESIDENT_MEMBRANE_MANIFEST" producing_language)" == Sounio ]] || {
  echo 'error: frozen Sounio resident-membrane runtime manifest is invalid' >&2
  exit 1
}
resident_membrane_expected_sha="$(manifest_value "$RESIDENT_MEMBRANE_MANIFEST" runtime_sha256)"
resident_membrane_runtime_dir="$ROOT_DIR/tools/loom/.runtime"
resident_membrane_content_dir="$resident_membrane_runtime_dir/sha256-$resident_membrane_expected_sha"
resident_membrane_content_output="$resident_membrane_content_dir/sounio-loom-resident-membrane-runtime"
resident_membrane_output="$resident_membrane_runtime_dir/sounio-loom-resident-membrane-runtime"
resident_membrane_stage_root="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-resident-install.XXXXXX")"
resident_membrane_stage="$resident_membrane_stage_root/sounio-loom-resident-membrane-runtime"
if [[ -n "${SOUNIO_LOOM_RESIDENT_MEMBRANE_PREBUILT:-}" ]]; then
  [[ -x "$SOUNIO_LOOM_RESIDENT_MEMBRANE_PREBUILT" ]] || {
    echo 'error: SOUNIO_LOOM_RESIDENT_MEMBRANE_PREBUILT is not executable' >&2
    exit 1
  }
  install -m 0755 "$SOUNIO_LOOM_RESIDENT_MEMBRANE_PREBUILT" \
    "$resident_membrane_stage"
else
  SOUNIO_LOOM_RESIDENT_MEMBRANE_SOUC="$subprocess_membrane_toolchain_root/bin/souc" \
    SOUNIO_LOOM_RESIDENT_MEMBRANE_OUTPUT="$resident_membrane_stage" \
    "$SCRIPT_DIR/build_sounio_loom_resident_membrane.sh"
fi
resident_membrane_actual_sha="$(sha256sum "$resident_membrane_stage" | awk '{print $1}')"
[[ "$resident_membrane_actual_sha" == "$resident_membrane_expected_sha" ]] || {
  echo 'error: rebuilt resident-membrane runtime failed frozen hash verification' >&2
  exit 1
}
command -v flock >/dev/null 2>&1 || {
  echo 'error: flock is required for resident-membrane runtime promotion' >&2
  exit 1
}
mkdir -p "$resident_membrane_runtime_dir"
(
  flock -x 8
  if [[ -e "$resident_membrane_content_output" ]]; then
    installed_sha="$(sha256sum "$resident_membrane_content_output" | awk '{print $1}')"
    [[ "$installed_sha" == "$resident_membrane_expected_sha" ]] || {
      echo 'error: content-addressed resident runtime is corrupt' >&2
      exit 1
    }
  else
    [[ ! -e "$resident_membrane_content_dir" ]] || {
      echo 'error: incomplete content-addressed resident runtime directory exists' >&2
      exit 1
    }
    mkdir "$resident_membrane_content_dir"
    install -m 0555 "$resident_membrane_stage" "$resident_membrane_content_output"
    chmod 0555 "$resident_membrane_content_dir"
  fi
  resident_link_tmp="$resident_membrane_runtime_dir/.resident-membrane-link.$$"
  ln -s "sha256-$resident_membrane_expected_sha/sounio-loom-resident-membrane-runtime" \
    "$resident_link_tmp"
  mv -Tf "$resident_link_tmp" "$resident_membrane_output"
) 8>"$resident_membrane_runtime_dir/.resident-membrane.lock"
[[ -f "$RESIDENT_MEMBRANE_V2_MANIFEST" && \
  "$(manifest_value "$RESIDENT_MEMBRANE_V2_MANIFEST" schema)" == \
    loom-resident-membrane-runtime-v2 && \
  "$(manifest_value "$RESIDENT_MEMBRANE_V2_MANIFEST" stage)" == \
    SOUNIO_RESIDENT_REALIZATION && \
  "$(manifest_value "$RESIDENT_MEMBRANE_V2_MANIFEST" producing_language)" == Sounio ]] || {
  echo 'error: frozen Sounio resident-membrane v2 runtime manifest is invalid' >&2
  exit 1
}
resident_membrane_v2_expected_sha="$(manifest_value "$RESIDENT_MEMBRANE_V2_MANIFEST" runtime_sha256)"
resident_membrane_v2_content_dir="$resident_membrane_runtime_dir/sha256-$resident_membrane_v2_expected_sha"
resident_membrane_v2_content_output="$resident_membrane_v2_content_dir/sounio-loom-resident-membrane-runtime-v2"
resident_membrane_v2_output="$resident_membrane_runtime_dir/sounio-loom-resident-membrane-runtime-v2"
resident_membrane_v2_stage_root="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-resident-v2-install.XXXXXX")"
resident_membrane_v2_stage="$resident_membrane_v2_stage_root/sounio-loom-resident-membrane-runtime-v2"
if [[ -n "${SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_PREBUILT:-}" ]]; then
  [[ -x "$SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_PREBUILT" ]] || {
    echo 'error: SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_PREBUILT is not executable' >&2
    exit 1
  }
  install -m 0755 "$SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_PREBUILT" \
    "$resident_membrane_v2_stage"
else
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_SOUC="$subprocess_membrane_toolchain_root/bin/souc" \
    SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_OUTPUT="$resident_membrane_v2_stage" \
    "$SCRIPT_DIR/build_sounio_loom_resident_membrane_v2.sh"
fi
resident_membrane_v2_actual_sha="$(sha256sum "$resident_membrane_v2_stage" | awk '{print $1}')"
[[ "$resident_membrane_v2_actual_sha" == "$resident_membrane_v2_expected_sha" ]] || {
  echo 'error: rebuilt resident-membrane v2 runtime failed frozen hash verification' >&2
  exit 1
}
(
  flock -x 8
  if [[ -e "$resident_membrane_v2_content_output" ]]; then
    installed_sha="$(sha256sum "$resident_membrane_v2_content_output" | awk '{print $1}')"
    [[ "$installed_sha" == "$resident_membrane_v2_expected_sha" ]] || {
      echo 'error: content-addressed resident v2 runtime is corrupt' >&2
      exit 1
    }
  else
    [[ ! -e "$resident_membrane_v2_content_dir" ]] || {
      echo 'error: incomplete content-addressed resident v2 runtime directory exists' >&2
      exit 1
    }
    mkdir "$resident_membrane_v2_content_dir"
    install -m 0555 "$resident_membrane_v2_stage" "$resident_membrane_v2_content_output"
    chmod 0555 "$resident_membrane_v2_content_dir"
  fi
  resident_v2_link_tmp="$resident_membrane_runtime_dir/.resident-membrane-v2-link.$$"
  ln -s "sha256-$resident_membrane_v2_expected_sha/sounio-loom-resident-membrane-runtime-v2" \
    "$resident_v2_link_tmp"
  mv -Tf "$resident_v2_link_tmp" "$resident_membrane_v2_output"
) 8>"$resident_membrane_runtime_dir/.resident-membrane-v2.lock"
[[ -f "$RESIDENT_MEMBRANE_V3_MANIFEST" && \
  "$(manifest_value "$RESIDENT_MEMBRANE_V3_MANIFEST" schema)" == \
    loom-resident-membrane-runtime-v3 && \
  "$(manifest_value "$RESIDENT_MEMBRANE_V3_MANIFEST" stage)" == \
    SOUNIO_RESIDENT_REALIZATION && \
  "$(manifest_value "$RESIDENT_MEMBRANE_V3_MANIFEST" producing_language)" == Sounio ]] || {
  echo 'error: frozen Sounio resident-membrane v3 runtime manifest is invalid' >&2
  exit 1
}
resident_membrane_v3_expected_sha="$(manifest_value "$RESIDENT_MEMBRANE_V3_MANIFEST" runtime_sha256)"
resident_membrane_v3_content_dir="$resident_membrane_runtime_dir/sha256-$resident_membrane_v3_expected_sha"
resident_membrane_v3_content_output="$resident_membrane_v3_content_dir/sounio-loom-resident-membrane-runtime-v3"
resident_membrane_v3_output="$resident_membrane_runtime_dir/sounio-loom-resident-membrane-runtime-v3"
resident_membrane_v3_stage_root="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-resident-v3-install.XXXXXX")"
resident_membrane_v3_stage="$resident_membrane_v3_stage_root/sounio-loom-resident-membrane-runtime-v3"
if [[ -n "${SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_PREBUILT:-}" ]]; then
  [[ -x "$SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_PREBUILT" ]] || {
    echo 'error: SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_PREBUILT is not executable' >&2
    exit 1
  }
  install -m 0755 "$SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_PREBUILT" \
    "$resident_membrane_v3_stage"
else
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_SOUC="$subprocess_membrane_toolchain_root/bin/souc" \
    SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_OUTPUT="$resident_membrane_v3_stage" \
    "$SCRIPT_DIR/build_sounio_loom_resident_membrane_v3.sh"
fi
resident_membrane_v3_actual_sha="$(sha256sum "$resident_membrane_v3_stage" | awk '{print $1}')"
[[ "$resident_membrane_v3_actual_sha" == "$resident_membrane_v3_expected_sha" ]] || {
  echo 'error: rebuilt resident-membrane v3 runtime failed frozen hash verification' >&2
  exit 1
}
(
  flock -x 8
  if [[ -e "$resident_membrane_v3_content_output" ]]; then
    installed_sha="$(sha256sum "$resident_membrane_v3_content_output" | awk '{print $1}')"
    [[ "$installed_sha" == "$resident_membrane_v3_expected_sha" ]] || {
      echo 'error: content-addressed resident v3 runtime is corrupt' >&2
      exit 1
    }
  else
    [[ ! -e "$resident_membrane_v3_content_dir" ]] || {
      echo 'error: incomplete content-addressed resident v3 runtime directory exists' >&2
      exit 1
    }
    mkdir "$resident_membrane_v3_content_dir"
    install -m 0555 "$resident_membrane_v3_stage" \
      "$resident_membrane_v3_content_output"
    chmod 0555 "$resident_membrane_v3_content_dir"
  fi
  resident_v3_link_tmp="$resident_membrane_runtime_dir/.resident-membrane-v3-link.$$"
  ln -s "sha256-$resident_membrane_v3_expected_sha/sounio-loom-resident-membrane-runtime-v3" \
    "$resident_v3_link_tmp"
  mv -Tf "$resident_v3_link_tmp" "$resident_membrane_v3_output"
) 8>"$resident_membrane_runtime_dir/.resident-membrane-v3.lock"
[[ -f "$RESIDENT_MEMBRANE_V5_MANIFEST" && \
  "$(manifest_value "$RESIDENT_MEMBRANE_V5_MANIFEST" schema)" == \
    loom-resident-membrane-runtime-v5 && \
  "$(manifest_value "$RESIDENT_MEMBRANE_V5_MANIFEST" stage)" == \
    SOUNIO_RESIDENT_REALIZATION && \
  "$(manifest_value "$RESIDENT_MEMBRANE_V5_MANIFEST" producing_language)" == Sounio ]] || {
  echo 'error: frozen Sounio resident-membrane v5 runtime manifest is invalid' >&2
  exit 1
}
resident_membrane_v5_expected_sha="$(manifest_value "$RESIDENT_MEMBRANE_V5_MANIFEST" runtime_sha256)"
resident_membrane_v5_content_dir="$resident_membrane_runtime_dir/sha256-$resident_membrane_v5_expected_sha"
resident_membrane_v5_content_output="$resident_membrane_v5_content_dir/sounio-loom-resident-membrane-runtime-v5"
resident_membrane_v5_output="$resident_membrane_runtime_dir/sounio-loom-resident-membrane-runtime-v5"
resident_membrane_v5_stage_root="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-resident-v5-install.XXXXXX")"
resident_membrane_v5_stage="$resident_membrane_v5_stage_root/sounio-loom-resident-membrane-runtime-v5"
if [[ -n "${SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_PREBUILT:-}" ]]; then
  [[ -x "$SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_PREBUILT" ]] || {
    echo 'error: SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_PREBUILT is not executable' >&2
    exit 1
  }
  install -m 0755 "$SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_PREBUILT" \
    "$resident_membrane_v5_stage"
else
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_SOUC="$subprocess_membrane_toolchain_root/bin/souc" \
    SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_OUTPUT="$resident_membrane_v5_stage" \
    "$SCRIPT_DIR/build_sounio_loom_resident_membrane_v5.sh"
fi
resident_membrane_v5_actual_sha="$(sha256sum "$resident_membrane_v5_stage" | awk '{print $1}')"
[[ "$resident_membrane_v5_actual_sha" == "$resident_membrane_v5_expected_sha" ]] || {
  echo 'error: rebuilt resident-membrane v5 runtime failed frozen hash verification' >&2
  exit 1
}
(
  flock -x 8
  if [[ -e "$resident_membrane_v5_content_output" ]]; then
    installed_sha="$(sha256sum "$resident_membrane_v5_content_output" | awk '{print $1}')"
    [[ "$installed_sha" == "$resident_membrane_v5_expected_sha" ]] || {
      echo 'error: content-addressed resident v5 runtime is corrupt' >&2
      exit 1
    }
  else
    [[ ! -e "$resident_membrane_v5_content_dir" ]] || {
      echo 'error: incomplete content-addressed resident v5 runtime directory exists' >&2
      exit 1
    }
    mkdir "$resident_membrane_v5_content_dir"
    install -m 0555 "$resident_membrane_v5_stage" \
      "$resident_membrane_v5_content_output"
    chmod 0555 "$resident_membrane_v5_content_dir"
  fi
  resident_v5_link_tmp="$resident_membrane_runtime_dir/.resident-membrane-v5-link.$$"
  ln -s "sha256-$resident_membrane_v5_expected_sha/sounio-loom-resident-membrane-runtime-v5" \
    "$resident_v5_link_tmp"
  mv -Tf "$resident_v5_link_tmp" "$resident_membrane_v5_output"
) 8>"$resident_membrane_runtime_dir/.resident-membrane-v5.lock"
SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_SOUC="$subprocess_membrane_toolchain_root/bin/souc" \
  "$SCRIPT_DIR/build_sounio_loom_kernel_peer_activation_capsule_current_frame.sh" \
  >/dev/null
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
