#!/usr/bin/env bash

set -euo pipefail
umask 077

CLIENT_PROTOCOL=3

usage() {
  cat <<'USAGE'
Usage: scripts/dev/install_sounio_coord_runtime.sh [options]

Install and atomically activate the coordination runtime shared by every
worktree attached to this repository.

Options:
  --source-root PATH       source bundle root (default: current worktree)
  --runtime-dir PATH       shared runtime root override
  --activate RUNTIME_ID    activate an already installed version
  --list                   list installed versions
  -h, --help               show this help
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

manifest_value() {
  local manifest="$1" key="$2"
  sed -n "s/^${key}=//p" "$manifest" | head -1
}

activate_runtime() {
  local runtime_id="$1" version_dir manifest protocol link_tmp
  version_dir="$RUNTIME_ROOT/versions/$runtime_id"
  manifest="$version_dir/manifest"
  [[ -f "$manifest" && -x "$version_dir/bin/sounio-coord-runtime" && \
    -f "$version_dir/hooks/sounio_coord_agent_hook_runtime.py" ]] || \
    die "installed runtime is incomplete: $runtime_id"
  protocol="$(manifest_value "$manifest" protocol_version)"
  [[ "$protocol" == "$CLIENT_PROTOCOL" ]] || \
    die "cannot activate protocol $protocol with installer protocol $CLIENT_PROTOCOL"
  if grep -q '^capability=agentd-transport-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-agentd-runtime" ]] || \
      die "installed runtime declares agentd transport but omits its implementation: $runtime_id"
  fi
  if grep -q '^capability=loom-kernel-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-loom-runtime" ]] || \
      die "installed runtime declares Loom but omits its OCaml kernel: $runtime_id"
  fi
  if grep -q '^capability=fleet-launcher-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-fleet-agent-runtime" ]] || \
      die "installed runtime declares the fleet launcher but omits its implementation: $runtime_id"
  fi
  if grep -q '^capability=fleet-proven-exit-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-fleet-agent-runtime" ]] || \
      die "installed runtime declares proven-exit recovery but omits its launcher: $runtime_id"
  fi
  if grep -q '^capability=fleet-home-isolation-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-fleet-agent-runtime" ]] || \
      die "installed runtime declares fleet HOME isolation but omits its launcher: $runtime_id"
  fi
  if grep -q '^capability=fleet-event-log-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-fleet-runtime" ]] || \
      die "installed runtime declares fleet reconciliation but omits its implementation: $runtime_id"
  fi
  if grep -q '^capability=fleet-tla-model-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-fleet-tla-sabotage" && \
      -f "$version_dir/formal/SounioFleet.tla" && \
      -f "$version_dir/formal/SounioFleet.cfg" ]] || \
      die "installed runtime declares the TLA+ fleet model but omits its bundle: $runtime_id"
  fi
  if grep -q '^capability=fleet-trace-refinement-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-fleet-trace-verify" ]] || \
      die "installed runtime declares fleet trace refinement but omits its verifier: $runtime_id"
  fi
  if grep -q '^capability=fleet-temporal-authority-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-fleet-runtime" && \
      -x "$version_dir/bin/sounio-fleet-trace-verify" ]] || \
      die "installed runtime declares temporal fleet authority but omits its implementation: $runtime_id"
  fi
  [[ ! -e "$RUNTIME_ROOT/current" || -L "$RUNTIME_ROOT/current" ]] || \
    die "refusing to replace non-symlink runtime path: $RUNTIME_ROOT/current"
  link_tmp="$RUNTIME_ROOT/.current.$$.$RANDOM"
  ln -s "versions/$runtime_id" "$link_tmp"
  mv -Tf "$link_tmp" "$RUNTIME_ROOT/current"
  printf 'ACTIVATED runtime_id=%s protocol=%s path=%s\n' \
    "$runtime_id" "$protocol" "$version_dir"
}

WORKTREE="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$WORKTREE" ]] || die "run this installer from a Git worktree"
WORKTREE="$(cd "$WORKTREE" && pwd -P)"
GIT_COMMON_DIR="$(git -C "$WORKTREE" rev-parse --git-common-dir 2>/dev/null || true)"
[[ -n "$GIT_COMMON_DIR" ]] || die "cannot resolve the shared Git directory"
case "$GIT_COMMON_DIR" in
  /*) ;;
  *) GIT_COMMON_DIR="$(cd "$WORKTREE/$GIT_COMMON_DIR" && pwd -P)" ;;
esac

SOURCE_ROOT="$WORKTREE"
RUNTIME_ROOT="${SOUNIO_COORD_RUNTIME_DIR:-$GIT_COMMON_DIR/sounio-coord-runtime}"
action=install
activate_id=''
while (($#)); do
  case "$1" in
    --source-root) (($# >= 2)) || die "$1 requires a value"; SOURCE_ROOT="$2"; shift 2 ;;
    --runtime-dir) (($# >= 2)) || die "$1 requires a value"; RUNTIME_ROOT="$2"; shift 2 ;;
    --activate) (($# >= 2)) || die "$1 requires a value"; action=activate; activate_id="$2"; shift 2 ;;
    --list) action=list; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown installer option: $1" ;;
  esac
done

SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd -P)"
mkdir -p "$RUNTIME_ROOT/versions"
RUNTIME_ROOT="$(cd "$RUNTIME_ROOT" && pwd -P)"
exec 9>"$RUNTIME_ROOT/.install.lock"
flock 9

if [[ "$action" == list ]]; then
  current=''
  [[ ! -e "$RUNTIME_ROOT/current" ]] || current="$(basename "$(readlink -f "$RUNTIME_ROOT/current")")"
  version_paths=("$RUNTIME_ROOT"/versions/*)
  for version_dir in "${version_paths[@]}"; do
    [[ -d "$version_dir" && -f "$version_dir/manifest" ]] || continue
    runtime_id="$(basename "$version_dir")"
    marker=no
    [[ "$runtime_id" != "$current" ]] || marker=yes
    printf 'RUNTIME runtime_id=%s current=%s protocol=%s runtime_version=%s source_sha=%s\n' \
      "$runtime_id" "$marker" \
      "$(manifest_value "$version_dir/manifest" protocol_version)" \
      "$(manifest_value "$version_dir/manifest" runtime_version)" \
      "$(manifest_value "$version_dir/manifest" source_sha)"
  done
  exit 0
fi

if [[ "$action" == activate ]]; then
  activate_runtime "$activate_id"
  exit 0
fi

runtime_source="$SOURCE_ROOT/scripts/dev/sounio_coord_runtime.sh"
hook_source="$SOURCE_ROOT/scripts/dev/sounio_coord_agent_hook_runtime.py"
causal_source="$SOURCE_ROOT/scripts/dev/sounio_coord_causal_runtime.py"
agentd_source="$SOURCE_ROOT/scripts/dev/sounio_coord_agentd.py"
fleet_source="$SOURCE_ROOT/scripts/dev/sounio_coord_fleet.py"
fleetd_source="$SOURCE_ROOT/scripts/dev/sounio_coord_fleetd.py"
fleet_model_source="$SOURCE_ROOT/formal/tla/SounioFleet.tla"
fleet_model_config="$SOURCE_ROOT/formal/tla/SounioFleet.cfg"
fleet_model_generator="$SOURCE_ROOT/scripts/dev/sounio_fleet_tla_sabotage.py"
fleet_trace_verifier="$SOURCE_ROOT/scripts/dev/sounio_fleet_trace_verify.py"
loom_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom.sh"
loom_project="$SOURCE_ROOT/tools/loom"
[[ -x "$runtime_source" ]] || die "runtime source missing or not executable: $runtime_source"
[[ -f "$hook_source" ]] || die "hook runtime source missing: $hook_source"
[[ -x "$causal_source" ]] || die "causal runtime source missing or not executable: $causal_source"
[[ -x "$agentd_source" ]] || die "agent supervisor source missing or not executable: $agentd_source"
[[ -x "$fleet_source" ]] || die "fleet launcher source missing or not executable: $fleet_source"
[[ -x "$fleetd_source" ]] || die "fleet reconciler source missing or not executable: $fleetd_source"
[[ -f "$fleet_model_source" ]] || die "fleet TLA+ model missing: $fleet_model_source"
[[ -f "$fleet_model_config" ]] || die "fleet TLC config missing: $fleet_model_config"
[[ -x "$fleet_model_generator" ]] || \
  die "fleet model sabotage generator missing or not executable: $fleet_model_generator"
[[ -x "$fleet_trace_verifier" ]] || \
  die "fleet trace verifier missing or not executable: $fleet_trace_verifier"
[[ -x "$loom_build_source" ]] || die "Loom build entrypoint missing or not executable: $loom_build_source"
[[ -f "$loom_project/src/loom.ml" && -f "$loom_project/src/loom_pty_stubs.c" && \
  -f "$loom_project/src/dune" && -f "$loom_project/dune-project" ]] || \
  die "Loom OCaml source bundle is incomplete: $loom_project"

version_output="$(cd "$WORKTREE" && "$runtime_source" runtime-version)"
protocol="$(sed -n 's/^protocol_version=//p' <<< "$version_output" | head -1)"
runtime_version="$(sed -n 's/^runtime_version=//p' <<< "$version_output" | head -1)"
[[ "$protocol" == "$CLIENT_PROTOCOL" ]] || \
  die "source protocol $protocol is incompatible with installer protocol $CLIENT_PROTOCOL"
[[ -n "$runtime_version" ]] || die "source runtime did not report a version"

agentd_version_output="$($agentd_source runtime-version)"
agentd_protocol="$(sed -n 's/^protocol_version=//p' <<< "$agentd_version_output" | head -1)"
[[ "$agentd_protocol" == 1 ]] || die "agent supervisor protocol must be 1"

fleet_version_output="$($fleet_source runtime-version)"
fleet_protocol="$(sed -n 's/^protocol_version=//p' <<< "$fleet_version_output" | head -1)"
[[ "$fleet_protocol" == 1 ]] || die "fleet launcher protocol must be 1"

fleetd_version_output="$($fleetd_source runtime-version)"
fleetd_protocol="$(sed -n 's/^protocol_version=//p' <<< "$fleetd_version_output" | head -1)"
[[ "$fleetd_protocol" == 1 ]] || die "fleet reconciler protocol must be 1"

"$loom_build_source" >/dev/null
loom_binary="$loom_project/_build/default/src/loom.exe"
[[ -x "$loom_binary" ]] || die "Loom build omitted its native executable"
loom_version_output="$($loom_binary runtime-version)"
loom_protocol="$(sed -n 's/^protocol_version=//p' <<< "$loom_version_output" | head -1)"
loom_language="$(sed -n 's/^language=//p' <<< "$loom_version_output" | head -1)"
[[ "$loom_protocol" == 1 && "$loom_language" == OCaml ]] || \
  die "Loom kernel must report protocol 1 and language OCaml"

bundle_sha="$(
  sha256sum "$runtime_source" "$hook_source" "$causal_source" "$agentd_source" \
    "$fleet_source" "$fleetd_source" "$fleet_model_source" \
    "$fleet_model_config" "$fleet_model_generator" "$fleet_trace_verifier" \
    "$loom_build_source" "$loom_project/dune-project" "$loom_project/src/dune" \
    "$loom_project/src/loom.ml" "$loom_project/src/loom_pty_stubs.c" | \
    awk '{print $1}' | sha256sum | awk '{print $1}'
)"
safe_version="$(printf '%s' "$runtime_version" | tr -c 'A-Za-z0-9._-' '_')"
runtime_id="p${protocol}-${safe_version}-${bundle_sha:0:12}"
version_dir="$RUNTIME_ROOT/versions/$runtime_id"
source_sha="$(git -C "$SOURCE_ROOT" rev-parse --short=12 HEAD 2>/dev/null || printf unknown)"

if [[ -d "$version_dir" ]]; then
  installed_sha="$(manifest_value "$version_dir/manifest" bundle_sha256)"
  [[ "$installed_sha" == "$bundle_sha" ]] || \
    die "runtime id collision with different bundle: $runtime_id"
else
  stage="$(mktemp -d "$RUNTIME_ROOT/.install.XXXXXX")"
  cleanup_stage() {
    [[ -z "${stage:-}" ]] || rm -rf "$stage"
  }
  trap cleanup_stage EXIT
  mkdir -p "$stage/bin" "$stage/hooks" "$stage/formal"
  install -m 0755 "$runtime_source" "$stage/bin/sounio-coord-runtime"
  install -m 0755 "$causal_source" "$stage/bin/sounio-coord-causal-runtime"
  install -m 0755 "$agentd_source" "$stage/bin/sounio-agentd-runtime"
  install -m 0755 "$fleet_source" "$stage/bin/sounio-fleet-agent-runtime"
  install -m 0755 "$fleetd_source" "$stage/bin/sounio-fleet-runtime"
  install -m 0755 "$fleet_model_generator" "$stage/bin/sounio-fleet-tla-sabotage"
  install -m 0755 "$fleet_trace_verifier" "$stage/bin/sounio-fleet-trace-verify"
  install -m 0755 "$loom_binary" "$stage/bin/sounio-loom-runtime"
  install -m 0644 "$fleet_model_source" "$stage/formal/SounioFleet.tla"
  install -m 0644 "$fleet_model_config" "$stage/formal/SounioFleet.cfg"
  install -m 0755 "$hook_source" "$stage/hooks/sounio_coord_agent_hook_runtime.py"
  {
    printf 'runtime_id=%s\n' "$runtime_id"
    printf 'protocol_version=%s\n' "$protocol"
    printf 'agentd_protocol_version=%s\n' "$agentd_protocol"
    printf 'fleet_protocol_version=%s\n' "$fleet_protocol"
    printf 'fleetd_protocol_version=%s\n' "$fleetd_protocol"
    printf 'loom_protocol_version=%s\n' "$loom_protocol"
    printf 'runtime_version=%s\n' "$runtime_version"
    printf 'bundle_sha256=%s\n' "$bundle_sha"
    printf 'source_sha=%s\n' "$source_sha"
    printf 'capability=causal-experiment-receipts-v1\n'
    printf 'capability=crash-recovery-v1\n'
    printf 'capability=agentd-transport-v1\n'
    printf 'capability=agentd-argv-attestation-v1\n'
    printf 'capability=agentd-tui-submit-v1\n'
    printf 'capability=agentd-logical-command-v1\n'
    printf 'capability=agentd-runtime-registration-v1\n'
    printf 'capability=loom-kernel-v1\n'
    printf 'capability=loom-cursor-replay-v1\n'
    printf 'capability=loom-exclusive-input-lease-v1\n'
    printf 'capability=loom-read-only-gui-v1\n'
    printf 'capability=loom-coord-transport-v1\n'
    printf 'capability=coord-reply-correlation-v1\n'
    printf 'capability=fleet-launcher-v1\n'
    printf 'capability=fleet-proven-exit-v1\n'
    printf 'capability=fleet-home-isolation-v1\n'
    printf 'capability=fleet-event-log-v1\n'
    printf 'capability=fleet-reconciler-v1\n'
    printf 'capability=fleet-linear-capability-v1\n'
    printf 'capability=fleet-ed25519-anchor-v1\n'
    printf 'capability=fleet-checkpoint-handoff-v1\n'
    printf 'capability=fleet-tla-model-v1\n'
    printf 'capability=fleet-trace-refinement-v1\n'
    printf 'capability=fleet-temporal-authority-v1\n'
    printf 'installed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$stage/manifest"
  mv "$stage" "$version_dir"
  stage=''
  trap - EXIT
  printf 'INSTALLED runtime_id=%s protocol=%s path=%s\n' \
    "$runtime_id" "$protocol" "$version_dir"
fi

activate_runtime "$runtime_id"
