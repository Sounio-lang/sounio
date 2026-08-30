#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
INSTALLER="$ROOT_DIR/scripts/dev/install_loom_hostd.sh"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-hostd-install.XXXXXX")"

fail() {
  printf 'sounio-loom-hostd-install-selftest: FAIL: %s test_root=%s\n' "$*" "$TEST_ROOT" >&2
  exit 1
}

cleanup() {
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" != 1 ]]; then
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

bash "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
bash "$ROOT_DIR/scripts/dev/build_sounio_loom_host_boot_reconciler.sh" >/dev/null
runtime="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
authority="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-host-boot-reconciler"
stage="$TEST_ROOT/stage"
mkdir -p "$stage"

first="$(bash "$INSTALLER" --install-root "$stage" \
  --runtime "$runtime" --authority "$authority" --user loom-test)"
[[ "$first" == *'activated=false automatic_lineage_resurrection=false'* ]] ||
  fail "staged installer widened activation: $first"

prefix="$stage/opt/sounio/loom-hostd"
unit="$stage/etc/systemd/system/sounio-loom-hostd.service"
manifest="$prefix/manifest.v1"
installed_runtime="$prefix/bin/sounio-loom-runtime"
installed_authority="$prefix/bin/sounio-loom-host-boot-reconciler"
[[ -x "$installed_runtime" && -x "$installed_authority" && -f "$unit" && -f "$manifest" ]] ||
  fail 'staged installation omitted an artifact'
[[ "$(sha256sum "$installed_authority" | cut -d ' ' -f 1)" == \
   '99f5062729a171ac2d8c1b9b181497fbe1b8c9317859ee0fdc4d2cd4acaedb5b' ]] ||
  fail 'installed Sounio authority hash drifted'
grep -Fxq 'language=OCaml' < <("$installed_runtime" runtime-version) ||
  fail 'installed runtime is not OCaml'
[[ "$(printf '0\n' | "$installed_authority")" == \
   'SOUNIO_HOST_BOOT_RECONCILER_SELFTEST PASS cases=14' ]] ||
  fail 'installed Sounio authority does not selftest'

grep -Fxq 'KillMode=process' "$unit" || fail 'unit would kill recovered lane processes'
grep -Fxq 'Restart=on-failure' "$unit" || fail 'unit does not restart after refusal or crash'
grep -Fxq 'NoNewPrivileges=true' "$unit" || fail 'unit lacks no-new-privileges boundary'
grep -Fxq 'ProtectSystem=strict' "$unit" || fail 'unit lacks strict filesystem protection'
grep -Fxq 'ExecStart=/opt/sounio/loom-hostd/bin/sounio-loom-runtime host-supervise --state-dir /var/lib/sounio/loom --service-enabled --apply' "$unit" ||
  fail 'unit is not wired to the Sounio-authorized supervisor'
grep -Fxq 'Environment=SOUNIO_LOOM_HOST_BOOT_AUTHORITY=/opt/sounio/loom-hostd/bin/sounio-loom-host-boot-reconciler' "$unit" ||
  fail 'unit omitted the frozen authority path'
grep -Fxq 'install_default=disabled' "$manifest" || fail 'install default is not disabled'
grep -Fxq 'service_enabled=false' "$manifest" || fail 'staged service was marked enabled'
grep -Fxq 'production_activation=false' "$manifest" || fail 'staged service was marked production-active'
[[ ! -e "$stage/etc/systemd/system/multi-user.target.wants/sounio-loom-hostd.service" ]] ||
  fail 'staged install created an enablement link'

unit_sha_before="$(sha256sum "$unit" | cut -d ' ' -f 1)"
manifest_sha_before="$(sha256sum "$manifest" | cut -d ' ' -f 1)"
bash "$INSTALLER" --install-root "$stage" --runtime "$runtime" \
  --authority "$authority" --user loom-test >/dev/null
[[ "$(sha256sum "$unit" | cut -d ' ' -f 1)" == "$unit_sha_before" && \
   "$(sha256sum "$manifest" | cut -d ' ' -f 1)" == "$manifest_sha_before" ]] ||
  fail 'staged reinstall was not byte-idempotent'

if bash "$INSTALLER" --install-root "$stage" --runtime "$runtime" \
  --authority "$authority" --user loom-test --activate \
  >"$TEST_ROOT/activate.out" 2>"$TEST_ROOT/activate.err"; then
  fail 'staged installer accepted activation'
fi
grep -q -- '--activate is forbidden for a staged install root' "$TEST_ROOT/activate.err" ||
  fail 'staged activation was refused for the wrong reason'

cp "$authority" "$TEST_ROOT/authority-mutant"
printf 'X' | dd of="$TEST_ROOT/authority-mutant" bs=1 seek=128 conv=notrunc status=none
chmod 0755 "$TEST_ROOT/authority-mutant"
mkdir -p "$TEST_ROOT/mutant-stage"
if bash "$INSTALLER" --install-root "$TEST_ROOT/mutant-stage" \
  --runtime "$runtime" --authority "$TEST_ROOT/authority-mutant" --user loom-test \
  >"$TEST_ROOT/mutant.out" 2>"$TEST_ROOT/mutant.err"; then
  fail 'installer admitted a mutated Sounio authority'
fi
grep -q 'Sounio authority hash drifted' "$TEST_ROOT/mutant.err" ||
  fail 'mutated authority was refused by the wrong boundary'

printf 'sounio-loom-hostd-install-selftest: PASS installer_transport=shell runtime_language=OCaml runtime_role=EFFECT_PARITY semantic_authority=Sounio action=9041 systemd_unit=PASS kill_mode=process restart=on-failure install_default=disabled staged_activation=DENIED mutated_authority=DENIED byte_idempotent=PASS python_executed=false rust_executed=false production_activation=false\n'
