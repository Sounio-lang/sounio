#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/elf_utils.sh"

ARTIFACT_ROOT="${SOUNIO_INSTALL_PATH_GATE_ARTIFACT_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-install-path-gate.XXXXXX")}"
PREFIX="$ARTIFACT_ROOT/prefix with spaces"
LOG_DIR="$ARTIFACT_ROOT/logs"
SUMMARY="$ARTIFACT_ROOT/summary.v1.tsv"
RESULTS="$ARTIFACT_ROOT/RESULTS.md"

mkdir -p "$LOG_DIR"

run_capture() {
  local name="$1"
  shift
  local stdout="$LOG_DIR/$name.stdout"
  local stderr="$LOG_DIR/$name.stderr"
  set +e
  "$@" >"$stdout" 2>"$stderr"
  local rc=$?
  set -e
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "${stdout#$ARTIFACT_ROOT/}" "${stderr#$ARTIFACT_ROOT/}" >>"$SUMMARY"
  return "$rc"
}

printf 'step\texit\tstdout_log\tstderr_log\n' >"$SUMMARY"

run_capture install-initial bash scripts/install.sh --prefix="$PREFIX" --force
run_capture install-force bash scripts/install.sh --prefix="$PREFIX" --force
run_capture souc-version "$PREFIX/bin/souc" --version
run_capture souc-info env SOUNIO_STDLIB_PATH="$PREFIX/lib/sounio/stdlib" "$PREFIX/bin/souc" info
run_capture hello-check env SOUNIO_STDLIB_PATH="$PREFIX/lib/sounio/stdlib" "$PREFIX/bin/souc" check examples/hello.sio
run_capture hello-run env SOUNIO_STDLIB_PATH="$PREFIX/lib/sounio/stdlib" "$PREFIX/bin/souc" run examples/hello.sio

raw_elf="$(sed -n 's/^raw_elf:[[:space:]]*//p' "$LOG_DIR/souc-info.stdout" | head -n 1)"
if [[ -z "$raw_elf" ]]; then
  echo "install path gate failed: souc info did not report raw_elf" >&2
  exit 1
fi

if [[ "$raw_elf" != "$PREFIX/bin/madaros-linux-x86_64" ]]; then
  echo "install path gate failed: installed souc did not resolve installed Madaros raw ELF" >&2
  exit 1
fi

if ! sounio_is_elf_binary "$raw_elf"; then
  echo "install path gate failed: installed Madaros raw ELF is not executable ELF: $raw_elf" >&2
  exit 1
fi

source_madaros=""
for cand in "$ROOT_DIR/artifacts/self-hosted/madaros" "$ROOT_DIR/bin/madaros-linux-x86_64"; do
  if sounio_is_elf_binary "$cand"; then
    source_madaros="$cand"
    break
  fi
done

if [[ -z "$source_madaros" ]]; then
  echo "install path gate failed: no checked Madaros source artifact found" >&2
  exit 1
fi

if [[ "$(sha256sum "$source_madaros" | awk '{print $1}')" != "$(sha256sum "$raw_elf" | awk '{print $1}')" ]]; then
  echo "install path gate failed: installed Madaros raw ELF hash differs from source artifact" >&2
  exit 1
fi

if ! grep -Fq 'check: OK' "$LOG_DIR/hello-check.stdout"; then
  echo "install path gate failed: installed souc did not check examples/hello.sio" >&2
  exit 1
fi

if ! grep -Fq 'Hello, Sounio' "$LOG_DIR/hello-run.stdout"; then
  echo "install path gate failed: installed souc did not run examples/hello.sio" >&2
  exit 1
fi

cat >"$RESULTS" <<EOF
# Sounio Install Path Gate

| Field | Value |
|---|---|
| artifact_root | \`$ARTIFACT_ROOT\` |
| prefix | \`$PREFIX\` |
| status | \`pass\` |

This gate validates the checked repo-artifact install path:

- \`scripts/install.sh --prefix <tmp with spaces> --force\`
- repeat install with \`--force\` over the same prefix
- installed \`bin/souc --version\`
- installed \`bin/souc info\` resolves the installed Madaros raw ELF
- installed Madaros raw ELF is executable ELF
- installed Madaros raw ELF SHA256 matches the checked source artifact
- installed \`bin/souc check examples/hello.sio\` succeeds with the installed stdlib path
- installed \`bin/souc run examples/hello.sio\` executes and prints \`Hello, Sounio\`

Raw logs are under \`logs/\`; machine-readable step exits are in \`summary.v1.tsv\`.
EOF

echo "Install path gate passed. See $RESULTS"
