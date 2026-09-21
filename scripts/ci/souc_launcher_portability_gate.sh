#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${SOUNIO_SOUC_PORTABILITY_WORK:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-souc-portability.XXXXXX")}"

if [[ -z "${SOUNIO_SOUC_PORTABILITY_WORK:-}" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

fail() {
  echo "[souc-portability] FAIL: $*" >&2
  exit 1
}

mkdir -p "$WORK/repo/bin" "$WORK/fake-bin"
cp "$ROOT_DIR/bin/souc" "$WORK/repo/bin/souc"

cat >"$WORK/repo/bin/souc-lean-single-x86_64" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
out="${2:?missing output path}"
cat >"$out" <<'PROGRAM'
#!/usr/bin/env bash
printf '%s\n' SOUC_BSD_MKTEMP_RUN_OK
PROGRAM
chmod +x "$out"
EOF
chmod +x "$WORK/repo/bin/souc-lean-single-x86_64"

# BSD mktemp requires the template Xs to be the final characters. Validate that
# contract, then delegate uniqueness to the host mktemp inside the gate's work
# directory so repeated launcher invocations cannot leak files into /tmp.
cat >"$WORK/fake-bin/mktemp" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
template="${1:-}"
case "$template" in
  *XXXXXX) ;;
  *)
    echo "bsd-mktemp: template must end in XXXXXX: $template" >&2
    exit 64
    ;;
esac
exec /usr/bin/mktemp "${SOUNIO_PORTABILITY_WORK:?}/generated.XXXXXX"
EOF
chmod +x "$WORK/fake-bin/mktemp"

printf '%s\n' 'fn main() -> i64 { 0 }' >"$WORK/probe.sio"

run_launcher() {
  PATH="$WORK/fake-bin:$PATH" \
    SOUNIO_PORTABILITY_WORK="$WORK" \
    SOUNIO_SOUC_ENGINE=lean_single \
    "$WORK/repo/bin/souc" "$@"
}

for attempt in 1 2; do
  output="$(run_launcher run "$WORK/probe.sio")" || fail "lean run attempt $attempt failed"
  [[ "$output" == "SOUC_BSD_MKTEMP_RUN_OK" ]] || fail "lean run attempt $attempt returned: $output"
  run_launcher check "$WORK/probe.sio" >/dev/null || fail "lean check attempt $attempt failed"
done

generated_count="$(find "$WORK" -maxdepth 1 -type f -name 'generated.*' | wc -l | tr -d ' ')"
[[ "$generated_count" == "2" ]] || fail "expected two unique run artifacts, found $generated_count"

# Negative control: recreate the old GNU-only templates and prove the BSD shim
# rejects both paths. A gate that cannot detect its target regression is noise.
sed \
  -e 's/souc-lean-run-XXXXXX/souc-lean-run-XXXXXX.elf/' \
  -e 's/souc-lean-check-XXXXXX/souc-lean-check-XXXXXX.elf/' \
  "$WORK/repo/bin/souc" >"$WORK/repo/bin/souc-bad"
chmod +x "$WORK/repo/bin/souc-bad"

for verb in run check; do
  set +e
  bad_output="$(PATH="$WORK/fake-bin:$PATH" \
    SOUNIO_PORTABILITY_WORK="$WORK" \
    SOUNIO_SOUC_ENGINE=lean_single \
    "$WORK/repo/bin/souc-bad" "$verb" "$WORK/probe.sio" 2>&1)"
  bad_rc=$?
  set -e
  [[ "$bad_rc" -ne 0 ]] || fail "negative $verb control unexpectedly passed"
  [[ "$bad_output" == *"template must end in XXXXXX"* ]] || fail "negative $verb control missed BSD diagnostic"
done

echo "[souc-portability] PASS: BSD mktemp run/check x2, unique artifacts=2, negative controls rejected"
