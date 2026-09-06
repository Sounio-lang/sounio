#!/usr/bin/env bash
# #2170: prove that four Madaros diagnostics have unique global identities.
#
# E208/E217/E218/E219 remain the published lean_single/catalogue identities.
# Madaros' unrelated reuses moved to E247/E248/E249/E250. This gate checks the
# semantic pairing, not just the presence of four fresh integers, and mutates
# every mapping back to its colliding number to prove the detector is live.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9

CHECKER="self-hosted/check/check.sio"
PARSER="self-hosted/parser/types.sio"
CATALOG="docs/llm-guide/error-catalog.md"
SOUC="${SOUNIO_DIAG_REMAP_SOUC:-}"
RUN_CONTROLS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checker) CHECKER="${2:?missing path after --checker}"; shift 2 ;;
    --parser) PARSER="${2:?missing path after --parser}"; shift 2 ;;
    --souc) SOUC="${2:?missing path after --souc}"; shift 2 ;;
    --control-child) RUN_CONTROLS=0; SOUC=""; shift ;;
    *) echo "diagnostic_remap_2170_gate: unknown argument: $1" >&2; exit 2 ;;
  esac
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/diagnostic-remap-2170.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "diagnostic_remap_2170: FAIL $*" >&2; exit 1; }

for path in "$CHECKER" "$PARSER" "$CATALOG"; do
  [[ -r "$path" ]] || fail "missing input $path"
done

python3 - "$CHECKER" "$PARSER" "$CATALOG" <<'PY' || exit 1
import pathlib, re, sys

checker_path, parser_path, catalog_path = map(pathlib.Path, sys.argv[1:4])
checker = checker_path.read_text(errors="replace")
parser = parser_path.read_text(errors="replace")
catalog = catalog_path.read_text(errors="replace")

expected = {
    247: (208, "ZD locus is not a well-formed sedenion pair"),
    248: (217, "f128/f256 value conversion is not implemented"),
    249: (218, "f128/f256 is reserved for compiler-owned format identity"),
    250: (219, "call to an `extern \\\"C\\\"` function the native backend does not implement"),
}
emit_counts = {247: 3, 248: 2, 249: 0, 250: 3}
failures = []

for new, (old, message) in expected.items():
    arm_prefix = f'else if code == {new} {{ print("{message}'
    if checker.count(arm_prefix) != 1:
        failures.append(f"E{new} semantic message arm missing or ambiguous")
    calls = len(re.findall(rf",\s*{new},\s*0,\s*0,\s*0\)", checker))
    if calls != emit_counts[new]:
        failures.append(f"E{new} emitter count {calls} != {emit_counts[new]}")
    if re.search(rf"code\s*==\s*{old}\b|,\s*{old},\s*0,\s*0,\s*0\)", checker):
        failures.append(f"old colliding E{old} still owns a Madaros arm or emitter")
    if not re.search(rf"^\|\s*E{new}\s*\|", catalog, re.MULTILINE):
        failures.append(f"E{new} catalogue row missing")
    explanation = pathlib.Path(f"docs/llm-guide/explanations/E{new}.md")
    if not explanation.is_file():
        failures.append(f"E{new} explanation missing")

# V0-B+: parser_reject_reserved_wide_float_path is a no-op; E249 is retained as a
# catalogue/message identity only (no live emitter). Do not require parser E249 prints.
if "fn parser_reject_reserved_wide_float_path" not in parser:
    failures.append("parser_reject_reserved_wide_float_path helper missing")
if parser.count('print("error[E249]")') != 0:
    failures.append("parser still emits live E249 after V0-B lift (expected 0)")
if "error[E218]" in parser:
    failures.append("old colliding E218 remains in parser reserved-wide-float path")

if failures:
    for item in failures:
        print(f"diagnostic_remap_2170: FAIL {item}", file=sys.stderr)
    raise SystemExit(1)

print("diagnostic_remap_2170: STATIC_PASS mappings=E208->E247,E217->E248,E218->E249(catalog_only),E219->E250")
PY

if [[ -n "$SOUC" ]]; then
  [[ -x "$SOUC" ]] || fail "--souc is not executable: $SOUC"

  live_refuse() {
    local code="$1" fixture="$2" label="$3"
    local log="$WORK/live-$code.log"
    "$SOUC" check "$fixture" >"$log" 2>&1
    local rc=$?
    if ! grep -Fq "error[E${code}]" "$log"; then
      sed 's/^/[live] /' "$log" >&2
      fail "$label did not emit error[E${code}] (rc=$rc)"
    fi
    if grep -Fq 'check: OK' "$log"; then
      fail "$label printed check: OK beside error[E${code}]"
    fi
    echo "diagnostic_remap_2170: LIVE_PASS $label=E${code} rc=$rc"
  }

  live_refuse 247 tests/audit/zd_mut_spine/exactly_private_locus_malformed.sio zd_locus
  # V0-B+: f128 source signatures are accepted; E249 is catalog-only.
  live_refuse 250 tests/compile-fail/extern_c_unimplemented_builtin.sio unsupported_extern
  echo "diagnostic_remap_2170: LIVE_BOUNDARY E248=structural_only E249=catalog_only_after_V0B"
else
  echo "diagnostic_remap_2170: LIVE_NOT_RUN provide --souc with a source-fresh Madaros wrapper"
fi

if [[ "$RUN_CONTROLS" -eq 1 ]]; then
  control_remap() {
    local new="$1" old="$2" kind="$3"
    local mutant="$WORK/${kind}-${new}-to-${old}.sio"
    if [[ "$kind" == "parser" ]]; then
      # No live parser E249 after V0-B; skip parser sabotage for 249.
      if [[ "$new" == "249" ]]; then
        echo "diagnostic_remap_2170: SABOTAGE_SKIP E249 parser (no live emitter after V0-B)"
        return 0
      fi
      sed "s/error\[E${new}\]/error[E${old}]/" "$PARSER" >"$mutant"
      if "$0" --control-child --checker "$CHECKER" --parser "$mutant" >"$WORK/control-$new.log" 2>&1; then
        fail "sabotage E${new}->E${old} was accepted"
      fi
    else
      if [[ "$new" == "249" ]]; then
        # Sabotage the catalogue message arm identity instead of a missing emitter.
        python3 - "$CHECKER" "$mutant" <<'PY'
import pathlib, sys
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
text = src.read_text()
old = 'else if code == 249 { print("f128/f256 is reserved for compiler-owned format identity; source values are unavailable in V0-A") }'
new = 'else if code == 218 { print("f128/f256 is reserved for compiler-owned format identity; source values are unavailable in V0-A") }'
if old not in text:
    raise SystemExit("cannot build E249 message-arm control")
dst.write_text(text.replace(old, new, 1))
PY
        if "$0" --control-child --checker "$mutant" --parser "$PARSER" >"$WORK/control-$new.log" 2>&1; then
          fail "sabotage E249 message arm -> E218 was accepted"
        fi
        echo "diagnostic_remap_2170: SABOTAGE_PASS E249->E218 message_arm rejected"
        return 0
      fi
      python3 - "$CHECKER" "$mutant" "$new" "$old" <<'PY'
import pathlib, sys
src, dst, new, old = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
text = src.read_text()
needle = f", {new}, 0, 0, 0)"
if needle not in text:
    raise SystemExit(f"cannot build control: {needle} absent")
dst.write_text(text.replace(needle, f", {old}, 0, 0, 0)", 1))
PY
      if "$0" --control-child --checker "$mutant" --parser "$PARSER" >"$WORK/control-$new.log" 2>&1; then
        fail "sabotage E${new}->E${old} was accepted"
      fi
    fi
    echo "diagnostic_remap_2170: SABOTAGE_PASS E${new}->E${old} rejected"
  }

  control_remap 247 208 checker
  control_remap 248 217 checker
  control_remap 249 218 checker
  control_remap 250 219 checker
fi

echo "diagnostic_remap_2170: PASS controls=$RUN_CONTROLS live=$([[ -n "$SOUC" ]] && echo executed || echo not_run)"
