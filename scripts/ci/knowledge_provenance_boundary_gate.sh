#!/usr/bin/env bash
# Accusation gate: a provenance mismatch at a Knowledge[...] fn boundary
# typechecks with no diagnostic, while a validity mismatch in the same
# syntactic position is refused.
#
# Basis: observer surface survey (artifacts/audit/observer_surface_survey_20260831.md)
# read statically that check_knowledge_type drops validity/provenance before
# TypeEntry (self-hosted/check/epistemic.sio:49) and that knowledge_meta_from_ty
# rebuilds every meta as always-valid/DERIVED (:497-530), so the
# provenance_subsumes call at the end of knowledge_call_boundary_compatible
# (epistemic.sio:908-924) compares DERIVED with DERIVED — the socket exists,
# the wire carries nothing.
#
# This is NOT a regression gate. It FAILS while the silence exists. When a
# provenance mismatch starts refusing, this gate turns green. Do not weaken
# the probes to pass; that is a language change.
#
# Engine/mode discipline (measured 2026-08-31, Slurm r770, from-source build):
#   - raw positional `souc <src> <out>` is the lean_single bootstrap engine;
#     Madaros speaks verbs (check/compile). Mode names here name ENGINES.
#   - E241 (unknown annotation component) refuses in Madaros `check` only.
#   - The committed bin/madaros-linux-x86_64 predates E241 entirely.
#   - lean_single's ty_eq is annotation-hash-based: an explicit eps bound on
#     the annotation trips its generic P0003 at the `let`. The lean probes
#     therefore carry no eps bound; the Madaros probes carry eps < 0.05 on
#     BOTH sides so the epsilon surface is satisfied and only the probed
#     slot (provenance, or validity for the controls) can disagree.
#
# Legs:
#   madaros-check / madaros-compile : silence + control
#   lean-raw                        : silence_lean + control_lean
#   madaros-check                   : literature probe + E241 fixture (arm presence)
# A control that fails to refuse voids its leg (verdict: error). A silence
# refusal only counts as enforcement if the diagnostic NAMES provenance.
#
# Authoritative path builds Madaros from source on a Slurm node. Local
# fallback uses committed binaries and is labeled STALE-SUSPECT.
#
# Artifact: artifacts/audit/knowledge_provenance_boundary/status.json
# Receipt:  tests/audit/KNOWLEDGE_PROVENANCE_BOUNDARY_SILENCE_2026-08-31.md
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SILENCE="tests/audit/knowledge_provenance_boundary_silence.sio"
CONTROL="tests/audit/knowledge_validity_boundary_control.sio"
SILENCE_LEAN="tests/audit/knowledge_provenance_boundary_silence_lean.sio"
CONTROL_LEAN="tests/audit/knowledge_validity_boundary_control_lean.sio"
LITERATURE="tests/audit/knowledge_literature_e241_probe.sio"
E241FIX="tests/compile-fail/knowledge_unknown_component_ident.sio"
OUT_DIR="${KNOWLEDGE_PROV_BOUNDARY_OUT:-$ROOT/artifacts/audit/knowledge_provenance_boundary}"
mkdir -p "$OUT_DIR"

# classify <log>: prints refused:<pattern> or silent
classify() {
  local log="$1"
  if grep -qiE 'temporal validity window' "$log"; then echo "refused:temporal-validity"; return; fi
  if grep -qiE 'error\[E241\]|unknown Knowledge annotation' "$log"; then echo "refused:e241"; return; fi
  if grep -qiE 'error\[E[0-9]+\]|error\[P[0-9]+\]|type error|parse error|refus' "$log"; then echo "refused:other"; return; fi
  echo "silent"
}

# classify_silence <log>: only a diagnostic NAMING provenance falsifies the
# silence. Any other refusal means the probe tripped an unrelated gate and
# the measurement is confounded — reported, never counted as enforcement.
classify_silence() {
  local log="$1"
  if grep -qiE 'error\[E[0-9]+\]|error\[P[0-9]+\]|type error|parse error|refus' "$log"; then
    if grep -qiE 'provenance' "$log"; then echo "refused:provenance"; return; fi
    echo "confounded:$(grep -oiE 'error\[E[0-9]+\]|error\[P[0-9]+\]|type error|parse error' "$log" | head -1)"; return
  fi
  echo "silent"
}

run_slurm() {
  tar -czf - -C "$ROOT" \
      self-hosted stdlib bin/souc bin/souc-linux-x86_64 bin/madaros scripts \
      "$SILENCE" "$CONTROL" "$SILENCE_LEAN" "$CONTROL_LEAN" "$LITERATURE" "$E241FIX" \
    | srun -p "${KNOWLEDGE_PROV_PARTITION:-cpu-ops}" -N1 -n1 -c8 \
        --mem=16G --time=00:20:00 --chdir=/tmp \
        --job-name=know-prov-boundary \
        bash -c '
          set -uo pipefail
          W=$(mktemp -d /tmp/know-prov.XXXXXX)
          cd "$W" || exit 1
          tar xzf - || { echo "REMOTE: untar failed" >&2; exit 1; }
          echo "REMOTE: host=$(hostname) nproc=$(nproc)" >&2
          export SOUNIO_STDLIB_PATH="$W/stdlib"
          export SOUNIO_BUILD_LOCK=/tmp/know-prov-build-$$.lock
          ulimit -s 524288 2>/dev/null || true
          t0=$SECONDS
          bash scripts/ci/build_modular_madaros.sh "$W/madaros.elf" > "$W/build.log" 2>&1
          brc=$?
          echo "REMOTE: build rc=$brc elapsed=$((SECONDS-t0))s" >&2
          if [ $brc -ne 0 ]; then tail -15 "$W/build.log" >&2; exit $brc; fi
          chmod +x "$W/madaros.elf"
          mkdir -p "$W/out"
          run_leg() {  # <engine> <verb-or-raw> <src> <logbase>
            local eng="$1" mode="$2" src="$3" b="$4"
            if [ "$eng" = madaros ]; then
              if [ "$mode" = check ]; then
                MADAROS_RAW_BIN="$W/madaros.elf" ./bin/souc check "$src" > "$W/out/$b.check.log" 2>&1
                echo "$?" > "$W/out/$b.check.rc"
              else
                MADAROS_RAW_BIN="$W/madaros.elf" ./bin/souc compile "$src" -o "$W/out/$b.elf" > "$W/out/$b.compile.log" 2>&1
                echo "$?" > "$W/out/$b.compile.rc"
              fi
            else
              ./bin/souc-linux-x86_64 "$src" "$W/out/$b.elf" > "$W/out/$b.raw.log" 2>&1
              echo "$?" > "$W/out/$b.raw.rc"
            fi
          }
          run_leg madaros check    tests/audit/knowledge_provenance_boundary_silence.sio      silence
          run_leg madaros compile  tests/audit/knowledge_provenance_boundary_silence.sio      silence
          run_leg madaros check    tests/audit/knowledge_validity_boundary_control.sio        control
          run_leg madaros compile  tests/audit/knowledge_validity_boundary_control.sio        control
          run_leg madaros check    tests/audit/knowledge_provenance_boundary_silence_lean.sio silence_lean
          run_leg lean    raw      tests/audit/knowledge_provenance_boundary_silence_lean.sio silence_lean
          run_leg lean    raw      tests/audit/knowledge_validity_boundary_control_lean.sio   control_lean
          run_leg madaros check    tests/audit/knowledge_literature_e241_probe.sio            literature
          run_leg madaros compile  tests/audit/knowledge_literature_e241_probe.sio            literature
          run_leg madaros check    tests/compile-fail/knowledge_unknown_component_ident.sio   e241fix
          for l in "$W"/out/*.rc; do echo "REMOTE: $(basename "$l")=$(cat "$l")" >&2; done
          tar -C "$W/out" -czf - .
          rm -rf "$W"
        ' > "$OUT_DIR/slurm_bundle.tar.gz"
  tar -C "$OUT_DIR" -xzf "$OUT_DIR/slurm_bundle.tar.gz"
}

run_local() {
  # shellcheck disable=SC1091
  source "$ROOT/scripts/lib/resolve_souc.sh"
  sounio_require_souc
  echo "LOCAL-committed-binaries: verdicts are STALE-SUSPECT (committed Madaros predates E241)"
  ulimit -s 524288 2>/dev/null || true
  env -u SOUC_BIN -u SOUNIO_SOUC_BIN "$SOUC_BIN" check "$ROOT/$SILENCE"  > "$OUT_DIR/silence.check.log" 2>&1; echo "$?" > "$OUT_DIR/silence.check.rc"
  env -u SOUC_BIN -u SOUNIO_SOUC_BIN "$SOUC_BIN" compile "$ROOT/$SILENCE" -o "$OUT_DIR/silence.elf" > "$OUT_DIR/silence.compile.log" 2>&1; echo "$?" > "$OUT_DIR/silence.compile.rc"
  env -u SOUC_BIN -u SOUNIO_SOUC_BIN "$SOUC_BIN" check "$ROOT/$CONTROL"  > "$OUT_DIR/control.check.log" 2>&1; echo "$?" > "$OUT_DIR/control.check.rc"
  env -u SOUC_BIN -u SOUNIO_SOUC_BIN "$SOUC_BIN" compile "$ROOT/$CONTROL" -o "$OUT_DIR/control.elf" > "$OUT_DIR/control.compile.log" 2>&1; echo "$?" > "$OUT_DIR/control.compile.rc"
  env -u SOUC_BIN -u SOUNIO_SOUC_BIN "$SOUC_BIN" check "$ROOT/$SILENCE_LEAN" > "$OUT_DIR/silence_lean.check.log" 2>&1; echo "$?" > "$OUT_DIR/silence_lean.check.rc"
  "$ROOT/bin/souc-linux-x86_64" "$ROOT/$SILENCE_LEAN" "$OUT_DIR/silence_lean.elf" > "$OUT_DIR/silence_lean.raw.log" 2>&1; echo "$?" > "$OUT_DIR/silence_lean.raw.rc"
  "$ROOT/bin/souc-linux-x86_64" "$ROOT/$CONTROL_LEAN" "$OUT_DIR/control_lean.elf" > "$OUT_DIR/control_lean.raw.log" 2>&1; echo "$?" > "$OUT_DIR/control_lean.raw.rc"
  env -u SOUC_BIN -u SOUNIO_SOUC_BIN "$SOUC_BIN" check "$ROOT/$LITERATURE" > "$OUT_DIR/literature.check.log" 2>&1; echo "$?" > "$OUT_DIR/literature.check.rc"
  env -u SOUC_BIN -u SOUNIO_SOUC_BIN "$SOUC_BIN" compile "$ROOT/$LITERATURE" -o "$OUT_DIR/literature.elf" > "$OUT_DIR/literature.compile.log" 2>&1; echo "$?" > "$OUT_DIR/literature.compile.rc"
  env -u SOUC_BIN -u SOUNIO_SOUC_BIN "$SOUC_BIN" check "$ROOT/$E241FIX"   > "$OUT_DIR/e241fix.check.log" 2>&1; echo "$?" > "$OUT_DIR/e241fix.check.rc"
}

if command -v srun >/dev/null 2>&1 && [[ "${KNOWLEDGE_PROV_SLURM:-1}" != "0" ]]; then
  run_slurm
else
  run_local
fi

# Transport guard: if any log is missing the measurement never happened.
# Classify nothing; fail as an infrastructure error, never as a verdict.
for l in silence.check silence.compile control.check control.compile \
         silence_lean.check silence_lean.raw control_lean.raw \
         literature.check literature.compile e241fix.check; do
  if [ ! -s "$OUT_DIR/$l.log" ]; then
    python3 - "$OUT_DIR/status.json" <<'PY'
import json, sys
with open(sys.argv[1], "w") as fh:
    json.dump({"status": "error", "reason": "transport failed — probe logs missing (Slurm/node/infra), no verdict measured"}, fh, indent=2)
PY
    echo "KNOWLEDGE_PROVENANCE_BOUNDARY status=error reason=\"transport failed, no verdict measured ($l)\"" >&2
    exit 2
  fi
done

# --- capability controls (each leg must prove it CAN refuse) ---
cap_e241=$(classify "$OUT_DIR/e241fix.check.log")                 # madaros-check parses & refuses Knowledge annotations
cap_eps=$(classify "$OUT_DIR/silence_lean.check.log")             # madaros-check call-boundary machinery fires (E036 epsilon leg)
cap_compile=$(classify "$OUT_DIR/literature.compile.log")         # madaros-compile propagates parse-level refusal
cap_lean=$(classify "$OUT_DIR/control_lean.raw.log")              # lean_single enforces the validity window
# --- the accusation legs ---
v_s_check=$(classify_silence "$OUT_DIR/silence.check.log")
v_s_compile=$(classify_silence "$OUT_DIR/silence.compile.log")
v_s_lean=$(classify_silence "$OUT_DIR/silence_lean.raw.log")
# --- the asymmetry report (validity under Madaros: measured, not verdict) ---
v_c_check=$(classify "$OUT_DIR/control.check.log")
v_c_compile=$(classify "$OUT_DIR/control.compile.log")
v_lit=$(classify "$OUT_DIR/literature.check.log")

echo "capability e241-fixture madaros-check:     $cap_e241"
echo "capability epsilon-boundary madaros-check: $cap_eps"
echo "capability literature madaros-compile:     $cap_compile"
echo "capability validity lean-raw:              $cap_lean"
echo "silence madaros-check:                     $v_s_check"
echo "silence madaros-compile:                   $v_s_compile"
echo "silence lean-raw:                          $v_s_lean"
echo "report validity madaros-check:             $v_c_check"
echo "report validity madaros-compile:           $v_c_compile"
echo "report literature madaros-check:           $v_lit"

verdict="fail"
reason=""
if [ "$cap_e241" != "refused:e241" ] || [ "$cap_lean" != "refused:temporal-validity" ] \
   || [ "$cap_eps" = "silent" ] || [ "$cap_compile" = "silent" ]; then
  verdict="error"; reason="capability control failed (e241=$cap_e241 eps=$cap_eps compile=$cap_compile lean=$cap_lean) — harness void"
elif [ "$v_s_check" = "refused:provenance" ] || [ "$v_s_compile" = "refused:provenance" ] || [ "$v_s_lean" = "refused:provenance" ]; then
  verdict="pass"; reason="provenance mismatch refused naming provenance (check=$v_s_check compile=$v_s_compile lean=$v_s_lean)"
elif [ "$v_s_check" != "silent" ] || [ "$v_s_compile" != "silent" ] || [ "$v_s_lean" != "silent" ]; then
  verdict="error"; reason="probe confounded by unrelated diagnostic (check=$v_s_check compile=$v_s_compile lean=$v_s_lean) — fix the probe"
else
  verdict="fail"; reason="provenance-only mismatch typechecks silently in every engine leg (accusation stands)"
fi

python3 - "$OUT_DIR/status.json" "$verdict" "$reason" \
        "$cap_e241" "$cap_eps" "$cap_compile" "$cap_lean" \
        "$v_s_check" "$v_s_compile" "$v_s_lean" \
        "$v_c_check" "$v_c_compile" "$v_lit" <<'PY'
import json, sys
path, verdict, reason = sys.argv[1:4]
v = sys.argv[4:14]
with open(path, "w") as fh:
    json.dump({"status": verdict, "reason": reason,
               "capability": {"e241_fixture_madaros_check": v[0], "epsilon_boundary_madaros_check": v[1],
                              "literature_madaros_compile": v[2], "validity_lean_raw": v[3]},
               "silence_legs": {"madaros_check": v[4], "madaros_compile": v[5], "lean_raw": v[6]},
               "asymmetry_report": {"validity_madaros_check": v[7], "validity_madaros_compile": v[8],
                                    "literature_madaros_check": v[9]}},
              fh, indent=2)
PY

echo "KNOWLEDGE_PROVENANCE_BOUNDARY status=$verdict reason=\"$reason\""
[ "$verdict" = "pass" ] && exit 0
[ "$verdict" = "fail" ] && exit 1
exit 2
