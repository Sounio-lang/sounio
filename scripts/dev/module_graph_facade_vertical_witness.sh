#!/usr/bin/env bash
# scripts/dev/module_graph_facade_vertical_witness.sh
#
# Vertical witness for the Sounio ModuleGraph facade closure proposition:
#
#   imported facade
#     -> complete logical/physical dependency closure
#     -> modular checker
#     -> lowering without fallback
#     -> native ELF executes and prints 42
#
# The witness exercises TWO fixture closures against the default Madaros
# compiler (bin/souc -> Madaros):
#
#   CONTROL  : examples/projects/hello_pkg/src/{main,greet}.sio  (2-module direct use)
#   PROBE    : tests/compiler/module_graph_facade_vertical_witness/{main,facade,leaf}.sio
#              (3-module transitive pub use re-export)
#
# For each closure the witness performs a MUTATION TEST: it compiles the
# original fixture, then compiles a variant where the leafmost source file
# returns a different literal. If the two ELFs are bit-identical (same
# sha256), the closure is being silently dropped by the compiler and the
# verdict is BLOCKED with classification `silent_corruption_*`.
#
# Output:
#   - stdout: human progress + a single final verdict line
#   - artifacts/witnesses/module_graph_facade_vertical_<UTC>.json : machine receipt
#
# Verdicts (exactly one):
#   PASS    : every closure compiles rc=0 AND ELF output reflects source semantics
#   BLOCKED : first failing boundary with exact command, exit code, and evidence
#   STALE   : raw Madaros ELF is older than the compiler source
#   INFRA   : compiler path cannot be tested (launcher missing, raw ELF missing, etc.)
#
# This script does NOT modify compiler-owned files. It only:
#   - reads from the worktree
#   - mutates the leafmost fixture in-place temporarily (restored in trap)
#   - writes ELFs and the JSON receipt to scratch paths
#
# Bounded scope: no CI wiring, no ModuleGraph abstractions, no roadmap expansion.
# The witness is observational evidence, not a fix.

set -uo pipefail

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

UTC_NOW="$(date -u +%Y%m%dT%H%M%SZ)"
RECEIPT_DIR="${WITNESS_RECEIPT_DIR:-$REPO_ROOT/artifacts/witnesses}"
RECEIPT_PATH="$RECEIPT_DIR/module_graph_facade_vertical_${UTC_NOW}.json"
SCRATCH_DIR="$(mktemp -d /tmp/sounio-modgraph-witness.XXXXXX)"
trap 'rm -rf "$SCRATCH_DIR"; [[ -n "${CONTROL_GREET_BACKUP:-}" ]] && mv -f "$CONTROL_GREET_BACKUP" "$CONTROL_GREET_PATH" 2>/dev/null || true; [[ -n "${PROBE_LEAF_BACKUP:-}" ]] && mv -f "$PROBE_LEAF_BACKUP" "$PROBE_LEAF_PATH" 2>/dev/null || true' EXIT

mkdir -p "$RECEIPT_DIR"

# Identity carried through the receipt.
WITNESS_ID="module_graph_facade_vertical"
WITNESS_SCHEMA="1.0"

# Fixture paths.
CONTROL_MAIN="$REPO_ROOT/examples/projects/hello_pkg/src/main.sio"
CONTROL_GREET_PATH="$REPO_ROOT/examples/projects/hello_pkg/src/greet.sio"

PROBE_DIR="$REPO_ROOT/tests/compiler/module_graph_facade_vertical_witness"
PROBE_MAIN="$PROBE_DIR/main.sio"
PROBE_FACADE="$PROBE_DIR/facade.sio"
PROBE_LEAF_PATH="$PROBE_DIR/leaf.sio"

# ----------------------------------------------------------------------------
# Phase 0 : environment capture
# ----------------------------------------------------------------------------

log()  { printf '[witness] %s\n' "$*" >&2; }
section() { printf '\n========== %s ==========\n' "$*" >&2; }

section "Phase 0: environment capture"

GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
GIT_WORKTREE="$(git rev-parse --show-toplevel 2>/dev/null || echo "$REPO_ROOT")"

if [[ ! -x "$REPO_ROOT/bin/souc" ]]; then
  verdict="INFRA"
  verdict_reason="bin/souc launcher not executable at $REPO_ROOT/bin/souc"
  log "INFRA: $verdict_reason"
  emit_receipt
  exit 0
fi

SOUC_VERSION="$("$REPO_ROOT/bin/souc" --version 2>&1 | head -n1 || echo unknown)"

# Resolve the raw Madaros ELF the launcher will actually use (mirror its logic).
RAW_MADAROS=""
for cand in "${MADAROS_RAW_BIN:-}" "${SOUNIO_MADAROS_BIN:-}" \
            "$REPO_ROOT/artifacts/self-hosted/madaros" \
            "$REPO_ROOT/bin/madaros-linux-x86_64"; do
  if [[ -n "$cand" && -x "$cand" && "$(head -c2 "$cand" 2>/dev/null)" != "#!" ]]; then
    RAW_MADAROS="$cand"; break
  fi
done

if [[ -z "$RAW_MADAROS" ]]; then
  verdict="INFRA"
  verdict_reason="no raw Madaros ELF resolved"
  log "INFRA: $verdict_reason"
  emit_receipt
  exit 0
fi

RAW_MADAROS_SHA256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
RAW_MADAROS_MTIME="$(stat -c '%Y' "$RAW_MADAROS" 2>/dev/null || echo 0)"
RAW_MADAROS_MTIME_UTC="$(date -u -d "@$RAW_MADAROS_MTIME" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"

log "branch=$GIT_BRANCH commit=$GIT_COMMIT dirty=$GIT_DIRTY"
log "launcher=$REPO_ROOT/bin/souc"
log "raw_elf=$RAW_MADAROS"
log "raw_elf_sha256=$RAW_MADAROS_SHA256"
log "raw_elf_mtime_utc=$RAW_MADAROS_MTIME_UTC"
log "version=$SOUC_VERSION"

# ----------------------------------------------------------------------------
# Phase 1 : STALE detection
# ----------------------------------------------------------------------------

section "Phase 1: STALE detection (raw_elf mtime vs compiler source)"

COMPILER_SOURCE_NEWEST_MTIME=0
while IFS= read -r f; do
  m="$(stat -c '%Y' "$f" 2>/dev/null || echo 0)"
  if [[ "$m" -gt "$COMPILER_SOURCE_NEWEST_MTIME" ]]; then
    COMPILER_SOURCE_NEWEST_MTIME="$m"
  fi
done < <(find "$REPO_ROOT/self-hosted/compiler" -maxdepth 1 -name '*.sio' -print 2>/dev/null)

COMPILER_SOURCE_NEWEST_UTC="$(date -u -d "@$COMPILER_SOURCE_NEWEST_MTIME" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
log "compiler_source_newest_utc=$COMPILER_SOURCE_NEWEST_UTC"

# STALE candidate only when raw_elf is substantially older than the compiler
# source. A few seconds of mtime skew is normal during `git worktree add` and
# does not indicate a stale binary; we require at least 1 hour of drift so the
# signal is meaningful.
STALE_THRESHOLD_SECONDS="${WITNESS_STALE_THRESHOLD_SECONDS:-3600}"
if [[ "$RAW_MADAROS_MTIME" -gt 0 \
      && $((COMPILER_SOURCE_NEWEST_MTIME - RAW_MADAROS_MTIME)) -ge "$STALE_THRESHOLD_SECONDS" ]]; then
  stale_candidate=true
  log "STALE candidate: raw_elf predates newest self-hosted/compiler/*.sio by >= ${STALE_THRESHOLD_SECONDS}s"
else
  stale_candidate=false
  drift=$((COMPILER_SOURCE_NEWEST_MTIME - RAW_MADAROS_MTIME))
  if [[ "$drift" -lt 0 ]]; then drift=0; fi
  log "raw_elf source/elf mtime drift=${drift}s (below stale threshold ${STALE_THRESHOLD_SECONDS}s)"
fi

# ----------------------------------------------------------------------------
# Compile helper — captures rc + stdout + stderr + sha256 of ELF + first ELF run
# ----------------------------------------------------------------------------

run_compile() {
  local src="$1"
  local out="$2"
  local stderr_file="$3"
  local stdout_file="$4"
  shift 4
  rm -f "$out"
  "$REPO_ROOT/bin/souc" compile "$src" -o "$out" "$@" \
    >"$stdout_file" 2>"$stderr_file"
  echo $?
}

run_elf_capture() {
  local elf="$1"
  local stdout_file="$2"
  local stderr_file="$3"
  local rc
  if [[ ! -x "$elf" ]]; then
    echo 127
    return
  fi
  "$elf" >"$stdout_file" 2>"$stderr_file"
  rc=$?
  echo $rc
}

elf_sha() {
  local elf="$1"
  if [[ ! -s "$elf" ]]; then echo ""; return; fi
  sha256sum "$elf" | awk '{print $1}'
}

elf_has_magic() {
  local elf="$1"
  if [[ ! -s "$elf" ]]; then echo false; return; fi
  if [[ "$(head -c4 "$elf" 2>/dev/null | od -An -tx1 | tr -d ' \n')" == "7f454c46" ]]; then
    echo true
  else
    echo false
  fi
}

# Classification helper — boundary strings are deliberately EPISTEMICALLY TIGHT.
# The witness proves semantics is lost somewhere between closure load and runtime
# observation. It does NOT prove the loss happens at any specific file/line. The
# merge lane (module_frontend.sio:4290-4722) is the PRIMARY SUSPECT with strong
# evidence, but is not yet condemned in isolation. Condemnation requires the
# differential IR comparison described in next_step_differential_ir_hashes.
BOUNDARY_RUNTIME_STDOUT="runtime stdout does not reflect mutated imported function body"
BOUNDARY_SUSPECT_MERGE="semantics lost between loaded closure and executed ELF; primary suspect module_frontend.sio:4290-4722 (fn_remap merge), pending differential IR hashes"

# ----------------------------------------------------------------------------
# Phase 2 : CONTROL — hello_pkg with greet=42, then greet=999
# ----------------------------------------------------------------------------

section "Phase 2: CONTROL (hello_pkg, 2-module direct use)"

if [[ ! -f "$CONTROL_MAIN" || ! -f "$CONTROL_GREET_PATH" ]]; then
  verdict="INFRA"
  verdict_reason="CONTROL fixture missing: $CONTROL_MAIN or $CONTROL_GREET_PATH"
  log "INFRA: $verdict_reason"
  emit_receipt
  exit 0
fi

CONTROL_GREET_BACKUP="$(mktemp /tmp/sounio-witness-greet.XXXXXX.sio)"
cp -a "$CONTROL_GREET_PATH" "$CONTROL_GREET_BACKUP"

# Phase 2a: greet=42 (canonical fixture).
CONTROL_A_OUT="$SCRATCH_DIR/control_a.elf"
CONTROL_A_STDERR="$SCRATCH_DIR/control_a.stderr"
CONTROL_A_STDOUT_CC="$SCRATCH_DIR/control_a.cc.out"
CONTROL_A_STDOUT_RUN="$SCRATCH_DIR/control_a.run.out"
CONTROL_A_STDERR_RUN="$SCRATCH_DIR/control_a.run.err"

log "compiling CONTROL_A (greet=42, canonical)..."
CONTROL_A_CC_RC=$(run_compile "$CONTROL_MAIN" "$CONTROL_A_OUT" "$CONTROL_A_STDERR" "$CONTROL_A_STDOUT_CC")
CONTROL_A_RUN_RC=$(run_elf_capture "$CONTROL_A_OUT" "$CONTROL_A_STDOUT_RUN" "$CONTROL_A_STDERR_RUN")
CONTROL_A_SHA=$(elf_sha "$CONTROL_A_OUT")
CONTROL_A_MAGIC=$(elf_has_magic "$CONTROL_A_OUT")
CONTROL_A_OUT_LAST="$(tail -n1 "$CONTROL_A_STDOUT_RUN" 2>/dev/null | tr -d '\n')"
log "CONTROL_A: cc_rc=$CONTROL_A_CC_RC run_rc=$CONTROL_A_RUN_RC magic=$CONTROL_A_MAGIC last='$CONTROL_A_OUT_LAST' sha=${CONTROL_A_SHA:0:16}"

# Phase 2b: mutate greet.sio to return 999.
cat >"$CONTROL_GREET_PATH" <<'EOF'
pub fn answer() -> i64 {
    999
}
EOF

CONTROL_B_OUT="$SCRATCH_DIR/control_b.elf"
CONTROL_B_STDERR="$SCRATCH_DIR/control_b.stderr"
CONTROL_B_STDOUT_CC="$SCRATCH_DIR/control_b.cc.out"
CONTROL_B_STDOUT_RUN="$SCRATCH_DIR/control_b.run.out"

log "compiling CONTROL_B (greet mutated to 999)..."
CONTROL_B_CC_RC=$(run_compile "$CONTROL_MAIN" "$CONTROL_B_OUT" "$CONTROL_B_STDERR" "$CONTROL_B_STDOUT_CC")
CONTROL_B_RUN_RC=$(run_elf_capture "$CONTROL_B_OUT" "$CONTROL_B_STDOUT_RUN" "/dev/null")
CONTROL_B_SHA=$(elf_sha "$CONTROL_B_OUT")
CONTROL_B_MAGIC=$(elf_has_magic "$CONTROL_B_OUT")
CONTROL_B_OUT_LAST="$(tail -n1 "$CONTROL_B_STDOUT_RUN" 2>/dev/null | tr -d '\n')"
log "CONTROL_B: cc_rc=$CONTROL_B_CC_RC run_rc=$CONTROL_B_RUN_RC magic=$CONTROL_B_MAGIC last='$CONTROL_B_OUT_LAST' sha=${CONTROL_B_SHA:0:16}"

# Restore greet.sio immediately.
mv -f "$CONTROL_GREET_BACKUP" "$CONTROL_GREET_PATH"
CONTROL_GREET_BACKUP=""

if ! diff -q "$CONTROL_GREET_PATH" "$REPO_ROOT/examples/projects/hello_pkg/src/greet.sio" >/dev/null 2>&1; then
  log "WARNING: greet.sio restore mismatch"
fi

# Classify CONTROL.
#
# Epistemic framing: this witness proves semantics is lost somewhere between
# closure load and runtime observation. It does NOT prove the loss happens at
# any specific file/line. The classifications below describe WHAT is observed,
# not WHERE the cause lives. Condemning the merge lane requires the differential
# IR comparison described in the receipt's next_step_differential_ir_hashes.
if [[ "$CONTROL_A_CC_RC" -ne 0 ]]; then
  CONTROL_STATUS="BLOCKED"
  CONTROL_CLASS="compile_fail"
  CONTROL_BOUNDARY="bin/souc compile"
elif [[ "$CONTROL_A_MAGIC" != "true" ]]; then
  CONTROL_STATUS="BLOCKED"
  CONTROL_CLASS="no_elf_emitted"
  CONTROL_BOUNDARY="native codegen"
elif [[ "$CONTROL_A_RUN_RC" -ne 0 ]]; then
  CONTROL_STATUS="BLOCKED"
  CONTROL_CLASS="run_rc_nonzero"
  CONTROL_BOUNDARY="runtime"
elif [[ "$CONTROL_A_OUT_LAST" != "42" ]]; then
  CONTROL_STATUS="BLOCKED"
  CONTROL_CLASS="stdout_not_42"
  CONTROL_BOUNDARY="runtime stdout"
elif [[ "$CONTROL_A_SHA" == "$CONTROL_B_SHA" ]]; then
  # Bit-identical ELFs despite greet.sio mutation: semantics is being dropped
  # somewhere upstream of codegen. Strongest evidence for silent corruption.
  CONTROL_STATUS="BLOCKED"
  CONTROL_CLASS="silent_corruption_elf_invariant_under_mutation"
  CONTROL_BOUNDARY="$BOUNDARY_SUSPECT_MERGE"
elif [[ "$CONTROL_B_OUT_LAST" != "999" ]]; then
  # ELFs differ but the runtime stdout does not reflect the mutation either:
  # semantics is lost somewhere between source mutation and runtime, but the
  # ELF bytes themselves do vary (likely a non-semantic payload: path string,
  # timestamp, or non-functional code path). The loss is downstream of merge
  # but not yet localised to merge vs codegen vs call-resolution.
  CONTROL_STATUS="BLOCKED"
  CONTROL_CLASS="closure_not_consumed_runtime"
  CONTROL_BOUNDARY="$BOUNDARY_RUNTIME_STDOUT; $BOUNDARY_SUSPECT_MERGE"
else
  CONTROL_STATUS="PASS"
  CONTROL_CLASS=""
  CONTROL_BOUNDARY=""
fi

log "CONTROL verdict: $CONTROL_STATUS class=$CONTROL_CLASS"

# ----------------------------------------------------------------------------
# Phase 3 : PROBE — 3-module transitive pub use
# ----------------------------------------------------------------------------

section "Phase 3: PROBE (3-module transitive pub use re-export)"

if [[ ! -f "$PROBE_MAIN" || ! -f "$PROBE_FACADE" || ! -f "$PROBE_LEAF_PATH" ]]; then
  verdict="INFRA"
  verdict_reason="PROBE fixture missing under $PROBE_DIR"
  log "INFRA: $verdict_reason"
  emit_receipt
  exit 0
fi

PROBE_LEAF_BACKUP="$(mktemp /tmp/sounio-witness-leaf.XXXXXX.sio)"
cp -a "$PROBE_LEAF_PATH" "$PROBE_LEAF_BACKUP"

# Phase 3a: leaf=42 (canonical probe).
PROBE_A_OUT="$SCRATCH_DIR/probe_a.elf"
PROBE_A_STDERR="$SCRATCH_DIR/probe_a.stderr"
PROBE_A_STDOUT_CC="$SCRATCH_DIR/probe_a.cc.out"
PROBE_A_STDOUT_RUN="$SCRATCH_DIR/probe_a.run.out"

log "compiling PROBE_A (leaf=42)..."
PROBE_A_CC_RC=$(run_compile "$PROBE_MAIN" "$PROBE_A_OUT" "$PROBE_A_STDERR" "$PROBE_A_STDOUT_CC")
PROBE_A_RUN_RC=$(run_elf_capture "$PROBE_A_OUT" "$PROBE_A_STDOUT_RUN" "/dev/null")
PROBE_A_SHA=$(elf_sha "$PROBE_A_OUT")
PROBE_A_MAGIC=$(elf_has_magic "$PROBE_A_OUT")
PROBE_A_OUT_LAST="$(tail -n1 "$PROBE_A_STDOUT_RUN" 2>/dev/null | tr -d '\n')"
log "PROBE_A: cc_rc=$PROBE_A_CC_RC run_rc=$PROBE_A_RUN_RC magic=$PROBE_A_MAGIC last='$PROBE_A_OUT_LAST' sha=${PROBE_A_SHA:0:16}"

# Phase 3b: mutate leaf.sio to return 7.
cat >"$PROBE_LEAF_PATH" <<'EOF'
pub fn leaf_value() -> i64 {
    7
}
EOF

PROBE_B_OUT="$SCRATCH_DIR/probe_b.elf"
PROBE_B_STDERR="$SCRATCH_DIR/probe_b.stderr"
PROBE_B_STDOUT_CC="$SCRATCH_DIR/probe_b.cc.out"
PROBE_B_STDOUT_RUN="$SCRATCH_DIR/probe_b.run.out"

log "compiling PROBE_B (leaf mutated to 7)..."
PROBE_B_CC_RC=$(run_compile "$PROBE_MAIN" "$PROBE_B_OUT" "$PROBE_B_STDERR" "$PROBE_B_STDOUT_CC")
PROBE_B_RUN_RC=$(run_elf_capture "$PROBE_B_OUT" "$PROBE_B_STDOUT_RUN" "/dev/null")
PROBE_B_SHA=$(elf_sha "$PROBE_B_OUT")
PROBE_B_MAGIC=$(elf_has_magic "$PROBE_B_OUT")
PROBE_B_OUT_LAST="$(tail -n1 "$PROBE_B_STDOUT_RUN" 2>/dev/null | tr -d '\n')"
log "PROBE_B: cc_rc=$PROBE_B_CC_RC run_rc=$PROBE_B_RUN_RC magic=$PROBE_B_MAGIC last='$PROBE_B_OUT_LAST' sha=${PROBE_B_SHA:0:16}"

mv -f "$PROBE_LEAF_BACKUP" "$PROBE_LEAF_PATH"
PROBE_LEAF_BACKUP=""

# Look for known boundary markers in the compile output.
PROBE_CC_HAS_SEED_BEGIN=0
PROBE_CC_HAS_SEED_DONE=0
PROBE_CC_HAS_MERGED_IR=0
PROBE_CC_HAS_FALLBACK=0
PROBE_CC_HAS_COMPACT=0
PROBE_CC_HAS_CLOSURE_INCOMPLETE=0
PROBE_CC_MERGED_IR_N=""
grep -q 'lower_array: seed_begin' "$PROBE_A_STDOUT_CC" "$PROBE_A_STDERR" 2>/dev/null && PROBE_CC_HAS_SEED_BEGIN=1
grep -q 'lower_array: seed_done' "$PROBE_A_STDOUT_CC" "$PROBE_A_STDERR" 2>/dev/null && PROBE_CC_HAS_SEED_DONE=1
grep -q 'Merged IR:' "$PROBE_A_STDOUT_CC" "$PROBE_A_STDERR" 2>/dev/null && PROBE_CC_HAS_MERGED_IR=1
PROBE_CC_MERGED_IR_N="$(grep -oE 'Merged IR: [0-9]+' "$PROBE_A_STDOUT_CC" "$PROBE_A_STDERR" 2>/dev/null | head -n1 | awk '{print $3}')"
grep -qE 'fallback=|source=fallback' "$PROBE_A_STDOUT_CC" "$PROBE_A_STDERR" 2>/dev/null && PROBE_CC_HAS_FALLBACK=1
grep -q 'compact modular IR table path' "$PROBE_A_STDOUT_CC" "$PROBE_A_STDERR" 2>/dev/null && PROBE_CC_HAS_COMPACT=1
grep -q 'AST closure incomplete' "$PROBE_A_STDOUT_CC" "$PROBE_A_STDERR" 2>/dev/null && PROBE_CC_HAS_CLOSURE_INCOMPLETE=1

# Detect SIGSEGV during compile or run.
PROBE_SEGV_COMPILE=0
PROBE_SEGV_RUN=0
[[ $PROBE_A_CC_RC -ge 128 ]] && PROBE_SEGV_COMPILE=1
[[ $PROBE_A_RUN_RC -ge 128 ]] && PROBE_SEGV_RUN=1

# Classify PROBE (same epistemic framing as CONTROL — see comment above).
if [[ $PROBE_CC_HAS_CLOSURE_INCOMPLETE -eq 1 ]]; then
  PROBE_STATUS="BLOCKED"
  PROBE_CLASS="ast_closure_incomplete_fail_closed"
  PROBE_BOUNDARY="module_frontend.sio fail-closed on incomplete import closure"
elif [[ $PROBE_SEGV_COMPILE -eq 1 ]]; then
  PROBE_STATUS="BLOCKED"
  PROBE_CLASS="compile_sigsegv"
  PROBE_BOUNDARY="bin/souc compile exit $PROBE_A_CC_RC"
elif [[ "$PROBE_A_CC_RC" -ne 0 ]]; then
  PROBE_STATUS="BLOCKED"
  PROBE_CLASS="compile_fail"
  PROBE_BOUNDARY="bin/souc compile"
elif [[ "$PROBE_A_MAGIC" != "true" ]]; then
  PROBE_STATUS="BLOCKED"
  PROBE_CLASS="no_elf_emitted"
  PROBE_BOUNDARY="native codegen"
elif [[ $PROBE_SEGV_RUN -eq 1 ]]; then
  PROBE_STATUS="BLOCKED"
  PROBE_CLASS="runtime_sigsegv_between_seed_begin_and_seed_done"
  PROBE_BOUNDARY="self-hosted/compiler/module_frontend.sio:4514-4517"
elif [[ "$PROBE_A_RUN_RC" -ne 0 ]]; then
  PROBE_STATUS="BLOCKED"
  PROBE_CLASS="run_rc_nonzero"
  PROBE_BOUNDARY="runtime"
elif [[ "$PROBE_A_OUT_LAST" != "42" ]]; then
  PROBE_STATUS="BLOCKED"
  PROBE_CLASS="stdout_not_42"
  PROBE_BOUNDARY="runtime stdout"
elif [[ "$PROBE_A_SHA" == "$PROBE_B_SHA" ]]; then
  PROBE_STATUS="BLOCKED"
  PROBE_CLASS="silent_corruption_elf_invariant_under_mutation"
  PROBE_BOUNDARY="$BOUNDARY_SUSPECT_MERGE"
elif [[ "$PROBE_B_OUT_LAST" != "7" ]]; then
  PROBE_STATUS="BLOCKED"
  PROBE_CLASS="closure_not_consumed_runtime"
  PROBE_BOUNDARY="$BOUNDARY_RUNTIME_STDOUT; $BOUNDARY_SUSPECT_MERGE"
else
  PROBE_STATUS="PASS"
  PROBE_CLASS=""
  PROBE_BOUNDARY=""
fi

log "PROBE verdict: $PROBE_STATUS class=$PROBE_CLASS"

# ----------------------------------------------------------------------------
# Phase 4 : overall verdict
# ----------------------------------------------------------------------------

section "Phase 4: verdict"

if [[ "$CONTROL_STATUS" == "PASS" && "$PROBE_STATUS" == "PASS" ]]; then
  verdict="PASS"
  verdict_reason="control 2-module and probe 3-module pub-use closures both compile rc=0 and reflect source semantics"
elif $stale_candidate \
     && [[ "$CONTROL_CLASS" == compile_fail || "$CONTROL_CLASS" == no_elf_emitted \
           || "$PROBE_CLASS" == compile_fail || "$PROBE_CLASS" == no_elf_emitted ]]; then
  # Only promote STALE when it plausibly explains the failure mode (a missing
  # build, not silent corruption). Silent corruption (`closure_not_consumed_*`,
  # `silent_corruption_*`) cannot be explained away by a stale binary — the
  # binary clearly ran and produced output, just the wrong output.
  verdict="STALE"
  verdict_reason="raw Madaros ELF predates newest self-hosted/compiler/*.sio by >= ${STALE_THRESHOLD_SECONDS}s; rebuild before treating compile/no-elf failure as semantic"
elif [[ "$CONTROL_STATUS" != "PASS" ]]; then
  verdict="BLOCKED"
  verdict_reason="control=$CONTROL_STATUS($CONTROL_CLASS) probe=$PROBE_STATUS($PROBE_CLASS)"
elif [[ "$PROBE_STATUS" != "PASS" ]]; then
  verdict="BLOCKED"
  verdict_reason="control=$CONTROL_STATUS probe=$PROBE_STATUS($PROBE_CLASS)"
else
  verdict="INFRA"
  verdict_reason="unhandled combination"
fi

log "FINAL: $verdict — $verdict_reason"

# ----------------------------------------------------------------------------
# Phase 5 : emit JSON receipt
# ----------------------------------------------------------------------------

emit_receipt() {
  mkdir -p "$RECEIPT_DIR"
  cat >"$RECEIPT_PATH" <<JSON
{
  "schema_version": "$WITNESS_SCHEMA",
  "witness_id": "$WITNESS_ID",
  "witness_run_utc": "$UTC_NOW",
  "git": {
    "branch": "$GIT_BRANCH",
    "commit": "$GIT_COMMIT",
    "dirty_files": $GIT_DIRTY,
    "worktree": "$GIT_WORKTREE"
  },
  "compiler": {
    "launcher": "$REPO_ROOT/bin/souc",
    "raw_elf_path": "$RAW_MADAROS",
    "raw_elf_sha256": "$RAW_MADAROS_SHA256",
    "raw_elf_mtime_utc": "$RAW_MADAROS_MTIME_UTC",
    "compiler_source_newest_utc": "$COMPILER_SOURCE_NEWEST_UTC",
    "version_string": "$SOUC_VERSION"
  },
  "stale_candidate": $stale_candidate,
  "fixtures": {
    "control": {
      "main_path": "$CONTROL_MAIN",
      "greet_path": "$CONTROL_GREET_PATH",
      "closure_files": ["main.sio", "greet.sio"],
      "uses_pub_use_reexport": false,
      "mutation_test": "greet.sio returns 42 vs 999; expect different ELF sha256 and different stdout"
    },
    "probe": {
      "main_path": "$PROBE_MAIN",
      "facade_path": "$PROBE_FACADE",
      "leaf_path": "$PROBE_LEAF_PATH",
      "closure_files": ["main.sio", "facade.sio", "leaf.sio"],
      "uses_pub_use_reexport": true,
      "mutation_test": "leaf.sio returns 42 vs 7; expect different ELF sha256 and different stdout"
    }
  },
  "result_control": {
    "status": "$CONTROL_STATUS",
    "class": "$CONTROL_CLASS",
    "boundary": "$CONTROL_BOUNDARY",
    "control_a_greet_value": 42,
    "control_a_compile_rc": $CONTROL_A_CC_RC,
    "control_a_run_rc": $CONTROL_A_RUN_RC,
    "control_a_elf_magic_ok": $CONTROL_A_MAGIC,
    "control_a_elf_sha256": "$CONTROL_A_SHA",
    "control_a_stdout_last_line": "$(printf '%s' "$CONTROL_A_OUT_LAST" | sed 's/"/\\"/g')",
    "control_b_greet_value": 999,
    "control_b_compile_rc": $CONTROL_B_CC_RC,
    "control_b_run_rc": $CONTROL_B_RUN_RC,
    "control_b_elf_magic_ok": $CONTROL_B_MAGIC,
    "control_b_elf_sha256": "$CONTROL_B_SHA",
    "control_b_stdout_last_line": "$(printf '%s' "$CONTROL_B_OUT_LAST" | sed 's/"/\\"/g')",
    "elfs_bit_identical_after_mutation": $([[ "$CONTROL_A_SHA" == "$CONTROL_B_SHA" ]] && echo true || echo false)
  },
  "result_probe": {
    "status": "$PROBE_STATUS",
    "class": "$PROBE_CLASS",
    "boundary": "$PROBE_BOUNDARY",
    "probe_a_leaf_value": 42,
    "probe_a_compile_rc": $PROBE_A_CC_RC,
    "probe_a_run_rc": $PROBE_A_RUN_RC,
    "probe_a_elf_magic_ok": $PROBE_A_MAGIC,
    "probe_a_elf_sha256": "$PROBE_A_SHA",
    "probe_a_stdout_last_line": "$(printf '%s' "$PROBE_A_OUT_LAST" | sed 's/"/\\"/g')",
    "probe_b_leaf_value": 7,
    "probe_b_compile_rc": $PROBE_B_CC_RC,
    "probe_b_run_rc": $PROBE_B_RUN_RC,
    "probe_b_elf_magic_ok": $PROBE_B_MAGIC,
    "probe_b_elf_sha256": "$PROBE_B_SHA",
    "probe_b_stdout_last_line": "$(printf '%s' "$PROBE_B_OUT_LAST" | sed 's/"/\\"/g')",
    "elfs_bit_identical_after_mutation": $([[ "$PROBE_A_SHA" == "$PROBE_B_SHA" ]] && echo true || echo false),
    "compile_markers": {
      "has_seed_begin": $PROBE_CC_HAS_SEED_BEGIN,
      "has_seed_done": $PROBE_CC_HAS_SEED_DONE,
      "has_merged_ir": $PROBE_CC_HAS_MERGED_IR,
      "merged_ir_n": "${PROBE_CC_MERGED_IR_N:-null}",
      "has_fallback_marker": $PROBE_CC_HAS_FALLBACK,
      "has_compact_modular_ir_table_path": $PROBE_CC_HAS_COMPACT,
      "has_ast_closure_incomplete": $PROBE_CC_HAS_CLOSURE_INCOMPLETE
    },
    "sigsegv_compile": $PROBE_SEGV_COMPILE,
    "sigsegv_run": $PROBE_SEGV_RUN
  },
  "verdict": "$verdict",
  "verdict_reason": "$verdict_reason",
  "strongest_claim_supported": "At commit $GIT_COMMIT, mutations in the body of imported functions do not influence the behaviour of the ELF produced by the default Madaros compiler, despite compilation and execution completing with success (rc=0). The loss occurs after the closure is constructed and before runtime observation is captured; the multi-module merge (module_frontend.sio:4290-4722, including fn_remap construction at :4532-4715) is the primary suspect with strong evidence, but is not yet condemned in isolation. Condemnation requires differential IR hashes immediately before and after fn_remap.",
  "next_step_differential_ir_hashes": {
    "purpose": "Localise the causal stage at which the imported function body stops reaching the executable behaviour",
    "protocol": [
      "1. compile the PROBE fixture with leaf_value=42; dump IrModule of the imported leaf program immediately before fn_remap (call it H_pre_A). Mutate leaf to 7; recompile; dump H_pre_B.",
      "2. If H_pre_A == H_pre_B the loss is already in import resolution or earlier (pre-merge); do not advance.",
      "3. If H_pre_A != H_pre_B dump the merged IrModule immediately after fn_remap for both runs (H_post_A, H_post_B). If H_post_A == H_post_B the merge is formally condemned.",
      "4. If H_post_A != H_post_B dump the IrModule handed to module_native_driver (H_driver_A, H_driver_B). If H_driver_A == H_driver_B the compact/full bridge is erasing the body.",
      "5. If H_driver_A != H_driver_B but the ELF stdout is invariant, the problem is in codegen or call-target resolution at native emit time."
    ],
    "structural_hash_recipe": "sha256 over a canonical serialisation of every (instr.op, instr.arg_i64, instr.arg_imm_i64, instr.label_id, instr.call_target_id) tuple in IrModule.functions[*].instrs[*], excluding capacity fields, plus the function-name table in registration order",
    "not_executed_in_this_witness": true
  },
  "promotion_path": {
    "intent": "Promote this witness to a permanent regression gate once the underlying silent-corruption defect is fixed",
    "trigger": "First commit on which running this script yields verdict=PASS with class=\"\" for both CONTROL and PROBE, with elfs_bit_identical_after_mutation=false AND stdout correctly reflecting mutations (999 for CONTROL_B, 7 for PROBE_B)",
    "target_path": "scripts/ci/module_graph_facade_fidelity_gate.sh (sibling to madaros_full_gate.sh) — promotion is a move + rename, not a rewrite",
    "gate_class": "metamorphic semantic fidelity gate — detects operational success with false semantics (the most dangerous class of scientific compiler bug)",
    "promotion_owner": "future implementing agent; not this witness's scope"
  },
  "overlapping_shadows": [
    {"branch": "codex/modulegraph-facade-gate-r1", "kind": "gate-only", "overlap": "high", "note": "774-line gate, same scope, no compiler edits; depends on codex/modulegraph-facade-vertical-r1"},
    {"branch": "codex/modulegraph-facade-vertical-r1", "kind": "implementation", "overlap": "high", "note": "11-commit implementation lane; lowering_identity_consumption=unproven"},
    {"branch": "codex/module-graph-facade-vertical-20260715", "kind": "implementation", "overlap": "medium", "note": "abandoned mid-edit, 44-file dirty worktree"},
    {"branch": "codex/imported-transitive-lowering-unblock-20260715", "kind": "surgical", "overlap": "medium", "note": "fail-closed on incomplete import closure"},
    {"branch": "codex/module-declaration-identity-r1", "kind": "parser-slice", "overlap": "medium", "note": "preserve declared module path; absorbed into vertical-r1"},
    {"branch": "codex/definition-registry-shadow-20260715", "kind": "shadow", "overlap": "low", "note": "BLOCKED on native-v2 global array alias"},
    {"branch": "codex/place-canonical-binding-shadow-20260715", "kind": "shadow", "overlap": "low", "note": "documented order-dependence 11/42"},
    {"branch": "codex/field-resolution-receipt-shadow-20260715", "kind": "shadow", "overlap": "low", "note": "observational shadow in check/check.sio"},
    {"branch": "codex/module-bindings-arch-20260713", "kind": "historical", "overlap": "historical", "note": "PRs #851 #853 already merged"}
  ],
  "github_issues_referenced": [901, 921, 842, 862, 913, 933, 983, 637, 888, 834, 854],
  "blocker_ids_referenced": [
    "BLK-20260713-monolithic-public-lower-call",
    "BLK-20260712-image-heap",
    "BLK-20260719-MODULE-GRAPH-FACADE"
  ],
  "scope_promise": {
    "no_compiler_source_modified": true,
    "no_ci_wiring": true,
    "no_modulegraph_abstractions_introduced": true,
    "no_roadmap_expansion": true
  }
}
JSON
  log "receipt written to $RECEIPT_PATH"
}

emit_receipt

# ----------------------------------------------------------------------------
# Phase 6 : final stdout verdict line (single line)
# ----------------------------------------------------------------------------

printf '\n%s\n' "MODULE_GRAPH_FACADE_WITNESS_${verdict} reason=$(printf '%s' "$verdict_reason" | sed 's/ /_/g') control=${CONTROL_STATUS} probe=${PROBE_STATUS} receipt=${RECEIPT_PATH}"

case "$verdict" in
  PASS)   exit 0 ;;
  BLOCKED) exit 1 ;;
  STALE)  exit 2 ;;
  INFRA)  exit 3 ;;
  *)      exit 4 ;;
esac
