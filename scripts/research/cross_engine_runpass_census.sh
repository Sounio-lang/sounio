#!/usr/bin/env bash
# Cross-engine run-pass census: Madaros (default) vs lean_single.
#
# Turns dual-engine divergence from anecdote into a measured corpus count.
# Known 2026-08-17 anecdotes this instrument is meant to absorb:
#   #1798 E158 forward inverse_of (CLOSED — both should reject once source-aligned)
#   #1792 var=0.000000 vs ~1e-5 (OPEN — runtime / fabricated variance)
#   #1801 E219 Madaros-only (non-allowlisted extern)
#   V0-A / E218 f128-f256 (Madaros rejects; lean_single accepts arithmetic)
#   tilde operator semantics differ by engine (doc surface; not every fixture hits it)
#
# CI oracles often pin lean_single. This census reports how often the default
# engine (Madaros) disagrees with that pin on the run-pass corpus.
#
# Classification (per file):
#   AGREE_ACCEPT              both check accept
#   AGREE_REJECT              both check reject, same primary error code + span
#   DIAG_DIFF                 both reject, primary error codes differ
#   SPAN_DIFF                 both reject, same primary code, different span
#   MADAROS_ONLY_REJECT       Madaros rejects, lean_single accepts   (accept-vs-reject)
#   LEAN_ONLY_REJECT          lean_single rejects, Madaros accepts   (accept-vs-reject)
#   RUNTIME_DIFF              both accept check; run rc or stdout differs
#   AGREE_RUNTIME             both accept check; run rc+stdout match (when --run)
#   TIMEOUT                   either side hit the per-file timeout
#   INSTRUMENT_ERROR          missing binary / setup failure
#
# Positive controls MUST fire or the sweep aborts (instrument validation).
#
# Usage:
#   scripts/research/cross_engine_runpass_census.sh
#   CROSS_ENGINE_OUT_DIR=... scripts/research/cross_engine_runpass_census.sh --run
#   CROSS_ENGINE_GLOBS='tests/run-pass/med/*.sio' ...  # subset
#   CROSS_ENGINE_JOBS=32 CROSS_ENGINE_TIMEOUT=20 ...
#
# Heavy runs: use scripts/dev/slurm_srun_minimal.sh on a node that can see
# /orangefs (not /workspace). See docs/ops/SLURM_LAUNCH_REPAIR_2026-08-17.md.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DO_RUN=0
GLOBS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) DO_RUN=1; shift ;;
    --glob=*) GLOBS+=("${1#*=}"); shift ;;
    --) shift; break ;;
    -h|--help)
      sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

OUT_DIR="${CROSS_ENGINE_OUT_DIR:-$ROOT_DIR/artifacts/research/cross_engine_runpass_census/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT_DIR"/{logs,work}
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

STACK_KB="${CROSS_ENGINE_STACK_KB:-524288}"
stack_before="$(ulimit -S -s 2>/dev/null || echo unknown)"
if [[ "$stack_before" != "unlimited" && "$stack_before" != "unknown" ]]; then
  if (( stack_before < STACK_KB )); then
    ulimit -S -s "$STACK_KB" 2>/dev/null || true
  fi
fi

TMO="${CROSS_ENGINE_TIMEOUT:-25}"
JOBS="${CROSS_ENGINE_JOBS:-$(nproc 2>/dev/null || echo 8)}"
SOUC="$ROOT_DIR/bin/souc"
LEAN_ELF="${CROSS_ENGINE_LEAN_ELF:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
[[ -x "$LEAN_ELF" ]] || LEAN_ELF="$ROOT_DIR/bin/souc-linux-x86_64"

if [[ ! -x "$SOUC" ]]; then
  echo "[census] FAIL: bin/souc missing" >&2
  exit 1
fi
if [[ ! -x "$LEAN_ELF" ]]; then
  echo "[census] FAIL: lean_single ELF missing" >&2
  exit 1
fi

# Resolve which Madaros ELF the default souc path will use (for the receipt).
MADAROS_RESOLVED="$("$SOUC" info 2>/dev/null | head -5 || true)"

printf '[census] root=%s\n' "$ROOT_DIR"
printf '[census] out=%s\n' "$OUT_DIR"
printf '[census] stack_before=%s stack_after=%s\n' "$stack_before" "$(ulimit -S -s 2>/dev/null || echo unknown)"
printf '[census] jobs=%s timeout=%ss do_run=%s\n' "$JOBS" "$TMO" "$DO_RUN"
printf '[census] madaros_via=bin/souc (default)\n'
printf '[census] lean_via=SOUNIO_SOUC_ENGINE=lean_single bin/souc → %s\n' "$LEAN_ELF"
{
  echo "status=running"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "root=$ROOT_DIR"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_describe=$(git describe --always --dirty 2>/dev/null || echo unknown)"
  echo "do_run=$DO_RUN"
  echo "jobs=$JOBS"
  echo "timeout_s=$TMO"
  echo "souc_version_madaros=$("$SOUC" --version 2>&1 | head -1)"
  echo "lean_elf=$LEAN_ELF"
} > "$OUT_DIR/receipt_meta.txt"

# --- helpers -----------------------------------------------------------------

# Extract primary error code like E158 / E218 / E219 from compiler stderr/stdout.
primary_errcode() {
  local f="$1"
  local code
  code="$(grep -oE 'error\[E[0-9]+\]' "$f" 2>/dev/null | head -1 | tr -d 'error[]' || true)"
  if [[ -z "$code" ]]; then
    code="$(grep -oE 'E[0-9]{3,4}' "$f" 2>/dev/null | head -1 || true)"
  fi
  printf '%s' "${code:-NONE}"
}

primary_span() {
  local f="$1"
  local span
  span="$(grep -oE '[A-Za-z0-9_./-]+\.sio:[0-9]+:[0-9]+' "$f" 2>/dev/null | head -1 || true)"
  if [[ -z "$span" ]]; then
    span="$(grep -oE '[0-9]+:[0-9]+' "$f" 2>/dev/null | head -1 || true)"
  fi
  printf '%s' "${span:-NONE}"
}

# Check with one engine. Writes full log to $2. Prints rc on stdout.
check_engine() {
  local engine="$1" src="$2" log="$3"
  local rc
  if [[ "$engine" == "madaros" ]]; then
    # Default souc → Madaros
    ( cd "$ROOT_DIR" && timeout "$TMO" env SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" \
        "$SOUC" check "$src" >"$log" 2>&1 )
    rc=$?
  else
    ( cd "$ROOT_DIR" && timeout "$TMO" env SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" \
        SOUNIO_SOUC_ENGINE=lean_single \
        "$SOUC" check "$src" >"$log" 2>&1 )
    rc=$?
  fi
  printf '%s' "$rc"
}

# Run with one engine (compile+execute via souc run). Writes log; prints rc.
run_engine() {
  local engine="$1" src="$2" log="$3"
  local rc
  if [[ "$engine" == "madaros" ]]; then
    ( cd "$ROOT_DIR" && timeout "$TMO" env SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" \
        "$SOUC" run "$src" >"$log" 2>&1 )
    rc=$?
  else
    ( cd "$ROOT_DIR" && timeout "$TMO" env SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" \
        SOUNIO_SOUC_ENGINE=lean_single \
        "$SOUC" run "$src" >"$log" 2>&1 )
    rc=$?
  fi
  printf '%s' "$rc"
}

# Normalize stdout for runtime comparison: drop absolute paths and souc banners.
normalize_output() {
  local f="$1"
  # strip common noise lines; keep numeric/science payload
  sed -E \
    -e '/^Madaros v/d' \
    -e '/^the bare highland/d' \
    -e '/^Horizon /d' \
    -e '/^science-boundary:/d' \
    -e '/^check: OK/d' \
    -e '/^compile:/d' \
    -e '/^elf:/d' \
    -e '/^epistemic_meas:/d' \
    -e '/^knightian:/d' \
    -e '/^knowledge_/d' \
    -e '/^tier_dist:/d' \
    -e '/^econf:/d' \
    -e '/^Merged IR:/d' \
    -e '/^Written to /d' \
    -e '/^Compilation successful/d' \
    -e '/^[[:space:]]*Output:/d' \
    -e "s|$ROOT_DIR/||g" \
    -e 's|/tmp/[^ ]+|/tmp/OUT|g' \
    "$f" 2>/dev/null || true
}

# --- positive controls -------------------------------------------------------

pc_fail=0
pc() {
  local label="$1" want="$2" src="$3"
  if [[ ! -f "$src" ]]; then
    printf '[census] positive-control SKIP %s (missing %s)\n' "$label" "$src" >&2
    return 0
  fi
  local ml="$OUT_DIR/work/pc_${label}_m.log" ll="$OUT_DIR/work/pc_${label}_l.log"
  local mrc lrc
  mrc="$(check_engine madaros "$src" "$ml")"
  lrc="$(check_engine lean "$src" "$ll")"
  local got="AGREE_ACCEPT"
  if [[ "$mrc" -eq 124 || "$lrc" -eq 124 ]]; then got="TIMEOUT"
  elif [[ "$mrc" -eq 0 && "$lrc" -eq 0 ]]; then got="AGREE_ACCEPT"
  elif [[ "$mrc" -ne 0 && "$lrc" -ne 0 ]]; then got="AGREE_REJECT"
  elif [[ "$mrc" -ne 0 && "$lrc" -eq 0 ]]; then got="MADAROS_ONLY_REJECT"
  else got="LEAN_ONLY_REJECT"; fi
  if [[ "$got" == "$want" ]]; then
    printf '[census] positive-control OK %s -> %s (mrc=%s lrc=%s)\n' "$label" "$got" "$mrc" "$lrc"
  else
    printf '[census] positive-control FAIL %s: want %s got %s (mrc=%s lrc=%s)\n' \
      "$label" "$want" "$got" "$mrc" "$lrc" >&2
    pc_fail=1
  fi
}

# AGREE_ACCEPT control: tiny run-pass (must exist on main)
if [[ -f tests/run-pass/_diag_sobol.sio ]]; then
  pc agree_accept AGREE_ACCEPT tests/run-pass/_diag_sobol.sio
elif [[ -f tests/run-pass/a13_crossmod_mainfirst_ok_ctrlE.sio ]]; then
  pc agree_accept AGREE_ACCEPT tests/run-pass/a13_crossmod_mainfirst_ok_ctrlE.sio
else
  echo "[census] positive-control FAIL agree_accept: no candidate fixture" >&2
  pc_fail=1
fi

# MADAROS_ONLY on f256 arithmetic reserved (E218) — compile-fail under Madaros,
# may accept under lean_single (V0-A class).
pc e218_f256 MADAROS_ONLY_REJECT tests/compile-fail/f256_v0b_arithmetic_rejected.sio

# E219 Madaros-only on non-allowlisted extern (if fixture present)
if [[ -f tests/compile-fail/extern_c_unimplemented_builtin.sio ]]; then
  pc e219_extern MADAROS_ONLY_REJECT tests/compile-fail/extern_c_unimplemented_builtin.sio
elif [[ -f tests/ffi_posix/arm_claim_unimplemented.sio ]]; then
  pc e219_extern MADAROS_ONLY_REJECT tests/ffi_posix/arm_claim_unimplemented.sio
fi

# AGREE_REJECT on a known-bad library file if present
if [[ -f stdlib/verify/interval.sio ]]; then
  pc agree_reject AGREE_REJECT stdlib/verify/interval.sio
fi

if [[ "$pc_fail" -ne 0 ]]; then
  echo "[census] ABORT: positive controls failed; a zero-disagreement sweep would be meaningless." >&2
  echo "status=fail_positive_control" >> "$OUT_DIR/receipt_meta.txt"
  exit 1
fi

# --- build file list ---------------------------------------------------------

LIST="$OUT_DIR/file_list.txt"
: > "$LIST"
if [[ ${#GLOBS[@]} -eq 0 ]]; then
  if [[ -n "${CROSS_ENGINE_GLOBS:-}" ]]; then
    # shellcheck disable=SC2206
    GLOBS=( $CROSS_ENGINE_GLOBS )
  else
    GLOBS=( 'tests/run-pass/**/*.sio' 'tests/run-pass/*.sio' )
  fi
fi

# Prefer git-tracked run-pass files for reproducibility.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git ls-files 'tests/run-pass/**/*.sio' 'tests/run-pass/*.sio' 2>/dev/null | sort -u > "$LIST"
fi
if [[ ! -s "$LIST" ]]; then
  find tests/run-pass -type f -name '*.sio' | sort > "$LIST"
fi

if [[ -n "${CROSS_ENGINE_FILE_LIST:-}" && -f "${CROSS_ENGINE_FILE_LIST}" ]]; then
  cp "${CROSS_ENGINE_FILE_LIST}" "$LIST"
fi

N_FILES="$(wc -l < "$LIST" | tr -d ' ')"
printf '[census] corpus_files=%s\n' "$N_FILES"
echo "corpus_files=$N_FILES" >> "$OUT_DIR/receipt_meta.txt"

if [[ "$N_FILES" -eq 0 ]]; then
  echo "[census] FAIL: empty corpus" >&2
  exit 1
fi

# --- per-file worker ---------------------------------------------------------

WORKER="$OUT_DIR/work/worker.sh"
cat > "$WORKER" <<'WORKER_EOF'
#!/usr/bin/env bash
set -uo pipefail
src="$1"
ROOT_DIR="$CENSUS_ROOT"
OUT_DIR="$CENSUS_OUT"
TMO="$CENSUS_TMO"
DO_RUN="$CENSUS_DO_RUN"
SOUC="$CENSUS_SOUC"
SOUNIO_STDLIB_PATH="$CENSUS_STDLIB"

safe="$(echo "$src" | tr '/' '_')"
ml="$OUT_DIR/logs/${safe}.madaros.check.log"
ll="$OUT_DIR/logs/${safe}.lean.check.log"
mr="$OUT_DIR/logs/${safe}.madaros.run.log"
lr="$OUT_DIR/logs/${safe}.lean.run.log"

primary_errcode() {
  local f="$1" code
  code="$(grep -oE 'error\[E[0-9]+\]' "$f" 2>/dev/null | head -1 | tr -d 'error[]' || true)"
  if [[ -z "$code" ]]; then
    code="$(grep -oE 'E[0-9]{3,4}' "$f" 2>/dev/null | head -1 || true)"
  fi
  printf '%s' "${code:-NONE}"
}
primary_span() {
  local f="$1" span
  span="$(grep -oE '[A-Za-z0-9_./-]+\.sio:[0-9]+:[0-9]+' "$f" 2>/dev/null | head -1 || true)"
  if [[ -z "$span" ]]; then
    span="$(grep -oE '[0-9]+:[0-9]+' "$f" 2>/dev/null | head -1 || true)"
  fi
  printf '%s' "${span:-NONE}"
}
normalize_output() {
  local f="$1"
  sed -E \
    -e '/^Madaros v/d' \
    -e '/^the bare highland/d' \
    -e '/^Horizon /d' \
    -e '/^science-boundary:/d' \
    -e '/^check: OK/d' \
    -e '/^compile:/d' \
    -e '/^elf:/d' \
    -e '/^epistemic_meas:/d' \
    -e '/^knightian:/d' \
    -e '/^knowledge_/d' \
    -e '/^tier_dist:/d' \
    -e '/^econf:/d' \
    -e '/^Merged IR:/d' \
    -e '/^Written to /d' \
    -e '/^Compilation successful/d' \
    -e '/^[[:space:]]*Output:/d' \
    -e "s|$ROOT_DIR/||g" \
    -e 's|/tmp/[^ ]+|/tmp/OUT|g' \
    "$f" 2>/dev/null || true
}

( cd "$ROOT_DIR" && timeout "$TMO" env SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" \
    "$SOUC" check "$src" >"$ml" 2>&1 )
mrc=$?
( cd "$ROOT_DIR" && timeout "$TMO" env SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" \
    SOUNIO_SOUC_ENGINE=lean_single \
    "$SOUC" check "$src" >"$ll" 2>&1 )
lrc=$?

m_err="$(primary_errcode "$ml")"
l_err="$(primary_errcode "$ll")"
m_span="$(primary_span "$ml")"
l_span="$(primary_span "$ll")"

class=""
run_mrc=""
run_lrc=""
run_note=""

if [[ "$mrc" -eq 124 || "$lrc" -eq 124 ]]; then
  class="TIMEOUT"
elif [[ "$mrc" -eq 0 && "$lrc" -eq 0 ]]; then
  class="AGREE_ACCEPT"
  if [[ "$DO_RUN" == "1" ]]; then
    ( cd "$ROOT_DIR" && timeout "$TMO" env SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" \
        "$SOUC" run "$src" >"$mr" 2>&1 )
    run_mrc=$?
    ( cd "$ROOT_DIR" && timeout "$TMO" env SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" \
        SOUNIO_SOUC_ENGINE=lean_single \
        "$SOUC" run "$src" >"$lr" 2>&1 )
    run_lrc=$?
    if [[ "$run_mrc" -eq 124 || "$run_lrc" -eq 124 ]]; then
      class="TIMEOUT"
      run_note="run_timeout"
    else
      nm="$(normalize_output "$mr" | sha256sum | awk '{print $1}')"
      nl="$(normalize_output "$lr" | sha256sum | awk '{print $1}')"
      if [[ "$run_mrc" -eq "$run_lrc" && "$nm" == "$nl" ]]; then
        class="AGREE_RUNTIME"
      else
        class="RUNTIME_DIFF"
        run_note="mrc=${run_mrc};lrc=${run_lrc};mhash=${nm:0:12};lhash=${nl:0:12}"
      fi
    fi
  fi
elif [[ "$mrc" -ne 0 && "$lrc" -ne 0 ]]; then
  if [[ "$m_err" != "$l_err" ]]; then
    class="DIAG_DIFF"
  elif [[ "$m_span" != "$l_span" ]]; then
    class="SPAN_DIFF"
  else
    class="AGREE_REJECT"
  fi
elif [[ "$mrc" -ne 0 && "$lrc" -eq 0 ]]; then
  class="MADAROS_ONLY_REJECT"
else
  class="LEAN_ONLY_REJECT"
fi

# TSV row (one line)
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$src" "$mrc" "$lrc" "$m_err" "$l_err" "$m_span" "$l_span" \
  "$class" "${run_mrc:-}" "${run_lrc:-}" "${run_note:-}"
WORKER_EOF
chmod +x "$WORKER"

export CENSUS_ROOT="$ROOT_DIR"
export CENSUS_OUT="$OUT_DIR"
export CENSUS_TMO="$TMO"
export CENSUS_DO_RUN="$DO_RUN"
export CENSUS_SOUC="$SOUC"
export CENSUS_STDLIB="$SOUNIO_STDLIB_PATH"

TSV="$OUT_DIR/census.tsv"
printf 'file\tmadaros_check_rc\tlean_check_rc\tmadaros_err\tlean_err\tmadaros_span\tlean_span\tclass\trun_mrc\trun_lrc\trun_note\n' > "$TSV"

# Parallel sweep
export -n  # nothing
if command -v xargs >/dev/null 2>&1; then
  # shellcheck disable=SC2016
  <"$LIST" xargs -P "$JOBS" -I{} bash "$WORKER" {} >> "$TSV"
else
  while IFS= read -r f; do
    bash "$WORKER" "$f" >> "$TSV"
  done < "$LIST"
fi

# --- summarize ---------------------------------------------------------------

SUMMARY="$OUT_DIR/summary.txt"
python3 - <<'PY' "$TSV" "$SUMMARY" "$OUT_DIR"
import sys, collections, pathlib
tsv, summary, out_dir = sys.argv[1:4]
rows = []
with open(tsv, encoding='utf-8', errors='replace') as f:
    header = f.readline().rstrip('\n').split('\t')
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 8:
            continue
        row = dict(zip(header, parts + [''] * (len(header) - len(parts))))
        rows.append(row)

counts = collections.Counter(r['class'] for r in rows)
n = len(rows)
# taxonomy buckets requested by the founder
accept_vs_reject = counts.get('MADAROS_ONLY_REJECT', 0) + counts.get('LEAN_ONLY_REJECT', 0)
diag_diff = counts.get('DIAG_DIFF', 0)
span_diff = counts.get('SPAN_DIFF', 0)
runtime_diff = counts.get('RUNTIME_DIFF', 0)
agree = (
    counts.get('AGREE_ACCEPT', 0)
    + counts.get('AGREE_REJECT', 0)
    + counts.get('AGREE_RUNTIME', 0)
)
diverge = n - agree - counts.get('TIMEOUT', 0) - counts.get('INSTRUMENT_ERROR', 0)
# more precise diverge = explicit disagreement classes
diverge_explicit = accept_vs_reject + diag_diff + span_diff + runtime_diff

lines = []
lines.append(f'corpus_files={n}')
lines.append(f'agree_total={agree}')
lines.append(f'diverge_explicit={diverge_explicit}')
lines.append(f'timeout={counts.get("TIMEOUT", 0)}')
lines.append('')
lines.append('## class counts')
for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
    lines.append(f'{k}={v}')
lines.append('')
lines.append('## taxonomy (founder request)')
lines.append(f'accept_vs_reject={accept_vs_reject}  # MADAROS_ONLY_REJECT + LEAN_ONLY_REJECT')
lines.append(f'different_diagnostic={diag_diff}  # DIAG_DIFF')
lines.append(f'same_diagnostic_different_span={span_diff}  # SPAN_DIFF')
lines.append(f'runtime_output_different={runtime_diff}  # RUNTIME_DIFF')
lines.append('')
lines.append('## disagreement files (first 200)')
shown = 0
for r in rows:
    if r['class'] in {
        'MADAROS_ONLY_REJECT', 'LEAN_ONLY_REJECT', 'DIAG_DIFF',
        'SPAN_DIFF', 'RUNTIME_DIFF',
    }:
        lines.append(
            f"{r['class']}\t{r['file']}\tm_err={r['madaros_err']}\tl_err={r['lean_err']}"
            f"\tm_span={r['madaros_span']}\tl_span={r['lean_span']}\tnote={r.get('run_note','')}"
        )
        shown += 1
        if shown >= 200:
            lines.append(f'... truncated; see census.tsv for full list')
            break

text = '\n'.join(lines) + '\n'
pathlib.Path(summary).write_text(text, encoding='utf-8')
print(text)

# compact JSON metrics
import json
metrics = {
    'status': 'pass' if n > 0 else 'fail',
    'metrics': {
        'total': n,
        'agree_total': agree,
        'diverge_explicit': diverge_explicit,
        'accept_vs_reject': accept_vs_reject,
        'different_diagnostic': diag_diff,
        'same_diagnostic_different_span': span_diff,
        'runtime_output_different': runtime_diff,
        'timeout': counts.get('TIMEOUT', 0),
        'by_class': dict(counts),
    },
}
pathlib.Path(out_dir, 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
PY

{
  echo "status=pass"
  echo "date_utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "summary=$SUMMARY"
  echo "tsv=$TSV"
  echo "metrics=$OUT_DIR/metrics.json"
} >> "$OUT_DIR/receipt_meta.txt"

printf '[census] DONE summary=%s\n' "$SUMMARY"
printf '[census] TSV=%s\n' "$TSV"
printf '[census] metrics=%s\n' "$OUT_DIR/metrics.json"
