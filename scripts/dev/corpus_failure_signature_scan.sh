#!/usr/bin/env bash
# corpus_failure_signature_scan.sh — issue #2306
#
# Compile every tests/run-pass program with the COMMITTED
# bin/madaros-linux-x86_64 (the surface the corpus baseline is measured
# against) and keep, per program: name, rc, first diagnostic line. The
# regression gate discards error text; this scan exists to answer the issue's
# question 1: do the ~363 compile failures collapse to a few causes?
#
# Same program filter as madaros_corpus_regression_gate.sh (//@ ignore in the
# first 8 lines, no fn main, //@ requires: gpu|llvm, //@ known-failure) so the
# measured set is comparable to the gate's. rc classes are kept distinct:
# 137 = OOM-killed (environment, not a verdict), 139 = compiler SEGV.
#
# Usage:
#   bash scripts/dev/corpus_failure_signature_scan.sh            # scan (Slurm if possible)
#   bash scripts/dev/corpus_failure_signature_scan.sh --cluster  # cluster artifacts/audit/corpus_2306/runs.tsv
#
# Slurm path ships the committed ELF; no build. JOBS defaults to 6 — the OOM
# killer takes compilers at high parallelism (measured, see issue #2306).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${CORPUS_2306_OUT:-$ROOT/artifacts/audit/corpus_2306}"
BASELINE="tests/madaros_corpus_baseline.txt"
JOBS="${SOUNIO_TEST_JOBS:-6}"
mkdir -p "$OUT_DIR"

# ---------------- cluster mode (offline, no compiler) ----------------
if [ "${1:-}" = "--cluster" ]; then
  RUNS="$OUT_DIR/runs.tsv"
  [ -s "$RUNS" ] || { echo "no runs.tsv at $RUNS" >&2; exit 2; }
  python3 - "$RUNS" "$BASELINE" "$OUT_DIR" <<'PY'
import re, sys, collections
runs_path, baseline_path, out_dir = sys.argv[1:4]

baseline = set()
for line in open(baseline_path):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    baseline.add(line)  # '<name>.sio <mode>'

def norm(sig):
    sig = re.sub(r"0x[0-9a-fA-F]+", "0xN", sig)
    sig = re.sub(r"\b\d+\b", "N", sig)
    sig = re.sub(r"\s+", " ", sig)
    return sig.strip()[:120]

rows = []
for line in open(runs_path):
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 3:
        rows.append((parts[0], parts[1], parts[2]))

fails_compile = [(n, rc, d) for n, rc, d in rows if rc != "0"]
new_failures = [(n, rc, d) for n, rc, d in fails_compile if f"{n} compile" not in baseline]

clusters = collections.defaultdict(list)
for n, rc, d in new_failures:
    key = ("rc=137 OOM" if rc == "137" else "rc=139 SEGV" if rc == "139" else norm(d) or "no-diagnostic")
    clusters[key].append(n)

with open(f"{out_dir}/clusters.md", "w") as fh:
    fh.write("# Corpus failure signature clusters — issue #2306\n\n")
    fh.write(f"scanned={len(rows)} failing_now={len(fails_compile)} "
             f"new_vs_baseline={len(new_failures)} baseline_size={len(baseline)}\n\n")
    fh.write("| count | signature | examples |\n|---|---:|---|\n")
    for sig, names in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        ex = ", ".join(names[:4])
        fh.write(f"| {len(names)} | `{sig}` | {ex} |\n")

print(f"scanned={len(rows)} failing_now={len(fails_compile)} new_vs_baseline={len(new_failures)}")
print(f"clusters={len(clusters)} -> {out_dir}/clusters.md")
for sig, names in sorted(clusters.items(), key=lambda kv: -len(kv[1]))[:12]:
    print(f"{len(names):5d}  {sig}")
PY
  exit $?
fi

# ---------------- scan mode ----------------
run_slurm() {
  tar -czf - -C "$ROOT" \
      bin/souc bin/madaros bin/madaros-linux-x86_64 stdlib \
      tests/run-pass "$BASELINE" \
    | srun -p "${CORPUS_2306_PARTITION:-bench}" -N1 -n1 -c"${SLURM_CPUS:-8}" \
        --mem=24G --time=00:45:00 --chdir=/tmp \
        --job-name=corpus2306-scan \
        bash -c '
          set -uo pipefail
          W=$(mktemp -d /tmp/corpus2306.XXXXXX)
          cd "$W" || exit 1
          tar xzf - || { echo "REMOTE: untar failed" >&2; exit 1; }
          echo "REMOTE: host=$(hostname) nproc=$(nproc)" >&2
          export SOUNIO_STDLIB_PATH="$W/stdlib"
          ulimit -s 524288 2>/dev/null || true
          ulimit -v unlimited 2>/dev/null || true
          MAD="$W/bin/madaros-linux-x86_64"
          chmod +x "$MAD" 2>/dev/null || true
          "$MAD" --version 2>&1 | head -1 | sed "s/^/REMOTE: /" >&2

          # Same program filter as madaros_corpus_regression_gate.sh
          mkdir -p "$W/out/logs" "$W/work"
          : > "$W/programs.txt"
          for f in tests/run-pass/*.sio; do
            head -n 8 "$f" | grep -qE "^//@([[:space:]]*)ignore\b" && continue
            grep -qE "^[[:space:]]*(pub[[:space:]]+)?fn[[:space:]]+main[[:space:]]*\(" "$f" || continue
            grep -qE "^//@ requires: (gpu|llvm)" "$f" && continue
            grep -qE "^//@ known-failure" "$f" && continue
            echo "$f" >> "$W/programs.txt"
          done
          echo "REMOTE: programs=$(wc -l < "$W/programs.txt")" >&2

          cat > "$W/one.sh" <<"ONE"
#!/usr/bin/env bash
f="$1"; W="$2"; MAD="$3"
name="$(basename "$f")"
elf="$W/work/${name%.sio}.elf"
timeout 180 "$MAD" compile "$f" -o "$elf" > "$W/out/logs/$name.log" 2>&1
rc=$?
diag=$(grep -m1 -iE "error|panic|refus|invalid|mismatch|lowering|segv|assert" "$W/out/logs/$name.log" | head -c 300)
[ -n "$diag" ] || diag=$(head -1 "$W/out/logs/$name.log" | head -c 300)
printf "%s\t%s\t%s\n" "$name" "$rc" "$diag" >> "$W/out/runs.tsv"
ONE
          chmod +x "$W/one.sh"
          : > "$W/out/runs.tsv"
          cat "$W/programs.txt" | xargs -P '"$JOBS"' -I{} "$W/one.sh" {} "$W" "$MAD"
          echo "REMOTE: done rows=$(wc -l < "$W/out/runs.tsv")" >&2
          tar -C "$W/out" -czf - runs.tsv
          rm -rf "$W" "$W/out"
        ' > "$OUT_DIR/scan_bundle.tar.gz"
  tar -C "$OUT_DIR" -xzf "$OUT_DIR/scan_bundle.tar.gz"
}

run_local() {
  local MAD="$ROOT/bin/madaros-linux-x86_64"
  [ -x "$MAD" ] || { echo "no committed Madaros at $MAD" >&2; exit 2; }
  ulimit -s 524288 2>/dev/null || true
  export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
  : > "$OUT_DIR/runs.tsv"
  for f in tests/run-pass/*.sio; do
    head -n 8 "$f" | grep -qE '^//@([[:space:]]*)ignore\b' && continue
    grep -qE '^[[:space:]]*(pub[[:space:]]+)?fn[[:space:]]+main[[:space:]]*\(' "$f" || continue
    grep -qE '^//@ requires: (gpu|llvm)' "$f" && continue
    grep -qE '^//@ known-failure' "$f" && continue
    echo "$f"
  done | xargs -P "$JOBS" -I{} bash -c '
    f="{}"; name="$(basename "$f")"
    timeout 180 "'"$ROOT"'/bin/madaros-linux-x86_64" compile "$f" -o /tmp/corpus2306-scan-$$.elf > /tmp/corpus2306-scan-$$.log 2>&1
    rc=$?
    diag=$(grep -m1 -iE "error|panic|refus|invalid|mismatch|lowering|segv|assert" /tmp/corpus2306-scan-$$.log | head -c 300)
    [ -n "$diag" ] || diag=$(head -1 /tmp/corpus2306-scan-$$.log | head -c 300)
    printf "%s\t%s\t%s\n" "$name" "$rc" "$diag" >> "'"$OUT_DIR"'/runs.tsv"
    rm -f /tmp/corpus2306-scan-$$.elf /tmp/corpus2306-scan-$$.log
  '
}

if command -v srun >/dev/null 2>&1 && [[ "${CORPUS_2306_SLURM:-1}" != "0" ]]; then
  run_slurm
else
  run_local
fi

rows=$(wc -l < "$OUT_DIR/runs.tsv" 2>/dev/null || echo 0)
echo "CORPUS2306_SCAN rows=$rows out=$OUT_DIR/runs.tsv"
[ "$rows" -gt 0 ] || exit 2
bash "$0" --cluster
