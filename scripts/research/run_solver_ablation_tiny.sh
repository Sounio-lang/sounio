#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-./bin/souc}"
SRC="benchmarks/solver/smt_ablation_tiny.sio"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${SOLVER_ABLATION_OUT_DIR:-/tmp/sounio-solver-ablation-tiny-$STAMP}"
mkdir -p "$OUT_DIR"

BIN="$OUT_DIR/smt_ablation_tiny"
CSV="$OUT_DIR/smt_ablation_tiny.csv"
SUMMARY="$OUT_DIR/summary.csv"
CONFIG_STATS="$OUT_DIR/config_stats.csv"
MANIFEST="$OUT_DIR/manifest.txt"

"$SOUC_BIN" compile "$SRC" -o "$BIN" > "$OUT_DIR/compile.log" 2>&1

start_ns="$(date +%s%N)"
"$BIN" > "$CSV"
run_status=$?
end_ns="$(date +%s%N)"
elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

{
  printf 'instance_id,score_mode,phase_mode,restart_budget,rows,sum_decisions,sum_conflicts,sum_restarts\n'
  awk -F, '
    /^[0-9]/ {
      k = $1 "," $4 "," $5 "," $6
      rows[k] += 1
      decisions[k] += $8
      conflicts[k] += $9
      restarts[k] += $10
    }
    END {
      for (k in rows) {
        print k "," rows[k] "," decisions[k] "," conflicts[k] "," restarts[k]
      }
    }
  ' "$CSV" | sort -t, -k1,1n -k2,2n -k3,3n -k4,4n
} > "$SUMMARY"

{
  printf 'instance_id,score_mode,phase_mode,restart_budget,rows,result_values,decisions_mean,decisions_min,decisions_max,conflicts_mean,conflicts_min,conflicts_max,restarts_mean,restarts_min,restarts_max\n'
  awk -F, '
    /^[0-9]/ {
      k = $1 "," $4 "," $5 "," $6
      if (rows[k] == 0) {
        decisions_min[k] = $8
        decisions_max[k] = $8
        conflicts_min[k] = $9
        conflicts_max[k] = $9
        restarts_min[k] = $10
        restarts_max[k] = $10
      }
      rows[k] += 1
      decisions_sum[k] += $8
      conflicts_sum[k] += $9
      restarts_sum[k] += $10
      if ($8 < decisions_min[k]) { decisions_min[k] = $8 }
      if ($8 > decisions_max[k]) { decisions_max[k] = $8 }
      if ($9 < conflicts_min[k]) { conflicts_min[k] = $9 }
      if ($9 > conflicts_max[k]) { conflicts_max[k] = $9 }
      if ($10 < restarts_min[k]) { restarts_min[k] = $10 }
      if ($10 > restarts_max[k]) { restarts_max[k] = $10 }
      rk = k SUBSEP $7
      if (!(rk in result_seen)) {
        result_seen[rk] = 1
        if (result_values[k] == "") {
          result_values[k] = $7
        } else {
          result_values[k] = result_values[k] "|" $7
        }
      }
    }
    END {
      for (k in rows) {
        printf "%s,%d,%s,%.6f,%d,%d,%.6f,%d,%d,%.6f,%d,%d\n",
          k,
          rows[k],
          result_values[k],
          decisions_sum[k] / rows[k],
          decisions_min[k],
          decisions_max[k],
          conflicts_sum[k] / rows[k],
          conflicts_min[k],
          conflicts_max[k],
          restarts_sum[k] / rows[k],
          restarts_min[k],
          restarts_max[k]
      }
    }
  ' "$CSV" | sort -t, -k1,1n -k2,2n -k3,3n -k4,4n
} > "$CONFIG_STATS"

madaros_hash="unavailable"
if [ -f artifacts/self-hosted/madaros ]; then
  madaros_hash="$(sha256sum artifacts/self-hosted/madaros | awk '{print $1}')"
fi

{
  printf 'schema=sounio.solver.smt_ablation_tiny.run.v1\n'
  printf 'timestamp_utc=%s\n' "$STAMP"
  printf 'source=%s\n' "$SRC"
  printf 'souc_bin=%s\n' "$SOUC_BIN"
  printf 'madaros_sha256=%s\n' "$madaros_hash"
  printf 'csv=%s\n' "$CSV"
  printf 'summary_csv=%s\n' "$SUMMARY"
  printf 'config_stats_csv=%s\n' "$CONFIG_STATS"
  printf 'compile_log=%s\n' "$OUT_DIR/compile.log"
  printf 'run_status=%s\n' "$run_status"
  printf 'elapsed_ms_total=%s\n' "$elapsed_ms"
  printf 'rows=%s\n' "$(awk '/^[0-9]/ { n += 1 } END { print n + 0 }' "$CSV")"
  printf 'note=%s\n' 'Tiny internal ablation smoke. Use for readiness plumbing only; not external benchmark or public novelty evidence.'
} > "$MANIFEST"

cat "$MANIFEST"
