#!/usr/bin/env bash
# seed_receipt_provenance_gate.sh
#
# Closes the hole left by canonical_compiler_gate.sh alone:
#
#   Self-repro (canonical) proves:
#       md5(committed ELF) == md5(ELF compiling current lean_single.sio)
#   That is STABILITY. It does not prove the ELF was *derived from* that
#   source via a recorded chain — a foreign fixed-point ELF that happens to
#   self-reproduce the source would also pass.
#
#   SeedReceipt provenance (this gate) proves, when a receipt is present:
#       receipt.source.sha256     == sha256(committed lean_single.sio)
#       receipt.output_seed.hash  == hash(committed seed ELF)
#       fixed_point.gk_md5        == fixed_point.gk_plus1_md5   (side by side)
#       generations[] records the chain (g0…gN) including the settle pair
#       fixed_point.gk_*          == committed seed hashes
#
# Policy (main must stay green without a receipt on day one):
#
#   1. Receipt PRESENT  → always hard-check against this tree (provenance).
#   2. Receipt ABSENT   → require only when the change set touches the seed
#      surface (lean_single.sio and/or the committed lean_single ELF /
#      receipt path). Other PRs and plain main/push runs PASS without a
#      receipt. Mutant control still runs every time.
#   3. SOUNIO_SEED_RECEIPT_REQUIRED=1 → missing receipt always FAIL
#      (optional later flip once a receipt is permanently on main).
#
# Why not "warn forever" or "bootstrap fake receipt":
#   Warn forever re-accumulates silent ELF swaps on non-touching paths less
#   than the seed surface, but still leaves seed-touching PRs unchecked if
#   people ignore warnings. A fake bootstrap receipt is a lying instrument.
#   Path-scoped require hits exactly the case the recipe exists for (#1750
#   class) and never paints main red for absence alone.
#
# Positive control (every invocation): mutant receipt with wrong
# source.sha256 MUST fail the checker.
#
# Committed receipt path (tracked; not under gitignored artifacts/):
#   bin/souc-lean-single-x86_64.SeedReceipt.json
#
# Override:
#   SOUNIO_SEED_RECEIPT_PATH
#   SOUNIO_SEED_RECEIPT_REQUIRED=0|1
#   SOUNIO_CANONICAL_SOUC / SOUNIO_SEED_ELF
#   CI_EVENT_NAME / CI_BASE_SHA / CI_HEAD_SHA  (PR path detection)
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo x)" != "Linux" ]]; then
  echo "[seed-provenance] SKIP: Linux-only (matches canonical gate surface)" >&2
  exit 0
fi

SRC="${SOUNIO_SEED_SOURCE:-self-hosted/compiler/lean_single.sio}"
SEED_DEFAULT="$ROOT_DIR/bin/souc-lean-single-x86_64"
[[ -x "$SEED_DEFAULT" ]] || SEED_DEFAULT="$ROOT_DIR/bin/souc-linux-x86_64"
SEED="${SOUNIO_SEED_ELF:-${SOUNIO_CANONICAL_SOUC:-$SEED_DEFAULT}}"
RECEIPT_DEFAULT="$ROOT_DIR/bin/souc-lean-single-x86_64.SeedReceipt.json"
RECEIPT="${SOUNIO_SEED_RECEIPT_PATH:-$RECEIPT_DEFAULT}"
REQUIRED="${SOUNIO_SEED_RECEIPT_REQUIRED:-0}"
CHECKER="$ROOT_DIR/scripts/dev/write_seed_receipt.py"
EVENT_NAME="${CI_EVENT_NAME:-${GITHUB_EVENT_NAME:-}}"
BASE_SHA="${CI_BASE_SHA:-}"
HEAD_SHA="${CI_HEAD_SHA:-HEAD}"

die() { echo "[seed-provenance] FAIL: $*" >&2; exit 1; }
note() { echo "[seed-provenance] $*"; }

[[ -f "$CHECKER" ]] || die "missing $CHECKER"
[[ -f "$SRC" ]] || die "missing source $SRC"
[[ -e "$SEED" ]] || die "missing seed ELF $SEED"

# ── Positive control: mutant with wrong source SHA must FAIL ───────────────
# If this control ever goes green, the provenance checker is a no-op.
run_mutant_control() {
  local work mutant
  work="$(mktemp -d /tmp/sounio-seed-prov-mutant.XXXXXX)"
  # Structurally valid receipt that LIES about source.sha256.
  mutant="$(python3 - "$work" "$SRC" "$SEED" <<'PY'
import hashlib, json, sys
from pathlib import Path
work, src_p, seed_p = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

def digest(p):
    data = p.read_bytes()
    return hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest(), len(data)

s_md5, s_sha, s_b = digest(src_p)
k_md5, k_sha, k_b = digest(seed_p)
lie = "a" * 64
assert lie != s_sha
receipt = {
    "schema": "sounio.SeedReceipt",
    "schema_version": 1,
    "receipt_utc": "1970-01-01T00:00:00Z",
    "source": {"path": str(src_p), "sha256": lie, "md5": s_md5, "bytes": s_b},
    "input_seed": {"path": str(seed_p), "sha256": k_sha, "md5": k_md5, "bytes": k_b, "role": "g0"},
    "generations": [
        {"gen": "g0", "gen_index": 0, "md5": k_md5, "sha256": k_sha, "path": "g0"},
        {"gen": "g1", "gen_index": 1, "md5": k_md5, "sha256": k_sha, "path": "g1"},
        {"gen": "g2", "gen_index": 2, "md5": k_md5, "sha256": k_sha, "path": "g2"},
    ],
    "fixed_point": {
        "criterion": "md5(g_k)==md5(g_{k+1})",
        "k": 1, "k_plus_1": 2,
        "gk_label": "g1", "gk_plus1_label": "g2",
        "gk_md5": k_md5, "gk_plus1_md5": k_md5,
        "gk_sha256": k_sha, "gk_plus1_sha256": k_sha,
        "md5_equal": True, "sha256_equal": True, "verified": True,
    },
    "output_seed": {"path": str(seed_p), "sha256": k_sha, "md5": k_md5, "bytes": k_b},
    "environment": {"placement": "unknown"},
    "checks": {},
    "limits": {"provenance_note": "mutant"},
}
path = work / "mutant_wrong_source.json"
path.write_text(json.dumps(receipt, indent=2) + "\n")
print(path)
PY
)"
  set +e
  python3 "$CHECKER" --check-against-tree "$mutant" --source "$SRC" --seed-elf "$SEED" \
    >"$work/mutant.out" 2>&1
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    cat "$work/mutant.out" >&2 || true
    rm -rf "$work"
    die "POSITIVE CONTROL BROKEN: mutant receipt with wrong source.sha256 was ACCEPTED.
  The provenance checker is a no-op. Do not trust this gate."
  fi
  if ! grep -Eqi 'PROVENANCE FAIL|source\.sha256|does not match' "$work/mutant.out"; then
    cat "$work/mutant.out" >&2 || true
    rm -rf "$work"
    die "POSITIVE CONTROL WEAK: mutant failed but without provenance/source message"
  fi
  note "POSITIVE CONTROL ok: mutant wrong-source receipt refused (rc=$rc)"
  rm -rf "$work"
}

run_mutant_control

# True when this change set touches the lean_single seed surface.
seed_surface_touched() {
  # Explicit override for local tests.
  if [[ "${SOUNIO_SEED_SURFACE_TOUCHED:-}" == "1" ]]; then
    return 0
  fi
  if [[ "${SOUNIO_SEED_SURFACE_TOUCHED:-}" == "0" ]]; then
    return 1
  fi
  if [[ "$EVENT_NAME" != "pull_request" ]]; then
    # push/schedule/local: do not require a receipt for absence alone
    return 1
  fi
  if [[ -z "$BASE_SHA" ]]; then
    note "pull_request without CI_BASE_SHA — cannot classify seed-surface touch; treating as not-touched (main-safe)"
    return 1
  fi
  local changed
  if ! changed="$(git -C "$ROOT_DIR" diff --name-only --diff-filter=ACMR "$BASE_SHA...$HEAD_SHA" 2>/dev/null)"; then
    note "git diff failed for seed-surface classify — treating as not-touched (main-safe)"
    return 1
  fi
  local p
  while IFS= read -r p; do
    [[ -z "$p" ]] && continue
    case "$p" in
      self-hosted/compiler/lean_single.sio|\
      bin/souc-lean-single-x86_64|\
      bin/souc-lean-single-x86_64.SeedReceipt.json)
        return 0
        ;;
    esac
  done <<<"$changed"
  return 1
}

# ── Missing receipt ────────────────────────────────────────────────────────
if [[ ! -f "$RECEIPT" ]]; then
  note "no committed SeedReceipt at $RECEIPT"
  note "self-repro (canonical_compiler_gate) still applies; provenance paper trail absent"
  if [[ "$REQUIRED" == "1" ]]; then
    die "SOUNIO_SEED_RECEIPT_REQUIRED=1 and receipt missing.
  Founder: produce one with
    SOUNIO_SEED_REFRESH_EXECUTE=1 bash scripts/dev/refresh_lean_seed.sh --execute --via-slurm
  then commit bin/souc-lean-single-x86_64.SeedReceipt.json next to the ELF
  (~5–15 min idle srun). docs/ops/LEAN_SINGLE_SEED_REFRESH.md"
  fi
  if seed_surface_touched; then
    die "this change set touches the lean_single seed surface but has no SeedReceipt.
  Touched surface: lean_single.sio and/or bin/souc-lean-single-x86_64(+.SeedReceipt.json).
  Self-repro alone does not prove the ELF came from this source.
  Produce and commit a receipt:
    SOUNIO_SEED_REFRESH_EXECUTE=1 \\
      bash scripts/dev/refresh_lean_seed.sh --execute --via-slurm \\
      --partition=cpu-ops --time=00:45:00
  Success: receipt fixed_point field shows
    gk_md5:       <H>
    gk_plus1_md5: <H>
  (identical by eye), then commit ELF + bin/souc-lean-single-x86_64.SeedReceipt.json
  (~5–15 min idle srun). docs/ops/LEAN_SINGLE_SEED_REFRESH.md"
  fi
  note "PASS (no receipt; change set does not touch seed surface; mutant control still ran)"
  exit 0
fi

# ── Receipt present: hard provenance check ─────────────────────────────────
note "checking $RECEIPT against source=$SRC seed=$SEED"
if ! python3 "$CHECKER" --check-against-tree "$RECEIPT" --source "$SRC" --seed-elf "$SEED"; then
  die "committed SeedReceipt does not match the tree (see checker output above).
  Either refresh the seed+receipt together, or remove a stale receipt.
  Recipe: docs/ops/LEAN_SINGLE_SEED_REFRESH.md"
fi

note "PASS: SeedReceipt provenance matches committed source + seed ELF"
exit 0
