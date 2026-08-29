#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_LIVE_GATES=1
ALLOW_NON_MAIN=0
ARTIFACT_ROOT="${SOUNIO_RELEASE_PRODUCTION_READINESS_ARTIFACT_ROOT:-/tmp/sounio-release-production-readiness-$(date -u +%Y%m%dT%H%M%SZ)}"
CLAIM_REGISTRY="${SOUNIO_PUBLIC_CLAIM_REGISTRY:-$ROOT_DIR/docs/serious-language/public-claim-registry.v1.tsv}"
BLOCKER_TSV="$ARTIFACT_ROOT/release_blockers.tsv"
SUMMARY_JSON="$ARTIFACT_ROOT/summary.v1.json"
LIVE_TSV="$ARTIFACT_ROOT/live_gates.tsv"

usage() {
  cat <<'USAGE'
Usage: scripts/ci/sounio_release_production_readiness_gate.sh [--skip-live-gates] [--allow-non-main]

Evaluates the broad Sounio release-production-ready contract. This is stricter
than Madaros production readiness:
  - current checkout must be clean and at origin/main,
  - Madaros production readiness must pass,
  - public claim closure and docs governance gates must pass,
  - release-critical product surfaces must not remain prototype/downgraded.

The gate is expected to fail until every release-critical blocker is either
closed with evidence or explicitly removed from the release support contract.

Use --allow-non-main only to validate changes to this gate on a feature branch.
Do not use it for release sign-off.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-live-gates)
      RUN_LIVE_GATES=0
      shift
      ;;
    --allow-non-main)
      ALLOW_NON_MAIN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$ARTIFACT_ROOT"

run_step() {
  local label="$1"
  shift
  local log="$ARTIFACT_ROOT/$label.log"
  echo "[release-production] >>> $label"
  if "$@" >"$log" 2>&1; then
    echo "[release-production] <<< $label PASS log=$log"
  else
    local rc=$?
    echo "[release-production] !!! $label FAIL rc=$rc log=$log" >&2
    tail -n 80 "$log" >&2 || true
    return "$rc"
  fi
}

declare -a LIVE_LABELS=()
declare -a LIVE_STATUSES=()
declare -a LIVE_LOGS=()

record_live_step() {
  local label="$1"
  local status="$2"
  local log="$3"
  LIVE_LABELS+=("$label")
  LIVE_STATUSES+=("$status")
  LIVE_LOGS+=("$log")
}

run_live_step() {
  local label="$1"
  shift
  local log="$ARTIFACT_ROOT/$label.log"
  if run_step "$label" "$@"; then
    record_live_step "$label" "pass" "$log"
  else
    record_live_step "$label" "fail" "$log"
  fi
}

write_live_tsv() {
  {
    echo -e "step\tstatus\tlog"
    local i
    for ((i = 0; i < ${#LIVE_LABELS[@]}; i++)); do
      echo -e "${LIVE_LABELS[$i]}\t${LIVE_STATUSES[$i]}\t${LIVE_LOGS[$i]}"
    done
  } > "$LIVE_TSV"
}

echo "== Sounio Release Production Readiness =="
echo "repo=$ROOT_DIR"
echo "branch=$(git branch --show-current 2>/dev/null || true)"
echo "head=$(git rev-parse HEAD 2>/dev/null || true)"
echo "origin_main=$(git rev-parse origin/main 2>/dev/null || true)"
echo "artifacts=$ARTIFACT_ROOT"

dirty_entries="$(git status --porcelain | wc -l | tr -d ' ')"
if [[ "$dirty_entries" != "0" ]]; then
  echo "status[checkout]=fail"
  echo "reason=dirty_checkout entries=$dirty_entries"
  git status --short
  exit 1
fi
echo "status[checkout]=pass"

head_sha="$(git rev-parse HEAD)"
origin_main_sha="$(git rev-parse origin/main)"
if [[ "$head_sha" != "$origin_main_sha" ]]; then
  if [[ "$ALLOW_NON_MAIN" == "1" ]]; then
    echo "status[origin_main]=warn"
    echo "reason=head_not_origin_main_allowed_for_branch_validation head=$head_sha origin_main=$origin_main_sha"
  else
    echo "status[origin_main]=fail"
    echo "reason=head_not_origin_main head=$head_sha origin_main=$origin_main_sha"
    exit 1
  fi
else
  echo "status[origin_main]=pass"
fi

if [[ ! -f "$CLAIM_REGISTRY" ]]; then
  echo "status[claim_registry]=fail"
  echo "reason=missing_claim_registry path=$CLAIM_REGISTRY"
  exit 2
fi

if [[ "$RUN_LIVE_GATES" == "1" ]]; then
  run_live_step madaros-production-ready env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
    bash "$ROOT_DIR/scripts/dev/madaros_readiness_status.sh" --no-audit --production-ready
  run_live_step docs-registry bash "$ROOT_DIR/scripts/dev/check_docs_registry.sh"
  run_live_step docs-consistency bash "$ROOT_DIR/scripts/dev/check_docs_consistency.sh"
  run_live_step website-docs-support env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_STDLIB_PATH \
    -u MADAROS_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
    bash "$ROOT_DIR/scripts/ci/sounio_website_docs_support_gate.sh"
  run_live_step package-support env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_STDLIB_PATH \
    -u MADAROS_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
    bash "$ROOT_DIR/scripts/ci/sounio_package_support_gate.sh"
  run_live_step stdlib-surface-support env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_STDLIB_PATH \
    -u MADAROS_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
    bash "$ROOT_DIR/scripts/ci/sounio_stdlib_surface_support_gate.sh"
  run_live_step direct-driver-support env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_STDLIB_PATH \
    -u MADAROS_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
    bash "$ROOT_DIR/scripts/ci/sounio_direct_driver_support_gate.sh"
  run_live_step editor-tooling-support env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_STDLIB_PATH \
    -u MADAROS_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
    bash "$ROOT_DIR/scripts/ci/sounio_editor_tooling_support_gate.sh"
  run_live_step install-support env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_STDLIB_PATH \
    -u MADAROS_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
    bash "$ROOT_DIR/scripts/ci/sounio_install_support_gate.sh"
  run_live_step serious-language-claim-closure env -u SOUC_BIN -u SOUNIO_STDLIB_PATH \
    bash "$ROOT_DIR/scripts/ci/serious_language_claim_closure_gate.sh"
  write_live_tsv
fi

set +e
python3 - "$CLAIM_REGISTRY" "$BLOCKER_TSV" "$SUMMARY_JSON" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

registry = Path(sys.argv[1])
blocker_tsv = Path(sys.argv[2])
summary_json = Path(sys.argv[3])

# Release-critical means the surface is part of the checked release support
# contract. Some broader research-frontier claims deliberately remain prototype
# in the repo, but then the release contract must cite a narrower closed support
# claim and public docs must keep the broader frontier explicitly out of scope.
# `direct_driver` is one such broad frontier; `direct_driver.support` is the
# bounded release-support claim. Requiring this support gate does not imply
# general direct-driver readiness or stability. It only locks in the named
# 24-fixture research cohort while broad direct_driver remains explicitly
# prototype/downgraded and unsupported for release purposes.
release_critical = {
    "binary.source": "checked modular Madaros prebuilt source-build evidence is not closed",
    "direct_driver.support": "checked bounded direct-driver support contract is not closed",
    "stdlib.surface": "checked bounded stdlib support contract is not closed",
    "tooling.package": "local package support contract is not closed",
    "tooling.editor": "checked editor-tooling support contract is not closed",
    "install": "installation is repo-artifact based, not a broad supported distribution path",
    "website.docs": "checked website/docs support contract is not closed",
}

allowed_levels = {"stable", "validated_research"}
allowed_closure = {"closed"}
rows: dict[str, dict[str, str]] = {}

with registry.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    required = [
        "claim_id",
        "claim_level",
        "closure_status",
        "evidence_kind",
        "evidence_ref",
        "spec_refs",
        "public_wording",
    ]
    if reader.fieldnames != required:
        raise SystemExit(f"claim registry header mismatch: {reader.fieldnames!r}")
    for row in reader:
        rows[row["claim_id"]] = {key: (value or "").strip() for key, value in row.items()}

blockers: list[dict[str, str]] = []
for claim_id, reason in release_critical.items():
    row = rows.get(claim_id)
    if row is None:
        blockers.append({
            "claim_id": claim_id,
            "claim_level": "missing",
            "closure_status": "missing",
            "evidence_ref": "-",
            "reason": f"release-critical claim missing from registry: {reason}",
        })
        continue
    if row["claim_level"] not in allowed_levels or row["closure_status"] not in allowed_closure:
        blockers.append({
            "claim_id": claim_id,
            "claim_level": row["claim_level"],
            "closure_status": row["closure_status"],
            "evidence_ref": row["evidence_ref"],
            "reason": reason,
        })

blocker_tsv.parent.mkdir(parents=True, exist_ok=True)
with blocker_tsv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["claim_id", "claim_level", "closure_status", "evidence_ref", "reason"],
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(blockers)

summary = {
    "schema": "sounio.release_production_readiness.v1",
    "status": "pass" if not blockers else "fail",
    "release_critical_claims": sorted(release_critical),
    "blocker_count": len(blockers),
    "blocker_tsv": str(blocker_tsv),
    "blockers": blockers,
}
summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print(f"release_critical_claims={len(release_critical)}")
print(f"blockers={len(blockers)}")
for blocker in blockers:
    print(
        "blocker "
        f"claim_id={blocker['claim_id']} "
        f"level={blocker['claim_level']} "
        f"closure={blocker['closure_status']} "
        f"evidence={blocker['evidence_ref']} "
        f"reason={blocker['reason']}"
    )

if blockers:
    raise SystemExit(1)
PY
classification_rc=$?
set -e

live_failures=0
for status in "${LIVE_STATUSES[@]}"; do
  if [[ "$status" != "pass" ]]; then
    live_failures=$((live_failures + 1))
  fi
done

echo "live_gates=$LIVE_TSV"
echo "summary=$SUMMARY_JSON"
if [[ "$live_failures" -gt 0 || "$classification_rc" -ne 0 ]]; then
  echo "status[release_production_ready]=fail"
  echo "reason=live_failures_${live_failures}_classification_rc_${classification_rc}"
  exit 1
fi

echo "status[release_production_ready]=pass"
