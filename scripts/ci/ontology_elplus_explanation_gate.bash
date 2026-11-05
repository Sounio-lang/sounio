#!/usr/bin/env bash
# Proof-carrying EL+ closure: independently validate reconstructed proof DAGs
# on both engines, prove the engines really differ, compare their derivations
# exactly, and exercise the shared-premise DAG regression.
#
# Engine selection belongs to this gate alone: an inherited compiler override
# can otherwise collapse both legs onto one engine and make the parity claim
# vacuous. Selection still pins a profile, not bytes: `bin/souc` resolves a
# local Madaros artifact ahead of the committed prebuilt, so each leg is
# additionally bound to its resolved raw artifact's path and SHA-256 (the
# documented committed digests) and passes the llms.txt hello preflight
# before any fixture output is trusted. Workflow wiring is intentionally
# deferred; run locally or from a maintainer-wired CI job as
# `bash scripts/ci/ontology_elplus_explanation_gate.bash`.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/elplus-explain.XXXXXX")"
trap 'rm -rf "$OUT"' EXIT

fail() { echo "ELPLUS_EXPLAIN_GATE_FAIL: $1" >&2; exit 1; }

for var in SOUC SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE MADAROS_RAW_BIN SOUNIO_MADAROS_BIN SOUNIO_STDLIB_PATH; do
  [[ -z "${!var:-}" ]] || fail "inherited override $var is set; this gate pins its own toolchain and stdlib"
done
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
SOUC="$ROOT_DIR/bin/souc"
[[ -x "$SOUC" ]] || fail "compiler launcher not executable: $SOUC"

sha256_of() { # <path>
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# Bind a leg to the exact compiler bytes that will produce its results. The
# engine selector alone does not identify bytes: `bin/souc` prefers a local
# Madaros artifact over the committed prebuilt. Resolve the raw artifact
# through the launcher's own receipt (`info` reports raw_elf for Madaros) or
# the launcher's structural lean_single resolution, hash the artifact — never
# the wrapper — and require the documented committed digest, per the llms.txt
# artifact-binding preflight.
bind_leg() { # <engine-label> <engine-selector>
  local label="$1" engine="$2"
  local raw expected digest info
  if [[ "$engine" == "madaros" ]]; then
    info="$(env SOUNIO_SOUC_ENGINE=madaros "$SOUC" info)" \
      || fail "engine=$label: launcher info failed; cannot bind the raw artifact"
    raw="$(printf '%s\n' "$info" | sed -n 's/^raw_elf:[[:space:]]*//p')"
    [[ -n "$raw" ]] || fail "engine=$label: adapter did not report raw_elf"
    [[ -f "$raw" ]] || fail "engine=$label: resolved raw artifact missing: $raw"
    expected="$(awk 'NR == 1 { print $1 }' "$ROOT_DIR/bin/madaros-linux-x86_64.sha256")" \
      || fail "engine=$label: cannot read the committed Madaros digest"
  else
    raw="$ROOT_DIR/bin/souc-lean-single-x86_64"
    [[ -x "$raw" ]] || raw="$ROOT_DIR/bin/souc-linux-x86_64"
    [[ -x "$raw" ]] || fail "engine=$label: no lean_single ELF resolves in this checkout"
    expected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_seed"]["sha256"])' "$ROOT_DIR/bin/souc-lean-single-x86_64.SeedReceipt.json")" \
      || fail "engine=$label: cannot read the lean_single seed receipt digest"
  fi
  case "$(cd "$(dirname "$raw")" && pwd -P)" in
    "$ROOT_DIR"*) ;;
    *) fail "engine=$label: resolved raw artifact escapes this checkout: $raw" ;;
  esac
  digest="$(sha256_of "$raw")" || fail "engine=$label: cannot hash $raw"
  [[ "$digest" == "$expected" ]] \
    || fail "engine=$label: raw artifact $raw sha256=$digest does not match the committed profile digest $expected; refusing to validate fixtures with unidentified compiler bytes"
  echo "ELPLUS_EXPLAIN_COMPILER_RECEIPT engine=$label raw=$raw sha256=$digest"
  # llms.txt hello preflight on the pinned engine: check, then run and
  # require the documented greeting, before any fixture output is trusted.
  env SOUNIO_SOUC_ENGINE="$engine" "$SOUC" check examples/hello.sio >"$OUT/$label.hello.check.log" 2>&1 \
    || { cat "$OUT/$label.hello.check.log" >&2; fail "engine=$label: hello check failed"; }
  env SOUNIO_SOUC_ENGINE="$engine" "$SOUC" run examples/hello.sio >"$OUT/$label.hello.run.log" 2>&1 \
    || { cat "$OUT/$label.hello.run.log" >&2; fail "engine=$label: hello run failed"; }
  grep -Fq 'Hello, Sounio' "$OUT/$label.hello.run.log" \
    || { cat "$OUT/$label.hello.run.log" >&2; fail "engine=$label: hello greeting missing"; }
}

run_leg() { # <engine-label> <engine-selector>
  local label="$1" engine="$2"
  bind_leg "$label" "$engine"
  local check_log="$OUT/$label.check.log" run_log="$OUT/$label.run.log" shared_log="$OUT/$label.shared.log" cycle_log="$OUT/$label.cycle.log"
  env SOUNIO_SOUC_ENGINE="$engine" "$SOUC" check tests/run-pass/ontology_elplus_explanation.sio >"$check_log" 2>&1 \
    || { cat "$check_log" >&2; fail "engine=$label: check failed"; }
  env SOUNIO_SOUC_ENGINE="$engine" "$SOUC" run tests/run-pass/ontology_elplus_explanation.sio >"$run_log" 2>&1 \
    || { cat "$run_log" >&2; fail "engine=$label: fixture run failed"; }
  grep -Fxq 'ONTOLOGY_ELPLUS_EXPLANATION_OK' "$run_log" \
    || { cat "$run_log" >&2; fail "engine=$label: success marker missing"; }
  python3 scripts/ci/ontology_elplus_explanation_oracle.py <"$run_log" >"$OUT/$label.oracle.log" \
    || { cat "$OUT/$label.oracle.log" >&2; fail "engine=$label: independent proof validation failed"; }
  grep -Fxq 'ONTOLOGY_ELPLUS_PROOF_ORACLE_NEGATIVE_OK rtos_source=refused seed_witnesses=refused unknown_kind=refused malformed_step=refused' "$OUT/$label.oracle.log" \
    || fail "engine=$label: oracle negative controls missing"
  grep -Fq 'ONTOLOGY_ELPLUS_PROOF_ORACLE_OK traces=7 steps=31 rules=0..11' "$OUT/$label.oracle.log" \
    || { cat "$OUT/$label.oracle.log" >&2; fail "engine=$label: unexpected proof census"; }
  cat "$OUT/$label.oracle.log"

  env SOUNIO_SOUC_ENGINE="$engine" "$SOUC" run tests/run-pass/ontology_elplus_explanation_shared_premise.sio >"$shared_log" 2>&1 \
    || { cat "$shared_log" >&2; fail "engine=$label: shared-premise regression failed"; }
  grep -Fxq 'ONTOLOGY_ELPLUS_SHARED_PREMISE_OK facts=8' "$shared_log" \
    || { cat "$shared_log" >&2; fail "engine=$label: shared-premise DAG was not minimal"; }

  env SOUNIO_SOUC_ENGINE="$engine" "$SOUC" run tests/run-pass/ontology_elplus_explanation_cycle.sio >"$cycle_log" 2>&1 \
    || { cat "$cycle_log" >&2; fail "engine=$label: cyclic provenance was not rejected"; }
  grep -Fxq 'ONTOLOGY_ELPLUS_EXPLANATION_CYCLE_OK rc=-4' "$cycle_log" \
    || { cat "$cycle_log" >&2; fail "engine=$label: cyclic-trace marker missing"; }

  local malformed_log="$OUT/$label.malformed.log"
  env SOUNIO_SOUC_ENGINE="$engine" "$SOUC" run tests/run-pass/ontology_elplus_explanation_malformed.sio >"$malformed_log" 2>&1 \
    || { cat "$malformed_log" >&2; fail "engine=$label: malformed provenance was not rejected"; }
  grep -Fxq 'ONTOLOGY_ELPLUS_EXPLANATION_MALFORMED_OK refused=6' "$malformed_log" \
    || { cat "$malformed_log" >&2; fail "engine=$label: malformed-provenance marker missing"; }

  grep -E '^(TRACE_|STEP |ONTOLOGY_ELPLUS_EXPLANATION_OK)' "$run_log" >"$OUT/$label.normalized"
}

run_leg madaros madaros
grep -Fq 'run_check_mode: verdict=0' "$OUT/madaros.check.log" \
  || { cat "$OUT/madaros.check.log" >&2; fail "madaros leg: Madaros check marker missing"; }
grep -Fq 'Compilation successful!' "$OUT/madaros.run.log" \
  || { cat "$OUT/madaros.run.log" >&2; fail "madaros leg: Madaros run marker missing"; }
if grep -Eq '^compile: fns=' "$OUT/madaros.check.log" "$OUT/madaros.run.log" "$OUT/madaros.shared.log" "$OUT/madaros.cycle.log" "$OUT/madaros.malformed.log"; then
  fail "madaros leg: lean_single compile marker present"
fi

run_leg lean_single lean_single
grep -Eq '^compile: fns=' "$OUT/lean_single.check.log" \
  || { cat "$OUT/lean_single.check.log" >&2; fail "lean_single leg: lean compile marker missing"; }
if grep -Fq 'run_check_mode:' "$OUT/lean_single.check.log" \
  || grep -Fq 'Compilation successful!' "$OUT/lean_single.run.log" "$OUT/lean_single.shared.log" "$OUT/lean_single.cycle.log" "$OUT/lean_single.malformed.log"; then
  fail "lean_single leg: Madaros marker present"
fi

# Both engines must accept the same derivations, not merely print the same
# success marker. Compare every trace and every emitted proof step exactly.
if ! cmp -s "$OUT/madaros.normalized" "$OUT/lean_single.normalized"; then
  diff -u "$OUT/madaros.normalized" "$OUT/lean_single.normalized" >&2 || true
  fail "Madaros and lean_single derivations differ"
fi

# The legacy engine remains independent: its standard temporal consumer
# detects accidental changes to the sensitive imported-module code layout.
env SOUNIO_SOUC_ENGINE=madaros "$SOUC" run examples/clinical/pathway_temporal_demo.sio >"$OUT/temporal.log" 2>&1 \
  || { cat "$OUT/temporal.log" >&2; fail "legacy temporal consumer failed"; }
grep -Fxq 'ALL PASS' "$OUT/temporal.log" \
  || { cat "$OUT/temporal.log" >&2; fail "legacy temporal marker missing"; }

echo 'ONTOLOGY_ELPLUS_EXPLAIN_GATE_OK traces=7 steps=31 rules=0..11 engines=madaros+lean_single receipts=bound shared_premise=8 cycles=refused malformed=refused legacy=temporal'
