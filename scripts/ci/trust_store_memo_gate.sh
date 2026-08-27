#!/usr/bin/env bash
# Trust-store memo gate (D16) — runs the memo's identity witness under Madaros.
#
# WHY THIS GATE EXISTS AT ALL: the witness it runs is annotated
# `//@ requires: madaros`, and the suite SKIPS such tests unless
# SOUNIO_MADAROS_AVAILABLE is set. The x509 stack does not typecheck under
# lean_single (bigint_mul arity, sha384/sha512 IVs, sct.sio [struct;8] vs
# [struct;9]), so the suite's stage2 engine cannot run it either way. Without
# this gate the trust store's only security witness never executes anywhere —
# which is exactly the state the memo shipped in.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

SRC=tests/run-pass/trust_store_load_cached_identity.sio
OUT="$(mktemp /tmp/trust_store_memo_out.XXXXXX)"
ERR="$(mktemp /tmp/trust_store_memo_err.XXXXXX)"
trap 'rm -f "$OUT" "$ERR"' EXIT

echo "== trust-store memo identity witness (Madaros) =="
echo "   note: the witness compares WITNESS_ROOTS (8) of ~146 roots field-for-field."
echo "   A full-bundle comparison needs a second ~1.3 GB load the arena cannot"
echo "   afford. OK here means the memo is installed, stable and byte-identical"
echo "   on the sampled roots -- not that all 146 were compared."
set +e
SOUNIO_SOUC_ENGINE=madaros ./bin/souc run "$SRC" >"$OUT" 2>"$ERR"
rc=$?
set -e

# exit 181 is `madaros: arena full` — uncatchable, and the specific failure
# this whole change is about. Name it rather than reporting a bare non-zero.
if [[ $rc -eq 181 ]] || grep -q 'arena full' "$OUT" "$ERR"; then
  echo "FAIL: arena exhausted (exit $rc) — the memo is not holding, or a" >&2
  echo "      second full load crept back into the witness. Each load costs" >&2
  echo "      ~1.3 GB of never-reclaimed arena; the witness must do ONE." >&2
  exit 1
fi

if ! grep -q 'trust_store_load_cached_identity: OK' "$OUT"; then
  echo "FAIL: identity witness did not report OK (rc=$rc)" >&2
  tail -40 "$ERR" >&2
  tail -40 "$OUT" >&2
  exit 1
fi

echo "TRUST_STORE_MEMO_GATE_OK (sampled witness: 8 roots)"
