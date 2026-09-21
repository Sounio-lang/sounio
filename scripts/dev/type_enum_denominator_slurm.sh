#!/usr/bin/env bash
# Type-enum denominator on Spurm. Login pod cannot see /orangefs or compute /tmp.
# Streams a minimal source tarball on srun stdin; pulls results back as base64.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_HOST="${1:-$ROOT/docs/audit/type_enum_denominator}"
mkdir -p "$OUT_HOST"
export SLURM_CONF="${SLURM_CONF:-/tmp/slurm-direct.conf}"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PAYLOAD="/tmp/type_enum_denom_payload_${STAMP}.tar.gz"
tar czf "$PAYLOAD" -C "$ROOT" \
  scripts/dev/type_enum_denominator_measure.py \
  self-hosted/parser/ast.sio \
  self-hosted/parser/parser.sio \
  self-hosted/check/types.sio \
  self-hosted/check/layout.sio \
  self-hosted/check/ownership.sio \
  self-hosted/check/check.sio \
  self-hosted/check/epistemic.sio \
  self-hosted/hlir/ir.sio \
  self-hosted/gpu/kernel_ir.sio \
  self-hosted/gpu/kernel_ir_wmma_lean.sio \
  self-hosted/compiler/parser.sio \
  self-hosted/compiler/parser_test.sio \
  self-hosted/compiler/lean_single.sio \
  self-hosted/lsp/hover.sio \
  self-hosted/bootstrap/bootstrap_v0.sio \
  self-hosted/ir/lower.sio \
  self-hosted/llvm/type_convert.sio \
  stdlib/compiler/types/type.sio \
  stdlib/compiler/transform/type_annotation.sio \
  tests/typekind

LOG="/tmp/type_enum_denom_${STAMP}.log"
cat "$PAYLOAD" | srun \
  --partition=cpu-ops --nodes=1 --ntasks=1 --time=00:15:00 --chdir=/tmp \
  --export=NONE,PATH=/usr/bin:/bin:/usr/local/bin,TMPDIR=/tmp,TMP=/tmp,TEMP=/tmp,HOME=/tmp \
  /bin/bash -lc "
set -euo pipefail
W=/tmp/denom_${STAMP}
mkdir -p \"\$W\" && cd \"\$W\"
cat > payload.tar.gz && tar xzf payload.tar.gz
export SOUNIO_ROOT=\"\$W\" DENOM_OUT=\"\$W/out\" DENOM_SHA='${SHA}'
mkdir -p \"\$DENOM_OUT\"
/usr/bin/python3 scripts/dev/type_enum_denominator_measure.py | tee \"\$DENOM_OUT/stdout.txt\"
mkdir -p /orangefs/training/sounio/type_enum_denominator/${STAMP} 2>/dev/null && \
  cp -a \"\$DENOM_OUT\"/. /orangefs/training/sounio/type_enum_denominator/${STAMP}/ || true
echo ___BEGIN_RESULTS_B64___
tar czf - -C \"\$DENOM_OUT\" . | base64 -w0
echo
echo ___END_RESULTS_B64___
hostname; date -u
" | tee "$LOG"

python3 - <<PY
from pathlib import Path
import base64, io, tarfile
text=Path("$LOG").read_text(errors="replace")
assert "___BEGIN_RESULTS_B64___" in text, "no results marker"
b64=text.split("___BEGIN_RESULTS_B64___",1)[1].split("___END_RESULTS_B64___",1)[0].strip()
out=Path("$OUT_HOST")
out.mkdir(parents=True, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(b64)), mode="r:gz").extractall(out)
print("EXTRACTED", out)
PY
echo "DONE stamp=$STAMP sha=$SHA out=$OUT_HOST log=$LOG"
