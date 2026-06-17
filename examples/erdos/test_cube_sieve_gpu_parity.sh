#!/usr/bin/env bash
# Gate for CPU-reference vs GPU-backend cube propagation parity.
#
# The default backend is the CPU producer so this gate is runnable without a GPU.
# A real RTX wrapper must implement the same CLI and will be checked by the same
# semantic comparator before any search ledger is trusted.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

PARITY="$ROOT/examples/erdos/cube_sieve_gpu_parity.py"
PRODUCER="$ROOT/examples/erdos/cube_sieve_propagation_manifest.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$PARITY" "$PRODUCER"
mkdir -p "$WORK"

cat > "$WORK/k6.edge" <<'EOF'
p edge 6 15
e 1 2
e 1 3
e 1 4
e 1 5
e 1 6
e 2 3
e 2 4
e 2 5
e 2 6
e 3 4
e 3 5
e 3 6
e 4 5
e 4 6
e 5 6
EOF

cat > "$WORK/k6.cube" <<'EOF'
0 0
1 1
2 2
3 3
4 4
EOF

echo "cube_sieve_gpu_parity_gate: workdir=$WORK"
python3 "$PARITY" "$WORK/k6.edge" 5 "$WORK/k6.cube" \
  --out-dir "$WORK/parity" > "$WORK/parity.out"
rg -q '^cube_sieve_gpu_parity v1$' "$WORK/parity.out"
rg -q '^trust_boundary=backend_untrusted__cpu_parity_only__drat_lrat_lean_verified_required$' \
  "$WORK/parity.out"
rg -q '^canonical_line_count=[1-9][0-9]*$' "$WORK/parity.out"
rg -q '^verified_claim=none$' "$WORK/parity.out"
rg -q '^geometry_claim=none$' "$WORK/parity.out"
rg -q '^promotion_gate=REJECT_NONE_PROOF_ARTIFACT$' "$WORK/parity.out"
rg -q '^promotable=0$' "$WORK/parity.out"
rg -q '^status=GPU_PARITY_PASS$' "$WORK/parity.out"
cmp "$WORK/parity/cpu.manifest" "$WORK/parity/backend.manifest"

cat > "$WORK/bad_backend.py" <<'PY'
#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

producer = Path(sys.argv[0]).with_name("cube_sieve_propagation_manifest.py")
proc = subprocess.run([sys.executable, str(producer), *sys.argv[1:]], check=True, text=True, stdout=subprocess.PIPE)
print(proc.stdout.replace("  final_domains=1,2,4,8,16,0", "  final_domains=1,2,4,8,16,1"), end="")
PY
cp "$PRODUCER" "$WORK/cube_sieve_propagation_manifest.py"
chmod +x "$WORK/bad_backend.py"

if python3 "$PARITY" "$WORK/k6.edge" 5 "$WORK/k6.cube" \
    --backend-producer "$WORK/bad_backend.py" > "$WORK/bad.out" 2>&1; then
  echo "error: parity accepted a corrupted backend manifest" >&2
  exit 1
fi
rg -q 'semantic mismatch|invalid manifest' "$WORK/bad.out"

cat > "$WORK/extra_backend.py" <<'PY'
#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

producer = Path(sys.argv[0]).with_name("cube_sieve_propagation_manifest.py")
proc = subprocess.run([sys.executable, str(producer), *sys.argv[1:]], check=True, text=True, stdout=subprocess.PIPE)
print(proc.stdout.replace("  claim=domain_propagation_result_only", "  claim=domain_propagation_result_only\n  backend_extra_line=forbidden"), end="")
PY
chmod +x "$WORK/extra_backend.py"

if python3 "$PARITY" "$WORK/k6.edge" 5 "$WORK/k6.cube" \
    --backend-producer "$WORK/extra_backend.py" > "$WORK/extra.out" 2>&1; then
  echo "error: parity accepted a backend manifest with extra unrecognized content" >&2
  exit 1
fi
rg -q 'canonical manifest mismatch' "$WORK/extra.out"

cat > "$WORK/promotable_backend.py" <<'PY'
#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

producer = Path(sys.argv[0]).with_name("cube_sieve_propagation_manifest.py")
proc = subprocess.run([sys.executable, str(producer), *sys.argv[1:]], check=True, text=True, stdout=subprocess.PIPE)
print(proc.stdout.replace("promotable=0", "promotable=1"), end="")
PY
chmod +x "$WORK/promotable_backend.py"

if python3 "$PARITY" "$WORK/k6.edge" 5 "$WORK/k6.cube" \
    --backend-producer "$WORK/promotable_backend.py" > "$WORK/promotable.out" 2>&1; then
  echo "error: parity accepted a backend manifest with promotable=1" >&2
  exit 1
fi
rg -q 'invalid manifest' "$WORK/promotable.out"

echo "cube_sieve_gpu_parity_gate: PASS"
