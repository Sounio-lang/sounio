# Madaros Imported Runtime Acceptance Fixtures

These fixtures are the dedicated issue #862 slice of
`scripts/ci/madaros_imported_runtime_acceptance_gate.sh`.

- `issue_862_positive.sio` combines a named selective import, a caller-local
  helper, and `print_f64`. Its executable must exit 0 and write exactly
  `0.500000` followed by one line feed.
- `issue_862_private_main.sio` imports a genuinely private function from its
  sibling leaf. Compilation must reject it with E175 and must not leave the
  requested ELF behind.

The same gate uses the existing issue #921 rational repro and the existing
issue #901 probability stdlib test directly. It does not build a compiler.
Point it at a node-local or otherwise source-fresh raw Madaros ELF and pin that
exact file by SHA-256:

```bash
raw=/path/to/source-fresh/madaros
sha=$(sha256sum "$raw" | awk '{print $1}')
SOUNIO_MADAROS_IMPORTED_RUNTIME_RAW_BIN="$raw" \
SOUNIO_MADAROS_IMPORTED_RUNTIME_EXPECTED_SHA256="$sha" \
  bash scripts/ci/madaros_imported_runtime_acceptance_gate.sh
```

Compact imported IR and fallback evidence are forbidden by the gate. Set
`SOUNIO_MADAROS_IMPORTED_RUNTIME_KEEP=1` to retain its temporary logs and ELF
artifacts for inspection.

Source freshness is established by the lane that builds the raw ELF; the gate
does not infer a source commit or perform a rebuild. It verifies the supplied
ELF before and after the matrix against the required SHA-256. The issue #901
receipt always records the merged function count and ELF size, and records code
and relocation counts from compiler logs or `readelf` when those surfaces are
available.
