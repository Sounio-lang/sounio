#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="$ROOT/bin/souc-linux-x86_64"

fail() {
    echo "FAIL: $1" >&2
    exit 1
}

echo "=== ontology_cli_smoke_gate ==="
echo

# 1. resolve
echo "[1/9] resolve QM:0000001"
out=$("$SOUC" ontology resolve QM:0000001 2>&1)
if ! echo "$out" | grep -q "curie: QM:0000001"; then fail "resolve curie"; fi
if ! echo "$out" | grep -q "label: Quantum mechanics"; then fail "resolve label"; fi
echo "  OK"

# 2. ancestors
echo "[2/9] ancestors QM:0000003"
out=$("$SOUC" ontology ancestors QM:0000003 2>&1)
if ! echo "$out" | grep -q "QM:0000003"; then fail "ancestors child"; fi
if ! echo "$out" | grep -q "QM:0000001"; then fail "ancestors parent"; fi
echo "  OK"

# 3. is-subclass
echo "[3/9] is-subclass"
out=$("$SOUC" ontology is-subclass QM:0000003 QM:0000001 2>&1)
if [[ "$out" != "yes" ]]; then fail "is-subclass positive"; fi
out=$("$SOUC" ontology is-subclass QM:0000001 QM:0000003 2>&1)
if [[ "$out" != "no" ]]; then fail "is-subclass negative"; fi
echo "  OK"

# 4. search
echo "[4/9] search 'Quantum'"
out=$("$SOUC" ontology search "Quantum" 2>&1)
if ! echo "$out" | grep -q "QM:0000001"; then fail "search result"; fi
if ! echo "$out" | grep -q " results"; then fail "search count"; fi
echo "  OK"

# 5. map
echo "[5/9] map SNOMED:73211009 --to HP"
out=$("$SOUC" ontology map SNOMED:73211009 --to HP 2>&1)
if ! echo "$out" | grep -q "SNOMED:73211009"; then fail "map source"; fi
if ! echo "$out" | grep -q "HP:0001626"; then fail "map target"; fi
echo "  OK"

# 6. list
echo "[6/9] list QM"
out=$("$SOUC" ontology list QM 2>&1)
if ! echo "$out" | grep -q "QM:0000001"; then fail "list content"; fi
if ! echo "$out" | grep -q " terms"; then fail "list count"; fi
echo "  OK"

# 7. validate
echo "[7/9] validate"
out=$("$SOUC" ontology validate 2>&1)
if ! echo "$out" | grep -q "VALID"; then fail "validate did not pass"; fi
echo "  OK"

# 8. count
echo "[8/9] count"
out=$("$SOUC" ontology count 2>&1)
if ! echo "$out" | grep -q "1008"; then fail "count total"; fi
out=$("$SOUC" ontology count QM 2>&1)
if ! echo "$out" | grep -q "112"; then fail "count prefix"; fi
echo "  OK"

# 9. format json + tui
echo "[9/10] format json resolve"
out=$("$SOUC" ontology --format json resolve QM:0000001 2>&1)
if ! echo "$out" | grep -q '"curie": "QM:0000001"'; then fail "json resolve curie"; fi
if ! echo "$out" | grep -q '"label": "Quantum mechanics"'; then fail "json resolve label"; fi
echo "  OK"

echo "[10/10] format tsv resolve"
out=$("$SOUC" ontology --format tsv resolve QM:0000001 2>&1)
if ! echo "$out" | grep -q $'QM:0000001\tQuantum mechanics'; then fail "tsv resolve"; fi
echo "  OK"

echo "[11/12] format tsv count"
out=$("$SOUC" ontology --format tsv count 2>&1)
if ! echo "$out" | grep -q "^1008$"; then fail "tsv count total"; fi
echo "  OK"

echo "[12/12] format tsv is-subclass"
out=$("$SOUC" ontology --format tsv is-subclass QM:0000003 QM:0000001 2>&1)
if [[ "$out" != "true" ]]; then fail "tsv is-subclass positive"; fi
echo "  OK"

echo
echo "PASS: all 12 ontology CLI smoke tests passed"
