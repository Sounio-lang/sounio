#!/usr/bin/env bash
# Reproduce the audit from the public PDG 2025 file.
set -euo pipefail
cd "$(dirname "$0")"
DB=${DB:-pdgall-2025-v0.2.3.sqlite}
[[ -f "$DB" ]] || curl -sL -o "$DB" https://pdg.lbl.gov/2025/api/pdgall-2025-v0.2.3.sqlite
echo "1a042ee01d3aad5d92a31a5209f5ff6720f7a60ee27fd9dbfd0d743127189a1d  $DB" | shasum -a 256 -c -
sqlite3 "$DB" < extract.sql > inputs.csv
c++ -std=c++2b -O2 -Wall -Wextra audit.cpp -o audit
./audit inputs.csv results.csv | tee output.txt
