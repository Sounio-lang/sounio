#!/usr/bin/env bash
# Download SWOW expansion datasets into data/processed/expansion/raw/
# DE: SWOW-DE2025 (R55, use R1-only downstream)
# ZH: SWOW-ZH24 (portal id; post-preprocessing = SWOW-ZH23 release)
# SL: CLARIN.SI (no SWOW portal needed)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW="$ROOT/data/processed/expansion/raw"
mkdir -p "$RAW/de" "$RAW/zh" "$RAW/sl"

echo "== SWOW typological expansion download =="
echo "ROOT=$ROOT"

# --- SL: CLARIN (reliable) ---
SL_STATS="$RAW/sl/SWOW-SL1.0_statistics_normalized.tsv"
if [[ ! -s "$SL_STATS" ]]; then
  echo "[SL] Downloading from CLARIN.SI hdl:11356/1980 ..."
  curl -fsSL \
    'https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1980/SWOW-SL1.0_statistics_normalized.tsv?sequence=4&isAllowed=y' \
    -o "$SL_STATS"
  curl -fsSL \
    'https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1980/README.txt?sequence=1&isAllowed=y' \
    -o "$RAW/sl/README.txt"
fi
echo "[SL] OK $(wc -c <"$SL_STATS") bytes  md5=$(md5sum "$SL_STATS" | awk '{print $1}')"

# --- DE: clone preprocessing code (data still from SWOW portal) ---
DE_CODE="$RAW/de/_swow_de_code"
if [[ ! -d "$DE_CODE/.git" ]]; then
  echo "[DE] Cloning SWOW-DE-2025-Code (R pipeline only; no CSV in repo) ..."
  git clone --depth 1 https://github.com/samuelae/SWOW-DE-2025-Code.git "$DE_CODE"
fi
echo "[DE] Code repo: $DE_CODE"
echo "[DE] Place trial export here after manual SWOW download:"
echo "      $RAW/de/SWOW_DE_2025_R55.csv"
echo "      (see $DE_CODE/01_Data/Final/readme.txt)"

# --- DE / ZH: SWOW portal (requires browser form; server often 500 from curl) ---
download_swow_playwright() {
  local dataset_id="$1"
  local out_zip="$2"
  if [[ -s "$out_zip" ]] && file "$out_zip" | grep -qi 'zip'; then
    echo "[$dataset_id] already present: $out_zip"
    return 0
  fi
  python3 - "$dataset_id" "$out_zip" <<'PY'
import sys, os, time
from playwright.sync_api import sync_playwright

dataset_id, out_zip = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(out_zip), exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    page.goto("https://smallworldofwords.org/en/project/research", wait_until="domcontentloaded", timeout=120000)
    time.sleep(2)
    page.locator(f"a.downloadLink[data-id='{dataset_id}']").first.click()
    page.wait_for_selector("#downloadForm", state="visible", timeout=30000)
    page.fill("#name", os.environ.get("SWOW_DOWNLOAD_NAME", "Research User"))
    page.fill("#email", os.environ.get("SWOW_DOWNLOAD_EMAIL", "research@example.com"))
    page.fill('textarea[name="message"]', "Typological expansion R1-only audit")
    with page.expect_response(lambda r: "download" in r.url, timeout=120000) as resp_info:
        page.click("#agreeButton")
    resp = resp_info.value
    body = resp.body()
    if resp.status != 200 or b"PK" not in body[:4]:
        print(f"ERROR: SWOW portal returned HTTP {resp.status} (not a zip).", file=sys.stderr)
        print(f"First bytes: {body[:80]!r}", file=sys.stderr)
        print("Manual: open https://smallworldofwords.org/en/project/research", file=sys.stderr)
        print(f"  download dataset {dataset_id} and place at: {out_zip}", file=sys.stderr)
        sys.exit(1)
    open(out_zip, "wb").write(body)
    browser.close()
print("saved", out_zip, os.path.getsize(out_zip))
PY
}

DE_ZIP="$RAW/de/SWOW-DE_2025.zip"
ZH_ZIP="$RAW/zh/SWOW-ZH23.zip"

if [[ "${SKIP_SWOW_PORTAL:-0}" != "1" ]]; then
  echo "[DE] Attempting SWOW-DE2025 ..."
  download_swow_playwright SWOW-DE2025 "$DE_ZIP" || true
  echo "[ZH] Attempting SWOW-ZH24 ..."
  download_swow_playwright SWOW-ZH24 "$ZH_ZIP" || true
fi

for f in "$DE_ZIP" "$ZH_ZIP"; do
  if [[ -s "$f" ]] && file "$f" | grep -qi zip; then
    echo "OK zip $f ($(wc -c <"$f") bytes)"
  else
    echo "MISSING or invalid: $f"
    echo "  -> Download manually from https://smallworldofwords.org/en/project/research"
    echo "  -> Or set SKIP_SWOW_PORTAL=1 after placing files by hand"
  fi
done

echo "Done. Next: python3 scripts/research/typological_expansion_preprocess.py"
