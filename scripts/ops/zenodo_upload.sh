#!/usr/bin/env bash
# scripts/ops/zenodo_upload.sh — Upload a preprint to Zenodo via REST API
#
# Usage:
#   export ZENODO_TOKEN="your-token-here"
#   bash scripts/ops/zenodo_upload.sh paper/preprint.pdf
#
# Prerequisites:
#   1. Create a Zenodo account at https://zenodo.org
#   2. Generate a token at https://zenodo.org/account/settings/applications/tokens/new/
#      with scopes: deposit:write, deposit:actions
#   3. Set ZENODO_TOKEN environment variable (never commit this)
#   4. Convert preprint to PDF first (e.g., pandoc paper/preprint.md -o paper/preprint.pdf)
#
# For testing, use the sandbox:
#   export ZENODO_SANDBOX=1
#   export ZENODO_TOKEN="sandbox-token"
#   bash scripts/ops/zenodo_upload.sh paper/preprint.pdf
#
# The script will:
#   1. Create a new deposit
#   2. Upload the PDF
#   3. Set metadata (title, authors, keywords, license)
#   4. Optionally publish (assigns DOI immediately)

set -euo pipefail

# --- Configuration -----------------------------------------------------------

FILE="${1:?Usage: $0 <file.pdf>}"

if [ ! -f "$FILE" ]; then
    echo "Error: File not found: $FILE"
    exit 1
fi

if [ -z "${ZENODO_TOKEN:-}" ]; then
    echo "Error: ZENODO_TOKEN not set."
    echo "  export ZENODO_TOKEN=\"your-token\""
    echo "  Get one at: https://zenodo.org/account/settings/applications/tokens/new/"
    exit 1
fi

if [ "${ZENODO_SANDBOX:-0}" = "1" ]; then
    API="https://sandbox.zenodo.org/api"
    echo "[sandbox mode] Using Zenodo sandbox (test DOIs with 10.5072 prefix)"
else
    API="https://zenodo.org/api"
fi

FILENAME=$(basename "$FILE")

# --- Metadata ----------------------------------------------------------------

METADATA='{
  "metadata": {
    "title": "Epistemic Shadow Lanes: Compiler-Generated Uncertainty Propagation for GPU Kernels",
    "upload_type": "publication",
    "publication_type": "preprint",
    "publication_date": "2026-03-21",
    "creators": [
      {
        "name": "Agourakis, Demetrios C.",
        "orcid": "0009-0001-8671-8878",
        "affiliation": "Pontifícia Universidade Católica de São Paulo (PUC-SP); Faculdade São Leopoldo Mandic"
      }
    ],
    "description": "We present a GPU compiler transformation that generates first-order GUM-style (JCGM 100:2008) uncertainty propagation alongside every arithmetic instruction in GPU kernel code. For each value register in emitted PTX, the compiler produces three shadow registers — uncertainty variance, validity predicate, and provenance bitset — that implement the law of propagation of uncertainty without programmer intervention. Implemented in the Sounio programming language compiler.",
    "access_right": "open",
    "license": "CC-BY-4.0",
    "keywords": [
      "uncertainty propagation",
      "GPU compiler",
      "GUM JCGM 100:2008",
      "epistemic computing",
      "shadow registers",
      "PTX codegen",
      "warp-vote ballot",
      "tensor core WMMA",
      "measurement uncertainty",
      "Sounio programming language"
    ],
    "subjects": [
      {"term": "Computer Science - Programming Languages", "identifier": "cs.PL", "scheme": "url"},
      {"term": "Computer Science - Distributed, Parallel, and Cluster Computing", "identifier": "cs.DC", "scheme": "url"}
    ],
    "language": "eng",
    "notes": "Preprint. Hardware benchmarks in progress. Code: https://github.com/agourakis82/sounio",
    "related_identifiers": [
      {
        "identifier": "https://github.com/agourakis82/sounio",
        "relation": "isSupplementTo",
        "resource_type": "software",
        "scheme": "url"
      }
    ]
  }
}'

# --- Step 1: Create deposit --------------------------------------------------

echo "Step 1: Creating deposit..."
DEPOSIT_RESPONSE=$(curl -s -H "Authorization: Bearer $ZENODO_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{}' \
    "$API/deposit/depositions")

DEPOSIT_ID=$(echo "$DEPOSIT_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
BUCKET_URL=$(echo "$DEPOSIT_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['links']['bucket'])" 2>/dev/null)

if [ -z "$DEPOSIT_ID" ] || [ "$DEPOSIT_ID" = "None" ]; then
    echo "Error: Failed to create deposit."
    echo "$DEPOSIT_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$DEPOSIT_RESPONSE"
    exit 1
fi

echo "  Deposit ID: $DEPOSIT_ID"
echo "  Bucket URL: $BUCKET_URL"

# --- Step 2: Upload file -----------------------------------------------------

echo "Step 2: Uploading $FILENAME ($(du -h "$FILE" | cut -f1))..."
UPLOAD_RESPONSE=$(curl -s --upload-file "$FILE" \
    -H "Authorization: Bearer $ZENODO_TOKEN" \
    "$BUCKET_URL/$FILENAME")

UPLOAD_KEY=$(echo "$UPLOAD_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('key',''))" 2>/dev/null)

if [ -z "$UPLOAD_KEY" ]; then
    echo "Error: Upload failed."
    echo "$UPLOAD_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$UPLOAD_RESPONSE"
    exit 1
fi

echo "  Uploaded: $UPLOAD_KEY"

# --- Step 3: Set metadata ----------------------------------------------------

echo "Step 3: Setting metadata..."
META_RESPONSE=$(curl -s -X PUT \
    -H "Authorization: Bearer $ZENODO_TOKEN" \
    -H "Content-Type: application/json" \
    -d "$METADATA" \
    "$API/deposit/depositions/$DEPOSIT_ID")

META_TITLE=$(echo "$META_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['metadata']['title'])" 2>/dev/null)

if [ -z "$META_TITLE" ]; then
    echo "Warning: Metadata update may have failed."
    echo "$META_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$META_RESPONSE"
else
    echo "  Title: $META_TITLE"
fi

# --- Step 4: Publish (optional) ----------------------------------------------

echo ""
echo "Deposit ready. Preview at:"
echo "  $API/deposit/depositions/$DEPOSIT_ID"
echo ""
read -r -p "Publish now? This assigns a DOI and is PERMANENT. [y/N] " CONFIRM

if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
    echo "Step 4: Publishing..."
    PUB_RESPONSE=$(curl -s -X POST \
        -H "Authorization: Bearer $ZENODO_TOKEN" \
        "$API/deposit/depositions/$DEPOSIT_ID/actions/publish")

    DOI=$(echo "$PUB_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['doi'])" 2>/dev/null)
    RECORD_URL=$(echo "$PUB_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['links']['record_html'])" 2>/dev/null)

    if [ -n "$DOI" ] && [ "$DOI" != "None" ]; then
        echo ""
        echo "=========================================="
        echo "  PUBLISHED"
        echo "  DOI: $DOI"
        echo "  URL: $RECORD_URL"
        echo "=========================================="
        echo ""
        echo "Add to CITATION.cff:"
        echo "  doi: $DOI"
    else
        echo "Error: Publish failed."
        echo "$PUB_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$PUB_RESPONSE"
        exit 1
    fi
else
    echo "Skipped publishing. You can publish later at:"
    echo "  https://zenodo.org/deposit/$DEPOSIT_ID"
fi
