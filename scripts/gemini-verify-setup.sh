#!/bin/bash
# gemini-verify-setup.sh - Verify Gemini CLI setup for Sounio development

# Use snap-installed gemini-cli (has OAuth credentials)
GEMINI_BIN="/snap/bin/gemini"

# Snap credentials location
SNAP_CREDS="$HOME/snap/gemini-cli/current/.gemini/oauth_creds.json"
OLD_CREDS="$HOME/.gemini/oauth_creds.json"

echo "======================================"
echo "   Gemini CLI Setup Verification"
echo "======================================"
echo ""

# Check binary
echo "--- Binary Status ---"
if [ -f "$GEMINI_BIN" ]; then
    echo "✓ Binary found at:"
    echo "  $GEMINI_BIN"
    VERSION=$($GEMINI_BIN --version 2>/dev/null)
    if [ -n "$VERSION" ]; then
        echo "✓ Version: $VERSION"
    fi
else
    echo "✗ Binary not found"
    exit 1
fi

echo ""
echo "--- Authentication Status ---"

# Check API Key
if [ -n "$GEMINI_API_KEY" ]; then
    echo "✓ GEMINI_API_KEY is set (API Key mode)"
    AUTH_METHOD="api_key"
fi

# Check Snap OAuth (primary)
if [ -f "$SNAP_CREDS" ]; then
    echo "✓ Snap OAuth credentials found"
    EXPIRY=$(cat "$SNAP_CREDS" | grep -o '"expiry_date": [0-9]*' | cut -d' ' -f2)
    if [ -n "$EXPIRY" ]; then
        EXPIRY_SEC=$((EXPIRY/1000))
        CURRENT_SEC=$(date +%s)
        if [ "$EXPIRY_SEC" -gt "$CURRENT_SEC" ]; then
            EXPIRY_HUMAN=$(date -d @$EXPIRY_SEC '+%Y-%m-%d %H:%M')
            echo "  ✓ Token valid until: $EXPIRY_HUMAN"
            AUTH_METHOD="oauth"
        else
            echo "  ✗ Token EXPIRED"
        fi
    fi
fi

# Check old OAuth (backup)
if [ -f "$OLD_CREDS" ]; then
    OLD_EXPIRY=$(cat "$OLD_CREDS" | grep -o '"expiry_date": [0-9]*' | cut -d' ' -f2)
    if [ -n "$OLD_EXPIRY" ]; then
        echo "ℹ Old OAuth creds exist (expired $(date -d @$((OLD_EXPIRY/1000)) '+%Y-%m-%d'))"
    fi
fi

echo ""
echo "--- API Test ---"

if [ -n "$GEMINI_API_KEY" ]; then
    echo "Testing with API Key..."
    RESULT=$($GEMINI_BIN -p "Say 'Gemini CLI is working!'" -m gemini-2.5-flash 2>&1)
    if echo "$RESULT" | grep -q "working"; then
        echo "✓ API connection successful!"
        echo ""
        echo "Output:"
        echo "$RESULT"
    else
        echo "✗ API connection failed"
        echo "Error: $(echo "$RESULT" | head -3)"
    fi
elif [ "$AUTH_METHOD" = "oauth" ]; then
    echo "Testing with OAuth (Ultra subscription)..."
    RESULT=$(GOOGLE_GENAI_USE_GCA=true $GEMINI_BIN -p "Say 'OAuth is working with Ultra subscription!'" -m gemini-2.5-flash 2>&1)
    if echo "$RESULT" | grep -q "working"; then
        echo "✓ OAuth connection successful!"
        echo "✓ Ultra subscription active"
        echo ""
        echo "Output:"
        echo "$RESULT"
    else
        echo "✗ OAuth connection failed"
        echo "Error: $(echo "$RESULT" | head -5)"
    fi
else
    echo "✗ No authentication method configured"
    echo "  Run: ./scripts/gemini-api-key-setup.sh"
fi

echo ""
echo "--- Available Scripts ---"
ls -1 $(dirname "$0")/gemini-*.sh 2>/dev/null | while read f; do
    echo "  $(basename $f)"
done

echo ""
echo "======================================"
echo "For setup help: ./scripts/gemini-api-key-setup.sh"
echo "======================================"
