#!/bin/bash
# gemini-api-key-setup.sh - Guide for setting up Gemini API key

cat << 'EOF'
=== Gemini API Key Setup ===

Your OAuth credentials have expired. For scripting, an API Key is recommended.

1. Get your API Key from Google AI Studio:
   https://aistudio.google.com/app/apikey

2. Set the environment variable (add to ~/.bashrc for persistence):
   export GEMINI_API_KEY='your-api-key-here'

3. Verify it works:
   ./scripts/gemini-verify-setup.sh

=== Quota Comparison ===

API Key (Free Tier):
- 1,000 requests/day
- Works in scripts/non-interactive mode
- No browser authentication needed

OAuth (Ultra Subscription - $249.99/mo):
- 2,000 requests/day  
- Requires interactive browser login
- Includes Deep Think mode, Veo 3 video

=== Using Ultra Subscription ===

If you want to use your Ultra subscription with OAuth:

1. Run interactively to refresh the OAuth token:
   node /snap/bin/gemini

2. Follow the browser authentication flow

3. Then set:
   export GOOGLE_GENAI_USE_GCA=true

4. Test: ./scripts/gemini-verify-setup.sh

=== Quick Test ===
EOF

# Check if API key is already set
if [ -n "$GEMINI_API_KEY" ]; then
    echo "✓ GEMINI_API_KEY is already set"
    echo "  Testing connection..."
    GEMINI_BIN="/snap/bin/gemini"
    "$GEMINI_BIN" -p "Hello" -m gemini-2.5-flash 2>&1 | head -5
else
    echo "✗ GEMINI_API_KEY is not set"
    echo "  Get your key at: https://aistudio.google.com/app/apikey"
fi
