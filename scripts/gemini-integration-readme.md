# Gemini CLI Integration for Sounio

Leverage your **$249.99/month Google AI Ultra** subscription within the Sounio development workflow.

## Quick Start

```bash
# 1. Ensure Gemini CLI is installed and authenticated
npm install -g @google/gemini-cli
gemini --login  # Use your Google AI Ultra account

# 2. Verify Ultra subscription
gemini /auth
# Should show: "Google AI Ultra — 2,000 requests/day"

# 3. Use the scripts
./scripts/gemini-analyze.sh self-hosted/compiler/lean_single.sio architecture
./scripts/gemini-generate-test.sh "propagate_bmi" "stdlib/epistemic/gum.sio" unit
```

## Scripts

### `gemini-analyze.sh` — Code Analysis

Analyze Sounio code with context-aware prompts.

```bash
./scripts/gemini-analyze.sh <file.sio> [context]

# Contexts:
#   architecture  — Module structure, refactoring opportunities
#   types         — Type system correctness, epistemic types
#   effects       — Effect declarations (with IO, Mut, Panic)
#   optimization  — Performance, codegen efficiency
#   general       — General quality review (default)

# Examples:
./scripts/gemini-analyze.sh stdlib/epistemic/knowledge.sio types
./scripts/gemini-analyze.sh self-hosted/check/check.sio architecture
```

**Uses:** `gemini-3.1-pro` or `gemini-3.1-pro-thinking` (Ultra quota)

---

### `gemini-generate-test.sh` — Test Generation

Generate Sounio test cases from existing code.

```bash
./scripts/gemini-generate-test.sh <function_name> <source_file> [test_type]

# Test types:
#   unit          — Standard run-pass test (default)
#   integration   — Cross-module integration test
#   compile-fail  — Tests that should fail compilation
#   fuzz          — Property-based/randomized tests

# Examples:
./scripts/gemini-generate-test.sh "sqrt_approx" "stdlib/epistemic/gum.sio" unit
./scripts/gemini-generate-test.sh "knowledge_add" "stdlib/epistemic/knowledge.sio" fuzz
```

**Output:** Saves to `tests/run-pass/` or `tests/compile-fail/`

---

### `gemini-offload.sh` — LLM Offload Integration

Drop-in replacement for `llm-offload.sh` that uses Ultra subscription.

```bash
./scripts/gemini-offload.sh <prompt-file> [mode]

# Modes:
#   review    — Code review with Sounio focus
#   scaffold  — Generate boilerplate
#   explain   — Deep explanation (thinking mode)
#   optimize  — Performance analysis (thinking mode)
#   math      — Mathematical reasoning (thinking mode)

# Example:
echo "Explain the 168 theorem proof" > /tmp/prompt.txt
./scripts/gemini-offload.sh /tmp/prompt.txt math
```

**Integration with existing workflow:**
```bash
# Instead of:
bash scripts/llm-offload.sh prompt.txt xai

# Use:
bash scripts/gemini-offload.sh prompt.txt review
```

---

### `gemini-check-stdlib.sh` — Batch Stdlib Analysis

Analyze multiple stdlib files efficiently.

```bash
./scripts/gemini-check-stdlib.sh [pattern] [max-files]

# Examples:
./scripts/gemini-check-stdlib.sh "*.sio" 5
./scripts/gemini-check-stdlib.sh "epistemic/*.sio" 10
```

**Features:**
- Rotates analysis types (syntax, effects, types, docs)
- Respects Ultra rate limits (60 RPM, 2,000/day)
- Skips oversized files (>100KB)

---

## Authentication

### Method 1: OAuth (Recommended — Uses Ultra Quota)

```bash
gemini --login
# Select "Login with Google"
# Use the Google account with your Ultra subscription
```

Quota: **2,000 requests/day**, **60 requests/minute**

### Method 2: API Key (Separate Billing)

```bash
export GEMINI_API_KEY="your-key-from-ai-studio"
gemini --auth api-key
```

⚠️ **Warning:** API keys use **separate pay-as-you-go billing**, not your Ultra subscription.

---

## Rate Limits & Quotas

| Tier | Requests/Day | RPM | Cost |
|------|--------------|-----|------|
| Free | 1,000 | 60 | $0 |
| Pro ($19.99/mo) | 1,500 | 60 | Included |
| **Ultra ($249.99/mo)** | **2,000** | **60** | **Included** |
| API Key | Varies | Varies | Pay-as-you-go |

### Optimizing Ultra Usage

```bash
# Use Flash for simple tasks (faster, same quota)
gemini --model gemini-3-flash --prompt "Quick syntax check"

# Reserve Pro/Thinking for complex analysis
gemini --model gemini-3.1-pro-thinking --prompt "Prove theorem..."

# Batch operations with sleep to respect RPM
for file in *.sio; do
    ./scripts/gemini-analyze.sh "$file" &
    sleep 1  # Max 60 RPM
done
```

---

## Integration with Existing Fleet

Update `.claude/offload-routing.md`:

```markdown
## Google AI Ultra (via Gemini CLI)

| Model | Context | Cost | Best For |
|-------|---------|------|----------|
| gemini-3.1-pro | 1M | Included in Ultra | Code review, architecture |
| gemini-3.1-pro-thinking | 1M | Included | Math, proofs, optimization |
| gemini-3-flash | 1M | Included | Fast iteration, bulk analysis |

### Usage
```bash
# Analyze Sounio code
./scripts/gemini-analyze.sh <file> [context]

# Generate tests
./scripts/gemini-generate-test.sh <func> <file> [type]

# Offload routing
./scripts/gemini-offload.sh <prompt> [mode]
```

### Quota
- 2,000 requests/day
- 60 requests/minute
- Reset: Midnight Pacific Time
```

---

## Troubleshooting

### "Products not yet authorized" Error

```bash
# Your Ultra subscription isn't linked to CLI
# Fix:
gemini --logout
gemini --login
# Use EXACT same account as https://one.google.com/about/google-ai-plans
```

### Wrong Quota (Showing 1,000 instead of 2,000)

```bash
# Check which account is logged in
gemini /auth

# Likely causes:
# 1. Using Workspace account (different quota system)
# 2. Using Pro instead of Ultra account
# 3. Not logged in

# Fix:
gemini --logout
gemini --login
```

### Rate Limit Errors (429)

```bash
# Check current usage
gemini /usage  # If available

# Wait for RPM reset (rolling 60-second window)
sleep 60

# Or switch to API key for overflow (separate billing)
export GEMINI_API_KEY="..."
```

---

## Advanced: Headless/CI Usage

For servers without browser OAuth:

```bash
# 1. Authenticate on local machine
gemini --login

# 2. Copy credentials to server
# Credentials stored in: ~/.config/gemini/credentials.json
scp ~/.config/gemini/credentials.json server:~/.config/gemini/

# 3. Use on server
gemini /auth  # Verify
./scripts/gemini-analyze.sh file.sio
```

---

## Cost Comparison

Analyzing `lean_single.sio` (~630KB, ~200K tokens):

| Method | Cost | Context |
|--------|------|---------|
| **Ultra CLI** | **$0** (in subscription) | ✅ Included |
| API Key (3.1 Pro) | ~$0.40-0.80 | Pay-as-you-go |
| Claude Opus 4.6 | ~$1.00 | Via API |
| Grok 4.1 Fast | ~$0.04 | Via API |

**Ultra advantage:** 2,000 analyses/month included vs. ~$800/month equivalent API cost.

---

## See Also

- `llm-offload.sh` — Multi-model consensus (API-based)
- `scripts/sounio-lint.sh` — Local linting (no quota)
- `scripts/resolve-souc.sh` — Compiler resolution
