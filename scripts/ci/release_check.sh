#!/bin/bash
# Release quality checklist for Demetrios v0.50.0

set -e

echo "═══════════════════════════════════════════════════════════"
echo "         DEMETRIOS v0.50.0 RELEASE CHECKLIST"
echo "═══════════════════════════════════════════════════════════"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

check() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
        ((PASS_COUNT++))
        return 0
    else
        echo -e "${RED}✗${NC} $2"
        ((FAIL_COUNT++))
        return 1
    fi
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARN_COUNT++))
}

cd /mnt/e/workspace/demetrios/compiler

echo ""
echo "1. BUILD CHECKS"
echo "───────────────────────────────────────────────────────────"

# Build in release mode
cargo build --release 2>/dev/null
check $? "Release build succeeds"

# Build documentation
cargo doc --no-deps 2>/dev/null
check $? "Documentation builds"

echo ""
echo "2. TEST CHECKS"
echo "───────────────────────────────────────────────────────────"

# Unit tests
cargo test --lib 2>/dev/null
check $? "Unit tests pass"

# Integration tests
cargo test --test '*' 2>/dev/null
check $? "Integration tests pass"

echo ""
echo "3. CODE QUALITY"
echo "───────────────────────────────────────────────────────────"

# Clippy (allow some warnings for now)
cargo clippy 2>/dev/null
check $? "Clippy analysis completes"

# Format check
if cargo fmt -- --check 2>/dev/null; then
    check 0 "Code is formatted"
else
    warn "Code formatting has issues (run cargo fmt)"
fi

echo ""
echo "4. DOCUMENTATION"
echo "───────────────────────────────────────────────────────────"

cd /mnt/e/workspace/demetrios

# README exists and is substantial
if [ -f "README.md" ] && [ $(wc -l < README.md) -gt 50 ]; then
    check 0 "README.md is substantial"
else
    warn "README.md might need more content"
fi

# CHANGELOG updated
if [ -f "CHANGELOG.md" ] && grep -q "0.50.0" CHANGELOG.md 2>/dev/null; then
    check 0 "CHANGELOG.md is updated for v0.50.0"
else
    warn "CHANGELOG.md needs v0.50.0 entry"
fi

# RELEASE_NOTES exists
if [ -f "RELEASE_NOTES.md" ]; then
    check 0 "RELEASE_NOTES.md exists"
else
    warn "RELEASE_NOTES.md is missing"
fi

# CONTRIBUTING exists
if [ -f "CONTRIBUTING.md" ]; then
    check 0 "CONTRIBUTING.md exists"
else
    warn "CONTRIBUTING.md is missing"
fi

echo ""
echo "5. SPECIFICATION FILES"
echo "───────────────────────────────────────────────────────────"

# Formal spec
if [ -f "spec/formal/semantic_types.tex" ]; then
    check 0 "Formal type theory specification exists"
else
    warn "Formal specification is missing"
fi

# Academic paper
if [ -f "docs/papers/semantic_types_paper.md" ]; then
    check 0 "Academic paper skeleton exists"
else
    warn "Paper skeleton is missing"
fi

echo ""
echo "6. NEW DAY 50 FILES"
echo "───────────────────────────────────────────────────────────"

cd /mnt/e/workspace/demetrios/compiler

# Profiling module
if [ -f "src/profiling/mod.rs" ]; then
    check 0 "Profiling module exists"
else
    warn "Profiling module is missing"
fi

# Optimized loader
if [ -f "src/ontology/loader/optimized.rs" ]; then
    check 0 "Optimized loader exists"
else
    warn "Optimized loader is missing"
fi

# SIMD operations
if [ -f "src/ontology/embedding/simd.rs" ]; then
    check 0 "SIMD operations exist"
else
    warn "SIMD operations are missing"
fi

# Semantic errors
if [ -f "src/diagnostics/semantic_errors.rs" ]; then
    check 0 "Semantic error messages exist"
else
    warn "Semantic error messages are missing"
fi

echo ""
echo "7. BENCHMARK SUITE"
echo "───────────────────────────────────────────────────────────"

# Ontology benchmarks
if [ -f "benches/ontology_bench.rs" ]; then
    check 0 "Ontology benchmark suite exists"
else
    warn "Ontology benchmarks are missing"
fi

# Check benchmarks compile
cargo check --bench ontology_bench 2>/dev/null
check $? "Benchmarks compile"

echo ""
echo "8. VERSION CHECK"
echo "───────────────────────────────────────────────────────────"

# Check Cargo.toml version
if grep -q 'version = "0.50.0"' Cargo.toml 2>/dev/null; then
    check 0 "Cargo.toml version is 0.50.0"
else
    warn "Cargo.toml version needs to be updated to 0.50.0"
fi

echo ""
echo "9. GIT STATUS"
echo "───────────────────────────────────────────────────────────"

cd /mnt/e/workspace/demetrios

# Check for git repo
if [ -d ".git" ]; then
    check 0 "Git repository exists"

    # Check branch
    BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    echo "   Current branch: $BRANCH"

    # Check for uncommitted changes
    if git diff --quiet 2>/dev/null && git diff --staged --quiet 2>/dev/null; then
        check 0 "No uncommitted changes"
    else
        warn "Uncommitted changes present"
    fi
else
    warn "Not a git repository"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "                    CHECKLIST SUMMARY"
echo "═══════════════════════════════════════════════════════════"
echo -e "${GREEN}Passed:${NC}  $PASS_COUNT"
echo -e "${RED}Failed:${NC}  $FAIL_COUNT"
echo -e "${YELLOW}Warnings:${NC} $WARN_COUNT"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}Release checklist passed!${NC}"
    exit 0
else
    echo -e "${RED}Release checklist has failures. Please fix before release.${NC}"
    exit 1
fi
