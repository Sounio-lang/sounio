#!/usr/bin/env bash
# public_claims_gate.sh -- the public surfaces may not bring back a claim that
# was retired because it did not match what is measured (TOUR.md section 6).
#
# WHY. On 2026-10-05 (main=99d078eb) the website, README, CITATION.cff,
# SCALE.md and formal/README.md still said, among other things: version
# "1.0.0-beta.5" (the release is 2.1.0, the compiler build Madaros v0.80.0);
# "5 backends ... Cranelift JIT" (Linux x86-64 only; the JIT is no longer
# shipped, CHANGELOG 2.1.0); "Dimensional mismatches are compile errors" (#2751,
# #2752); "251/251, 100% pass rate" (a 2026-05-12 result that predates
# module-privacy enforcement); "GPU acceleration" (PTX for empty-bodied kernels
# only); "zero sorry / fully proved" (61 axioms, 741 native_decide); paths
# under crates/souc that no longer exist; "2000-line C compiler" (stage0.c is
# 2,792 lines and is not on the `make build` path). Each was fixed once by hand.
# This gate keeps them fixed.
#
# SCOPE. website/src (except content/blog, which holds dated posts), README.md,
# CITATION.cff, SCALE.md, formal/README.md, TOUR.md. Matching is per line and
# case-insensitive.
#
# ESCAPES.
#   * A line that quotes history may carry the marker `public-claims: historical`
#     (in a comment, or in prose) on the same line. That is the only general
#     escape, and it is visible in review.
#   * "Cranelift JIT" is retired only when presented as shipped. A line that
#     also says it is retired / no longer shipped (in any of the site locales)
#     is the correction, not the claim.
#
# USAGE.
#   bash scripts/ci/public_claims_gate.sh               # scan this checkout
#   bash scripts/ci/public_claims_gate.sh --root DIR    # scan another tree
#   bash scripts/ci/public_claims_gate.sh --selftest [REF]
#     Proves the gate fails when it should: the current tree must pass, each
#     retired phrase planted in a scratch tree must fail, both escapes must
#     pass, and the files as they were at REF (default 99d078eb7, the last main
#     before the fix) must fail. If REF is not in the clone, that last control
#     is reported as SKIP rather than passed.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SELF="$ROOT_DIR/scripts/ci/public_claims_gate.sh"

PATHS=(website/src README.md CITATION.cff SCALE.md formal/README.md TOUR.md)
EXCLUDE_PREFIX="website/src/content/blog/"
MARKER='public-claims: historical'

# id|ERE (case-insensitive)|a literal sample the selftest plants
RETIRED=(
  'version-1.0.0-beta|1\.0\.0-beta|souc 1.0.0-beta.5'
  'stdlib-251|251[[:space:]]*/[[:space:]]*251|251/251 stdlib tests'
  'stdlib-100pct|100% pass rate|a 100% pass rate'
  'cranelift-jit-shipped|Cranelift JIT|an optional Cranelift JIT profile'
  'five-backends|5 backends|5 backends: x86_64, ARM64, PTX, WASM, JIT'
  'compile-time-dimensional|compile-time[[:space:]]+dimensional[[:space:]]+analysis|compile-time dimensional analysis'
  'dimensional-compile-errors|Dimensional mismatches are compile errors|Dimensional mismatches are compile errors'
  'gpu-acceleration|GPU acceleration|native GPU acceleration'
  'zero-sorry|zero[[:space:]]+`?sorry|zero `sorry` anywhere'
  'fully-proved|fully proved|all theorems are fully proved'
  'crates-souc|crates/souc|crates/souc/src/check/mod.rs'
  'c-compiler-2000|2000-line C compiler|a 2000-line C compiler'
)
# A "Cranelift JIT" line that also says this is the correction.
CRANELIFT_RETIRED='retired|no longer|not shipped|removed|não é mais|ya no se|δεν διατίθεται πλέον|不再|もう出荷されていません|廃止'

scan() {  # scan <root>  -> prints offending file:line lines, returns 1 if any
  local root="$1" hits=0 entry id re f
  local -a files=()
  for p in "${PATHS[@]}"; do
    if [[ -d "$root/$p" ]]; then
      while IFS= read -r f; do files+=("${f#"$root"/}"); done \
        < <(find "$root/$p" -type f | LC_ALL=C sort)
    elif [[ -f "$root/$p" ]]; then
      files+=("$p")
    fi
  done
  if ((${#files[@]} == 0)); then
    echo "public_claims_gate: nothing to scan under $root" >&2
    return 2
  fi
  local -a scanned=()
  for f in "${files[@]}"; do
    [[ "$f" == "$EXCLUDE_PREFIX"* ]] && continue
    scanned+=("$f")
  done
  for entry in "${RETIRED[@]}"; do
    id="${entry%%|*}"; re="${entry#*|}"; re="${re%|*}"
    # One grep per phrase over every file; -H keeps the path even for one file.
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      local rest="$line"                      # path:line:text
      local text="${rest#*:}"; text="${text#*:}"
      [[ "$text" == *"$MARKER"* ]] && continue
      if [[ "$id" == cranelift-jit-shipped ]] && grep -qiE "$CRANELIFT_RETIRED" <<<"$text"; then
        continue
      fi
      printf '  %s  [%s]\n' "$(cut -c1-180 <<<"$rest")" "$id"
      hits=$((hits + 1))
    done < <(cd "$root" && printf '%s\0' "${scanned[@]}" \
               | xargs -0 -r grep -HInEi -- "$re" 2>/dev/null)
  done
  ((hits == 0))
}

run_gate() {  # run_gate <root>
  local root="$1" out rc
  out="$(scan "$root")"; rc=$?
  if ((rc == 2)); then return 2; fi
  if ((rc != 0)); then
    echo "public_claims_gate: FAIL -- a retired public claim is back:" >&2
    printf '%s\n' "$out" >&2
    echo "" >&2
    echo "  Fix the claim to what is measured (TOUR.md section 6). If the line only" >&2
    echo "  quotes history, append the marker \`$MARKER\` to that line." >&2
    return 1
  fi
  echo "public_claims_gate: PASS -- no retired public claim in ${PATHS[*]} (blog excluded)"
  return 0
}

selftest() {
  local ref="${1:-99d078eb7}" fails=0 w rc entry id re
  w="$(mktemp -d "${TMPDIR:-/tmp}/public_claims_selftest.XXXXXX")"
  trap 'rm -rf "$w"' RETURN

  expect() {  # expect <name> <pass|fail> <root>
    local name="$1" want="$2" root="$3" got out
    out="$(run_gate "$root" 2>&1)"; rc=$?
    got=pass; ((rc != 0)) && got=fail
    ((rc == 2)) && got="error"
    if [[ "$got" == "$want" ]]; then
      printf '  ok    %-44s expected %s, got %s\n' "$name" "$want" "$got"
    else
      printf '  FAIL  %-44s expected %s, got %s\n' "$name" "$want" "$got"
      printf '%s\n' "$out" | sed 's/^/        /'
      fails=$((fails + 1))
    fi
  }

  echo "== public_claims_gate selftest"
  expect "current tree" pass "$ROOT_DIR"

  # Each retired phrase, planted alone, must fail.
  for entry in "${RETIRED[@]}"; do
    id="${entry%%|*}"; local sample="${entry##*|}"
    rm -rf "$w/plant"; mkdir -p "$w/plant/website/src"
    printf 'Sounio ships %s today.\n' "$sample" > "$w/plant/README.md"
    expect "planted: $id" fail "$w/plant"
  done

  # Escapes must pass.
  rm -rf "$w/esc"; mkdir -p "$w/esc/website/src/content/blog"
  printf 'v1.0.0-beta.5 was released 2026-03-05. <!-- %s -->\n' "$MARKER" > "$w/esc/README.md"
  printf 'The Cranelift JIT is no longer shipped (CHANGELOG 2.1.0).\n' > "$w/esc/TOUR.md"
  printf 'O Cranelift JIT não é mais distribuído.\n' > "$w/esc/website/src/a.mdx"
  printf '251/251 in May. Dated post.\n' > "$w/esc/website/src/content/blog/old.md"
  expect "escapes: marker, retirement, blog" pass "$w/esc"

  # The marker must be on the SAME line.
  rm -rf "$w/near"; mkdir -p "$w/near/website/src"
  printf '%s\nGPU acceleration for everyone.\n' "<!-- $MARKER -->" > "$w/near/README.md"
  expect "marker on the previous line does not escape" fail "$w/near"

  # The files as they were before the fix must fail.
  if git -C "$ROOT_DIR" rev-parse --verify -q "$ref^{commit}" >/dev/null; then
    mkdir -p "$w/ref"
    git -C "$ROOT_DIR" archive "$ref" -- "${PATHS[@]}" 2>/dev/null | tar -x -C "$w/ref"
    expect "tree at $ref (before the fix)" fail "$w/ref"
    echo "  -- what the gate reports on $ref (first 15 lines):"
    run_gate "$w/ref" 2>&1 | grep -E '^\s+\S+:[0-9]+:' | head -15 | sed 's/^/  /'
    echo "     total offending lines at $ref: $(run_gate "$w/ref" 2>&1 | grep -cE '^\s+\S+:[0-9]+:')"
  else
    echo "  SKIP  tree at $ref: commit not in this clone (needs fetch-depth: 0)"
  fi

  if ((fails)); then
    echo "public_claims_gate selftest: FAIL ($fails control(s) misbehaved)"
    return 1
  fi
  echo "public_claims_gate selftest: OK"
  return 0
}

case "${1:-}" in
  --selftest) selftest "${2:-}" ;;
  --root)     run_gate "$(cd "${2:?--root needs a directory}" && pwd)" ;;
  "")         run_gate "$ROOT_DIR" ;;
  *)          echo "usage: $0 [--root DIR | --selftest [REF]]" >&2; exit 2 ;;
esac
