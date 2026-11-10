#!/usr/bin/env bash
# Structural caller controls: no compiler or scientific calculation is executed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/collision/a" "$WORK/collision/b" "$WORK/diagnostic" "$WORK/empty" "$WORK/fragment" "$WORK/positive" "$WORK/nested" "$WORK/hanging"
printf '```sio\nfn bad() { INVALID }\n```\n' > "$WORK/collision/a/shared.md"
printf '```sio\nfn good() { 0 }\n```\n' > "$WORK/collision/b/shared.md"
printf '```sio\nfn bad() { INVALID }\n```\n' > "$WORK/diagnostic/unique.md"
printf '```sio\nlet value = 0\n```\n' > "$WORK/fragment/fragment.md"
printf '```sio\nfn good() { 0 }\n```\n' > "$WORK/positive/good.md"
printf '```sio\nfn bad() { INVALID }\n```sio\nfn good() { 0 }\n```\n' > "$WORK/nested/nested.md"
printf '```sio\nfn good() { 0 }\n```\n' > "$WORK/hanging/hanging.md"
cat > "$WORK/checker" <<'CHECKER'
#!/usr/bin/env bash
case "${STUB_MODE:-normal}" in
  diagnostic) echo 'parse error: INVALID'; echo 'check: FAILED'; exit 0 ;;
  positive_error) echo 'check: OK'; echo 'error[E137]: INVALID'; exit 0 ;;
  silent) exit 0 ;;
  fatal_positive) echo "check: OK"; echo "segmentation fault (core dumped)"; exit 0 ;;
  hanging) sleep 5; echo "check: OK"; exit 0 ;;
  failed_positive) echo 'check: OK'; exit 1 ;;
esac
if grep -q INVALID "$2"; then echo 'check: FAILED'; exit 1; fi
echo 'check: OK'
CHECKER
chmod +x "$WORK/checker"
expect_reject() {
  if SOUC="$WORK/checker" bash "$ROOT/scripts/ci/check_doc_snippets.sh" "$1" > "$WORK/out" 2>&1; then
    echo "SELFTEST_FAIL: accepted $1" >&2; exit 1
  fi
}
expect_reject "$WORK/collision"
grep -Fq '1 pass, 1 fail, 0 skip, 2 total' "$WORK/out"
expect_reject "$WORK/nested"
grep -Fq "nested Sounio fence" "$WORK/out"
export STUB_MODE=hanging SNIPPET_TIMEOUT_SECONDS=1
expect_reject "$WORK/hanging"
grep -Fq "0 pass, 1 fail" "$WORK/out"
unset STUB_MODE SNIPPET_TIMEOUT_SECONDS
expect_reject "$WORK/empty"
expect_reject "$WORK/fragment"
for mode in diagnostic positive_error silent failed_positive fatal_positive; do
  export STUB_MODE="$mode"
  expect_reject "$WORK/diagnostic"
done
unset STUB_MODE
SOUC="$WORK/checker" bash "$ROOT/scripts/ci/check_doc_snippets.sh" "$WORK/positive" > "$WORK/out" 2>&1
grep -Fxq 'GATE_MEASURED assertions=1' "$WORK/out"
echo DOC_SNIPPET_CHECKER_SELFTEST_PASS
# Exercise explicit contracts in a separate synthetic repository.
SAMPLE="$WORK/sample-repo"
mkdir -p "$SAMPLE/scripts/ci" "$SAMPLE/docs/training/rosetta" "$SAMPLE/website/src/content/docs" "$SAMPLE/tests/run-pass" "$SAMPLE/scripts/lib"
cp "$ROOT/scripts/ci/check_doc_snippets.sh" "$SAMPLE/scripts/ci/check_doc_snippets.sh"
cp "$ROOT/scripts/lib/classify_compile.sh" "$SAMPLE/scripts/lib/classify_compile.sh"
cp "$ROOT/scripts/lib/resolve_souc.sh" "$SAMPLE/scripts/lib/resolve_souc.sh"
printf 'fn good() { 0 }\n' > "$WORK/good.sio"
printf '```sio\nfn good() { 0 }\n```\n' > "$SAMPLE/docs/training/rosetta/sample.md"
hash="$(sha256sum "$WORK/good.sio" | cut -d' ' -f1)"
contract="$SAMPLE/scripts/ci/doc_snippet_contracts.tsv"
printf 'docs/training/rosetta/sample.md\t0\t%s\tstandalone\t-\t-\tpositive\t-\n' "$hash" > "$contract"
SOUC="$WORK/checker" bash "$SAMPLE/scripts/ci/check_doc_snippets.sh" > "$WORK/contract.log" 2>&1
grep -Fq 'SELECTED_SAMPLE_ONLY snippet_checks=1' "$WORK/contract.log"
cp "$contract" "$WORK/contract.good"
reject_contract() {
  if SOUC="$WORK/checker" bash "$SAMPLE/scripts/ci/check_doc_snippets.sh" > "$WORK/contract.log" 2>&1; then echo 'SELFTEST_FAIL: bad contract accepted'; exit 1; fi
}
cat "$WORK/contract.good" >> "$contract"
reject_contract
cp "$WORK/contract.good" "$contract"
printf '```sio\nfn changed() { 0 }\n```\n' > "$SAMPLE/docs/training/rosetta/sample.md"
reject_contract
grep -Fq 'snippet hash drift' "$WORK/contract.log"
printf '```sio\nfn good() { 0 }\n```\n```sio\nfn added() { 0 }\n```\n' > "$SAMPLE/docs/training/rosetta/sample.md"
reject_contract
grep -Fq 'missing selected-document contract' "$WORK/contract.log"
printf '```sio\nfn good() { 0 }\n```\n' > "$SAMPLE/docs/training/rosetta/sample.md"
cp "$WORK/good.sio" "$SAMPLE/tests/run-pass/linked.sio"
printf 'docs/training/rosetta/sample.md\t0\t%s\tlinked-fixture\ttests/run-pass/linked.sio\t%s\tpositive\t-\n' "$hash" "$hash" > "$contract"
SOUC="$WORK/checker" bash "$SAMPLE/scripts/ci/check_doc_snippets.sh" > "$WORK/contract.log" 2>&1
grep -Fq 'FIXTURE_CHECK_PASS (DISPLAYED SNIPPET NOT VALIDATED)' "$WORK/contract.log"
grep -Fq 'snippet_checks=0 fixture_checks=1' "$WORK/contract.log"
printf 'docs/training/rosetta/sample.md\t0\t%s\tcontextual\t-\t-\tnone\tRequires prior context\n' "$hash" > "$contract"
reject_contract
grep -Fq 'no measured checks' "$WORK/contract.log"
echo DOC_SNIPPET_CONTRACT_SELFTEST_PASS
# Intentional-negative contracts require an exact diagnostic and refusal status.
cat > "$WORK/refusal-checker" <<'CHECKER'
#!/usr/bin/env bash
echo 'error[E036]: confidence bound is not tight'
case "${REFUSAL_MODE:-good}" in good) exit 1;; zero) exit 0;; wrong) echo 'check: OK'; exit 1;; timeout) exit 124;; crash) exit 139;; fatal) echo "segmentation fault (core dumped)"; exit 1;; oom) echo Killed; exit 1;; esac
CHECKER
chmod +x "$WORK/refusal-checker"
printf 'docs/training/rosetta/sample.md\t0\t%s\tnegative\t-\t-\tnegative\tE036\n' "$hash" > "$contract"
SOUC="$WORK/refusal-checker" bash "$SAMPLE/scripts/ci/check_doc_snippets.sh" > "$WORK/contract.log" 2>&1
grep -Fq 'SNIPPET_CHECK_PASS kind=negative' "$WORK/contract.log"
for mode in zero wrong timeout crash fatal oom; do
  if REFUSAL_MODE="$mode" SOUC="$WORK/refusal-checker" bash "$SAMPLE/scripts/ci/check_doc_snippets.sh" > "$WORK/contract.log" 2>&1; then echo "SELFTEST_FAIL: invalid negative $mode accepted"; exit 1; fi
done
printf 'docs/training/rosetta/sample.md\t0\t%s\tlinked-fixture\ttests/run-pass/linked.sio\t%s\tnegative\tconfidence|epsilon|ε|E036|bound is not tight\n' "$hash" "$hash" > "$contract"
SOUC="$WORK/refusal-checker" bash "$SAMPLE/scripts/ci/check_doc_snippets.sh" > "$WORK/contract.log" 2>&1
grep -Fq 'FIXTURE_CHECK_PASS (DISPLAYED SNIPPET NOT VALIDATED)' "$WORK/contract.log"
echo DOC_SNIPPET_NEGATIVE_CONTRACT_SELFTEST_PASS

if SOUC="$WORK/missing-compiler" bash "$ROOT/scripts/ci/check_doc_snippets.sh" "$WORK/positive" > "$WORK/missing.log" 2>&1; then echo SELFTEST_FAIL_missing_compiler_fell_back; exit 1; fi
grep -Fq "SOUC_BIN is set but cannot be used" "$WORK/missing.log"
echo DOC_SNIPPET_CANONICAL_RESOLVER_SELFTEST_PASS
