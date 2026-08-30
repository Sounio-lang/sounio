#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat >"$TMP/gh" <<'FAKE_GH'
#!/usr/bin/env bash
set -euo pipefail

json=""
jq_expr=""
if [[ "${FAKE_GH_FAIL:-0}" == "1" ]]; then
    echo "simulated gh failure" >&2
    exit 42
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --json)
            json="$2"
            shift 2
            ;;
        --jq)
            jq_expr="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

case "$json:$jq_expr" in
    state:*)
        echo "OPEN"
        ;;
    files:*)
        printf '%s\n' "${FAKE_PR_FILES:-self-hosted/ir/lower.sio}"
        ;;
    statusCheckRollup:*IN_PROGRESS*)
        echo "0"
        ;;
    statusCheckRollup:*FAILURE*)
        echo "0"
        ;;
    statusCheckRollup:*join*)
        if [[ "${FAKE_CHECKS:-impact_only}" == "impact_full" ]]; then
            printf 'Impact\nFull Test Suite\n'
        else
            printf 'Impact\n'
        fi
        ;;
    statusCheckRollup:*Full\ Test\ Suite*)
        if [[ "${FAKE_CHECKS:-impact_only}" == "impact_full" ]]; then
            echo "SUCCESS"
        else
            echo "null"
        fi
        ;;
    *)
        echo "unexpected fake gh query: json=$json jq=$jq_expr" >&2
        exit 97
        ;;
esac
FAKE_GH
chmod +x "$TMP/gh"

PATH="$TMP:$PATH" FAKE_PR_FILES='self-hosted/ir/lower.sio' FAKE_CHECKS=impact_only \
    "$ROOT/scripts/dev/pr_merge_ready.sh" 2144 >"$TMP/missing.out" 2>"$TMP/missing.err" && {
        echo "expected self-hosted compiler change to require Full Test Suite" >&2
        exit 1
    }

grep -q "Full Test Suite" "$TMP/missing.err"
grep -q "absent" "$TMP/missing.err"

PATH="$TMP:$PATH" FAKE_PR_FILES='self-hosted/ir/lower.sio' FAKE_CHECKS=impact_full \
    "$ROOT/scripts/dev/pr_merge_ready.sh" 2144 >"$TMP/ready.out" 2>"$TMP/ready.err"

grep -q "MERGE-READY" "$TMP/ready.out"

PATH="$TMP:$PATH" FAKE_GH_FAIL=1 \
    "$ROOT/scripts/dev/pr_merge_ready.sh" 2144 >"$TMP/gh_fail.out" 2>"$TMP/gh_fail.err" && {
        echo "expected gh failure to make readiness fail" >&2
        exit 1
    }

grep -q "simulated gh failure" "$TMP/gh_fail.err"

echo "PR_MERGE_READY_SELFTEST_OK"
