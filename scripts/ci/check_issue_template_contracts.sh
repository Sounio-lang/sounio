#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail=0

has_real_ripgrep() {
  command -v rg >/dev/null 2>&1 && rg --version 2>/dev/null | grep -qi '^ripgrep '
}

matches_pattern() {
  local pattern="$1"
  local file="$2"
  if has_real_ripgrep; then
    rg -n -- "$pattern" "$file" >/dev/null 2>&1
  else
    grep -n -E -- "$pattern" "$file" >/dev/null 2>&1
  fi
}

require_file() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo "Missing required issue governance file: $file"
    fail=1
  fi
}

require_pattern() {
  local pattern="$1"
  local file="$2"
  local message="$3"
  if ! matches_pattern "$pattern" "$file"; then
    echo "$message"
    fail=1
  fi
}

require_file ".github/ISSUE_TEMPLATE/bug_report.yml"
require_file ".github/ISSUE_TEMPLATE/feature_request.yml"
require_file ".github/ISSUE_TEMPLATE/release_readiness_regression.yml"
require_file ".github/ISSUE_TEMPLATE/config.yml"
require_file ".github/workflows/issue-pr-automation.yml"
require_file ".github/workflows/release-gate.yml"

require_pattern 'id:[[:space:]]+release_scope' ".github/ISSUE_TEMPLATE/bug_report.yml" \
  "bug_report.yml must include id: release_scope"
require_pattern 'id:[[:space:]]+severity' ".github/ISSUE_TEMPLATE/bug_report.yml" \
  "bug_report.yml must include id: severity"
require_pattern 'id:[[:space:]]+confidence' ".github/ISSUE_TEMPLATE/bug_report.yml" \
  "bug_report.yml must include id: confidence"
require_pattern 'id:[[:space:]]+effort' ".github/ISSUE_TEMPLATE/bug_report.yml" \
  "bug_report.yml must include id: effort"
require_pattern 'id:[[:space:]]+traceability_ids' ".github/ISSUE_TEMPLATE/bug_report.yml" \
  "bug_report.yml must include id: traceability_ids"

require_pattern 'id:[[:space:]]+release_scope' ".github/ISSUE_TEMPLATE/feature_request.yml" \
  "feature_request.yml must include id: release_scope"
require_pattern 'id:[[:space:]]+severity' ".github/ISSUE_TEMPLATE/feature_request.yml" \
  "feature_request.yml must include id: severity"
require_pattern 'id:[[:space:]]+confidence' ".github/ISSUE_TEMPLATE/feature_request.yml" \
  "feature_request.yml must include id: confidence"
require_pattern 'id:[[:space:]]+effort' ".github/ISSUE_TEMPLATE/feature_request.yml" \
  "feature_request.yml must include id: effort"
require_pattern 'id:[[:space:]]+traceability_ids' ".github/ISSUE_TEMPLATE/feature_request.yml" \
  "feature_request.yml must include id: traceability_ids"
require_pattern 'id:[[:space:]]+acceptance' ".github/ISSUE_TEMPLATE/feature_request.yml" \
  "feature_request.yml must include id: acceptance"

require_pattern 'id:[[:space:]]+gate_command' ".github/ISSUE_TEMPLATE/release_readiness_regression.yml" \
  "release_readiness_regression.yml must include id: gate_command"
require_pattern 'id:[[:space:]]+failing_command' ".github/ISSUE_TEMPLATE/release_readiness_regression.yml" \
  "release_readiness_regression.yml must include id: failing_command"
require_pattern 'id:[[:space:]]+failing_output' ".github/ISSUE_TEMPLATE/release_readiness_regression.yml" \
  "release_readiness_regression.yml must include id: failing_output"

require_pattern 'blank_issues_enabled:[[:space:]]+false' ".github/ISSUE_TEMPLATE/config.yml" \
  "config.yml must disable blank issues"
require_pattern 'discussions' ".github/ISSUE_TEMPLATE/config.yml" \
  "config.yml must link GitHub Discussions"
require_pattern 'security/policy' ".github/ISSUE_TEMPLATE/config.yml" \
  "config.yml must link security policy"

require_pattern 'pull_request_target' ".github/workflows/issue-pr-automation.yml" \
  "issue-pr-automation workflow must run on pull_request_target"
require_pattern 'actions/stale@' ".github/workflows/issue-pr-automation.yml" \
  "issue-pr-automation workflow must use actions/stale"
require_pattern 'Release' ".github/workflows/release-gate.yml" \
  "release-gate workflow must include release readiness"

if [[ $fail -ne 0 ]]; then
  exit 1
fi

echo "Issue template contract check passed."
