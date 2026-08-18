#!/bin/bash
# Sounio Validation CI/CD Script
#
# This script integrates check_sounio.sh into CI/CD pipelines
# Usage in CI:
#   - GitHub Actions: Add as a step
#   - GitLab CI: Add as a job script
#   - Jenkins: Add as a pipeline step
#   - Local pre-commit: Run before commits

set -e

# Colors for CI output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VALIDATOR_SCRIPT="check_sounio.sh"
MAX_ERRORS=0  # Fail CI if errors > 0
SKIP_PATTERNS=("*.disabled" "*.backup" "*.old")  # Files to skip
# Directories containing intentionally invalid Sounio (error tests, compile-fail tests)
SKIP_DIRS=("tests/error_audit" "tests/compile-fail" "tests/ui" "archive")

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Function to print success message
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Function to print error message
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to print warning message
print_warning() {
    echo -e "${YELLOW}⚠  $1${NC}"
}

# Function to find all .sio files in the repository
find_sounio_files() {
    local files=()
    
    # Use find to get all .sio files
    while IFS= read -r -d '' file; do
        # Check if file should be skipped
        local skip=false
        for pattern in "${SKIP_PATTERNS[@]}"; do
            if [[ "$file" == $pattern ]]; then
                skip=true
                break
            fi
        done
        
        # Check directory skip patterns
        for dir_pattern in "${SKIP_DIRS[@]}"; do
            if [[ "$file" == *"$dir_pattern"* ]]; then
                skip=true
                break
            fi
        done
        if [ "$skip" = false ]; then
            files+=("$file")
        fi
    done < <(find . -name "*.sio" -type f -print0 2>/dev/null || true)
    
    echo "${files[@]}"
}

# Function to validate a single file
validate_file() {
    local file="$1"
    
    echo -e "\n${BLUE}Checking:${NC} $file"
    
    if [ ! -f "$file" ]; then
        print_warning "File not found: $file"
        return 1
    fi
    
    # Run the validator
    local output
    output=$(bash "$VALIDATOR_SCRIPT" "$file" 2>&1)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # Check for warnings
        if grep -q "⚠" <<<"$output"; then
            print_warning "File has warnings (but no errors)"
            echo "$output" | grep -A2 -B2 "⚠"
            return 0  # Warnings don't fail CI by default
        else
            print_success "File passed validation"
            return 0
        fi
    else
        print_error "File failed validation"
        echo "$output" | tail -20  # Show last 20 lines of output
        return 1
    fi
}

# Function to run validation on all files
validate_all_files() {
    print_header "SOUNIO VALIDATION - CI/CD"
    
    # Check if validator script exists
    if [ ! -f "$VALIDATOR_SCRIPT" ]; then
        print_error "Validator script not found: $VALIDATOR_SCRIPT"
        echo "Please ensure check_sounio.sh is in the current directory"
        exit 1
    fi
    
    # Make sure validator is executable
    chmod +x "$VALIDATOR_SCRIPT"
    
    # Find all Sounio files
    print_header "Finding Sounio files"
    local files
    files=$(find_sounio_files)
    local file_count=$(echo "$files" | wc -w)
    
    if [ "$file_count" -eq 0 ]; then
        print_warning "No .sio files found to validate"
        return 0
    fi
    
    echo "Found $file_count .sio file(s) to validate"
    
    # Validate each file
    print_header "Validating files"
    local error_count=0
    local warning_count=0
    local success_count=0
    
    for file in $files; do
        if validate_file "$file"; then
            success_count=$((success_count + 1))
        else
            error_count=$((error_count + 1))
        fi
    done
    
    # Summary
    print_header "VALIDATION SUMMARY"
    echo "Total files: $file_count"
    echo -e "${GREEN}Passed: $success_count${NC}"
    
    if [ $error_count -gt 0 ]; then
        echo -e "${RED}Failed: $error_count${NC}"
    else
        echo -e "${GREEN}Failed: $error_count${NC}"
    fi
    
    # Determine CI result
    if [ $error_count -gt $MAX_ERRORS ]; then
        print_error "CI validation failed: $error_count error(s) exceed maximum ($MAX_ERRORS)"
        echo -e "\n${RED}========================================${NC}"
        echo -e "${RED}  CI PIPELINE FAILED${NC}"
        echo -e "${RED}========================================${NC}"
        return 1
    else
        print_success "CI validation passed!"
        echo -e "\n${GREEN}========================================${NC}"
        echo -e "${GREEN}  CI PIPELINE PASSED${NC}"
        echo -e "${GREEN}========================================${NC}"
        return 0
    fi
}

# Function to create GitHub Actions workflow file
create_github_actions_workflow() {
    local workflow_dir=".github/workflows"
    mkdir -p "$workflow_dir"
    
    cat > "$workflow_dir/sounio-validation.yml" << 'EOF'
name: Sounio Validation

on:
  push:
    branches: [ main, master, develop ]
  pull_request:
    branches: [ main, master, develop ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday at midnight

jobs:
  validate:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
      
    - name: Set up Sounio environment
      run: |
        # Make validator executable
        chmod +x check_sounio.sh
        
        # Display Sounio version info if available
        if [ -f "bin/souc" ]; then
          echo "Sounio compiler found"
          bin/souc --version || true
        fi
    
    - name: Run Sounio validation
      run: |
        # Run the CI validation script
        bash scripts/ci/sounio_validation.sh
        
    - name: Upload validation report
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: sounio-validation-report
        path: |
          validation_report.txt
        retention-days: 7
EOF
    
    print_success "Created GitHub Actions workflow: $workflow_dir/sounio-validation.yml"
}

# Function to create GitLab CI configuration
create_gitlab_ci_config() {
    cat > ".gitlab-ci.yml" << 'EOF'
stages:
  - validate

sounio-validation:
  stage: validate
  image: ubuntu:latest
  script:
    - apt-get update && apt-get install -y bash
    - chmod +x check_sounio.sh
    - bash scripts/ci/sounio_validation.sh
  artifacts:
    when: always
    paths:
      - validation_report.txt
    expire_in: 1 week
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
    - if: $CI_COMMIT_TAG
EOF
    
    print_success "Created GitLab CI configuration: .gitlab-ci.yml"
}

# Function to create pre-commit hook
create_pre_commit_hook() {
    local hook_dir=".git/hooks"
    if [ ! -d "$hook_dir" ]; then
        print_error "Not a git repository"
        return 1
    fi
    
    cat > "$hook_dir/pre-commit" << 'EOF'
#!/bin/bash
# Sounio pre-commit validation hook

set -e

echo "🔍 Running Sounio validation..."

# Get staged .sio files
FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.sio$')

if [ -z "$FILES" ]; then
    echo "No .sio files staged for commit"
    exit 0
fi

ERRORS=0

for file in $FILES; do
    if [ ! -f "$file" ]; then
        continue
    fi
    
    echo "\nChecking: $file"
    
    if bash check_sounio.sh "$file" 2>/dev/null; then
        echo "✅ $file passed"
    else
        echo "❌ $file failed"
        ERRORS=$((ERRORS + 1))
    fi

done

if [ $ERRORS -gt 0 ]; then
    echo "\n❌ $ERRORS file(s) failed validation"
    echo "Please fix the errors before committing"
    echo "Run './check_sounio.sh <file>' for details"
    exit 1
else
    echo "\n✅ All staged .sio files passed validation"
    exit 0
fi
EOF
    
    chmod +x "$hook_dir/pre-commit"
    print_success "Created pre-commit hook: $hook_dir/pre-commit"
}

# Function to generate validation report
generate_report() {
    local report_file="validation_report.txt"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    echo "Sounio Validation Report" > "$report_file"
    echo "=======================" >> "$report_file"
    echo "Generated: $timestamp" >> "$report_file"
    echo "" >> "$report_file"
    
    # Find all files
    local files
    files=$(find_sounio_files)
    
    for file in $files; do
        echo "File: $file" >> "$report_file"
        echo "----------------------------------------" >> "$report_file"
        
        if bash "$VALIDATOR_SCRIPT" "$file" >> "$report_file" 2>&1; then
            echo "Status: PASS" >> "$report_file"
        else
            echo "Status: FAIL" >> "$report_file"
        fi
        
        echo "" >> "$report_file"
    done
    
    print_success "Generated validation report: $report_file"
}

# Main function
main() {
    local command="${1:-validate}"
    
    case "$command" in
        "validate")
            validate_all_files
            ;;
        "github")
            create_github_actions_workflow
            ;;
        "gitlab")
            create_gitlab_ci_config
            ;;
        "pre-commit")
            create_pre_commit_hook
            ;;
        "report")
            generate_report
            ;;
        "help" | "--help" | "-h")
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  validate     - Run validation on all .sio files (default)"
            echo "  github       - Create GitHub Actions workflow"
            echo "  gitlab       - Create GitLab CI configuration"
            echo "  pre-commit   - Create git pre-commit hook"
            echo "  report       - Generate validation report"
            echo "  help         - Show this help message"
            ;;
        *)
            print_error "Unknown command: $command"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"