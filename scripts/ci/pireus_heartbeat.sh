#!/usr/bin/env bash
# Pireus Heartbeat Monitor - read-only cluster state collection
# Exit codes: 0=success, 1=Slurm unavailable, 2=command failure, 3=invalid mode

set -euo pipefail

# Configuration
RECEIPT_DIR="${PIREUS_HEARTBEAT_RECEIPT_DIR:-/tmp/pireus-heartbeat}"
DRY_RUN="${PIREUS_HEARTBEAT_DRY_RUN:-false}"
EVIDENCE_DIRS="${PIREUS_EVIDENCE_DIRS:-tools/pireus/evidence:tools/pireus/continuity/runtime:tools/pireus/continuity/recovered}"

# Evidence directory configuration (colon-separated)
IFS=':' read -ra EVIDENCE_PATHS <<< "$EVIDENCE_DIRS"

# Helper functions
log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

check_command() {
    if ! command -v "$1" &>/dev/null; then
        return 1
    fi
    return 0
}

collect_slurm_state() {
    local output_file="$1"
    local json_file="$2"
    
    log "Collecting Slurm state..."
    
    # Check if Slurm is available
    if ! check_command squeue; then
        log "WARNING: squeue not found - Slurm may not be available"
        echo "verdict: NO_SLURM" > "$output_file"
        return 1
    fi
    
    {
        echo "=== Slurm Queue State (all jobs) ==="
        echo "Collected at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Hostname: $(hostname)"
        echo ""
        
        # All jobs with detailed format - filter Pireus-related
        echo "--- Active Jobs ---"
        if squeue -a -o "%.18i %.40j %.12u %.10T %.10M %.6D %.18R %.15b %.50k" --noheader 2>/dev/null | \
           tee >(grep -iE 'pireus|inkling|sglang' || true) >(wc -l | xargs printf "Total jobs: %s\n" >&2) > /dev/null; then
            squeue -a -o "%.18i %.40j %.12u %.10T %.10M %.6D %.18R %.15b %.50k" --noheader
        else
            echo "No jobs in queue or squeue failed"
        fi
        echo ""
        
        # Recent job history (last 26 hours)
        echo "--- Recent Job History (26h window) ---"
        if check_command sacct; then
            sacct -S now-26hours --format=JobID,JobName%40,User,NodeList,Start,End,State,ExitCode,Elapsed 2>/dev/null || echo "sacct query failed"
        else
            echo "sacct not available"
        fi
        echo ""
        
        # Cluster info
        echo "--- Cluster Information ---"
        if check_command sinfo; then
            sinfo 2>/dev/null || echo "sinfo query failed"
        else
            echo "sinfo not available"
        fi
        echo ""
        
    } | tee "$output_file"
}

collect_gpu_state() {
    local output_file="$1"
    
    log "Collecting GPU state..."
    
    {
        echo "=== NVIDIA GPU State ==="
        echo ""
        
        if check_command nvidia-smi; then
            echo "--- GPU Status ---"
            nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu --format=csv 2>/dev/null || echo "nvidia-smi query failed"
            echo ""
            
            echo "--- Running Compute Processes ---"
            nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "No compute processes or query failed"
            echo ""
        else
            echo "nvidia-smi not available"
            echo ""
        fi
        
    } | tee -a "$output_file"
}

collect_evidence_state() {
    local output_file="$1"
    local json_file="$2"
    
    log "Collecting evidence directory state..."
    
    local evidence_found="false"
    local evidence_json_file="${json_file}.evidence.tmp"
    
    # Initialize evidence JSON array
    echo '[]' > "$evidence_json_file"
    
    {
        echo "=== Pireus Evidence Directories ==="
        echo ""
        
        for evidence_path in "${EVIDENCE_PATHS[@]}"; do
            if [[ -d "$evidence_path" ]]; then
                echo "--- Directory: $evidence_path ---"
                echo "Exists: yes"
                
                # Count files
                local file_count
                file_count=$(find "$evidence_path" -type f 2>/dev/null | wc -l)
                echo "File count: $file_count"
                
                # List recent files (last 7 days, limit to 50)
                echo "Recent files (last 7 days, max 50):"
                find "$evidence_path" -type f -mtime -7 -printf '%T+ %p %s bytes\n' 2>/dev/null | sort -r | head -50 || echo "  (none or find failed)"
                
                # Compute SHA256 checksums for key files
                echo ""
                echo "SHA256 checksums (key files):"
                find "$evidence_path" -type f \( -name "*.txt" -o -name "*.json" -o -name "*.py" -o -name "*.sio" \) -mtime -7 2>/dev/null | \
                    head -20 | while read -r file; do
                    if [[ -f "$file" ]]; then
                        sha256sum "$file" 2>/dev/null || echo "  $file: (checksum failed)"
                    fi
                done
                
                echo ""
                evidence_found="true"
                
                # Append to JSON array
                jq --arg path "$evidence_path" --argjson count "$file_count" \
                    '. + [{path: $path, exists: true, file_count: $count}]' \
                    "$evidence_json_file" > "${evidence_json_file}.2"
                mv "${evidence_json_file}.2" "$evidence_json_file"
            else
                echo "--- Directory: $evidence_path ---"
                echo "Exists: no"
                echo ""
                
                # Append to JSON array
                jq --arg path "$evidence_path" \
                    '. + [{path: $path, exists: false}]' \
                    "$evidence_json_file" > "${evidence_json_file}.2"
                mv "${evidence_json_file}.2" "$evidence_json_file"
            fi
        done
        
    } >> "$output_file"
    
    # Read evidence array from file and update main JSON
    local evidence_array
    evidence_array=$(cat "$evidence_json_file")
    jq --argjson evidence "$evidence_array" '. + {evidence_directories: $evidence}' "$json_file" > "${json_file}.tmp"
    mv "${json_file}.tmp" "$json_file"
    rm -f "$evidence_json_file"
    
    echo "$evidence_found"
}

determine_verdict() {
    local slurm_output="$1"
    local evidence_found="$2"
    local json_file="$3"
    
    local verdict="UNKNOWN"
    local reason=""
    
    # Check for NO_SLURM verdict
    if grep -q "verdict: NO_SLURM" "$slurm_output" 2>/dev/null; then
        verdict="NO_SLURM"
        reason="Slurm commands not available"
    # Check for failed jobs (non-zero exit codes in recent history)
    elif grep -E 'FAILED|CANCELLED|TIMEOUT|NODE_FAIL|BOOT_FAIL|OUT_OF_MEMORY' "$slurm_output" &>/dev/null && \
         grep -iE 'pireus|inkling|sglang' "$slurm_output" &>/dev/null; then
        verdict="FAILED_JOBS"
        reason="Pireus-related jobs with failure states detected in last 26 hours"
    # Check for running Pireus jobs
    elif grep -iE 'pireus|inkling|sglang' "$slurm_output" &>/dev/null && \
         grep -E 'RUNNING|PENDING|CONFIGURING' "$slurm_output" &>/dev/null; then
        verdict="RUNNING"
        reason="Pireus-related jobs currently running or pending"
    # Check for evidence but no active jobs
    elif [[ "$evidence_found" == "true" ]]; then
        verdict="IDLE"
        reason="Evidence directories present but no active Pireus jobs"
    # No evidence and no jobs
    else
        verdict="NO_EVIDENCE"
        reason="No Pireus-related jobs and no evidence directories found"
    fi
    
    log "Verdict: $verdict - $reason"
    
    # Update JSON with verdict
    jq --arg verdict "$verdict" --arg reason "$reason" \
       '. + {verdict: $verdict, reason: $reason}' "$json_file" > "${json_file}.tmp"
    mv "${json_file}.tmp" "$json_file"
    
    echo "$verdict"
}

generate_markdown_summary() {
    local json_file="$1"
    local md_file="$2"
    
    log "Generating markdown summary..."
    
    local verdict
    verdict=$(jq -r '.verdict' "$json_file")
    local reason
    reason=$(jq -r '.reason' "$json_file")
    local hostname
    hostname=$(jq -r '.hostname' "$json_file")
    local timestamp
    timestamp=$(jq -r '.timestamp' "$json_file")
    
    {
        echo "# Pireus Heartbeat Report"
        echo ""
        echo "**Verdict:** \`$verdict\`"
        echo ""
        echo "**Reason:** $reason"
        echo ""
        echo "---"
        echo ""
        echo "## System Information"
        echo ""
        echo "- **Hostname:** $hostname"
        echo "- **Timestamp:** $timestamp"
        echo "- **Collection Mode:** $(if [[ "$DRY_RUN" == "true" ]]; then echo "DRY_RUN"; else echo "LIVE"; fi)"
        echo ""
        
        echo "## Verdict Rules"
        echo ""
        echo "- **RUNNING**: Pireus-related jobs (pireus/inkling/sglang) are RUNNING or PENDING"
        echo "- **IDLE**: Evidence directories exist but no active Pireus jobs"
        echo "- **FAILED_JOBS**: Pireus-related jobs with failure states in last 26h"
        echo "- **NO_EVIDENCE**: No Pireus jobs and no evidence directories found"
        echo "- **NO_SLURM**: Slurm commands unavailable"
        echo ""
        
        echo "## Evidence Directories"
        echo ""
        local evidence_count
        evidence_count=$(jq -r '.evidence_directories | length' "$json_file")
        if [[ "$evidence_count" -gt 0 ]]; then
            echo "| Path | Exists | File Count |"
            echo "|------|--------|------------|"
            jq -r '.evidence_directories[] | "| \(.path) | \(.exists) | \(.file_count // "N/A") |"' "$json_file"
        else
            echo "*No evidence directories configured*"
        fi
        echo ""
        
        echo "## Full Report"
        echo ""
        echo "See the uploaded artifact for complete Slurm queue state, GPU status, and evidence checksums."
        echo ""
        
    } > "$md_file"
    
    log "Markdown summary written to $md_file"
}

dry_run_mode() {
    log "DRY_RUN mode enabled - simulating collection without Slurm"
    
    mkdir -p "$RECEIPT_DIR"
    
    local output_file="$RECEIPT_DIR/slurm_state.txt"
    local json_file="$RECEIPT_DIR/receipt.json"
    local md_file="$RECEIPT_DIR/receipt.md"
    
    # Create mock data
    {
        echo "=== DRY_RUN MODE ==="
        echo "Collected at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Hostname: $(hostname)"
        echo ""
        echo "This is a dry-run simulation. No actual Slurm queries were executed."
        echo ""
    } > "$output_file"
    
    # Create base JSON
    jq -n --arg hostname "$(hostname)" --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{hostname: $hostname, timestamp: $timestamp, dry_run: true}' > "$json_file"
    
    # Collect evidence state (reuse actual function)
    local evidence_found
    evidence_found=$(collect_evidence_state "$output_file" "$json_file")
    
    # Create verdict
    determine_verdict "$output_file" "$evidence_found" "$json_file"
    
    # Generate markdown
    generate_markdown_summary "$json_file" "$md_file"
    
    log "DRY_RUN complete. Receipt in $RECEIPT_DIR"
    cat "$md_file"
    exit 0
}

main() {
    log "Starting Pireus Heartbeat Monitor"
    
    # Handle dry-run mode
    if [[ "$DRY_RUN" == "true" ]]; then
        dry_run_mode
    fi
    
    # Create receipt directory
    mkdir -p "$RECEIPT_DIR"
    
    local output_file="$RECEIPT_DIR/slurm_state.txt"
    local json_file="$RECEIPT_DIR/receipt.json"
    local md_file="$RECEIPT_DIR/receipt.md"
    
    # Create base JSON with metadata
    jq -n --arg hostname "$(hostname)" --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{hostname: $hostname, timestamp: $timestamp}' > "$json_file"
    
    # Collect Slurm state
    local slurm_exit=0
    collect_slurm_state "$output_file" "$json_file" || slurm_exit=$?
    
    # Collect GPU state
    collect_gpu_state "$output_file" || true
    
    # Collect evidence state
    local evidence_found
    evidence_found=$(collect_evidence_state "$output_file" "$json_file")
    
    # Determine verdict
    determine_verdict "$output_file" "$evidence_found" "$json_file"
    
    # Generate markdown summary
    generate_markdown_summary "$json_file" "$md_file"
    
    log "Heartbeat collection complete. Receipt in $RECEIPT_DIR"
    
    # Print summary to stdout
    cat "$md_file"
    
    # Exit with Slurm availability code if Slurm was unavailable
    if [[ $slurm_exit -ne 0 ]]; then
        log "Exiting with code 1 (Slurm unavailable)"
        exit 1
    fi
    
    exit 0
}

# Entry point
main "$@"
