#!/usr/bin/env bash
# Pireus Heartbeat Monitor - read-only cluster state collection
# Exit codes: 0=success, 1=Slurm unavailable, 2=command failure, 3=invalid mode

set -euo pipefail

# Configuration
RECEIPT_DIR="${PIREUS_HEARTBEAT_RECEIPT_DIR:-/tmp/pireus-heartbeat}"
DRY_RUN="${PIREUS_HEARTBEAT_DRY_RUN:-false}"
# Default evidence paths: CLUSTER-side paths outside the runner checkout
# These prove actual Pireus execution on the cluster, not just repo contents
EVIDENCE_DIRS="${PIREUS_EVIDENCE_DIRS:-/workspace/.wt/pireus-integration-20260906/tools/pireus/continuity/validation}"

# Evidence directory configuration (colon-separated)
IFS=':' read -ra EVIDENCE_PATHS <<< "$EVIDENCE_DIRS"

# Freshness threshold (seconds) - 26 hours
FRESHNESS_THRESHOLD_HOURS=26
FRESHNESS_THRESHOLD_SECONDS=$((FRESHNESS_THRESHOLD_HOURS * 3600))

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
        
        # Recent job history (last 26 hours) with WorkDir
        echo "--- Recent Job History (26h window) ---"
        if check_command sacct; then
            sacct -S now-26hours --format=JobID,JobName%40,User,NodeList,Start,End,State,ExitCode,Elapsed,WorkDir%80 2>/dev/null || echo "sacct query failed"
        else
            echo "sacct not available"
        fi
        echo ""
        
        # Extract Pireus/Inkling WorkDirs for evidence collection
        echo "--- Pireus/Inkling Working Directories ---"
        if check_command sacct; then
            sacct -S now-26hours --format=WorkDir%200 --noheader 2>/dev/null | \
                grep -iE 'pireus|inkling|sglang' | sort -u || echo "No Pireus/Inkling work directories found"
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

get_directory_freshness() {
    local dir_path="$1"
    
    if [[ ! -d "$dir_path" ]]; then
        echo "null"
        return
    fi
    
    # Find newest file mtime in the directory (recursively)
    local newest_mtime
    newest_mtime=$(find "$dir_path" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1)
    
    if [[ -z "$newest_mtime" ]]; then
        echo "null"
        return
    fi
    
    # Convert to integer (strip decimal)
    newest_mtime=${newest_mtime%.*}
    
    # Calculate age in seconds
    local current_time
    current_time=$(date +%s)
    local age_seconds=$((current_time - newest_mtime))
    
    # Return JSON object with mtime, age, and human-readable timestamp
    local newest_timestamp
    newest_timestamp=$(date -u -d "@${newest_mtime}" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "unknown")
    
    jq -n --argjson mtime "$newest_mtime" \
          --argjson age_seconds "$age_seconds" \
          --arg timestamp "$newest_timestamp" \
          --argjson threshold "$FRESHNESS_THRESHOLD_SECONDS" \
          '{mtime: $mtime, age_seconds: $age_seconds, timestamp: $timestamp, is_fresh: ($age_seconds < $threshold)}'
}

collect_evidence_state() {
    local output_file="$1"
    local json_file="$2"
    local slurm_output="$3"
    
    log "Collecting evidence directory state..."
    
    local evidence_found="false"
    local evidence_json_file="${json_file}.evidence.tmp"
    
    # Initialize evidence JSON array
    echo '[]' > "$evidence_json_file"
    
    # Extract Pireus/Inkling WorkDirs from Slurm output
    local workdir_list=()
    if [[ -f "$slurm_output" ]]; then
        while IFS= read -r workdir; do
            workdir=$(echo "$workdir" | xargs)  # trim whitespace
            if [[ -n "$workdir" && -d "$workdir" ]]; then
                workdir_list+=("$workdir")
            fi
        done < <(grep -A 100 "Pireus/Inkling Working Directories" "$slurm_output" | grep -v "^---" | grep -v "Working Directories" | grep -v "sacct not available" | grep -v "^$" | grep -iE 'pireus|inkling|sglang' || true)
    fi
    
    # Combine configured evidence paths with discovered WorkDirs
    local all_paths=("${EVIDENCE_PATHS[@]}" "${workdir_list[@]}")
    
    {
        echo "=== Pireus Evidence Directories ==="
        echo ""
        echo "Freshness threshold: ${FRESHNESS_THRESHOLD_HOURS}h (${FRESHNESS_THRESHOLD_SECONDS}s)"
        echo ""
        
        for evidence_path in "${all_paths[@]}"; do
            if [[ -d "$evidence_path" ]]; then
                echo "--- Directory: $evidence_path ---"
                echo "Exists: yes"
                
                # Count files
                local file_count
                file_count=$(find "$evidence_path" -type f 2>/dev/null | wc -l)
                echo "File count: $file_count"
                
                # Get freshness
                local freshness_json
                freshness_json=$(get_directory_freshness "$evidence_path")
                local newest_timestamp
                newest_timestamp=$(echo "$freshness_json" | jq -r '.timestamp')
                local age_seconds
                age_seconds=$(echo "$freshness_json" | jq -r '.age_seconds')
                local is_fresh
                is_fresh=$(echo "$freshness_json" | jq -r '.is_fresh')
                
                echo "Newest file: $newest_timestamp (age: ${age_seconds}s, fresh: $is_fresh)"
                
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
                
                # Append to JSON array with freshness
                jq --arg path "$evidence_path" \
                   --argjson count "$file_count" \
                   --argjson freshness "$freshness_json" \
                   '. + [{path: $path, exists: true, file_count: $count, freshness: $freshness}]' \
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
    else
        # Check if there are any Pireus-related jobs in the last 26h (active or completed)
        # Look in the job history section, not the entire output
        local has_recent_pireus_job=false
        if grep -A 100 "Recent Job History" "$slurm_output" 2>/dev/null | grep -iE 'pireus|inkling|sglang' &>/dev/null; then
            has_recent_pireus_job=true
        fi
        
        # Check if there are currently running or pending Pireus jobs
        local has_active_pireus_job=false
        if grep -A 50 "Active Jobs" "$slurm_output" 2>/dev/null | grep -iE 'pireus|inkling|sglang' | grep -E 'RUNNING|PENDING|CONFIGURING' &>/dev/null; then
            has_active_pireus_job=true
        fi
        
        # Check evidence freshness (any directory with fresh evidence)
        local has_fresh_evidence=false
        if [[ -f "$json_file" ]]; then
            has_fresh_evidence=$(jq -r '[.evidence_directories[]? | select(.exists == true) | .freshness.is_fresh] | any' "$json_file" 2>/dev/null || echo "false")
        fi
        
        # FRESHNESS rule: RUNNING requires active/recent job AND fresh evidence
        if [[ "$has_active_pireus_job" == "true" && "$has_fresh_evidence" == "true" ]]; then
            verdict="RUNNING"
            reason="Pireus jobs active and evidence fresh (< ${FRESHNESS_THRESHOLD_HOURS}h)"
        elif [[ "$has_active_pireus_job" == "true" && "$has_fresh_evidence" == "false" ]]; then
            verdict="RUNNING"
            reason="Pireus jobs active but evidence stale (≥ ${FRESHNESS_THRESHOLD_HOURS}h) - possible issue"
        elif [[ "$has_recent_pireus_job" == "true" && "$has_fresh_evidence" == "true" ]]; then
            verdict="RUNNING"
            reason="Pireus jobs ended within ${FRESHNESS_THRESHOLD_HOURS}h and evidence fresh"
        # Check for failed jobs
        elif grep -E 'FAILED|CANCELLED|TIMEOUT|NODE_FAIL|BOOT_FAIL|OUT_OF_MEMORY' "$slurm_output" &>/dev/null && \
             grep -iE 'pireus|inkling|sglang' "$slurm_output" &>/dev/null; then
            verdict="FAILED_JOBS"
            reason="Pireus-related jobs with failure states detected in last ${FRESHNESS_THRESHOLD_HOURS}h"
        # Check for stale evidence without recent jobs
        elif [[ "$evidence_found" == "true" && "$has_fresh_evidence" == "false" ]]; then
            verdict="IDLE"
            reason="Evidence directories present but stale (≥ ${FRESHNESS_THRESHOLD_HOURS}h) and no recent jobs"
        # Evidence exists but no recent jobs
        elif [[ "$evidence_found" == "true" ]]; then
            verdict="IDLE"
            reason="Evidence directories present but no Pireus jobs in last ${FRESHNESS_THRESHOLD_HOURS}h"
        # No evidence and no jobs
        else
            verdict="NO_EVIDENCE"
            reason="No Pireus-related jobs and no evidence directories found"
        fi
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
        echo "- **RUNNING**: Pireus job active/recent (< ${FRESHNESS_THRESHOLD_HOURS}h) AND evidence fresh (< ${FRESHNESS_THRESHOLD_HOURS}h)"
        echo "- **IDLE**: Evidence directories exist but stale (≥ ${FRESHNESS_THRESHOLD_HOURS}h) or no recent jobs"
        echo "- **FAILED_JOBS**: Pireus-related jobs with failure states in last ${FRESHNESS_THRESHOLD_HOURS}h"
        echo "- **NO_EVIDENCE**: No Pireus jobs and no evidence directories found"
        echo "- **NO_SLURM**: Slurm commands unavailable"
        echo ""
        
        echo "## Evidence Directories"
        echo ""
        local evidence_count
        evidence_count=$(jq -r '.evidence_directories | length' "$json_file")
        if [[ "$evidence_count" -gt 0 ]]; then
            echo "| Path | Exists | Files | Newest | Age | Fresh |"
            echo "|------|--------|-------|--------|-----|-------|"
            jq -r '.evidence_directories[] | 
                if .exists == true then
                    "| \(.path) | \(.exists) | \(.file_count) | \(.freshness.timestamp // "N/A") | \(.freshness.age_seconds // 0)s | \(.freshness.is_fresh // false) |"
                else
                    "| \(.path) | \(.exists) | N/A | N/A | N/A | N/A |"
                end' "$json_file"
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
        echo "--- Pireus/Inkling Working Directories ---"
        echo "No WorkDirs in dry-run mode"
        echo ""
    } > "$output_file"
    
    # Create base JSON
    jq -n --arg hostname "$(hostname)" --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{hostname: $hostname, timestamp: $timestamp, dry_run: true}' > "$json_file"
    
    # Collect evidence state (reuse actual function)
    local evidence_found
    evidence_found=$(collect_evidence_state "$output_file" "$json_file" "$output_file")
    
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
    
    # Collect evidence state (passing slurm output for WorkDir extraction)
    local evidence_found
    evidence_found=$(collect_evidence_state "$output_file" "$json_file" "$output_file")
    
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
