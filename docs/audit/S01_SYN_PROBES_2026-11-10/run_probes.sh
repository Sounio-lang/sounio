#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
CASES="$HERE/cases.tsv"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-$HERE/runs/$RUN_ID}"
MADAROS_RAW_BIN="${MADAROS_RAW_BIN:-}"
MADAROS_BUILD_LOG="${MADAROS_BUILD_LOG:-}"
TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"

if [[ -z "$MADAROS_RAW_BIN" || ! -x "$MADAROS_RAW_BIN" ]]; then
    echo "error: MADAROS_RAW_BIN must name an executable current-source Madaros ELF" >&2
    exit 64
fi
MADAROS_RAW_BIN="$(cd "$(dirname "$MADAROS_RAW_BIN")" && pwd)/$(basename "$MADAROS_RAW_BIN")"
if [[ -z "$MADAROS_BUILD_LOG" || ! -s "$MADAROS_BUILD_LOG" ]]; then
    echo "error: MADAROS_BUILD_LOG must name the log that produced MADAROS_RAW_BIN" >&2
    exit 64
fi
MADAROS_BUILD_LOG="$(cd "$(dirname "$MADAROS_BUILD_LOG")" && pwd)/$(basename "$MADAROS_BUILD_LOG")"

SOURCE_HEAD="$(git -C "$ROOT" rev-parse HEAD)"
SOURCE_STATUS="$(git -C "$ROOT" status --porcelain --untracked-files=normal)"
SOURCE_TREE="clean"
if [[ -n "$SOURCE_STATUS" ]]; then
    SOURCE_TREE="dirty"
fi
WORKTREE_DIFF_SHA="$(git -C "$ROOT" diff --binary | sha256sum | cut -d' ' -f1)"
INDEX_DIFF_SHA="$(git -C "$ROOT" diff --cached --binary | sha256sum | cut -d' ' -f1)"

if [[ -e "$OUT_DIR" ]]; then
    echo "error: refusing to reuse existing OUT_DIR: $OUT_DIR" >&2
    exit 73
fi

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/artifacts" "$OUT_DIR/home-madaros" \
    "$OUT_DIR/home-lean" "$OUT_DIR/tmp-madaros" "$OUT_DIR/tmp-lean" \
    "$OUT_DIR/attestation"
cd "$ROOT"

sha256_file() {
    if [[ -f "$1" ]]; then
        sha256sum "$1" | cut -d' ' -f1
    else
        printf '%s' '-'
    fi
}

MADAROS_SHA="$(sha256_file "$MADAROS_RAW_BIN")"
LEAN_BIN="$ROOT/bin/souc-lean-single-x86_64"
if [[ ! -x "$LEAN_BIN" ]]; then
    echo "error: lean_single ELF is missing or not executable: $LEAN_BIN" >&2
    exit 74
fi
LEAN_SHA="$(sha256_file "$LEAN_BIN")"
TIMEOUT_PATH="$(command -v "$TIMEOUT_BIN")"

printf '%s\n' "$SOURCE_STATUS" >"$OUT_DIR/attestation/source-status.txt"
git -C "$ROOT" diff --binary >"$OUT_DIR/attestation/source-worktree.patch"
git -C "$ROOT" diff --cached --binary >"$OUT_DIR/attestation/source-index.patch"
cp "$MADAROS_BUILD_LOG" "$OUT_DIR/attestation/madaros-build.log"

if ! bash "$ROOT/scripts/ci/canonical_compiler_gate.sh" \
    >"$OUT_DIR/attestation/lean-single-gate.stdout" \
    2>"$OUT_DIR/attestation/lean-single-gate.stderr"; then
    echo "error: canonical lean_single gate failed" >&2
    exit 75
fi
if ! MADAROS_RAW_BIN="$MADAROS_RAW_BIN" "$ROOT/bin/madaros" info \
    >"$OUT_DIR/attestation/madaros-info.stdout" \
    2>"$OUT_DIR/attestation/madaros-info.stderr"; then
    echo "error: Madaros identity probe failed" >&2
    exit 76
fi
ATTESTED_MADAROS="$(sed -n 's/^raw_elf:[[:space:]]*//p' \
    "$OUT_DIR/attestation/madaros-info.stdout")"
if [[ "$ATTESTED_MADAROS" != "$MADAROS_RAW_BIN" ]]; then
    echo "error: Madaros launcher did not attest the requested raw ELF" >&2
    exit 77
fi
set +e
env -i PATH=/usr/local/bin:/usr/bin:/bin LC_ALL=C \
    SOUNIO_SOUC_ENGINE=lean_single bash -x "$ROOT/bin/souc" --version \
    >"$OUT_DIR/attestation/lean-route.stdout" \
    2>"$OUT_DIR/attestation/lean-route.stderr"
set -e
if ! grep -Fq -- "exec $LEAN_BIN --version" \
    "$OUT_DIR/attestation/lean-route.stderr"; then
    echo "error: bin/souc did not route lean_single to the attested ELF" >&2
    exit 78
fi

cat >"$OUT_DIR/manifest.tsv" <<EOF
field	value
schema_version	1
run_id	$RUN_ID
source_head	$SOURCE_HEAD
source_tree	$SOURCE_TREE
worktree_diff_sha256	$WORKTREE_DIFF_SHA
index_diff_sha256	$INDEX_DIFF_SHA
source_status_sha256	$(sha256_file "$OUT_DIR/attestation/source-status.txt")
cases_sha256	$(sha256_file "$CASES")
runner_sha256	$(sha256_file "$HERE/run_probes.sh")
build_madaros_script_sha256	$(sha256_file "$ROOT/scripts/ci/build_modular_madaros.sh")
madaros_main_source_sha256	$(sha256_file "$ROOT/self-hosted/compiler/main.sio")
lean_single_source_sha256	$(sha256_file "$ROOT/self-hosted/compiler/lean_single.sio")
bin_souc_sha256	$(sha256_file "$ROOT/bin/souc")
bin_madaros_sha256	$(sha256_file "$ROOT/bin/madaros")
resolve_souc_sha256	$(sha256_file "$ROOT/scripts/lib/resolve_souc.sh")
resolve_madaros_sha256	$(sha256_file "$ROOT/scripts/lib/resolve_madaros.sh")
madaros_path	$MADAROS_RAW_BIN
madaros_sha256	$MADAROS_SHA
madaros_build_log_sha256	$(sha256_file "$OUT_DIR/attestation/madaros-build.log")
madaros_science_boundary_mode	off
madaros_science_boundary_scope	syntax-probe-only-no-claim-contract
lean_single_path	$LEAN_BIN
lean_single_sha256	$LEAN_SHA
hostname	$(hostname)
uname	$(uname -srm)
slurm_job_id	${SLURM_JOB_ID:-not-slurm}
timeout_path	$TIMEOUT_PATH
timeout_sha256	$(sha256_file "$TIMEOUT_PATH")
EOF

RESULTS="$OUT_DIR/results.tsv"
printf '%s\n' $'rule_id\tcase_id\tmode\tengine\tcompiler_path\tcompiler_sha256\tsource_head\tsource_tree\tfixture_sha256\texpected_stdout_sha256\tfront_end_rc\texecution_rc\tfront_stdout_sha256\tfront_stderr_sha256\texecution_stdout_sha256\texecution_stderr_sha256\tartifact_sha256\tclassification\tcoverage\tlog_dir\tnote' >"$RESULTS"

case_count=0
failure_count=0
declare -A seen_rules=()

while IFS=$'\t' read -r rule_id case_id mode fixture expected_stdout coverage note; do
    [[ "$rule_id" == "rule_id" ]] && continue
    [[ -z "$rule_id" ]] && continue
    case_count=$((case_count + 1))
    if [[ -n "${seen_rules[$rule_id]:-}" ]]; then
        echo "error: duplicate rule in cases.tsv: $rule_id" >&2
        exit 65
    fi
    seen_rules[$rule_id]=1
    source_file="$ROOT/$fixture"
    if [[ ! -f "$source_file" ]]; then
        echo "error: missing fixture: $fixture" >&2
        exit 66
    fi
    if [[ "$mode" != "check" && "$mode" != "run" ]]; then
        echo "error: invalid mode '$mode' for $case_id" >&2
        exit 67
    fi
    expected_file="-"
    expected_sha="-"
    if [[ "$mode" == "run" ]]; then
        expected_file="$ROOT/$expected_stdout"
        if [[ ! -f "$expected_file" ]]; then
            echo "error: missing expected stdout: $expected_stdout" >&2
            exit 70
        fi
        expected_sha="$(sha256_file "$expected_file")"
    elif [[ "$expected_stdout" != "-" ]]; then
        echo "error: check case must use '-' for expected stdout: $case_id" >&2
        exit 71
    fi
    fixture_sha="$(sha256_file "$source_file")"

    for engine in madaros lean_single; do
        log_dir="$OUT_DIR/logs/$case_id/$engine"
        mkdir -p "$log_dir"
        artifact="$OUT_DIR/artifacts/${case_id}-${engine}.elf"
        front_stdout="$log_dir/front.stdout"
        front_stderr="$log_dir/front.stderr"
        execution_stdout="$log_dir/execution.stdout"
        execution_stderr="$log_dir/execution.stderr"
        : >"$execution_stdout"
        : >"$execution_stderr"
        rm -f -- "$artifact"

        if [[ "$engine" == "madaros" ]]; then
            compiler_path="$MADAROS_RAW_BIN"
            compiler_sha="$MADAROS_SHA"
            home_dir="$OUT_DIR/home-madaros"
            tmp_dir="$OUT_DIR/tmp-madaros"
            engine_env=(SOUNIO_SOUC_ENGINE=madaros MADAROS_RAW_BIN="$MADAROS_RAW_BIN")
            compiler_driver="$ROOT/bin/madaros"
            compiler_driver_args=(--science-boundary off)
        else
            compiler_path="$LEAN_BIN"
            compiler_sha="$LEAN_SHA"
            home_dir="$OUT_DIR/home-lean"
            tmp_dir="$OUT_DIR/tmp-lean"
            engine_env=(SOUNIO_SOUC_ENGINE=lean_single)
            compiler_driver="$ROOT/bin/souc"
            compiler_driver_args=()
        fi

        set +e
        if [[ "$mode" == "check" ]]; then
            env -i PATH=/usr/local/bin:/usr/bin:/bin LC_ALL=C HOME="$home_dir" \
                TMPDIR="$tmp_dir" SOUNIO_STDLIB_PATH="$ROOT/stdlib" \
                "${engine_env[@]}" "$TIMEOUT_BIN" 180 "$compiler_driver" \
                "${compiler_driver_args[@]}" \
                check "$source_file" >"$front_stdout" 2>"$front_stderr"
            front_rc=$?
            execution_rc="-"
        else
            env -i PATH=/usr/local/bin:/usr/bin:/bin LC_ALL=C HOME="$home_dir" \
                TMPDIR="$tmp_dir" SOUNIO_STDLIB_PATH="$ROOT/stdlib" \
                "${engine_env[@]}" "$TIMEOUT_BIN" 180 "$compiler_driver" \
                "${compiler_driver_args[@]}" \
                compile "$source_file" -o "$artifact" >"$front_stdout" 2>"$front_stderr"
            front_rc=$?
            execution_rc="-"
            if [[ $front_rc -eq 0 && -s "$artifact" ]]; then
                chmod +x -- "$artifact"
                if [[ $? -eq 0 && -x "$artifact" ]]; then
                    env -i PATH=/usr/local/bin:/usr/bin:/bin LC_ALL=C HOME="$home_dir" \
                        TMPDIR="$tmp_dir" "$TIMEOUT_BIN" 60 "$artifact" \
                        >"$execution_stdout" 2>"$execution_stderr"
                    execution_rc=$?
                else
                    execution_rc=126
                fi
            fi
        fi
        set -e

        classification="WITNESS_PASS"
        if [[ $front_rc -ne 0 ]]; then
            classification="FAIL_FRONT_END"
        elif [[ "$mode" == "run" && ! -x "$artifact" ]]; then
            classification="FAIL_NO_ARTIFACT"
        elif [[ "$mode" == "run" && "$execution_rc" != "0" ]]; then
            classification="FAIL_EXECUTION"
        elif [[ "$mode" == "run" ]] && ! cmp -s -- "$expected_file" "$execution_stdout"; then
            classification="FAIL_STDOUT_BYTES"
        fi
        if [[ "$classification" != "WITNESS_PASS" ]]; then
            failure_count=$((failure_count + 1))
        fi

        rel_log_dir="${log_dir#"$ROOT/"}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$rule_id" "$case_id" "$mode" "$engine" "$compiler_path" \
            "$compiler_sha" "$SOURCE_HEAD" "$SOURCE_TREE" "$fixture_sha" \
            "$expected_sha" "$front_rc" "$execution_rc" "$(sha256_file "$front_stdout")" \
            "$(sha256_file "$front_stderr")" "$(sha256_file "$execution_stdout")" \
            "$(sha256_file "$execution_stderr")" "$(sha256_file "$artifact")" \
            "$classification" "$coverage" "$rel_log_dir" "$note" >>"$RESULTS"
    done
done <"$CASES"

if [[ $case_count -ne 16 || ${#seen_rules[@]} -ne 16 ]]; then
    echo "error: expected exactly 16 rules, found $case_count" >&2
    exit 68
fi
for n in $(seq -f '%03g' 1 16); do
    rule="SYN-$n"
    if [[ -z "${seen_rules[$rule]:-}" ]]; then
        echo "error: missing rule: $rule" >&2
        exit 69
    fi
done

printf 'cases\t%s\nfailures\t%s\nresults\t%s\n' "$case_count" "$failure_count" "$RESULTS"
if [[ $failure_count -ne 0 ]]; then
    exit 1
fi
