run_test() {
    local file="$1"
    local idx="$2"
    local output_file="$TEST_TMP/result_$idx.json"
    local basename
    basename="$(basename "$file")"
    local rel_file="${file#$ROOT_DIR/}"
    
    local is_run_pass=false
    local is_compile_fail=false
    local is_typecheck_fail=false
    local is_ignored=false
    local is_check_only=false
    local is_known_failure=false
    local is_flaky=false
    local is_vacuous_baseline=false
    local timeout_val=30
    local skip_if=""
    local requires=""
    local known_reason=""
    local unknown_expect=""
    # //@ xfail-expect: / //@ xfail-exit: pin HOW a //@ known-failure fails,
    # in every mode (filtered or not, manifest loaded or not). See the
    # enforcement block after the run for the contract.
    local -a xfail_expect=()
    local xfail_exit=""
    local xfail_error=""
    local raw_exit=""
    local inline_known_failure=false
    
    # Parse annotations
    while IFS= read -r line; do
        if [[ ! "$line" =~ ^[[:space:]]*//@\  && ! "$line" =~ ^[[:space:]]*//([[:space:]]|$) && ! "$line" =~ ^[[:space:]]*$ ]]; then
            break
        fi
        # Only a line that STARTS with `//@ ` is an annotation. The case arms
        # below match by substring, so without this a header comment that
        # merely mentions one -- "The engine-forcing //@ ignore is removed",
        # "Deliberately no `//@ requires: gpu`" -- was read as that annotation
        # (measured 2026-10-06: it ignored ffi_system_exec.sio and overwrote
        # gpu_kernel_lane_loop.sio's `requires: madaros`).
        [[ "$line" =~ ^[[:space:]]*//@\  ]] || continue
        # Fail closed on invented stdout assertions. `expect-stdout-contains`
        # was silently ignored because the harness only extracted
        # `expect-stdout:`; the same hole would swallow `expected-output` or
        # `expect-stdout-has`. Key extraction is identifier-only; the payload
        # is still read by parameter expansion below (the vacuous-regex bug).
        local expect_line="${line%"${line##*[![:space:]]}"}"
        expect_line="${expect_line#"${expect_line%%[![:space:]]*}"}"
        expect_line="${expect_line%$'\r'}"
        if [[ "$expect_line" == "//@ expect"* || "$expect_line" == "//@ expected"* ]]; then
            local expect_key="${expect_line#//@ }"
            expect_key="${expect_key%%:*}"
            expect_key="${expect_key%% *}"
            case "$expect_key" in
                expect-stdout|expect-stdout-contains) ;;
                *)
                    if [[ -z "$unknown_expect" ]]; then
                        unknown_expect="$expect_key"
                    fi
                    ;;
            esac
        fi
        case "$line" in
            *"//@ run-pass"*) is_run_pass=true ;;
            *"//@ compile-fail"*) is_compile_fail=true ;;
            *"//@ typecheck-fail"*) is_typecheck_fail=true ;;
            *"//@ ignore"*) is_ignored=true ;;
            *"//@ check-only"*) is_check_only=true ;;
            *"//@ xfail-expect:"*)
                if [[ "$line" =~ xfail-expect:[[:space:]]*(.*[^[:space:]])[[:space:]]*$ ]]; then
                    xfail_expect+=("${BASH_REMATCH[1]%$'\r'}")
                else
                    xfail_error="${xfail_error:-empty //@ xfail-expect: pattern}"
                fi
                ;;
            *"//@ xfail-exit:"*)
                # A process status is 0-255. Anything else could never match,
                # so it would read as a pin while pinning nothing: refuse it.
                # Canonicalised (base 10, no leading zeros) for the comparison.
                if [[ "$line" =~ xfail-exit:[[:space:]]*([0-9]{1,3})[[:space:]]*$ ]] && ((10#${BASH_REMATCH[1]} <= 255)); then
                    xfail_exit="$((10#${BASH_REMATCH[1]}))"
                else
                    xfail_error="${xfail_error:-//@ xfail-exit: needs one exit code in 0-255}"
                fi
                ;;
            *"//@ known-failure"*) 
                is_known_failure=true
                inline_known_failure=true
                if [[ "$line" =~ known-failure:[[:space:]]*(.+) ]]; then
                    known_reason="${BASH_REMATCH[1]}"
                fi
                ;;
            *"//@ flaky"*) is_flaky=true ;;
            *"//@ timeout"*) 
                if [[ "$line" =~ ([0-9]+) ]]; then
                    timeout_val="${BASH_REMATCH[1]}"
                fi
                ;;
            *"//@ skip-if"*)
                if [[ "$line" =~ skip-if:[[:space:]]*(.+) ]]; then
                    skip_if="${BASH_REMATCH[1]}"
                fi
                ;;
            *"//@ requires"*)
                if [[ "$line" =~ requires:[[:space:]]*(.+) ]]; then
                    requires="${BASH_REMATCH[1]}"
                fi
                ;;
        esac
    done < "$file"

    if [[ -n "${KNOWN_FAILURE_MAP[$rel_file]:-}" ]]; then
        is_known_failure=true
        known_reason="${known_reason:-hardened diagnostics blocker manifest}"
    fi

    # See the VACUOUS_BASELINE_MAP loading comment above. Membership is
    # recorded regardless of exit code, same as is_known_failure -- whether it
    # counts as a tolerated failure (vxfail) or a "now passes, shrink the
    # baseline" notice (vxpas) is decided once exit_code is known below.
    if [[ "$VACUOUS_REFRESH" != "1" && -n "${VACUOUS_BASELINE_MAP[$rel_file]:-}" ]]; then
        is_vacuous_baseline=true
    fi

    # Check filter
    if ! test_matches_filter "$basename"; then
        return
    fi

    if [[ -n "$unknown_expect" ]]; then
        echo "{\"status\":\"fail\",\"category\":\"fail\",\"name\":\"$basename\",\"relfile\":\"$rel_file\",\"time\":0,\"output\":\"unknown annotation: $unknown_expect (expected: expect-stdout|expect-stdout-contains)\",\"idx\":$idx}" > "$output_file"
        return
    fi

    # An xfail pin is a claim about a declared known failure. Malformed, or
    # stranded on a test that declares no known failure, it would pin nothing,
    # so refuse it instead of letting it read as protection.
    if [[ -z "$xfail_error" ]] && ! $inline_known_failure && { ((${#xfail_expect[@]} > 0)) || [[ -n "$xfail_exit" ]]; }; then
        xfail_error="//@ xfail-expect/xfail-exit without an inline //@ known-failure"
    fi
    if [[ -n "$xfail_error" ]]; then
        echo "{\"status\":\"fail\",\"category\":\"fail\",\"name\":\"$basename\",\"relfile\":\"$rel_file\",\"time\":0,\"output\":\"bad xfail pin: $xfail_error\",\"idx\":$idx}" > "$output_file"
        return
    fi
    
    # Check ignored
    if $is_ignored; then
        echo "{\"status\":\"skip\",\"reason\":\"ignored\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"
        return
    fi
    
    # Check skip-if
    if [[ -n "$skip_if" ]]; then
        case "$skip_if" in
            no-gpu) [[ -z "${SOUNIO_GPU_AVAILABLE:-}" ]] && { echo "{\"status\":\"skip\",\"reason\":\"skip-if:no-gpu\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"; return; } ;;
            no-llvm) [[ -z "${SOUNIO_LLVM_AVAILABLE:-}" ]] && { echo "{\"status\":\"skip\",\"reason\":\"skip-if:no-llvm\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"; return; } ;;
            ci-only) [[ -n "${CI:-}" ]] && { echo "{\"status\":\"skip\",\"reason\":\"skip-if:ci-only\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"; return; } ;;
            # An unrecognized skip-if value must not fall through silently: that
            # would let a typo (e.g. `no-gpu` misspelled) run the test unguarded
            # while the annotation reads as if it were gating something -- the
            # same "guard that asserts nothing" defect this file exists to remove.
            *)
                echo "{\"status\":\"fail\",\"category\":\"fail\",\"name\":\"$basename\",\"output\":\"unknown skip-if: $skip_if (expected: no-gpu|no-llvm|ci-only)\",\"idx\":$idx}" > "$output_file"
                return
                ;;
        esac
    fi

    # Check requires
    if [[ -n "$requires" ]]; then
        case "$requires" in
            gpu) [[ -z "${SOUNIO_GPU_AVAILABLE:-}" ]] && { echo "{\"status\":\"skip\",\"reason\":\"requires:gpu\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"; return; } ;;
            llvm) [[ -z "${SOUNIO_LLVM_AVAILABLE:-}" ]] && { echo "{\"status\":\"skip\",\"reason\":\"requires:llvm\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"; return; } ;;
            # `requires: madaros` — feature lives only in the modular Madaros compiler
            # (check.sio), not in the lean_single bootstrap that builds the suite's
            # stage2 binary. Skipped unless SOUNIO_MADAROS_AVAILABLE is set (a future
            # Madaros-based test job sets it). Tracked: Madaros-official migration.
            madaros) [[ -z "${SOUNIO_MADAROS_AVAILABLE:-}" ]] && { echo "{\"status\":\"skip\",\"reason\":\"requires:madaros\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"; return; } ;;
            # `requires: lean_single` — the mirror of the above: the feature lives
            # only in the lean_single bootstrap, so the test must NOT run on the
            # Madaros job. Refinement subtyping is the case that needed this:
            # lean_single evaluates the predicate and emits E208/E209, while
            # Madaros carries the diagnostic (E042, "value does not satisfy the
            # refinement predicate") but never reaches it, because it relates a
            # refinement type to its base in NEITHER direction -- measured
            # 2026-09-03: `fn f(x: Positive) -> i32 { x }` is E008 and
            # `let p: Positive = 0` is E001, a bare type mismatch where the
            # predicate should have spoken. Without this arm such a test would
            # fail on the Madaros job for a reason unrelated to what it asserts.
            lean_single)
                # Gated on what will ACTUALLY run, not on a declaration. The
                # madaros arm above trusts SOUNIO_MADAROS_AVAILABLE, which is a
                # statement of intent; a local checkout with a built Madaros and
                # that variable unset runs Madaros anyway, and the test would
                # then fail with "missing error: ..." as though the compiler were
                # wrong instead of the test being inapplicable. Second clause is
                # bin/souc's own rule, restated once: it picks Madaros when a
                # local artifact exists and no explicit engine was handed in.
                if [[ -n "${SOUNIO_MADAROS_AVAILABLE:-}" ]] \
                   || { [[ -z "${SOUNIO_TEST_SOUC_BIN:-}" ]] && [[ -x "$ROOT_DIR/artifacts/self-hosted/madaros" ]]; }; then
                    echo "{\"status\":\"skip\",\"reason\":\"requires:lean_single\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"
                    return
                fi
                ;;
            # `requires: slow` — tests that legitimately need minutes, not
            # seconds (full GRI-Mech kinetics integrations, PINN training
            # loops, a Lyapunov spectrum). The default Full Test Suite job
            # runs on a shared GHA runner where these routinely exceed their
            # own generous `//@ timeout:` -- not because they are wrong, but
            # because they are slow. Skipped unless SOUNIO_SLOW_TESTS_AVAILABLE
            # is set; the nightly slow-lane job sets it, with a much larger
            # job timeout budget.
            slow) [[ -z "${SOUNIO_SLOW_TESTS_AVAILABLE:-}" ]] && { echo "{\"status\":\"skip\",\"reason\":\"requires:slow\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"; return; } ;;
            # `requires: vendor-material` — the test reads third-party material
            # (vendor PDFs, datafiles) that the repository cannot carry, named one
            # path per line in its `.args` sidecar (repo-relative or absolute).
            # Skipped, naming the first missing path, unless every path exists;
            # then the header's mode (check-only) applies, and
            # scripts/ci/pireus_argv_tests_gate.sh runs it with that argv. A
            # missing sidecar is a test-authoring error, not a skip (#2820).
            vendor-material)
                local vm_args="${file}.args"
                if [[ ! -f "$vm_args" ]]; then
                    echo "{\"status\":\"fail\",\"category\":\"fail\",\"name\":\"$basename\",\"output\":\"requires: vendor-material but no .args sidecar names the material\",\"idx\":$idx}" > "$output_file"
                    return
                fi
                local vm_missing="" vm_line vm_path
                while IFS= read -r vm_line || [[ -n "$vm_line" ]]; do
                    [[ -z "$vm_line" ]] && continue
                    vm_path="$vm_line"; [[ "$vm_path" == /* ]] || vm_path="$ROOT_DIR/$vm_path"
                    if [[ ! -e "$vm_path" ]]; then vm_missing="$vm_line"; break; fi
                done < "$vm_args"
                if [[ -n "$vm_missing" ]]; then
                    echo "{\"status\":\"skip\",\"reason\":\"requires:vendor-material (missing $vm_missing)\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"
                    return
                fi
                ;;
            # An unrecognized requires value must not fall through silently: a typo
            # (e.g. `requires: madros`) would otherwise run the test against
            # whatever engine is present instead of being gated as intended, with
            # the annotation asserting nothing -- indistinguishable from the
            # vacuous-match defect this PR exists to remove.
            *)
                echo "{\"status\":\"fail\",\"category\":\"fail\",\"name\":\"$basename\",\"output\":\"unknown requires: $requires (expected: gpu|llvm|madaros|lean_single|slow|vendor-material)\",\"idx\":$idx}" > "$output_file"
                return
                ;;
        esac
    fi
    
    # Skip tests with no annotation
    if ! $is_run_pass && ! $is_compile_fail && ! $is_check_only && ! $is_typecheck_fail; then
        echo "{\"status\":\"skip\",\"reason\":\"no-annotation\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"
        return
    fi
    
    # Read expected patterns
    local expect_stdout=()
    local expect_stdout_contains=()
    local error_patterns=()
    while IFS= read -r line; do
        if [[ ! "$line" =~ ^[[:space:]]*//@\  && ! "$line" =~ ^[[:space:]]*//([[:space:]]|$) && ! "$line" =~ ^[[:space:]]*$ ]]; then
            break
        fi
        # Extraction by parameter expansion, not regex: the pattern that
        # follows "//@ expect-stdout: " / "//@ error-pattern: " often contains
        # regex metacharacters (`[`, `(`, ...), and quoting the whole =~
        # pattern (as this used to) makes bash treat it as a literal string
        # instead of a regex, so the capture group never captures and
        # BASH_REMATCH[1] is always empty -- every expect-stdout/error-pattern
        # assertion then matched vacuously. Parameter expansion has no
        # metacharacter class to get this wrong for either annotation.
        if [[ "$line" == "//@ expect-stdout: "* ]]; then
            expect_stdout+=("${line#*//@ expect-stdout: }")
        fi
        if [[ "$line" == "//@ expect-stdout-contains: "* ]]; then
            expect_stdout_contains+=("${line#*//@ expect-stdout-contains: }")
        fi
        if [[ "$line" == "//@ error-pattern: "* ]]; then
            error_patterns+=("${line#*//@ error-pattern: }")
        fi
    done < "$file"
    
    # Execute test
    local output=""
    local exit_code=0
    local test_output=""
    local start_time end_time
    
    start_time=$(date +%s)
    
    if $is_run_pass || $is_check_only; then
        if $is_check_only; then
            output=$(timeout "$timeout_val" "$SOUC_BIN" check "$file" 2>&1) || exit_code=$?
            raw_exit=$exit_code
            if [[ $exit_code -eq 124 ]]; then
                test_output="check timed out after ${timeout_val}s"
            elif [[ $exit_code -ne 0 ]]; then
                test_output="check exited $exit_code"
            fi
        else
            output=$(timeout "$timeout_val" "$SOUC_BIN" run "$file" 2>&1) || exit_code=$?
            raw_exit=$exit_code
            if [[ $exit_code -eq 124 ]]; then
                test_output="run timed out after ${timeout_val}s"
            elif [[ $exit_code -ne 0 ]]; then
                # Keep the numeric verdict AND a snippet of compiler/program
                # output. "run exited 1" alone hid compile-fail vs main().
                # No pipeline: pipefail + head -c would fail on long output.
                raw="${output//$'\n'/ | }"
                if ((${#raw} > 160)); then
                    snippet="${raw: -160}"
                else
                    snippet="$raw"
                fi
                test_output="run exited $exit_code"
                if [[ -n "$snippet" ]]; then
                    test_output="$test_output | $snippet"
                fi
            fi
        fi
        
        # Check expected stdout patterns.
        #
        # PIPEFAIL RULE (2026-08-17): never feed a captured string to a
        # verdict-carrying `grep -q` through `echo "$x" | grep -q ...`.
        # `grep -q` exits on first match and closes the pipe; under `set -o pipefail`
        # an `echo` still flushing a large output then fails the whole pipeline,
        # and `if ! ...` reads that as the pattern being absent. The here-string form
        # has no writer process to lose writes. Guarded by scripts/ci/sigpipe_hygiene_gate.sh.
        if [[ $exit_code -eq 0 ]]; then
            for pattern in "${expect_stdout[@]}"; do
                if ! grep -qF -- "$pattern" <<<"$output"; then
                    exit_code=1
                    test_output="missing stdout: $pattern"
                    break
                fi
            done
            if [[ $exit_code -eq 0 ]]; then
                for pattern in "${expect_stdout_contains[@]}"; do
                    if ! grep -qF -- "$pattern" <<<"$output"; then
                        exit_code=1
                        test_output="missing stdout contains: $pattern"
                        break
                    fi
                done
            fi
        fi
        
    elif $is_compile_fail; then
        local tmp_out
        local compile_exit_code=0
        local check_error_patterns=false
        tmp_out="$(mktemp /tmp/sounio-cf-XXXXXX.elf)"
        output=$(timeout "$timeout_val" "$SOUC_BIN" compile "$file" -o "$tmp_out" 2>&1) || compile_exit_code=$?
        raw_exit=$compile_exit_code
        rm -f "$tmp_out"

        if [[ $compile_exit_code -eq 124 ]]; then
            test_output="compile timed out after ${timeout_val}s"
            exit_code=1
        # Shells encode signal termination as 128 + signal. A crash is never
        # a valid compile-fail rejection, even when its output matches.
        elif [[ $compile_exit_code -ge 128 && $compile_exit_code -le 192 ]]; then
            local signal_num=$((compile_exit_code - 128))
            test_output="compile terminated by signal ${signal_num} (exit ${compile_exit_code})"
            exit_code=1
        elif [[ $compile_exit_code -ge 125 && $compile_exit_code -le 127 ]]; then
            test_output="compile harness exited ${compile_exit_code}"
            exit_code=1
        elif [[ $compile_exit_code -eq 0 ]] && grep -qF "typecheck: failed" <<<"$output"; then
            check_error_patterns=true
            exit_code=0
        elif [[ $compile_exit_code -eq 0 ]]; then
            test_output="expected compile failure but passed"
            exit_code=1
        else
            check_error_patterns=true
            exit_code=0
        fi

        if $check_error_patterns; then
            for pattern in "${error_patterns[@]}"; do
                if ! grep -qiF -- "$pattern" <<<"$output"; then
                    exit_code=1
                    test_output="missing error: $pattern"
                    break
                fi
            done
        fi
    elif $is_typecheck_fail; then
        # Proof-carrying tests: the illegal inference must be rejected by the
        # TYPE CHECKER (`souc check`), and the reason MUST be pinned via
        # //@ error-pattern. Running `check` (not `compile`) is deliberate:
        # `check` runs the boundary-preserving visibility/type pass, whereas
        # `compile` can "pass" on unrelated backend / missing-main failures
        # without ever exercising the guarantee (see audit 2026-07-24).
        # Parse the pinned pattern(s) locally rather than reusing the shared
        # error_patterns array above: this path runs `check`, not `compile`,
        # a distinct contract (see the audit 2026-07-24 note above) that
        # deserves its own local list rather than sharing state with the
        # compile-fail path.
        local tf_patterns=()
        local pline
        while IFS= read -r pline; do
            case "$pline" in
                *"//@ error-pattern: "*) tf_patterns+=("${pline#*//@ error-pattern: }") ;;
            esac
        done < <(head -n 20 "$file")

        output=$(timeout "$timeout_val" "$SOUC_BIN" check "$file" 2>&1) || exit_code=$?
        raw_exit=$exit_code
        if [[ $exit_code -eq 124 ]]; then
            test_output="check timed out after ${timeout_val}s"
        elif [[ $exit_code -eq 0 ]]; then
            test_output="expected typecheck failure but passed"
            exit_code=1
        else
            exit_code=0  # nonzero check exit == the expected rejection
            # A typecheck-fail test MUST pin the reason; no error-pattern is vacuous.
            if [[ ${#tf_patterns[@]} -eq 0 ]]; then
                exit_code=1
                test_output="typecheck-fail requires //@ error-pattern to pin the diagnostic"
            else
                for pattern in "${tf_patterns[@]}"; do
                    if ! grep -qiF -- "$pattern" <<<"$output"; then
                        exit_code=1
                        test_output="missing error: $pattern"
                        break
                    fi
                done
            fi
        fi
    fi

    end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # A manifest entry that pinned its failure mode (path|substring) only covers a
    # failure whose test_output matches; anything else is a fresh fail, not a repeat
    # of the audited one. Checked here, before is_known_failure is consulted below,
    # so unmatched entries fall straight through to the ordinary fail path.
    if $is_known_failure && [[ $exit_code -ne 0 ]]; then
        expected_reason="${KNOWN_FAILURE_REASON_MAP[$rel_file]:-}"
        if [[ -n "$expected_reason" ]] && ! grep -qF -- "$expected_reason" <<<"$test_output"; then
            is_known_failure=false
            test_output="known-failure reason mismatch: expected '$expected_reason', got: $test_output"
        fi
    fi

    # Inline xfail pins (//@ xfail-expect: TEXT, repeatable; //@ xfail-exit: N).
    # Unlike the manifest pin above, these are read from the test itself and
    # enforced in every mode, including --filter runs where the manifest is not
    # loaded. A declared known failure counts as xfail only if it fails THE
    # DECLARED WAY: the raw exit code of the souc invocation equals N (124 = the
    # harness timeout) and every TEXT occurs in the full compiler/program output.
    # Only for a timeout (raw exit 124, when souc printed nothing final) is the
    # harness verdict searched too; otherwise a pin could match text the harness
    # wrote itself ("run exited 139", "missing stdout: ..."). A crash, a
    # different exit code or a different failure message is a fresh FAIL; a
    # pass is still XPAS.
    if $is_known_failure && [[ $exit_code -ne 0 ]] && { ((${#xfail_expect[@]} > 0)) || [[ -n "$xfail_exit" ]]; }; then
        local xfail_mismatch=""
        if [[ -n "$xfail_exit" && "$raw_exit" != "$xfail_exit" ]]; then
            xfail_mismatch="exit ${raw_exit:-none}, expected $xfail_exit"
        fi
        if [[ -z "$xfail_mismatch" ]]; then
            local xfail_haystack="$output"
            if [[ "$raw_exit" == "124" ]]; then
                xfail_haystack="$output"$'\n'"$test_output"
            fi
            local xfail_pat
            for xfail_pat in "${xfail_expect[@]}"; do
                if ! grep -qF -- "$xfail_pat" <<<"$xfail_haystack"; then
                    xfail_mismatch="missing '$xfail_pat'"
                    break
                fi
            done
        fi
        if [[ -n "$xfail_mismatch" ]]; then
            is_known_failure=false
            test_output="known-failure expectation mismatch ($xfail_mismatch), got: $test_output"
        fi
    fi

    # Determine final status
    local status=""
    local category=""

    if [[ $exit_code -eq 0 ]]; then
        if $is_known_failure; then
            status="xpas"
            category="known-failure"
        elif $is_vacuous_baseline; then
            status="vxpas"
            category="vacuous-baseline"
        elif $is_flaky; then
            status="pass"
            category="flaky"
        else
            status="pass"
            category="pass"
        fi
    else
        if $is_known_failure; then
            status="xfail"
            category="known-failure"
        elif $is_vacuous_baseline; then
            status="vxfail"
            category="vacuous-baseline"
        elif $is_flaky; then
            status="fail"
            category="flaky"
        else
            status="fail"
            category="fail"
        fi
    fi

    # Parse agent witness from raw output (if present)
    local agent_witness=""
    agent_witness=$(echo "$output" | sed -n 's/^agent_witness=//p' | head -n 1)

    # Escape output for JSON
    local escaped_output
    escaped_output=$(echo "$test_output" | sed 's/"/\\"/g' | tr '\n' ' ' | sed 's/  */ /g' | head -c 200)

    local json="{\"status\":\"$status\",\"category\":\"$category\",\"name\":\"$basename\",\"relfile\":\"$rel_file\",\"time\":$duration,\"output\":\"$escaped_output\",\"idx\":$idx"
    if [[ -n "$agent_witness" ]]; then
        json="$json,\"agent_witness\":$agent_witness"
    fi
    json="$json}"
    echo "$json" > "$output_file"
    if [[ -n "${SOUNIO_TEST_RESULTS_DIR:-}" ]]; then
        printf '%s\n' "$output" > "$SOUNIO_TEST_RESULTS_DIR/result_$idx.log"
        cp "$output_file" "$SOUNIO_TEST_RESULTS_DIR/"
    fi
}

