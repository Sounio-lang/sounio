#!/usr/bin/env bash
# Structural QIR tape-order checks. Invalid fixtures must compile then fail at runtime.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/lib/resolve_souc.sh"
sounio_require_souc
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
work="$(mktemp -d "${TMPDIR:-/tmp}/qir-base-order.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for name in qir_order_rejection qir_repeat_measure_rejection qir_disjoint_gate_rejection qir_nonfinite_rejection qir_malformed_emit qir_base_valid qir_base_diagnostic qir_malformed_tape qir_rotation_valid qir_phase_valid; do
    case "$name" in
        qir_base_valid|qir_base_diagnostic|qir_malformed_tape|qir_rotation_valid|qir_phase_valid) input="$ROOT/tests/stdlib/quantum/$name.sio" ;;
        *) input="$ROOT/tests/stdlib/quantum/$name.sio.in" ;;
    esac
    cp "$input" "$work/$name.sio"
    if ! "$SOUC_BIN" compile "$work/$name.sio" -o "$work/$name.elf" >"$work/$name.compile.log" 2>&1 || [[ ! -s "$work/$name.elf" ]]; then
        echo "FAIL: $name did not compile" >&2
        cat "$work/$name.compile.log" >&2
        exit 1
    fi
    chmod +x "$work/$name.elf"
    rc=0
    "$work/$name.elf" >"$work/$name.runtime.log" 2>&1 || rc=$?
    case "$name" in
        qir_base_valid)
            if [[ $rc -ne 0 ]] || ! grep -q VALID_BELL_QIR_EMITTED "$work/$name.runtime.log" || ! awk '
                /call void @__quantum__qis__h__body/ { if (c || m || h) exit 1; h++ }
                /call void @__quantum__qis__cnot__body/ { if (h != 1 || m || c) exit 1; c++ }
                /call void @__quantum__qis__mz__body/ { if (c != 1 || m >= 2) exit 1; m++ }
                END { if (h != 1 || c != 1 || m != 2) exit 1 }
            ' "$work/$name.runtime.log"; then
                echo "FAIL: valid Bell QIR" >&2; cat "$work/$name.runtime.log" >&2; exit 1
            fi
            for required in \
                'call void @__quantum__qis__h__body(ptr null)' \
                'call void @__quantum__qis__cnot__body(ptr null, ptr inttoptr (i64 1 to ptr))' \
                'declare void @__quantum__qis__mz__body(ptr, ptr writeonly) #1' \
                'attributes #1 = { "irreversible" }' \
                '!llvm.module.flags = !{!0, !1, !2, !3}' \
                '!0 = !{i32 1, !"qir_major_version", i32 2}' \
                '!1 = !{i32 7, !"qir_minor_version", i32 0}' \
                '!2 = !{i32 1, !"dynamic_qubit_management", i1 false}' \
                '!3 = !{i32 1, !"dynamic_result_management", i1 false}'; do
                if ! grep -Fq "$required" "$work/$name.runtime.log"; then
                    echo "FAIL: valid Bell QIR missing $required" >&2; exit 1
                fi
            done ;;
        qir_phase_valid)
            if [[ $rc -ne 0 ]] || ! grep -Fq PHASE_QIR_EMITTED "$work/$name.runtime.log" || ! awk '
                /call void @__quantum__qis__h__body/ { if (m) exit 1; h++ }
                /call void @__quantum__qis__rz__body/ { if (h != 1 || m) exit 1; rz++ }
                /call void @__quantum__qis__mz__body/ { if (h != 2 || rz != 1) exit 1; m++ }
                END { if (h != 2 || rz != 1 || m != 1) exit 1 }
            ' "$work/$name.runtime.log"; then
                echo "FAIL: phase QIR" >&2; cat "$work/$name.runtime.log" >&2; exit 1
            fi
            grep -Fq "call void @__quantum__qis__rz__body(double 3.141592653589, ptr null)" "$work/$name.runtime.log" || { echo "FAIL: phase Rz operand" >&2; exit 1; } ;;
        qir_rotation_valid)
            if [[ $rc -ne 0 ]] || ! grep -Fq ROTATION_QIR_EMITTED "$work/$name.runtime.log" || ! awk '
                /call void @__quantum__qis__ry__body/ { if (rz || m || ry) exit 1; ry++ }
                /call void @__quantum__qis__rz__body/ { if (ry != 1 || m || rz) exit 1; rz++ }
                /call void @__quantum__qis__mz__body/ { if (rz != 1 || m) exit 1; m++ }
                END { if (ry != 1 || rz != 1 || m != 1) exit 1 }
            ' "$work/$name.runtime.log"; then
                echo "FAIL: rotation QIR" >&2; cat "$work/$name.runtime.log" >&2; exit 1
            fi
            grep -Fq "call void @__quantum__qis__ry__body(double 1.570796326794, ptr null)" "$work/$name.runtime.log" || { echo "FAIL: rotation Ry operand" >&2; exit 1; }
            grep -Fq "call void @__quantum__qis__rz__body(double 0.200000000000, ptr null)" "$work/$name.runtime.log" || { echo "FAIL: rotation Rz operand" >&2; exit 1; } ;;
        qir_nonfinite_rejection)
            if [[ $rc -ne 1 ]] || ! grep -Fq NONFINITE_PRECHECK_OK "$work/$name.runtime.log" || grep -q "QIR_EMITTED\|define i64 @qsoir" "$work/$name.runtime.log"; then
                echo "FAIL: non-finite rejection" >&2; cat "$work/$name.runtime.log" >&2; exit 1
            fi ;;
        qir_malformed_tape)
            if [[ $rc -ne 0 ]] || ! grep -q MALFORMED_TAPE_REJECTED "$work/$name.runtime.log"; then
                echo "FAIL: malformed tape diagnostics" >&2; cat "$work/$name.runtime.log" >&2; exit 1
            fi ;;
        qir_base_diagnostic)
            if [[ $rc -ne 0 ]] || ! grep -q QIR_BASE_DIAGNOSTICS_OK "$work/$name.runtime.log"; then
                echo "FAIL: order diagnostics" >&2; cat "$work/$name.runtime.log" >&2; exit 1
            fi ;;
        *)
            if [[ $rc -ne 1 ]] || grep -q "QIR_EMITTED\|define i64 @qsoir" "$work/$name.runtime.log"; then
                echo "FAIL: $name was not rejected (rc=$rc)" >&2; cat "$work/$name.runtime.log" >&2; exit 1
            fi ;;
    esac
    echo "PASS: $name"
done
if ! "$SOUC_BIN" compile "$ROOT/tests/stdlib/quantum/qir_output_labels.sio" -o "$work/labels.elf" >"$work/labels.compile.log" 2>&1 || [[ ! -s "$work/labels.elf" ]]; then
    echo 'FAIL: labels fixture did not compile' >&2; cat "$work/labels.compile.log" >&2; exit 1
fi
chmod +x "$work/labels.elf"
"$work/labels.elf" > "$work/labels.log"
awk '
/QIR_LABELS_EMPTY_BEGIN/ { section="empty"; next }
/QIR_LABELS_EMPTY_END/ { section=""; next }
/QIR_LABELS_MANY_BEGIN/ { section="many"; next }
/QIR_LABELS_MANY_END/ { section=""; next }
section == "empty" && /@qir_tuple_label = private constant \[2 x i8\] c"t\\00"/ { empty_label++ }
section == "empty" && /call void @__quantum__rt__tuple_record_output\(i64 0, ptr @qir_tuple_label\)/ { empty_tuple++ }
section == "empty" && /call void @__quantum__rt__result_record_output/ { bad=1 }
section == "many" && /@qir_tuple_label = private constant \[2 x i8\] c"t\\00"/ { many_label++ }
section == "many" && /call void @__quantum__rt__tuple_record_output\(i64 12, ptr @qir_tuple_label\)/ { many_tuple++ }
section == "many" && /@qir_result_label_[0-9]+ = private constant/ { defs++ }
section == "many" && /call void @__quantum__rt__result_record_output/ { calls++ }
section == "many" && /@qir_result_label_10 = private constant \[4 x i8\] c"r10\\00"/ { two_digit++ }
section == "many" && /call void @__quantum__rt__result_record_output\(ptr inttoptr \(i64 10 to ptr\), ptr @qir_result_label_10\)/ { ordered++ }
section == "many" {
    for (i=0; i<12; i++) {
        def="@qir_result_label_" i " = private constant [" (length("" i)+2) " x i8] c\"r" i "\\00\""
        handle=(i==0 ? "null" : "inttoptr (i64 " i " to ptr)")
        call="call void @__quantum__rt__result_record_output(ptr " handle ", ptr @qir_result_label_" i ")"
        if ($0 == def) found_def[i]++
        if (index($0, call)) found_call[i]++
    }
}
END { for (i=0; i<12; i++) if (found_def[i] != 1 || found_call[i] != 1) bad=1; if (bad || empty_label != 1 || empty_tuple != 1 || many_label != 1 || many_tuple != 1 || defs != 12 || calls != 12 || two_digit != 1 || ordered != 1) exit 1 }
' "$work/labels.log" || { echo 'FAIL: output labels' >&2; exit 1; }
echo 'PASS: qir_output_labels'
