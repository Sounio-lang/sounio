#!/usr/bin/env bash
# Sprint 111 gate: F# NativeKernel — .NET NativeLibrary wrapper for Sounio .so
# Tests: NativeKernel.fs exists, .fsproj includes it, builds, symbol check
set -o pipefail

PASS=0; FAIL=0; NOT_RUN=0

run_grep() {
    local name="$1" file="$2" pattern="$3"
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "PASS $name"
        PASS=$((PASS+1))
    else
        echo "FAIL $name (pattern not found: $pattern)"
        FAIL=$((FAIL+1))
    fi
}

run_file_exists() {
    local name="$1" file="$2"
    if [ -f "$file" ]; then
        echo "PASS $name"
        PASS=$((PASS+1))
    else
        echo "FAIL $name (file not found: $file)"
        FAIL=$((FAIL+1))
    fi
}

INTEROP=interop/fsharp/Sounio.Interop

# ── T1: NativeKernel.fs exists ────────────────────────────────────────────────
run_file_exists T1_NativeKernel_exists "$INTEROP/NativeKernel.fs"

# ── T2: .fsproj includes NativeKernel.fs ──────────────────────────────────────
run_grep T2_fsproj_includes_NativeKernel "$INTEROP/Sounio.Interop.fsproj" "NativeKernel.fs"

# ── T3-T6: NativeKernel.fs key APIs ──────────────────────────────────────────
run_grep T3_NativeLibrary_Load "$INTEROP/NativeKernel.fs" "NativeLibrary.Load"
run_grep T4_GetExport "$INTEROP/NativeKernel.fs" "NativeLibrary.GetExport"
run_grep T5_CallI64 "$INTEROP/NativeKernel.fs" "member.*CallI64"
run_grep T6_CallF64 "$INTEROP/NativeKernel.fs" "member.*CallF64"
run_grep T7_IDisposable "$INTEROP/NativeKernel.fs" "interface IDisposable"
run_grep T8_NativeLibrary_Free "$INTEROP/NativeKernel.fs" "NativeLibrary.Free"
run_grep T9_HasExport "$INTEROP/NativeKernel.fs" "HasExport"
run_grep T10_Marshal_GetDelegate "$INTEROP/NativeKernel.fs" "Marshal.GetDelegateForFunctionPointer"

# ── T11: dotnet build ─────────────────────────────────────────────────────────
if command -v dotnet &>/dev/null; then
    build_out=$(dotnet build "$INTEROP/Sounio.Interop.fsproj" --nologo 2>&1)
    build_ec=$?
    if [ $build_ec -eq 0 ]; then
        echo "PASS T11_dotnet_build"
        PASS=$((PASS+1))
    else
        echo "FAIL T11_dotnet_build"
        echo "$build_out" | tail -10
        FAIL=$((FAIL+1))
    fi
else
    echo "NOT_RUN T11_dotnet_build (dotnet not in PATH)"
    NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "Sprint 111 gate: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN"
if [ $FAIL -gt 0 ]; then exit 1; fi
exit 0
