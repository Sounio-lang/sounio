#!/usr/bin/env python3
"""
Smoke test for the Sounio kernel/session embedding ABI.

Exercises the full lifecycle over SNIO binary protocol:
  1. SESSION_CREATE  (msg 14)  → get session_id
  2. KERNEL_DESCRIBE (msg 16)  → register a kernel
  3. KERNEL_EXECUTE  (msg 17)  → run it
  4. KERNEL_OUTPUT   (msg 18)  → read structured output
  5. KERNEL_DIAGNOSTICS (msg 19) → read diagnostics
  6. KERNEL_ARTIFACTS   (msg 20) → read artifacts
  7. SESSION_DESTROY (msg 15)  → release session
  8. SHUTDOWN        (msg 4)   → stop server

Strategy: file-based I/O to work around JIT stdout pipe buffering.
All request messages are pre-built into a stdin file, souc reads from
the file, and stdout is captured to a temp file for parsing.

Usage:
  python3 kernel_session_smoke.py [path/to/souc] [path/to/serve.sio]
"""

import io
import os
import struct
import subprocess
import sys
import tempfile
import time

MAGIC = b"SNIO"

# Message types
MSG_RESULT          = 2
MSG_ERROR           = 3
MSG_SHUTDOWN        = 4
MSG_SESSION_CREATE  = 14
MSG_SESSION_DESTROY = 15
MSG_KERNEL_DESCRIBE = 16
MSG_KERNEL_EXECUTE  = 17
MSG_KERNEL_OUTPUT   = 18
MSG_KERNEL_DIAG     = 19
MSG_KERNEL_ARTIFACTS = 20


def make_msg(msg_type: int, payload: bytes) -> bytes:
    """Build an SNIO message frame."""
    return MAGIC + struct.pack("<BI", msg_type, len(payload)) + payload


def read_response(f):
    """Read one SNIO response from a file-like object.
    Returns (msg_type_str, payload_values | error_str)."""
    magic = f.read(4)
    if len(magic) < 4:
        return ("eof", None)
    assert magic == MAGIC, f"Bad magic: {magic!r}"
    msg_type = struct.unpack("<B", f.read(1))[0]
    body_len = struct.unpack("<I", f.read(4))[0]

    if msg_type == MSG_RESULT:
        count = struct.unpack("<q", f.read(8))[0]
        values = [struct.unpack("<q", f.read(8))[0] for _ in range(count)]
        return ("result", values)
    elif msg_type == MSG_ERROR:
        err_len = struct.unpack("<H", f.read(2))[0]
        err_msg = f.read(err_len).decode("utf-8", errors="replace")
        return ("error", err_msg)
    elif msg_type == MSG_SHUTDOWN:
        return ("shutdown", None)
    else:
        body = f.read(body_len)
        return ("unknown", body)


def build_request_sequence() -> bytes:
    """Pre-build the full SNIO request sequence.

    We assume session_id=1 and kernel_id=0 since we're the first session
    and first kernel. The server assigns IDs deterministically.
    """
    buf = b""

    # R1: SESSION_CREATE (flags=0)
    buf += make_msg(MSG_SESSION_CREATE, struct.pack("<qq", 0, 0))

    # R2: KERNEL_DESCRIBE (session_id=1, path="examples/hello.sio", flags=0)
    path_bytes = b"examples/hello.sio"
    buf += make_msg(MSG_KERNEL_DESCRIBE,
                    struct.pack("<q", 1) +
                    struct.pack("<H", len(path_bytes)) + path_bytes +
                    struct.pack("<q", 0))

    # R3: KERNEL_EXECUTE (session_id=1, kernel_id=0, args=[42, 7])
    buf += make_msg(MSG_KERNEL_EXECUTE,
                    struct.pack("<qqq", 1, 0, 2) +
                    struct.pack("<q", 42) + struct.pack("<q", 7))

    # R4: KERNEL_OUTPUT (session_id=1)
    buf += make_msg(MSG_KERNEL_OUTPUT, struct.pack("<q", 1))

    # R5: KERNEL_DIAGNOSTICS (session_id=1)
    buf += make_msg(MSG_KERNEL_DIAG, struct.pack("<q", 1))

    # R6: KERNEL_ARTIFACTS (session_id=1)
    buf += make_msg(MSG_KERNEL_ARTIFACTS, struct.pack("<q", 1))

    # R7: SESSION_DESTROY (session_id=1)
    buf += make_msg(MSG_SESSION_DESTROY, struct.pack("<q", 1))

    # R8: SESSION_CREATE again (flags=1) — test slot reuse
    buf += make_msg(MSG_SESSION_CREATE, struct.pack("<qq", 0, 1))

    # R9: KERNEL_DESCRIBE on session 2 (valid file, tests multi-session kernels)
    path2 = b"examples/hello.sio"
    buf += make_msg(MSG_KERNEL_DESCRIBE,
                    struct.pack("<q", 2) +
                    struct.pack("<H", len(path2)) + path2 +
                    struct.pack("<q", 0))

    # R10: KERNEL_EXECUTE on session 2, kernel 0 — placeholder dispatch
    buf += make_msg(MSG_KERNEL_EXECUTE,
                    struct.pack("<qqq", 2, 0, 3) +
                    struct.pack("<q", 10) + struct.pack("<q", 20) + struct.pack("<q", 30))

    # R11: KERNEL_DIAGNOSTICS on session 2 (should have info diagnostic from placeholder exec)
    buf += make_msg(MSG_KERNEL_DIAG, struct.pack("<q", 2))

    # R12: SESSION_DESTROY (session_id=2)
    buf += make_msg(MSG_SESSION_DESTROY, struct.pack("<q", 2))

    # R13: SHUTDOWN
    buf += make_msg(MSG_SHUTDOWN, b"")

    return buf


def main():
    souc = sys.argv[1] if len(sys.argv) > 1 else "artifacts/omega/souc-bin/souc-linux-x86_64-jit"
    serve_sio = sys.argv[2] if len(sys.argv) > 2 else "self-hosted/interop/serve_entry.sio"

    # Build all requests upfront
    requests = build_request_sequence()

    with tempfile.NamedTemporaryFile(prefix="snio_in_", delete=False) as fin:
        fin.write(requests)
        stdin_path = fin.name

    stdout_path = tempfile.mktemp(prefix="snio_out_")

    print(f"[smoke] souc: {souc}")
    print(f"[smoke] serve: {serve_sio}")
    print(f"[smoke] stdin:  {stdin_path} ({len(requests)} bytes)")
    print(f"[smoke] stdout: {stdout_path}")
    print(f"[smoke] Starting server (JIT warmup may take 60s+)...")

    try:
        with open(stdin_path, "rb") as fin_handle, \
             open(stdout_path, "wb") as fout_handle:
            result = subprocess.run(
                [souc, "run", serve_sio],
                stdin=fin_handle,
                stdout=fout_handle,
                stderr=subprocess.PIPE,
                timeout=300,
            )

        if result.returncode != 0:
            stderr_text = result.stderr.decode("utf-8", errors="replace")[:1000]
            print(f"[smoke] souc exited with code {result.returncode}")
            print(f"[smoke] stderr: {stderr_text}")

        # Parse output
        out_size = os.path.getsize(stdout_path)
        print(f"[smoke] Output file: {out_size} bytes")

        if out_size == 0:
            print("[smoke] FATAL: No output produced")
            sys.exit(2)

        with open(stdout_path, "rb") as f:
            responses = []
            while f.tell() < out_size:
                resp = read_response(f)
                if resp[0] == "eof":
                    break
                responses.append(resp)

        print(f"[smoke] Parsed {len(responses)} responses")
        for i, (kind, val) in enumerate(responses):
            print(f"  [{i}] {kind}: {val}")

    except subprocess.TimeoutExpired:
        print("[smoke] FATAL: souc timed out after 300s")
        sys.exit(2)
    finally:
        # Cleanup temp files
        for p in [stdin_path, stdout_path]:
            try:
                os.unlink(p)
            except OSError:
                pass

    # Validate responses
    # Expected: ready_signal, R1..R10 responses = 11 total
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}")
            failed += 1

    print("\n[smoke] === Validation ===")

    # Response 0: ready signal
    check("ready signal present", len(responses) >= 1)
    if len(responses) >= 1:
        check("ready signal is result", responses[0][0] == "result")

    # Response 1: SESSION_CREATE → session_id
    if len(responses) >= 2:
        kind, val = responses[1]
        check("T1: SESSION_CREATE result", kind == "result")
        sid = val[0] if kind == "result" and len(val) >= 1 else -1
        check("T1: session_id > 0", sid > 0)
    else:
        check("T1: response present", False)
        sid = -1

    # Response 2: KERNEL_DESCRIBE
    if len(responses) >= 3:
        kind, val = responses[2]
        check("T2: KERNEL_DESCRIBE result", kind == "result")
        if kind == "result" and len(val) >= 1:
            check("T2: describe.ok == 1", val[0] == 1)
            kid = val[1] if len(val) > 1 else -1
            check("T2: kernel_id >= 0", kid >= 0)
        else:
            check("T2: describe.ok == 1", False)
            check("T2: kernel_id >= 0", False)
    else:
        check("T2: response present", False)

    # Response 3: KERNEL_EXECUTE
    if len(responses) >= 4:
        kind, val = responses[3]
        check("T3: KERNEL_EXECUTE result", kind == "result")
        check("T3: returns values", kind == "result" and len(val) >= 1)
    else:
        check("T3: response present", False)

    # Response 4: KERNEL_OUTPUT
    if len(responses) >= 5:
        kind, val = responses[4]
        check("T4: KERNEL_OUTPUT result", kind == "result")
        if kind == "result":
            check("T4: output.session_id matches", len(val) >= 1 and val[0] == sid)
            check("T4: output.exec_count >= 1", len(val) >= 3 and val[2] >= 1)
        else:
            check("T4: output.session_id matches", False)
            check("T4: output.exec_count >= 1", False)
    else:
        check("T4: response present", False)

    # Response 5: KERNEL_DIAGNOSTICS (may have info diag from placeholder exec)
    if len(responses) >= 6:
        kind, val = responses[5]
        check("T5: KERNEL_DIAGNOSTICS result", kind == "result")
        if kind == "result" and len(val) >= 1:
            # Placeholder execute adds an INFO diagnostic (code=102)
            if val[0] == 1 and len(val) >= 5:
                check("T5: diag is INFO (not ERROR)", val[1] == 3)
                check("T5: diag.code == 102 (placeholder)", val[4] == 102)
            else:
                check("T5: diagnostics present", val[0] >= 0)
        else:
            check("T5: diagnostics readable", False)
    else:
        check("T5: response present", False)

    # Response 6: KERNEL_ARTIFACTS
    if len(responses) >= 7:
        kind, val = responses[6]
        check("T6: KERNEL_ARTIFACTS result", kind == "result")
        check("T6: artifacts.count == 0", kind == "result" and len(val) >= 1 and val[0] == 0)
    else:
        check("T6: response present", False)

    # Response 7: SESSION_DESTROY
    if len(responses) >= 8:
        kind, val = responses[7]
        check("T7: SESSION_DESTROY result", kind == "result")
    else:
        check("T7: response present", False)

    # Response 8: SESSION_CREATE (slot reuse)
    if len(responses) >= 9:
        kind, val = responses[8]
        check("T8: second SESSION_CREATE result", kind == "result")
        sid2 = val[0] if kind == "result" and len(val) >= 1 else -1
        check("T8: second session_id > 0", sid2 > 0)
        check("T8: second session_id != first", sid2 != sid)
    else:
        check("T8: response present", False)

    # Response 9: KERNEL_DESCRIBE on session 2
    if len(responses) >= 10:
        kind, val = responses[9]
        check("T9: second KERNEL_DESCRIBE result", kind == "result")
        if kind == "result" and len(val) >= 2:
            check("T9: describe.ok == 1", val[0] == 1)
            check("T9: kernel_id == 0", val[1] == 0)
        else:
            check("T9: describe.ok == 1", False)
            check("T9: kernel_id == 0", False)
    else:
        check("T9: response present", False)

    # Response 10: KERNEL_EXECUTE on session 2 — placeholder result
    if len(responses) >= 11:
        kind, val = responses[10]
        check("T10: second KERNEL_EXECUTE result", kind == "result")
        check("T10: returns value", kind == "result" and len(val) >= 1)
    else:
        check("T10: response present", False)

    # Response 11: KERNEL_DIAGNOSTICS on session 2 — should have info diagnostic
    if len(responses) >= 12:
        kind, val = responses[11]
        check("T11: KERNEL_DIAGNOSTICS result", kind == "result")
        if kind == "result" and len(val) >= 1:
            check("T11: diagnostics.count == 1 (placeholder info)", val[0] == 1)
            # Diagnostic wire layout: [count, severity, line, col, code, ...]
            if len(val) >= 5:
                check("T11: diag.severity == 3 (INFO)", val[1] == 3)
                check("T11: diag.code == 102 (NO_COMPILED_CODE)", val[4] == 102)
            else:
                check("T11: diag fields present", False)
        else:
            check("T11: diagnostics.count == 1", False)
    else:
        check("T11: response present", False)

    # Response 12: SESSION_DESTROY (second)
    if len(responses) >= 13:
        kind, val = responses[12]
        check("T12: second SESSION_DESTROY result", kind == "result")
    else:
        check("T12: response present", False)

    # Response 13: SHUTDOWN
    if len(responses) >= 14:
        kind, val = responses[13]
        check("T13: SHUTDOWN", kind == "shutdown")
    else:
        check("T13: response present", False)

    print(f"\n[smoke] Results: {passed} PASS, {failed} FAIL, {passed + failed} total")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
