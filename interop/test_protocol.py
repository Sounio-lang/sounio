#!/usr/bin/env python3
"""
Quick end-to-end test for the Sounio IPC protocol.
Tests the SNIO binary wire protocol by spawning souc --serve
and sending CALL_FUNC, INFO, CAPABILITIES, HEALTH, INIT,
GPU_CAPS, DIAGNOSTICS, STATS, COMPILE, and SHUTDOWN messages.

Usage: python3 interop/test_protocol.py
"""

import struct
import subprocess
import sys
import os

MAGIC = b"SNIO"
MSG_CALL_FUNC = 1
MSG_RESULT = 2
MSG_ERROR = 3
MSG_SHUTDOWN = 4
MSG_INFO = 5
MSG_CAPABILITIES = 6
MSG_COMPILE = 7
MSG_DIAGNOSTICS = 9
MSG_HEALTH = 10
MSG_GPU_CAPS = 11
MSG_INIT = 12
MSG_STATS = 13

def write_call_func(stdin, func_name: str, args: list[int]):
    name_bytes = func_name.encode("utf-8")
    name_len = len(name_bytes)
    arg_count = len(args)
    body_len = 2 + name_len + 8 + arg_count * 8

    stdin.write(MAGIC)
    stdin.write(struct.pack("<B", MSG_CALL_FUNC))
    stdin.write(struct.pack("<I", body_len))
    stdin.write(struct.pack("<H", name_len))
    stdin.write(name_bytes)
    stdin.write(struct.pack("<q", arg_count))
    for a in args:
        stdin.write(struct.pack("<q", a))
    stdin.flush()

def write_simple_msg(stdin, msg_type):
    """Send a message with empty payload (INFO, CAPABILITIES, HEALTH, etc.)"""
    stdin.write(MAGIC)
    stdin.write(struct.pack("<B", msg_type))
    stdin.write(struct.pack("<I", 0))
    stdin.flush()

def write_shutdown(stdin):
    stdin.write(MAGIC)
    stdin.write(struct.pack("<B", MSG_SHUTDOWN))
    stdin.write(struct.pack("<I", 0))
    stdin.flush()

def read_response(stdout):
    magic = stdout.read(4)
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic!r}")

    msg_type = struct.unpack("<B", stdout.read(1))[0]
    body_len = struct.unpack("<I", stdout.read(4))[0]

    if msg_type == MSG_RESULT:
        count = struct.unpack("<q", stdout.read(8))[0]
        values = []
        for _ in range(count):
            v = struct.unpack("<q", stdout.read(8))[0]
            values.append(v)
        return ("RESULT", values)

    elif msg_type == MSG_ERROR:
        err_len = struct.unpack("<H", stdout.read(2))[0]
        err_msg = stdout.read(err_len).decode("utf-8")
        return ("ERROR", err_msg)

    elif msg_type == MSG_SHUTDOWN:
        return ("SHUTDOWN", None)

    else:
        stdout.read(body_len)
        return ("UNKNOWN", msg_type)

def main():
    souc = os.path.join(os.path.dirname(__file__), "..",
                        "artifacts/omega/souc-bin/souc-linux-x86_64-jit")
    if not os.path.exists(souc):
        souc = "./artifacts/omega/souc-bin/souc-linux-x86_64-jit"

    main_sio = "self-hosted/compiler/main.sio"

    print("=== Sounio IPC Protocol Test ===")
    print(f"souc: {souc}")
    print(f"main: {main_sio}")
    print()

    # Start souc --serve
    proc = subprocess.Popen(
        [souc, "run", main_sio, "--", "--serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdin = proc.stdin
    stdout = proc.stdout

    passed = 0
    failed = 0

    try:
        # Read ready signal
        resp = read_response(stdout)
        assert resp[0] == "RESULT", f"Expected RESULT ready signal, got {resp}"
        print("[ready] Server started OK")

        # Test 1: ping
        write_call_func(stdin, "ping", [])
        resp = read_response(stdout)
        if resp == ("RESULT", [1]):
            print("[PASS] T1: ping -> 1")
            passed += 1
        else:
            print(f"[FAIL] T1: ping -> {resp}")
            failed += 1

        # Test 2: sum
        write_call_func(stdin, "sum", [100, 200, 300])
        resp = read_response(stdout)
        if resp == ("RESULT", [600]):
            print("[PASS] T2: sum([100,200,300]) -> 600")
            passed += 1
        else:
            print(f"[FAIL] T2: sum -> {resp}")
            failed += 1

        # Test 3: dot_product
        write_call_func(stdin, "dot_product", [1, 2, 3, 10, 20, 30])
        resp = read_response(stdout)
        expected = 1*10 + 2*20 + 3*30  # = 140
        if resp == ("RESULT", [expected]):
            print(f"[PASS] T3: dot_product([1,2,3],[10,20,30]) -> {expected}")
            passed += 1
        else:
            print(f"[FAIL] T3: dot_product -> {resp}, expected {expected}")
            failed += 1

        # Test 4: vec_add
        write_call_func(stdin, "vec_add", [1, 2, 3, 10, 20, 30])
        resp = read_response(stdout)
        if resp == ("RESULT", [11, 22, 33]):
            print("[PASS] T4: vec_add([1,2,3],[10,20,30]) -> [11,22,33]")
            passed += 1
        else:
            print(f"[FAIL] T4: vec_add -> {resp}")
            failed += 1

        # Test 5: vec_scale
        write_call_func(stdin, "vec_scale", [5, 1, 2, 3])
        resp = read_response(stdout)
        if resp == ("RESULT", [5, 10, 15]):
            print("[PASS] T5: vec_scale(5, [1,2,3]) -> [5,10,15]")
            passed += 1
        else:
            print(f"[FAIL] T5: vec_scale -> {resp}")
            failed += 1

        # Test 6: unknown function
        write_call_func(stdin, "nonexistent", [42])
        resp = read_response(stdout)
        if resp[0] == "ERROR" and "unknown function" in resp[1]:
            print(f"[PASS] T6: unknown function -> error: {resp[1]}")
            passed += 1
        else:
            print(f"[FAIL] T6: unknown function -> {resp}")
            failed += 1

        # Test 7: dot_product odd arg count
        write_call_func(stdin, "dot_product", [1, 2, 3])
        resp = read_response(stdout)
        if resp[0] == "ERROR":
            print(f"[PASS] T7: dot_product(odd args) -> error: {resp[1]}")
            passed += 1
        else:
            print(f"[FAIL] T7: dot_product(odd) -> {resp}")
            failed += 1

        # Test 8: INFO message — should return 6 i64 values
        write_simple_msg(stdin, MSG_INFO)
        resp = read_response(stdout)
        if resp[0] == "RESULT" and len(resp[1]) == 6:
            abi_ver = resp[1][0]
            print(f"[PASS] T8: INFO -> 6 values (abi={abi_ver}, rt={resp[1][1]}.{resp[1][2]}.{resp[1][3]})")
            passed += 1
        else:
            print(f"[FAIL] T8: INFO -> {resp}")
            failed += 1

        # Test 9: CAPABILITIES message — should return single bitmask
        write_simple_msg(stdin, MSG_CAPABILITIES)
        resp = read_response(stdout)
        if resp[0] == "RESULT" and len(resp[1]) == 1 and resp[1][0] > 0:
            caps = resp[1][0]
            flags = []
            if caps & 1: flags.append("FFI")
            if caps & 2: flags.append("EXPORT")
            if caps & 4: flags.append("PROFILING")
            print(f"[PASS] T9: CAPABILITIES -> {caps} ({'+'.join(flags)})")
            passed += 1
        else:
            print(f"[FAIL] T9: CAPABILITIES -> {resp}")
            failed += 1

        # Test 10: HEALTH message — should return 1 (ok)
        write_simple_msg(stdin, MSG_HEALTH)
        resp = read_response(stdout)
        if resp[0] == "RESULT" and len(resp[1]) == 1 and resp[1][0] == 1:
            print("[PASS] T10: HEALTH -> 1 (ok)")
            passed += 1
        else:
            print(f"[FAIL] T10: HEALTH -> {resp}")
            failed += 1

        # Test 11: INIT message — should return 1 (ack)
        write_simple_msg(stdin, MSG_INIT)
        resp = read_response(stdout)
        if resp[0] == "RESULT" and len(resp[1]) == 1 and resp[1][0] == 1:
            print("[PASS] T11: INIT -> 1 (ack)")
            passed += 1
        else:
            print(f"[FAIL] T11: INIT -> {resp}")
            failed += 1

        # Test 12: GPU_CAPS message — should return capability bitmask (no GPU)
        write_simple_msg(stdin, MSG_GPU_CAPS)
        resp = read_response(stdout)
        if resp[0] == "RESULT" and len(resp[1]) == 1:
            print(f"[PASS] T12: GPU_CAPS -> {resp[1][0]} (no GPU expected)")
            passed += 1
        else:
            print(f"[FAIL] T12: GPU_CAPS -> {resp}")
            failed += 1

        # Test 13: DIAGNOSTICS message — should return 0 (empty)
        write_simple_msg(stdin, MSG_DIAGNOSTICS)
        resp = read_response(stdout)
        if resp[0] == "RESULT" and len(resp[1]) == 1 and resp[1][0] == 0:
            print("[PASS] T13: DIAGNOSTICS -> 0 (empty)")
            passed += 1
        else:
            print(f"[FAIL] T13: DIAGNOSTICS -> {resp}")
            failed += 1

        # Test 14: STATS message — should return message count
        write_simple_msg(stdin, MSG_STATS)
        resp = read_response(stdout)
        if resp[0] == "RESULT" and len(resp[1]) == 1 and resp[1][0] > 0:
            print(f"[PASS] T14: STATS -> message_count={resp[1][0]}")
            passed += 1
        else:
            print(f"[FAIL] T14: STATS -> {resp}")
            failed += 1

        # Test 15: COMPILE message (placeholder) — should return 0
        write_simple_msg(stdin, MSG_COMPILE)
        resp = read_response(stdout)
        if resp[0] == "RESULT" and len(resp[1]) == 1:
            print(f"[PASS] T15: COMPILE -> {resp[1][0]} (placeholder)")
            passed += 1
        else:
            print(f"[FAIL] T15: COMPILE -> {resp}")
            failed += 1

        # Shutdown
        write_shutdown(stdin)
        resp = read_response(stdout)
        if resp[0] == "SHUTDOWN":
            print("[PASS] T16: shutdown")
            passed += 1
        else:
            print(f"[FAIL] T16: shutdown -> {resp}")
            failed += 1

    except Exception as e:
        print(f"[ERROR] {e}")
        failed += 1
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()

    print()
    print(f"Results: {passed} PASS, {failed} FAIL out of {passed + failed}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
