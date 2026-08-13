#!/usr/bin/env python3
"""Confere que o limitador de taxa do inject_san respeita o alvo, contra um
destino de loopback que so descarta (sem FPGA envolvida)."""
import socket
import subprocess
import sys
import threading
import time

PORT = 52001
N_SAMPLES = 200000
N_POINTS = 7
TAXA_ALVO_GBPS = 2.0

recebido = [0, 0.0, 0.0]  # bytes, t0, t1


def sumidouro():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 << 20)
    s.bind(("127.0.0.1", PORT))
    s.settimeout(8)
    n = 0
    t0 = None
    t1 = None
    try:
        while True:
            d = s.recvfrom(9200)[0]
            if t0 is None:
                t0 = time.monotonic()
            t1 = time.monotonic()
            n += len(d)
    except socket.timeout:
        pass
    recebido[0] = n
    recebido[1] = t0 or 0.0
    recebido[2] = t1 or 0.0
    s.close()


def main():
    t = threading.Thread(target=sumidouro, daemon=True)
    t.start()
    time.sleep(0.3)

    r = subprocess.run(
        ["./inject_san", "127.0.0.1", str(PORT), str(N_SAMPLES), str(N_POINTS),
         "140", "50002", str(TAXA_ALVO_GBPS)],
        capture_output=True, text=True,
    )
    print(r.stdout)
    if r.returncode != 0:
        print("injetor falhou:", r.stderr)
        return 1
    t.join(timeout=9)

    bytes_, t0, t1 = recebido
    if bytes_ == 0 or t1 <= t0:
        print("PACE_FAIL nada recebido de forma mensuravel")
        return 1
    dt = t1 - t0
    gbps_medido = bytes_ * 8 / dt / 1e9
    razao = gbps_medido / TAXA_ALVO_GBPS
    ok = 0.5 <= razao <= 2.0  # tolerancia larga: e' um teste de sanidade, nao de precisao
    print(f"PACE_{'PASS' if ok else 'FAIL'} alvo={TAXA_ALVO_GBPS:.2f} Gbit/s "
          f"medido={gbps_medido:.2f} Gbit/s razao={razao:.2f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
