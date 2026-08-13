#!/usr/bin/env python3
"""Teste de laco fechado do inject_san.

Captura o datagrama de verdade na rede (loopback) e confere, campo a campo,
que os bytes no fio sao o empacotamento esperado: 4 amostras de 128 bits por
beat, cada uma com n_conf campos Q0.15 de 15 bits, o campo k em [15k+14:15k].

Reimplementa o LCG do injetor de forma independente, para nao validar o
codigo contra ele mesmo.
"""
import socket
import subprocess
import sys
import threading

PORT = 51999
N_SAMPLES = 8
N_POINTS = 7
N_CONF = N_POINTS - 1
LANES = 4

capturado = []


def receber():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", PORT))
    s.settimeout(6)
    try:
        capturado.append(s.recvfrom(9000)[0])
    except socket.timeout:
        pass
    s.close()


def lcg_stream():
    """Mesmo LCG do injetor, reimplementado do zero."""
    st = 12345
    while True:
        st = (st * 1103515245 + 12345) & 0xFFFFFFFF
        yield (st >> 17) & 0x7FFF


def get_field(rec: bytes, k: int) -> int:
    """Le o campo k de 15 bits do registro de 128 bits."""
    v = int.from_bytes(rec, "little")
    return (v >> (15 * k)) & 0x7FFF


def main():
    t = threading.Thread(target=receber, daemon=True)
    t.start()
    import time
    time.sleep(0.4)

    r = subprocess.run(
        ["./inject_san", "127.0.0.1", str(PORT), str(N_SAMPLES), str(N_POINTS), "140", "50001"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("injetor falhou:", r.stderr.strip())
        return 1
    t.join(timeout=7)

    if not capturado:
        print("LOOPBACK_FAIL nada capturado")
        return 1
    d = capturado[0]

    esperado_beats = (N_SAMPLES + LANES - 1) // LANES
    if len(d) != esperado_beats * 64:
        print(f"LOOPBACK_FAIL tamanho={len(d)} esperado={esperado_beats * 64}")
        return 1

    gen = lcg_stream()
    erros = 0
    for s in range(N_SAMPLES):
        beat, lane = s // LANES, s % LANES
        rec = d[beat * 64 + lane * 16: beat * 64 + lane * 16 + 16]
        for k in range(N_CONF):
            esperado = next(gen)
            visto = get_field(rec, k)
            if visto != esperado:
                if erros < 5:
                    print(f"  divergencia s={s} k={k}: fio={visto} esperado={esperado}")
                erros += 1

    # os 15 bits altos de cada registro de 128 tem de estar zerados
    # (n_conf*15 = 90 bits usados de 128)
    for s in range(N_SAMPLES):
        beat, lane = s // LANES, s % LANES
        rec = int.from_bytes(d[beat * 64 + lane * 16: beat * 64 + lane * 16 + 16], "little")
        if rec >> (15 * N_CONF):
            print(f"  lixo nos bits altos da amostra {s}")
            erros += 1

    print(f"LOOPBACK_{'FAIL' if erros else 'PASS'} "
          f"bytes={len(d)} beats={esperado_beats} campos={N_SAMPLES * N_CONF} erros={erros}")
    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(main())
