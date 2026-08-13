// inject_san.c — host de injecao para a variante em rede do SAN scan.
//
// Envia a coorte pela fibra de 100G em UDP; o networklayer do VNx entrega o
// payload ao kernel por AXI-Stream. O host NAO participa do caminho de dados
// da FPGA — ele so gera e transmite.
//
// O empacotamento e' bit-a-bit identico ao do artefato aceito (T3_GREEN):
//   * 1 amostra = 128 bits, com n_conf campos Q0.15 de 15 bits, campo k em
//     [15k+14 : 15k]
//   * 1 beat = 512 bits = 4 amostras (LANES=4)
// A coorte e' o mesmo LCG deterministico do tb_san_scan, para o resultado ser
// conferivel contra o golden model sem depender dos artefatos T3.
//
// build: gcc -O2 -o inject_san inject_san.c
// uso:   ./inject_san <ip-fpga> <porta> <n_samples> <n_points> [beats-por-pacote]
//
// cluster-ops 2026-08-13
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define LANES 4
#define BEAT_BYTES 64          /* 512 bits */
#define SAMPLE_BITS 128

/* grava um campo de 15 bits na posicao k do registro de 128 bits (little-endian
   por bit, igual ao ap_uint do HLS: bit 0 = byte 0, bit menos significativo) */
static void put_field(uint8_t *rec, int k, uint16_t v) {
    for (int b = 0; b < 15; b++) {
        int bit = 15 * k + b;
        if (v & (1u << b)) rec[bit >> 3] |= (uint8_t)(1u << (bit & 7));
        else               rec[bit >> 3] &= (uint8_t)~(1u << (bit & 7));
    }
}

/* mesmo LCG do testbench: cohort deterministica e reproduzivel */
static uint32_t lcg_state = 12345u;
static uint32_t lcg(void) { lcg_state = lcg_state * 1103515245u + 12345u; return lcg_state; }

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
          "uso: %s <ip-fpga> <porta> <n_samples> <n_points> [beats-por-pacote] [porta-origem]\n"
          "  n_points 5 = geometria ResNet-50, 7 = ViT-large\n"
          "  porta-origem default 50000; tem de casar com theirPort na tabela\n"
          "  de sockets do networklayer (ver ctl_san.cpp)\n", argv[0]);
        return 2;
    }
    const char *ip = argv[1];
    int port       = atoi(argv[2]);
    long n_samples = atol(argv[3]);
    int n_points   = atoi(argv[4]);
    /* MTU 9214 no fabric -> 9000 bytes de payload cabem 140 beats com folga */
    int bpp        = (argc > 5) ? atoi(argv[5]) : 140;
    int sport      = (argc > 6) ? atoi(argv[6]) : 50000;
    int n_conf     = n_points - 1;

    if (n_points < 2 || n_points > 8) { fprintf(stderr, "n_points fora de 2..8\n"); return 2; }
    if (bpp < 1 || bpp * BEAT_BYTES > 9000) { fprintf(stderr, "beats-por-pacote fora de 1..140\n"); return 2; }

    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) { perror("socket"); return 1; }
    /* porta de origem FIXA: a tabela de sockets do networklayer casa por
       (theirIP, theirPort, myPort). Com porta efemera o casamento muda a cada
       execucao e o UDP do VNx descarta em silencio. */
    struct sockaddr_in src; memset(&src, 0, sizeof src);
    src.sin_family = AF_INET; src.sin_addr.s_addr = htonl(INADDR_ANY);
    src.sin_port = htons((uint16_t)sport);
    if (bind(fd, (struct sockaddr *)&src, sizeof src) < 0) { perror("bind"); close(fd); return 1; }
    /* buffer de socket grande: a 100G o kernel enfileira rapido */
    int sndbuf = 16 << 20; setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof sndbuf);
    struct sockaddr_in dst; memset(&dst, 0, sizeof dst);
    dst.sin_family = AF_INET; dst.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, ip, &dst.sin_addr) != 1) { fprintf(stderr, "ip invalido\n"); return 2; }

    /* buffer de envio: um pacote = bpp beats */
    size_t pkt_bytes = (size_t)bpp * BEAT_BYTES;
    uint8_t *pkt = calloc(1, pkt_bytes);
    if (!pkt) { perror("calloc"); return 1; }

    long n_beats = (n_samples + LANES - 1) / LANES;
    long enviados = 0, pacotes = 0, bytes = 0;
    double t0 = now_s();

    long b = 0;
    while (b < n_beats) {
        int nb = (int)((n_beats - b < bpp) ? (n_beats - b) : bpp);
        memset(pkt, 0, (size_t)nb * BEAT_BYTES);
        for (int i = 0; i < nb; i++) {
            uint8_t *beat = pkt + (size_t)i * BEAT_BYTES;
            for (int p = 0; p < LANES; p++) {
                long s = (b + i) * LANES + p;
                uint8_t *rec = beat + p * (SAMPLE_BITS / 8);
                if (s < n_samples)
                    for (int k = 0; k < n_conf; k++)
                        put_field(rec, k, (uint16_t)(lcg() >> 17)); /* 15 bits */
            }
        }
        ssize_t w = sendto(fd, pkt, (size_t)nb * BEAT_BYTES, 0,
                           (struct sockaddr *)&dst, sizeof dst);
        if (w < 0) { perror("sendto"); free(pkt); close(fd); return 1; }
        bytes += w; pacotes++; enviados += (long)nb * LANES;
        b += nb;
    }
    double dt = now_s() - t0;

    printf("INJECT_SAN origem=:%d destino=%s:%d n_samples=%ld n_points=%d beats=%ld\n",
           sport, ip, port, n_samples, n_points, n_beats);
    printf("  pacotes=%ld  bytes=%ld  tempo=%.3fs\n", pacotes, bytes, dt);
    printf("  taxa=%.1f Msamples/s  %.2f Gbit/s\n",
           n_samples / dt / 1e6, bytes * 8.0 / dt / 1e9);
    /* baseline a bater: 511 Msamples/s via DMA (T3_GREEN) */
    free(pkt); close(fd);
    return 0;
}
