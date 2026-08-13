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
//        [porta-origem] [taxa-alvo-Gbit/s]
//
// SEMPRE um pacote por chamada de sendto — NUNCA sendmmsg. Uma versao
// anterior usava sendmmsg (varios datagramas por syscall) para reduzir o
// custo de troca de contexto; mesmo em lotes de so' 8, PACEADOS a apenas
// 5 Gbit/s, isso travou o networklayer do VNx (perda de pacote seguida de
// kernel que nunca mais completa, so recuperavel com systemctl restart
// u250-vnx.service). O culpado nao e' a taxa media, e' a rajada DENTRO da
// chamada: sendmmsg entrega N datagramas ao driver de uma vez, sem o
// espacamento natural que o custo de syscall do sendto da de graca. O
// networklayer do VNx tem FIFOs pequenas e nenhum controle de fluxo — nao
// absorve rajada nenhuma, so' o ritmo de um-de-cada-vez.
//
// A taxa-alvo (se dada) pacea via atraso entre chamadas de sendto, com o
// mesmo token bucket de antes — so' que agora sobre pacotes individuais,
// nunca sobre lotes.
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
          "uso: %s <ip-fpga> <porta> <n_samples> <n_points> [beats-por-pacote]\n"
          "  [porta-origem] [taxa-alvo-Gbit/s]\n"
          "  n_points 5 = geometria ResNet-50, 7 = ViT-large\n"
          "  porta-origem default 50000; tem de casar com theirPort na tabela\n"
          "  de sockets do networklayer (ver ctl_san.cpp)\n"
          "  taxa-alvo omitida ou 0 = ritmo natural do sendto (~13 Gbit/s,\n"
          "  provado seguro); so use taxa-alvo para ir MAIS DEVAGAR que isso\n", argv[0]);
        return 2;
    }
    const char *ip = argv[1];
    int port       = atoi(argv[2]);
    long n_samples = atol(argv[3]);
    int n_points   = atoi(argv[4]);
    /* MTU 9214 no fabric -> 9000 bytes de payload cabem 140 beats com folga */
    int bpp        = (argc > 5) ? atoi(argv[5]) : 140;
    int sport      = (argc > 6) ? atoi(argv[6]) : 50000;
    double taxa_gbps = (argc > 7) ? atof(argv[7]) : 0.0;  /* 0 = ritmo natural do sendto */
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
    int sndbuf = 4 << 20; setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof sndbuf);
    struct sockaddr_in dst; memset(&dst, 0, sizeof dst);
    dst.sin_family = AF_INET; dst.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, ip, &dst.sin_addr) != 1) { fprintf(stderr, "ip invalido\n"); return 2; }

    long n_beats = (n_samples + LANES - 1) / LANES;

    /* --- FASE 1: empacota TUDO na memoria, fora do cronometro ---
       Medir geracao junto com transmissao mede o empacotador, nao o caminho
       de dados: put_field faz 15 operacoes de bit por campo, e isso domina.
       Primeira medicao ingenua deu 7,3 Msamples/s — era o custo do bit-twiddling
       no host, nao a capacidade da fibra nem da FPGA. */
    size_t total = (size_t)n_beats * BEAT_BYTES;
    uint8_t *buf = calloc(1, total);
    if (!buf) { perror("calloc"); return 1; }
    double tp0 = now_s();
    for (long s = 0; s < n_samples; s++) {
        uint8_t *rec = buf + (s / LANES) * BEAT_BYTES + (s % LANES) * (SAMPLE_BITS / 8);
        for (int k = 0; k < n_conf; k++)
            put_field(rec, k, (uint16_t)(lcg() >> 17)); /* 15 bits */
    }
    double tpack = now_s() - tp0;

    /* --- FASE 2: so a transmissao entra no cronometro ---
       um pacote por chamada de sendto, sempre. Com taxa-alvo, atrasa antes
       de cada chamada para nao passar do orcamento de bytes/tempo. */
    double taxa_Bps = taxa_gbps * 1e9 / 8.0;
    long pacotes = 0, bytes = 0;
    double t0 = now_s();
    for (long b = 0; b < n_beats; ) {
        long nb = (n_beats - b < bpp) ? (n_beats - b) : bpp;
        size_t pkt_bytes = (size_t)nb * BEAT_BYTES;

        if (taxa_gbps > 0.0) {
            double alvo_t = (bytes + (long)pkt_bytes) / taxa_Bps;
            double agora = now_s() - t0;
            if (alvo_t > agora) {
                double atraso = alvo_t - agora;
                struct timespec ts = { (time_t)atraso, (long)((atraso - (long)atraso) * 1e9) };
                nanosleep(&ts, NULL);
            }
        }

        ssize_t w = sendto(fd, buf + (size_t)b * BEAT_BYTES, pkt_bytes, 0,
                           (struct sockaddr *)&dst, sizeof dst);
        if (w < 0) { perror("sendto"); free(buf); close(fd); return 1; }
        bytes += w; pacotes++; b += nb;
    }
    double dt = now_s() - t0;

    printf("INJECT_SAN origem=:%d destino=%s:%d n_samples=%ld n_points=%d beats=%ld\n",
           sport, ip, port, n_samples, n_points, n_beats);
    printf("  empacotamento (fora do cronometro): %.3fs = %.1f Msamples/s\n",
           tpack, n_samples / tpack / 1e6);
    printf("  transmissao (%s): pacotes=%ld bytes=%ld tempo=%.4fs\n",
           taxa_gbps > 0.0 ? "com pace" : "ritmo natural do sendto", pacotes, bytes, dt);
    printf("  taxa=%.1f Msamples/s  %.2f Gbit/s\n",
           n_samples / dt / 1e6, bytes * 8.0 / dt / 1e9);
    /* referencias: 511 Msamples/s via DMA (T3_GREEN); teto da fibra de 100G
       com 128 bits por amostra = 781 Msamples/s a taxa de linha */
    free(buf); close(fd);
    return 0;
}
