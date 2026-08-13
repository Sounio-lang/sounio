// Confere que o empacotamento do inject_san e' identico ao do testbench aceito.
// Reimplementa o empacotamento do tb (via ap_uint) em C puro e compara byte a
// byte com o do injetor, para a mesma sequencia de valores.
//
// build: gcc -O2 -o check_pack check_pack.c
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define LANES 4
#define BEAT_BYTES 64

/* versao do injetor */
static void put_field(uint8_t *rec, int k, uint16_t v) {
    for (int b = 0; b < 15; b++) {
        int bit = 15 * k + b;
        if (v & (1u << b)) rec[bit >> 3] |= (uint8_t)(1u << (bit & 7));
        else               rec[bit >> 3] &= (uint8_t)~(1u << (bit & 7));
    }
}

/* versao de referencia: escreve o registro de 128 bits como dois uint64,
   espelhando rec(15k+14, 15k) = q  do ap_uint no testbench */
static void ref_record(uint8_t *out, const uint16_t *q, int n_conf) {
    uint64_t lo = 0, hi = 0;
    for (int k = 0; k < n_conf; k++) {
        uint64_t v = (uint64_t)(q[k] & 0x7FFF);
        int base = 15 * k;
        if (base + 15 <= 64) {
            lo |= v << base;
        } else if (base >= 64) {
            hi |= v << (base - 64);
        } else {                        /* campo cruza a fronteira de 64 bits */
            int lo_bits = 64 - base;
            lo |= (v & ((1ULL << lo_bits) - 1)) << base;
            hi |= v >> lo_bits;
        }
    }
    for (int i = 0; i < 8; i++) { out[i] = (uint8_t)(lo >> (8 * i)); out[8 + i] = (uint8_t)(hi >> (8 * i)); }
}

int main(void) {
    int falhas = 0, casos = 0;
    for (int n_points = 5; n_points <= 7; n_points += 2) {
        int n_conf = n_points - 1;
        for (int t = 0; t < 2000; t++) {
            uint16_t q[8];
            for (int k = 0; k < n_conf; k++) q[k] = (uint16_t)((t * 7919 + k * 613) & 0x7FFF);
            uint8_t a[16] = {0}, b[16] = {0};
            for (int k = 0; k < n_conf; k++) put_field(a, k, q[k]);
            ref_record(b, q, n_conf);
            casos++;
            if (memcmp(a, b, 16) != 0) {
                if (falhas < 3) {
                    printf("  FALHA n_points=%d t=%d\n    injetor:", n_points, t);
                    for (int i = 15; i >= 0; i--) printf("%02x", a[i]);
                    printf("\n    referen:");
                    for (int i = 15; i >= 0; i--) printf("%02x", b[i]);
                    printf("\n");
                }
                falhas++;
            }
        }
    }
    printf("CHECK_PACK_%s  casos=%d falhas=%d\n", falhas ? "FAIL" : "PASS", casos, falhas);
    return falhas ? 1 : 0;
}
