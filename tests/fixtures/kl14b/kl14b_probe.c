/* KL-14b probe shared library — one symbol, no libc dependency beyond CRT. */
long kl14b_add(long a, long b) {
    return a + b;
}
