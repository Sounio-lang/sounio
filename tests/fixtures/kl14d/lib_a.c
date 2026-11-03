/* KL-14d multi-DT_NEEDED probe A. */
long kl14d_a(long x) { return x + 1; }
long kl14d_interpose(void) { return 101; }
