# M6: per-head scratch in the paged device kernel

## Cause owned

The kernel was already parallel (grid nrows x nheads, one
thread per block) but sco had no head dimension: every head of
a row shared sco[row*maxblocks*B + t] with no cross-block sync.
All M4/M5 green lanes rested on benign block scheduling.

## Change

- ptx: sco base (row*nheads+head)*sstride via existing %r_rh
  (1 line) + header dims.
- host: dc alloc n*heads*maxb*bb f64 (ll_attn_paged_rows_gpu).
- cu oracle mirrored to the new layout.

## Receipts

- CPU: SERVEPUMP/BATCHCONT/PAGEDRUNS/PAGEDGPU PASS.
- RTX4000 Ada 5860: serve-path 3/3 PASS + PAGEDGPU-DEV PASS
  (proof == n*heads, kernel really ran) + PAGEDGPU PASS.
- FPE seen mid-lane was a missing micro-paged fixture in the
  relay tarball, not the kernel; restaged and green.
