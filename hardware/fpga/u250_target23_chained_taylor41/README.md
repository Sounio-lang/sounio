# Target-23 chained Taylor-41 kernel

This kernel replays two disjoint 843-step partitions of the exact F192
center-radius chain. Each 224-bit signed value is transported as seven
little-endian 32-bit limbs. The two partition outputs concatenate to the
16,860-word CPU transcript.

The kernel implements strict Picard self-map checks, Taylor degree 40 with a
componentwise order-41 Lagrange remainder, a logarithmic-norm radius bound, an
integer exponential majorant, and up to 42 event bisections. A partition emits
an explicit refusal marker if any obligation fails.

`testbench.cpp` compares both partitions word for word in HLS CSim. `host.cpp`
performs the same comparison through native XRT and accepts an optional device
index and partition. With two visible cards, run partition 0 on device 0 and
partition 1 on device 1 concurrently; omitting the partition replays both
sequentially. `build_xclbin.sh` generates the RTL with the reviewed 100 MHz HLS
schedule and asks the physical linker for 10 MHz, the lowest frequency accepted
by this U250 platform. Override these independently with `HLS_CLOCK_MHZ` and
`KERNEL_FREQ_MHZ`. The first physical link attempt at 100 MHz was retained as a
timing-closure failure rather than presented as a bitstream. The AXI input and
output masters map to DDR banks 0 and 1.

Passing this kernel certifies only the frozen target-23 center chain. It does
not prove a full leaf, global H-PG, novelty priority, or an open problem.
