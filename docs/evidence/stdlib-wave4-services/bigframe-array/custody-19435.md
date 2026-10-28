# Durable custody — integrated frontier

Slurm 19435 completed with exit 0 in 44 seconds, using 512 MiB and 2 CPUs; batch MaxRSS was 366544 KiB. It produced the archive on DL380p, verified gzip integrity and SHA256, and retained the global Slurm phase/resource table.

The completed archive was relayed as user sounio (UID/GID 1000) through DL380 to OrangeFS. The destination readback checksum matches the Slurm-generated source checksum.

Archive:
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/integrated-frontier-export01/integrated-frontier.tar.gz`

SHA256:
`ea122de0ca5fbc2d6d674fc6bbacc80e0a74a16eb73fae8210784ef0d93dee2d`

Size: 357893323 bytes. Destination sidecars ARCHIVE.sha256 and TRANSFER.verify preserve the readback.

Contents include the sealed integrated compiler source/raw/seed, immutable services overlay and runtime archives, positive-preflight failures, fourteen Worker refusal controls, BigFrame original/reversible marker source and binaries, ABI and minimal array contrasts, the portable 19432 source/ELFs/logs, private GCC13 sysroot, GNU time and Node tools, source/tool/input/output hash guards, Slurm receipts and the immutable b23fdd316d diagnostic patch. Early duplicated normal source/source02 trees are excluded; root source plus original overlays reconstruct them, while source03 is retained directly.

The original C++ oracle built on DL380 by 19415 remains separately at:
/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/bigframe-abi07/oracle01/reference

It is byte-identical to the private-sysroot build 19422 on DL380p: both SHA256 values are 289a50d2d07b7dc9a91ad394214b031e2e041762c81c133e14be93361685f4a6. This is a toolchain recovery receipt; it does not qualify Sounio WorkerFn or array-cast semantics.

The archive preserves failures. Collection/custody success does not convert the eight portable SIGSEGV cases or six additional wrong-bit cases into behavior passes. Production stdlib/compiler/runtime and the original scientific witness remain unchanged by b23fdd316d and this documentation follow-up.
