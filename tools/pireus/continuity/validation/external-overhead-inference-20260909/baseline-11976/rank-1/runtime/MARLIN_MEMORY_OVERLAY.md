# Pinned Marlin placeholder allocation correction

The unmodified serving job11878 stopped at35.583 GiB host MemAvailable on
Spark8e54, before completed model construction/weight loading. Slurm recorded
ExitCode75:0 and both host grants/protected-memory checks passed afterward.

The installed ModelOpt source on both Sparks is SHA256
0b87d0fb15b1929f490fbc6f224ab5e1e0e11fee848d01e5ee4c511b4c0f4343,
matching SGLang commit a74222ef6e690f851e2e4ff1c0be7dc1357be313.
create_weights allocates two swizzled blockscale copies for Marlin even though
the Marlin finalize and apply branches consume the unswizzled/repacked scale
parameters and return before the other backends' swizzled-scale paths.
The checkpoint contains78 corresponding FP8 scale tensors totaling15703474176
bytes; equal TP2 partition predicts7.3125 GiB of avoidable copies per rank.
This is a shape-derived estimate, not a full-model measured saving.

patch_marlin_placeholders.py changes only the two allocation conditions
within ModelOptNvFp4FusedMoEMethod.create_weights. It refuses any source hash
other than the installed pin. The patched source SHA256 is
26b999da4d72fd238c32331782f72b6aa110165adec96fb8885f4252a5a7099c.
The installer writes a content-addressed, read-only source overlay; Apptainer
binds that one file only after hash verification. The original SIF, checkpoint,
quantization and production kernels are unchanged. This is an explicit local
runtime patch on top of the pinned base, not an unmodified-runtime claim.

Job11882 passed real GB10 controls on both nodes: exact consumed/repacked
parameter bits,356864 output components per node with diagnostic non-atomic
reduction, and a consumed-scale corruption negative. The stock SM121 Marlin
path uses atomic reduction; baseline repeats and baseline/challenger outputs
vary. Those observations are retained without claiming stock bit determinism.
The diagnostic non-atomic override is confined to the test process and is
not part of the production overlay. Jobs11880 and11881 remain failed evidence.

The full model still requires a fresh guarded load, cache allocation and
generation. No serving or eight-proposal acceptance follows from this layer test.

The overlay is staged in the Slurm user-owned /scratch/pireus/cache;
the root-owned runtime-tools directory remains read-only to the job.
Job11884 verified the mounted patched source hash on both nodes while the
other inspected source hashes stayed unchanged. Job11883 was cancelled
during checkpoint hashing to correct this installation-path issue.
