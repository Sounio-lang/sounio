# Scalar cast and process termination components

Slurm18756 preserved a minimal nested read_f64 index cast: compile0/native139. Original BigFrame ptrace observed f64 bits1.0 used as an integer array index. Newly synthesized unresolved read_f64 slots now receive returns_float=1; existing declarations retain their metadata.

Slurm18828 paired components before: lower compile0/native31, exit compile0/native42. Slurm18820 after: both compile0/native0, with RAW_READ_FLOAT_METADATA_OK and FATAL_EXIT_GROUP_EMISSION_OK. Source and seed guards passed. Earlier18814/18817 used the wrong seed CLI and failedE221; they are preserved harness failures. The legacy seed requires positional SOURCE OUTPUT.

nc_emit_exit_code emits exit_group231; normal entry trampolines also terminate the process. The component tests actual helper emission, not live-thread termination. A rebuilt compiler replay of the minimal cast, original BigFrame and active-thread process exit remains required. Slurm18829 builds a copy of the original capacity snapshot with only the raw read_f64 metadata correction, preserving workload and assertions.

Reproduce scheduled components: set SEED_RAW_BIN to the qualified absolute seed and run bash scripts/ci/wave4_scalar_abi_components.sh /absolute/fresh-output after (or before). Private probes are appended only to copied test snapshot modules. Production visibility is unchanged. This is a source/component receipt, not stdlib acceptance.

Tracked-script release Slurm18844 COMPLETED0:0 (2m46) reproduced both modes: after lower/exit compile0 native0, before compile0 native31/42, all guarded. The script additionally requires each after marker, which is present in the qualified native logs. Final compiler and active-worker termination gates remain pending.
