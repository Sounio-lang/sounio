# Refuse publishing native artifacts with missing function bodies

The x86 preview emitter already counted missing non-builtin function bodies and emitted NATIVE_REFUSAL_SUMMARY empty_stub_ud2. It still left compile_ok true, so the driver could publish an ELF containing UD2 and report compilation success. The emitter now makes that existing refusal fail the output transaction. The diagnostic continues to name the unsupported body.

Slurm18686 executed the actual emitter before and after this isolated change using the sealed seed9cf. The same driver constructed a module containing a real main and an empty unknown function. Before the change it obtained a bad ELF and its assertion failed (driver exit1). After the change the emitter returned status12 and no bad.elf existed. The driver then replaced that unknown function with the real str_len builtin; emission succeeded and the produced main executable exited17 as specified. The driver itself exited0 with NATIVE_MISSING_BODY_REFUSAL_OK. Source and control manifests were unchanged after execution, and the outer Slurm job completed0:0.

Exact custody:
- Overlay archive native-empty-guard-v2-controls.tar.gz SHA256 61dfb9af629e44ae7db6f0a6b0d8354989e836e1d21651a2471fa33b993f4779.
- Before emitter SHA256 52bad673bfcce51f5fcd243ab8407070086bdf0c8af0772c81692dd479e00167; after SHA256 7f1b5b508ed30fde9a7bacbf4081aebca5c499743339435969cca1711b5d2da8.
- Identical driver SHA256 340b1783ee325ed091bcdca04914f3e95e17849ff42242b61615752ecbfe04f4.
- Receipts: /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/native-empty-guard/results/{before,after}; complete inputs are in INPUTS.sha256 and final guards.

This qualifies the missing-body output refusal in the executed emitter component. It does not implement the unsupported body, qualify a rebuilt integrated compiler, or close the original str/BigFrame/Neuro/PINN workload failures. Capacity WIP is deliberately not included in this source change.
