# Scalar uncertainty-shadow dominance frontier

This is a compiler IR diagnostic, not a production repair or whole-stdlib result. All runs use experimental raw04 a4f5e8d655754202493bbb04d8fbb15890b10e8fc861ac3dc60438452704c333. No lowerer, numerical library, assertion in an original regression, or compiler policy was changed.

Slurm19737 completed0 in4s. A small Sounio case originated with darwin-llm sounio_verificado (reported check-valid804ms), then retained the parameter alias and conditional assignment in a compact fixture:

```sounio
var s = seed
if s == 0 { s = 1 }
s = s * 3 + 1
```

The C++23 reference independently evaluates the two tested inputs, with compile-time and native checks. The Sounio programs check and compile successfully in LOOP_REGIONS0 and1; the positive executables exit0, while changing exactly the second expected assertion makes both negative controls exit1. C++ reference stdout, source, source hashes and the original phase outputs are retained. These value checks pass even though the IR diagnostic below fails; they do not establish correct uncertainty metadata.

The already qualified C++23 CFG auditor accepts the complete source-stage schema2 capture, then reports three missing must-def facts in core: reg8 at instructions30,40,45. Reg8 is the f64 zero/variance shadow defined only in the seed==0 arm (instruction6). The branch at4 can skip that definition. The main function has no missing facts. The auditor exits0 for successful diagnostic collection even when CFG_UNDEFINED rows exist; it is not a zero-defect acceptance exit. No parser refusal or partial trace was accepted.

Slurm19729 completed0 in2s. On the frozen WIDE projection used by19724, toggling only SOUNIO_OCP_SKIP_DEDUP_IMM left the complete capture and the three missing facts per core unchanged. This frozen projection is a diagnostic input, not an assertion that its body equals the current library after later sampler changes.

Slurm19731 separately completed0. Its ordered-path receipt explicitly records optimize=0 in the default compile, so the optional module cleanup did not run in that baseline. An explicit -O run reduces the diagnostic to one missing fact per core. The input/source/compiler/auditor guards passed. Thus the default missing definitions precede optional cleanup; the observations do not by themselves distinguish source lowering from a miscompiled producer of the lowering code.

Readonly source inspection supplies a candidate mechanism: a parameter identifier with no variance binding leaves a mutable alias untracked; a conditional literal assignment emits a new zero variance register and the fallback assignment path publishes it into the local metadata map. A later arithmetic expression reads that branch-local register. The executed source emit_f64_const allocates a fresh register, so this receipt does not attribute the symptom to a constant cache. A fix must preserve the meaning of untracked values, Knowledge/correlation/Hessian information, ownership of mutable cells, the channel budget and definitions on all paths. No such fix is claimed here.

Slurm19739 completed0 in27s and preserved sources, C++ oracle/auditor, exact compiler, toolchain, ELF binaries, positive/negative outputs, the two optimizer comparisons and original before/final manifests. Archive:

- /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/root/scalar-shadow-frontier01/scalar-shadow-frontier.tar.gz
- SHA256 1584191be158b2d8e3afb19d972c1e118c22679f943fca9411253b06a19c89be
- 127533916 bytes.

Separate Slurm19741 completed0 in4s, re-read the archive hash and complete tar census on the worker with OrangeFS already mounted. This is a separate process readback, not a cross-node availability test. The runtime DSO hash was captured at export time and is explicitly not a retroactive pre-execution guard; original manifests remain unchanged. Compact Git receipts omit ELF/oracle binaries that are present in the durable archive.

The lowerer is under another lane's active claim. Review proposals and further numerical/metadata controls remain separate from permission to edit that file, a candidate build, actual compiler replay and whole-stdlib acceptance.
