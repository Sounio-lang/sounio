# Worker capability admission: issuance deliberately closed

This component introduces compiler-owned WorkerFnI64 and WorkerFnF64 identities. Their TypeKind tags100/101 are appended after the existing tags96..99. They are incompatible with each other, ordinary integers/floats/functions, unknown and error carriers. Source casts and record literals cannot construct them.

The constructor front end requires a visible ordinary named function item with an exact unary i64-to-i64 or f64-to-f64 signature. Local function aliases, capturing closures, methods, kernels, extern roots, ABI mismatches and declared effects outside local mutation/panic/division are refused. This is signature admission only: it does not yet prove a transitive runtime-state-free closure.

**No native capability is issued by this component.** Lowering explicitly refuses both constructors and both raw extractors with a WorkerFn diagnostic and no ELF. Forged casts and record literals are also refused independently of checker admission. Protected identities2001/2002/2017/2018 are centralized for the later implementation; the native module currently supplies identities, not a certificate.

The raw extractor remains unconditionally closed until the separately qualified loader-origin helper is adopted and the final proof is connected. The future canonical origin is the physical stdlib/ffi/worker.sio file. A textual module namespace cannot grant extraction authority.

## Actual components

Slurm **18903 COMPLETED0**, elapsed1m16:

- Actual check::types/compat component executes WORKER_OPAQUE_TYPE_COMPONENT_OK.
- It verifies existing wide-format identity tags, distinct new tags100/101, same-kind compatibility, cross-ABI refusal, primitive refusal and refusal before unknown/error wildcards.
- Actual checker accepts two named scalar candidates and rejects eight adversarial cases with a WorkerFn diagnostic: wrong root ABI, capture, local alias, forged cast, raw extraction, declared Alloc, mismatched capability ABI and forged struct.
- C++23 strict verifier rejects seven wrong-output mutations. Parser, undefined-API and generic unsupported refusals cannot substitute for a WorkerFn refusal.
- All46existing loan cases and their seven wrong-output controls remain valid.
- Final source/feature/seed/reference guards pass.
- Checker ELF SHA256 a5f3663180712ef238f80d3e373fcaee981b78a562a9ecbb8be2a4485144bbbd.
- Type component ELF SHA256536687183b625cefaf21e6833c7c87561307808b3b67a594539047cf3aa26c50.
- Overlay SHA256 cde7fa3d06a95eb51e67674a3cb41c0a072af2bf75649c81cdf83b06c2680a4b.

Slurm **18914 COMPLETED0**, elapsed2m43:

- Actual parser/specializer/lowerer/native-emitter component refuses all ten capability-related inputs with an explicit WorkerFn native-capability refusal and no ELF. This includes the two signature candidates: no native certificate is implemented.
- An ordinary source function still compiles and executes WORKER_CLOSED_CONTROL_OK.
- Final source/feature/seed guards pass.
- Actual pipeline ELF SHA256 b86be5590041807192599d2a665d7700fb29a448d4b9e6d08a764b9927df3520.
- Ordinary control ELF SHA256 c96ff92ca1d1d3a585a4573883f8447e556eee13b6612548658d4e30ef2f5d3e.
- Overlay SHA25642f147e4cd69ac04a8b087fd645dacf787683ecbcc7974fd2a393ed058dd1886.

Durable directories are /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-front-v1 and worker-native-closed-v1.

## Remaining implementation

The complete feature still needs the loader-issued physical-origin check, checker finalization of source closure requests after all bodies are checked, final-IR capability value-flow and transitive closure proofs, isolated worker clones, and proof-controlled omission of region operations and stack-store barriers inside those clones. The original serial helpers must preserve their barriers.

Capability provenance must be validated at every raw extraction, not merely at constructor calls. Cached/serialized IR cannot restore a certificate from a flag. Unsupported capability carriers and unknown/mixed provenance remain refused until proved.

The services pool APIs, normal scalar closures and thread migration are separate consumer gates. This receipt is not a WorkerFn native-feature or parallel-runtime acceptance claim.
