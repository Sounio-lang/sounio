# Standalone generic enum payload qualification

This tranche specializes explicit generic enum declarations in a standalone source module. It is not the imported-core or HeapVec qualification.

The parser retains ordered enum type parameters. The specializer clones real variants and recursively substitutes their payload types, assigns an unspellable concrete nominal identity, and rewrites qualified constructors and patterns under an expected concrete type. Cache equality uses the declaration slot and the complete bounded type key. A constructor for another enum or an unsupported identity refuses instead of erasing the payload. Existing generic-function identities retain their previous spelling for previously admitted shapes.

Slurm 18607 built the exact source overlay a7efc304d40b649a3d5716d756f7a05ddf8394c74f14b39ff124f45d4e1a44df using seed9cf. The raw compiler SHA256 is 48ebc7e96b0f07e5927d7b91fcfe432d2889fe0bff3e16fc978c3581957e4643. Source-manifest digest: e72a781edd4376c7d95b2b6297cbe035b01fdcb85d787d90aa244ba6cb713056.

The original simultaneous WaveOption<i64>/WaveOption<f64>/WaveResult<i64,f64> consumer and concrete owned-enum transfer both changed from CHECK1 on c23 to CHECK0/COMPILE0/NATIVE0. The identifier argument boundary makes one linear-consumption decision, rather than consuming once while typing and again at the call boundary.

Slurm18657 and18661 exercised actual nested Option<Result<OwnedToken,i64>>, Unit payloads and a shared borrow of an enum with an integer payload. Both the distinct Wave names and the canonical Option/Result spellings executed natively with the expected marker. Five invalid programs reject with the expected ownership/type/specialization diagnostic: duplicated owner, unconsumed owner, wrong concrete instance, wrong payload type and wrong declaration.

The type-key function was independently executed using the exact seed in18626: eighteen distinct AST shapes, exact array extents, reference/pointer mutability, tuple and function shapes, and order-independent effect sets passed. Unsupported or incomplete type forms refuse. This component result is separate from Gen2 compiler validation.

Open prerequisites remain explicit:
- Multiple loaded source modules are deliberately refused when generic enum templates need this path. Exact per-occurrence module ownership and named/glob/reexport resolution are a following tranche.
- A reference stored inside an enum payload remains E264. The pending borrowed-payload source is retained in fixtures as an intended positive prerequisite, not reclassified as a negative acceptance test.
- Returned Option<&T>/Option<&!T> needs persistent owner-bound loans and stable value-cell access before HeapVec can qualify.
- Raw typed moves, initialized-slot invariants, nonaliasing aggregate transfer, wide values, capturing closures and storage of borrowed T have no qualification claim here.
- Whole-stdlib acceptance and Gen2 fixed-point validation remain open.

Closed receipts are under /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/{generic-enum-active,generic-enum-controls,generic-enum-canonical-controls,generic-type-keys-seed}. No original stdlib positive test was weakened or relabeled.

## Imported declarations by physical module identity

The following tranche retains each Item's exact loaded source-module index before merging. Generic function/struct clones and generated concrete enums keep their declaration's index. Checker and lowering prefer this per-occurrence index over the older name-only ownership registry. Enum lookup uses the loader's per-use physical target map, distinguishes explicit named imports from glob imports, follows public reexports, rejects private access and ambiguous declarations, and includes the declaration slot in the concrete identity.

Slurm18673 passed the seed component that specializes equal-named enums from distinct source modules through a reexport and verifies distinct concrete symbols and exact generated declaration owners. The first18672 component compile failure is retained.

Slurm18679 built generic-module-v3-overlay SHA256 3d864fc798303ba76bc033572a01930b36c5908956b68a15a4efc801698dc130. Raw compiler SHA256 e9ad6cde0328e1dbf249da726fb44a798b0dd575236d5ec4040cd309f558b03e; complete source-manifest SHA256 ac431f502a2b4f860911f567569a31c73f674bfb9f53222784b3bf0a3eb7116e. Standalone generic/owned enum controls and the imported/reexported and named-over-glob consumers each passed check, compile and native execution. Wrong concrete declaration, private imported/unimported constructors and ambiguous glob imports rejected. Source, build-input and raw-compiler final guards passed; WORKER_STATUS=0. The worker collected diagnostic exits rather than enforcing the matrix, so these explicit exit files and native markers are the acceptance evidence.

The one-module refusal described above is superseded for this tested imported-declaration subset. Imported core Option/Result with returned borrowed payloads, persistent owner loans, raw typed storage, HeapVec and full-suite acceptance remain unqualified. Capacity/native WIP used by this sealed build is recorded separately and is not part of this source tranche. Current tracked source must be rebuilt as a coherent integration before whole-stdlib acceptance.
