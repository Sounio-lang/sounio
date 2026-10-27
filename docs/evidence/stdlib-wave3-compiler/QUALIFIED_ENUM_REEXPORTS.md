# Canonical generic enum paths through public reexports

A consumer of a public facade can name the enum's canonical defining path. The earlier imported-enum resolver searched only the consumer's direct use paths. With core::lib reexporting core::option::Option, an annotation core::option::Option<i64> did not resolve and the checker saw the prefix core before the specialization poison refused compilation. The original core witness is retained by the core lane, Slurm18753, with E018/E009 and failed compilation; unqualified exports passing do not replace it.

The resolver now follows the loader's exact physical target map through public reexports of the requested terminal name. It preserves declaration-slot identity, retains ambiguity refusal, and never traverses a private dependency as an exported canonical namespace. This adds no runtime representation.

Slurm18764 executed the same AST specialization driver before and after this change, using sealed seed9cf. Before: native exit1 at the specialization-poison assertion. After: native exit0 and GENERIC_ENUM_QUALIFIED_REEXPORT_AST_OK. The driver verifies that a qualified parameter's concrete symbol equals the actual defining enum's generated symbol and differs from a homonymous enum in another module. A second invocation changes the facade's public use to a private use; the qualified type must again poison specialization. Both source and seed final guards passed. The earlier18762 stopped before compilation because its overlay manifest incorrectly hashed itself; that failed receipt remains intact.

Custody: /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/generic-qualified-component-v2/results/{before,after}. Overlay SHA256 be55d4cecc41fe37a4e8d4d87cb41d25746830b084962c3652e54771cf3269ce. After specializer SHA256 e497a3b45039f2b7facc2dc6b0594e42f4c23c554ba2c8172c686bfaed7482b9. Driver SHA256 af05ca6a34b9b60501844c244641fa8f1cc83005a1da389afc03094330bca968.

This is an executed specializer component result. A rebuilt integrated compiler must replay the unchanged core full-qualified witness before that user-facing path is accepted. Returned reference payloads and HeapVec remain separate prerequisites.
