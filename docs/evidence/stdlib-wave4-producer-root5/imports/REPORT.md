# ItemKind exports: exact CausalGraph coverage

The collector contained ItemClaim twice and omitted ItemCausalGraph. Replace only the second, redundant arm with an explicit CausalGraph no-op. Scientific graph metadata remains outside ordinary symbol exports. No wildcard or exhaustiveness relaxation.

The gate imports the actual ROOT5 parser AST, SymbolKind and resolve::imports modules and calls public collect_module_exports. It constructs the actual Item and ItemList records, enumerates all 28 actual ItemKind variants and private/public visibility, and inspects pool base/count/delta, exported SymbolKind, Name bytes and dropped count. No projected enum or ABI is used.

Slurm19806 compiled the before/after modules using both seed9cf and final private ccfa. The final producer correctly diagnoses the BEFORE collector as nonexhaustive, although the preexisting imported-error policy returns0 and emits an unqualified ELF. That ELF was never executed. AFTER has no error diagnostics with either producer. Legacy assignment warnings remain unchanged.

The first native attempts in19806 returned126 because positional producer output lacks executable mode. Slurm19807 chmodded the retained byte-identical ELFs and ran AFTER old/final and BEFORE old: all native exits0 and independent C++23 oracle56/56, with stdout byte-identical across all three. No recompilation or source change was required.

Slurm19810 verified all hashes again and exercised C++ wrong-kind, private-export, missing-row and duplicate-output controls: all four refused; valid56rows accepted. Native payload resource/status/output receipts are retained. The C++ reference derives enumeration order from the actual AST/SymbolKind sources and applies an explicit independent expected export mapping.

Setup predecessor19805 used an unset TIME variable and did not invoke the producers. This technical failure is not a semantic baseline. The actual compile evidence is19806; actual native evidence19807; protocol controls19810.

Source before SHA256 cca2bdf1b35d1ed3eff4aaab0cebb145ddef30aeee8edfc698edb269af7e1b98.
Source after SHA256 35ff54f4a9d85be019cebe7f9c8f5881a3cf3a393265e229a3281ea1bb53918e.
Old producer9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a.
Final producer ccfaec48d1bae176091ef1f3401c2d863c1f05284390f3c50ca9d66e94c7114f.

All gates ran on Slurm dl380,1CPU/8GiB. Scratch custody:
 /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-imports-real02

This qualifies the narrow collector repair only. ROOT5 main build19789 remains FAILED/noRAW_READY; no modern bootstrap or historical scientific replay was run by this gate.

