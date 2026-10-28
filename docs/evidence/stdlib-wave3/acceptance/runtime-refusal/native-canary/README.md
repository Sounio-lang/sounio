# Native Sounio canary for runtime refusal

Slurm19352 checks and compiles an actual Sounio program with sealed raw826d9c18 and then runs its native ELF through the canonical runner. The declared exit1/diagnostic case passes. A wrong diagnostic annotation and an unexpected successful native exit are both rejected. All three cases have check0/compile0; native exits are1/1/0, respectively. No numerical calculation or library-correctness claim is attached to this operational canary.

The local-model proposal omitted string quotes and failed E137; it was corrected before this Slurm qualification. Retained ELF hashes and full capture records remain under runtime-refusal-native01/results.
