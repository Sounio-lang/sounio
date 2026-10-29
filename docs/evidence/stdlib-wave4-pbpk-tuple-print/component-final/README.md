# Published component and durable custody

Committed candidate: `101c26fbaa`, helper SHA256 `f5a20a8c978477127b3ab2fe4f7f6994815003e670fed21f55764fd424d4879e`.

Slurm19772 COMPLETED0:0 in9 seconds. `gate.stdout` is the published gate marker; `gate/results/STATUS` is0. Both modern checks and the legacy compile/native phase passed. The full compile output and all source/tool/ELF guards are in the archive; the compact receipt keeps their exact tool, ELF and extracted Worker-prefix identities plus native/check stdout.

Slurm19773 COMPLETED0:0 in15 seconds exported the six complete component attempts, including failed investigations, exact sources, recipes, outputs and ELFs, plus copies of the executed raw04 compiler and legacy9cf seed. Source manifests were verified, the archive census and SHA readback succeeded, copied compiler/seed bytes matched the executed files, and before/after input manifests matched on the real mounted OrangeFS filesystem.

Archive:
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/pbpk-binding-stamp-export01/pbpk-binding-stamp.tar.gz`

SHA256 `5855633059667b4f8a2c64724f69d055d9112789774d1c6134711089c223cb05`.

This qualifies the helper and the controlled AST publication/source-profile component. It does not qualify the complete checker, original PBPK compilation or a newly rebuilt compiler. See `../CHECKER_DELTA.md` for those remaining gates and the preserved modern-compiler SIG139.
