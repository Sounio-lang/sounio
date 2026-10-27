# Isolated lower.sio authorization and narrow-memory adoption

Parent relayed the user's explicit approval at 2026-10-27T14:16Z: “Autorizar a edição isolada dos trechos separados”.

This permits separate regions in this lane's own worktree for the approved stdlib plan. It does not permit editing the v13 lane's checkout, branch, cast-lowering regions, or main. Coordination scope/check retained the expected active whole-file conflict; the explicit user exception permits this disjoint change. The scope request and claim attempt were recorded outside Git under /workspace/.tmp/stdlib-wave3-compiler/LOWER_EXCEPTION_{SCOPE,CHECK,CLAIM}.txt and announced to the active owner on the coordination bus.

This tranche adopts the exact narrow-memory producer already tested in the sealed Slurm 18159 diagnostic. Original lower.sio SHA256: bc8c89852a89e6455be3c59a23d4b0db349fc61c64dcde4dc8676df2d1ea6031. Adopted SHA256: 023be9fd1c6c7426a170b569c01459d0da1eb842031a46229b6d062ed19cfaf9.

lower_cast_expr_ref is byte-identical before/after (SHA256 3277cee760a2a1e48d128e161d930ef9a793ae3a7ea73b62b665293fa2195df4). The exact change adds raw-pointer memory-width metadata to local/capture tracking and load/store IR flags; references retain canonical word-cell semantics.

The diagnostic artifact is evidence for the proposed behavior, not the final source-qualified compiler. A fresh rebuild from this committed adoption, preserved source and seed manifests, and the original twelve width controls is required before acceptance. Raw-pointer fields and arbitrary returned-pointer inference remain outside this narrow metadata coverage.

The subsequent protected-print tranche changes only the helper for compiler-generated print identities and synthesized print/println/newline lookup sites. It uses the approved unspellable @native: namespace; explicit source calls and the floating formatter arithmetic remain unchanged. Exact diff retained as PROTECTED_PRINT_ADOPTION.diff outside Git. Fresh before/after exact-output and original runaway tests remain required.
