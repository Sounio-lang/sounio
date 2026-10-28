# Repair declared mutation and public value contracts

The144-program activation exposed missing Mut declarations in a borrowed DataFrame sort helper, covariance setters/propagation, and GUM covariance construction. Those functions now declare the mutation they perform. Perm128 and its fixed value array are public, matching the existing public permutation functions and documented caller construction. The approximation and uncertain-octonion tests now declare the already-required Approx and NonAssoc effects. Algorithms, tolerances and expected numerical values are unchanged.

Slurm19196 replays the12 selected failures through the canonical E2E runner in isolated source copies:10pass/2fail/0skip. The remaining errors identify GUM's private cov_set and the octonion caller effect. After the corresponding declarations,19207 replays exactly those two:2pass/0fail/0skip. Both jobs have zero selection/source identity errors. The10 earlier successful cases are not relabeled as an integrated12-case job.

All12 selected original runtime checks now have passing receipts. Root-wide integration and the remaining activation/whole-suite failures remain open. The compiler in both jobs is the sealed bd4ba240 raw; no Python/Rust/Julia numerical implementation was executed.
