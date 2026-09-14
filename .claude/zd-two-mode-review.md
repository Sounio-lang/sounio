# ZD two-mode corollary — 2026-09-06

Worktree: /workspace/.wt/zd-two-mode-20260906
Branch: research/zd-two-mode-20260906
Base: b959a02f4f6bed27bf2a0c11aed9a0620edae069
Validation: locally Lean-checked; no CI or Sounio runtime claim. Publication state is recorded by Git.

## Added files
- formal/lean4/SounioZDTwoMode.lean: abstract integer recurrence, modes, closed form, initial equality suffices, initial recovery, iff for first-step homogeneity.
- formal/lean4/SounioZDTwoModeBridge.lean: concrete P3 sums, link to both existing recurrences, all-level solution, initial-only cp2 hypothesis, exact W12 seed, W12 all levels, insufficiency of s3 alone.

## Verification
Lean 4.33.0, commit d8b18978322de05a8f3dba51ef03cf5461676c17.
Existing cached SounioZDFiberAntisym.olean was stale: cp2_level_recursion unknown.
Recompiled current SounioZDFiberAntisym.lean to isolated .validation directory: exit 0.
This reused other existing imported olean dependencies; it was not a clean rebuild of the complete Lean tree.
New abstract module: exit 0; bridge final build: exit 0.
Axiom output for main corollaries: propext, Classical.choice, Quot.sound. W12 seed: propext only. No sorryAx or new axioms. Seed uses kernel decide, not native_decide.
Evidence: .validation/algebra.log, dependency.log, dependency.exit, bridge-2.log, bridge-2.exit.
Recheck script: bash .validation/verify.sh (run in tmux remotely).

Source SHA256:
- SounioZDTwoMode.lean: 95defe16c4e7fda08ccf771e2ec06c264599e4b791056dbba52a09fb06d68858
- SounioZDTwoModeBridge.lean: ebbea85f55cc4851bd099b3b71015131e08fd1de64f0f1c61bb7170b72866a65
- SounioZDFiberAntisym.lean: 7df8174484c807055717eab477932aaf9715e75164b9fb831239b7bb4fa3286d

## Orthogonal reviews
- xai / Grok 4.5, math-review of abstract module: OK, no mathematical issue identified; .validation/review.txt, /tmp/llm-offload-yz1rJA.
- xai / Grok 4.5 and Qwen 3 235B: supplementary whole-draft read, .validation/review-full.txt. Preceded the doc-comment placement correction; not a final syntax check.
- Qwen 3 235B via OpenRouter, math-review of final combined source: all four checks OK; .validation/review-final.txt, /tmp/llm-offload-QgnI9B.
- Z.AI unavailable (account rate restriction); DeepSeek fallback unavailable (authentication). Neither counted as a pass. Qwen supplied the independent second provider.
Reviews are secondary evidence; Lean compilation supplies formal checking.

## Scope
A corollary of the existing transfer, not claimed as new mathematics in the literature.
For every natural i: s(3+i,12)-s(3+i,1)=1152*(8^i-4^i).
The seed difference is (D,C)=(0,192), established directly from P3 by decide.
This classifies the two-observable trajectory, not all labels, algebraic equivalence or the full spectrum.
No existing research primitive, source theorem or shared checkout file was modified.
The broader mapping from arbitrary label bits to initial (D,C) remains unsolved by this work.
