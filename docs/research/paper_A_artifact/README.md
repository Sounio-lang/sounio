<!-- docs:meta
topic_id: repo.docs.research.paper-a-artifact.readme
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-artifact.readme
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Paper A artifact — *Manufacturing Precision Is a Type Error: Compile-Time Anti-Garbling for Uncertainty-Typed Languages*

Everything the paper cites that runs on this tree, with the values it cites and the command that
regenerates them. `bash docs/research/paper_A_artifact/reproduce.sh` runs all of it (~10 min; the
Lean gate is ~3 min of that) and **fails** on any mismatch. It also refuses to run on anything but
the committed compiler (`SOUNIO_REQUIRE_COMMITTED_MADAROS=1`): on 2026-08-31 a day of measurements
was silently made on a stale local build, so the artifact checks the binary's identity first.

## Identity

| what | value |
|---|---|
| repository | `Sounio-lang/sounio`, branch `main` |
| compiler | `bin/madaros-linux-x86_64` (committed prebuilt, Madaros v0.80.0); `md5sum` printed by `reproduce.sh` |
| second engine | `SOUNIO_SOUC_ENGINE=lean_single` — every Sounio program below is byte-identical under both engines except `vancomycin_auc_epistemic.sio` (builtin `Knowledge<f64>`; lean_single only, #1706) |
| Lean | 4.33.1 (`formal/lean4/lean-toolchain`), Mathlib-free |
| cohort seed | LCG `seed = 20260831`, `next = (seed·1103515245 + 12345) mod 2³¹`; Monte Carlo stream seed `8311971` |

## What regenerates what

| paper claim | artifact | recorded value |
|---|---|---|
| §6 metatheory: Lemma 1 (general), Lemma 2, NS progress/preservation, exactness preservation, Theorem 6.4, partition lemma, sign theorems, x+x/opaque/let witnesses | `formal/lean4/EpistemicEffectsNS.lean` via `scripts/ci/ns_metatheory_lean_gate.sh` | `NS_METATHEORY_LEAN_GATE_PASS`, 16 theorems in the axiom footprint, ⊆ {propext, Quot.sound, Classical.choice}, sorry-free |
| §7/§8.2 analysis-level prototypes | `docs/research/sounio/noise_symbols.sio`, `ns_dataflow.sio`, `ns_contract.sio` | souc-green; `ns_contract` 5/5 PASS |
| §8.4 RQ4 flip rate, scenario B (interval sum, ρ = 1) | `docs/research/sounio/rq4_vanco_two_compartment_flip.sio` | `B_silenced=311` of `B_true_warn=909` (34.2 %), `B_var_ratio_permille=500` |
| §8.4 RQ4 flip rate, scenario A (phase partition, Cov < 0) | same | `silenced_sum=0`, `spurious_naive=1894`, `var_ratio_sum_permille=1204`, `var_ratio_naive_permille=300662` |
| §6.4 (v) Monte Carlo adequacy | `docs/research/sounio/rq4_vanco_mc_adequacy.sio` | `var_mc_over_t_permille=999`, `warn_mcsd=911`, `warn_mcq=877`, `var_n_over_mc_permille=300917` |
| §5.5 the implemented escape valve (exact propagation) | `stdlib/epistemic/affine.sio` + `tests/run-pass/affine_*.sio` | three `… PASS` lines |
| §8.4 clinical receipt AUC 450 ± 44, CI [361, 539] → WARN | `examples/vancomycin_auc_affine.sio` (both engines); `examples/vancomycin_auc_epistemic.sio` (lean_single) | `GUM_GATE=WARN_SUBTHERAPEUTIC_POSSIBLE` |
| §8.2 the compile-time E230 rule and its sabotage gate | **not on `main`** — the NS wire (N1–N4, `noise_sets.sio`, E230 at `kadd`/`kmul`, `SOUNIO_NS_DISABLE`, `scripts/ci/ns_antigarbling_gate.sh`) is on the integration branch rebased as `fable/ns-wire-rebase-20260831`, pending codex-2's review | see that PR; the artifact does not pretend the rule is in `main` |

## Honest boundaries (the paper's §6.4/§6.5/§10, in artifact terms)

- The Lean file mechanizes the **core calculus**; the correspondence between the calculus and the
  production checker is a manual argument backed by the four controls and the sabotage gate.
- Every §8 measurement is of the **intraprocedural** checker (§5.6 summaries are not implemented).
- The cohort is synthetic (a deterministic LCG grid over plausible ranges), not a registry; the
  Monte Carlo draws from the same four sources the type system labels — there is no oracle.
- `add_correlated` (`stdlib/epistemic/gum_supplement1.sio`) is in-tree and unit-tested only; the
  proved-disjoint certificate path is not implemented. The implemented escape valve is the affine
  type, which computes the exact sum instead of asking for ρ.

## Reviews on record

`paper_A_ns_metatheory_xai_review_2026-08-30.md` (Lean: Grok 4.5, 4.6, 4.6-on-fixes, Kimi K3),
`paper_A_prose_review_grok_2026-08-31.md` (text: merit 2, all three "changes that would raise the
score" acted on the same day).
