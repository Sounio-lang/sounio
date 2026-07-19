<!-- docs:meta
topic_id: repo.docs.gpu.dyck-ssm-poc
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.gpu.dyck-ssm-poc
-->

# "Trainable to converse?" — a proof-of-concept: the octonion SSM as a next-token model

The question: could the non-associative octonion SSM be **trained for next-token prediction** (the core of
a conversational model), and does non-associativity **help on nesting structure** — the one principled
hope, since `BRACKETING_TASK.md` showed the octonion associator reads bracketing? We built the minimal
honest test on **Dyck-k** (balanced nested brackets), the canonical language where predicting the
**closing** bracket type requires tracking the nesting stack.

## Setup
All three models share embedding, nonlinearity, readout, optimizer, and **matched state dimension 32**;
they differ only in the state-mixing algebra:
- **OCT** `h_t = tanh(A ⊗ h_{t-1} + B x_t)` — octonion product (**non-associative**), 4 channels ×8.
- **QUAT** — same with quaternion product (**associative** hypercomplex control), 8 channels ×4.
- **REAL** `h_t = tanh(M h_{t-1} + B x_t)` — real matrix (**associative** linear SSM).
Trained with BPTT + Adam (400 iters). The octonion BPTT is finite-difference gradient-checked (err 1e-8).

## Result (Dyck-3, test set)
| Model | loss | next-tok acc | **closing-bracket acc** |
|---|---|---|---|
| OCT octonion (non-assoc) | 1.341 | 49.8% | **88.6%** |
| QUAT quaternion (assoc) | 1.312 | 50.2% | 88.8% |
| REAL linear matrix (assoc) | 1.278 | 50.8% | 89.7% |

(Overall next-tok ≈ 50% is the irreducible entropy of *opening*-bracket choices; the informative metric is
the **closing**-bracket accuracy, where the stack determines the answer.)

## Two honest conclusions
1. **Trainable — yes.** The non-associative octonion SSM trains through BPTT and *learns the nesting*:
   88.6% closing-bracket accuracy (chance among 3 types ≈ 33%), gradient-checked. Mechanically, it could
   be trained on language.
2. **Non-associativity does not help — here.** OCT (88.6%) ≈ QUAT (88.8%) ≈ REAL (89.7%); if anything the
   associative baseline edges ahead. No non-associative advantage appeared for sequential nesting. This is
   consistent with `OCTONION_SIGNATURE_BRIDGE.md`: the octonion associator is a *specific* (static,
   alternating G₂-3-form) operation, not a general nesting/higher-structure tool — and bounded-depth Dyck
   is **regular**, so an associative recurrence already suffices.

## Verdict on "trainable to converse?"
Mechanically yes; **but no demonstrated benefit** over associative models, and a real cost: octonion
non-associativity **breaks the associative parallel scan** that makes modern SSMs (S5/Mamba) fast, so
training is sequential BPTT — far slower at scale. On this evidence there is **no compelling path to
conversation** for the non-associative model versus associative baselines. We did **not** search for a
sequential task engineered to favor it (that would p-hack the positive). The honest place the whole
program lands: the octonion tensor-core machinery is a real, trainable, exactly-differentiable
sequence-model primitive whose non-associativity pays off **only** for static alternating-associator
signals (`NONASSOC_HEADTOHEAD`, `BRACKETING`), not as a general language/sequence advantage. Harness
`dyck_ssm.py`.
