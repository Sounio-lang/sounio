`llm::epistemic_clinical_pbpk::EpistemicDecisionReceipt` is now opaque: its
fields are private and only `clinical_pbpk_evaluate_gate` can build one, so a
caller can no longer write a receipt by hand with `authorized: true` (lie
corpus item 1c). Read the fields through the `epr_*` accessors. Acting on a
decision takes `epr_authorization(r)`, which re-derives the `SafeAuthorization`
from the enclosure and window stored in the receipt; the `epr_authorized` bool
is for reporting only and is not accepted where a `SafeAuthorization` is
required. Pinned by `tests/compile-fail/lie01e_decision_receipt_cannot_be_forged.sio`
(E176), `tests/compile-fail/lie01f_bool_is_not_an_authorization.sio` (E009) and
`tests/run-pass/lie01_pbpk_sound_auc.sio`.
