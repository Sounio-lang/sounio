`llm::epistemic_clinical_pbpk::EpistemicDecisionReceipt` is now opaque: its
fields are private and only `clinical_pbpk_evaluate_gate` can build one, so a
caller can no longer write a receipt by hand with `authorized: true` (lie
corpus item 1c). Read the fields through the `epr_*` accessors. Acting on a
decision takes `epr_authorization(r)`, which now returns an opaque
`PbpkAuthorization` re-derived from the enclosure and window stored in the
receipt. A generic `SafeAuthorization` (which anyone can build from
`se_point`) is not accepted where a `PbpkAuthorization` is required; the
converse is available as `pbpk_auth_as_generic`. `epr_can_authorize(r)` reports,
without panicking, whether `epr_authorization` would succeed. The
`epr_authorized` bool is for reporting only. Pinned by
`tests/compile-fail/lie01e_decision_receipt_cannot_be_forged.sio` (E176),
`lie01f_bool_is_not_an_authorization.sio` (E009),
`lie01g_generic_token_is_not_pbpk_authorization.sio` (E009),
`lie01h_pbpk_authorization_cannot_be_forged.sio` (E176),
`tests/run-pass/lie01_pbpk_sound_auc.sio` and the regime re-derivation check in
`tests/llm/llm_phase7_blindada.sio`. Synthetic research witness only.
