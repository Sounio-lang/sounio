The user wants a hostile review of a specific code snippet and architectural description for an interval transport worker.
Author: Demetrios Chiuratto Agourakis.
Context: PLs, Lean 4, clinical pharmacology (though this specific artifact is a Python interval transport worker).

Input artifact description:
1. Partial-branch hulls (`event_time`, `event_derivative`, `event_normal`) are only emitted if `upward_returns` exist AND `not unresolved`.
2. Independent verifier checks: if status is `TRANSPORT_REFUSED`, then `terminal_domain_cover_certified` must not be True, and `event_time`, `event_derivative`, `event_normal` must NOT be present. If they are, `fail(...)`.
3. A negative mutation test inserts a partial hull and verifies it's rejected.
4. Fail-closed: All unexpected exceptions terminate without a receipt. Single-threaded, one-shot Slurm job. Global hooks restored by `finally`. Imports and frozen prior receipt bound by SHA-256.

Questions to address:
1. Is the partial-hull under-approximation fixed for every refused result?
2. Does this preserve useful hulls for complete returns and `FINAL_SYMBOLIC_DEPENDENCE_LOST` (which has complete domain cover but fails 6-variable acceptance gate)?
3. Are there any remaining BLOCKERS? (Do not invent clinical implications, it's a dynamical-systems falsifier).

Let's analyze the logic:
- Implementation logic:
  `if upward_returns and not unresolved:` -> emits partial hulls.
- If there are unresolved regions, `not unresolved` is False, so it does not emit the hulls. Thus, status will likely be `TRANSPORT_REFUSED`.
- If it emits the partial hulls, does the verifier allow it?
  Verifier: `if transport.get("status") == "TRANSPORT_REFUSED":`
    checks `event_time` etc. If present, `fail()`.
  So if status is `REFUSED`, it forbids the keys.

Wait! Look closely at the questions:
1. Is the partial-hull under-approximation fixed for *every* refused result?
- In the verifier, it only fails if `transport.get("status") == "TRANSPORT_REFUSED"`.
- What if the status is NOT `TRANSPORT_REFUSED`, but the domain is actually unresolved, yet it outputs the partial hulls?
- Wait, if `upward_returns and not unresolved`, the implementation emits the hulls. What status does it get? If `not unresolved` is True, then it's either a successful return or a `FINAL_SYMBOLIC_DEPENDENCE_LOST`.
- But what if `unresolved` is True, and `upward_returns` is True? The implementation does *not* emit the partial hulls. But does the implementation *also* set the status to something other than `TRANSPORT_REFUSED`?
If the status is `TRANSPORT_REFUSED`, the verifier rejects the partial hulls.
If the status is something else, the verifier allows the partial hulls. Does the verifier check if a `TRANSPORT_REFUSED` status is correctly assigned when `unresolved` is True? The prompt says: "The independent exact-rational verifier now rejects every refused receipt that contains any of those fields".
Wait, what if a receipt has a non-refused status, but it actually was a partial return? The prompt says: "The prior hostile review found that a partially refused domain could still emit top-level hulls over only the successful branches. The implementation now does: `if upward_returns and not unresolved: emit hulls`".
If `unresolved` is True, it doesn't emit the hulls. But does it guarantee that the status is set to `TRANSPORT_REFUSED`? The verifier relies on the `status` field to gate the rejection of the partial hulls.
What if the implementation forgets to emit the partial hulls (due to `unresolved`), but ALSO forgets to set the status to `TRANSPORT_REFUSED`, setting it to `SUCCESS` instead?
Wait, the prompt says "A partially refused domain". If it's partially refused, does the implementation correctly flag the status?
If the verifier only looks at `status == "TRANSPORT_REFUSED"`, it completely trusts the `status` field. If the implementation incorrectly sets `status = "SUCCESS"` but the computation actually had `unresolved` regions, the verifier will accept it, and it will accept the partial hulls if they were emitted.
Even if the partial hulls are NOT emitted, the verifier must check if the status is correctly set. The verifier doesn't have access to `upward_returns` or `unresolved` unless they are in the receipt.

Let's read the verifier snippet carefully:
```python
if transport.get("status") == "TRANSPORT_REFUSED":
    if transport.get("terminal_domain_cover_certified") is True:
        fail("refused transport claims a terminal domain cover")
    for key in ("event_time", "event_derivative", "event_normal"):
        if key in transport:
            fail(f"refused transport exposes a partial {key} hull")
```
This verifier ONLY checks what a `TRANSPORT_REFUSED` receipt must NOT contain. It does NOT verify that a partial domain cover *must* result in `TRANSPORT_REFUSED`. If the implementation has a bug where it sets `status = "SUCCESS"` (or `FINAL_SYMBOLIC_DEPENDENCE_LOST`) on a partial return, the verifier will happily accept it.
The verifier is checking the `TRANSPORT_REFUSED` case, but what if the status string changes? `status == "TRANSPORT_REFUSED"`.
Is there a `status` enum or constant? Using raw strings in the verifier is fragile.

Wait, what about `FINAL_SYMBOLIC_DEPENDENCE_LOST`? The question asks: "Does this preserve useful hulls for complete returns and for the distinct `FINAL_SYMBOLIC_DEPENDENCE_LOST` result, which has a complete domain cover but fails only the six-variable acceptance gate?"
If the status is `FINAL_SYMBOLIC_DEPENDENCE_LOST`, the verifier does NOT fail the presence of `event_time`, `event_derivative`, `event_normal`.
So yes, it preserves useful hulls for that status.

But wait! What exactly *is* the `status` field? The verifier trusts the `status` field unconditionally. If the implementation computes `upward_returns`, determines `unresolved` is True, but the code that sets `status = "TRANSPORT_REFUSED"` has a bug, or if a new partial status is introduced, the verifier might be bypassed.
More importantly, look at the verifier's failure condition:
```python
if transport.get("status") == "TRANSPORT_REFUSED":
    if transport.get("terminal_domain_cover_certified") is True:
```
Wait, if `status == "TRANSPORT_REFUSED"`, `terminal_domain_cover_certified` must not be True.
What if `status != "TRANSPORT_REFUSED"`? Can it have `terminal_domain_cover_certified = False`?
If `status == "FINAL_SYMBOLIC_DEPENDENCE_LOST"`, does the verifier check if `terminal_domain_cover_certified` is True?
The prompt says: "FINAL_SYMBOLIC_DEPENDENCE_LOST result, which has a complete domain cover but fails only the six-variable acceptance gate".
If `FINAL_SYMBOLIC_DEPENDENCE_LOST` has a complete domain cover, its `terminal_domain_cover_certified` SHOULD be True!
Does the verifier check this? No! The verifier ONLY checks `terminal_domain_cover_certified` when `status == "TRANSPORT_REFUSED"`.
It does NOT check that `FINAL_SYMBOLIC_DEPENDENCE_LOST` or `SUCCESS` implies `terminal_domain_cover_certified == True`.
This is a massive MAJOR or BLOCKER issue. The verifier validates the consistency of the *bad* state, but does not validate the consistency of the *good* states. A receipt could claim `status = SUCCESS` with `terminal_domain_cover_certified = False` and `event_time = empty`, and the verifier passes.

Also, look at the mutation test: "A dedicated negative mutation inserts a partial hull and must be rejected." But does it mutate the status? If the mutation just inserts a partial hull into a `TRANSPORT_REFUSED` receipt, it tests the verifier's `fail()` branch. But what if the mutation changes the status of a partial receipt to `SUCCESS`? Is that caught?
Wait, "A dedicated negative mutation inserts a partial hull and must be rejected." This doesn't test the state consistency logic completely.

Let's look at Question 1: "Is the partial-hull under-approximation fixed for every refused result?"
Yes, syntactically. If it claims to be refused, it cannot have the hulls.
But is the *definition* of "refused result" sound? The verifier accepts ANY receipt where `status != "TRANSPORT_REFUSED"`. If an incomplete result is falsely labeled as `SUCCESS` or `FINAL_SYMBOLIC_DEPENDENCE_LOST`, the verifier will NOT flag it.

Let's check the exact prompt text:
"The prior hostile review found that a partially refused domain could still emit top-level hulls over only the successful branches."
"The independent exact-rational verifier now rejects every refused receipt that contains any of those fields:"
Wait, if the implementation code is:
`if upward_returns and not unresolved:` -> emits hulls.
What does it do when `unresolved` is True? It doesn't emit the hulls. Does it emit `status = "TRANSPORT_REFUSED"`? The verifier assumes that `status` correctly reflects the `unresolved` state.
Can the verifier mathematically or logically deduce `status`? No, it just reads a string.

Look at:
```python
if transport.get("status") == "TRANSPORT_REFUSED":
    if transport.get("terminal_domain_cover_certified") is True:
        fail(...)
```
What if the status is an unexpected string? e.g. "TRANSPORT_REJECTED" or "UNRESOLVED"?
The verifier does NOT fail if the status is unknown.
It should be a strict match against allowed statuses.
If `status not in {"SUCCESS", "FINAL_SYMBOLIC_DEPENDENCE_LOST", "TRANSPORT_REFUSED"}`, `fail("unknown status")`.

What if `status == "TRANSPORT_REFUSED"` but `terminal_domain_cover_certified` is missing (not True, but absent or False)?
The verifier doesn't check anything else for `TRANSPORT_REFUSED`.
If `status == "SUCCESS"`, there are NO positive constraints checked by the verifier.
The verifier *must* check that `status == "SUCCESS"` or `FINAL_SYMBOLIC_DEPENDENCE_LOST` implies `terminal_domain_cover_certified is True`.
Because otherwise, the verifier is useless for certifying that a *successful* transport actually covered the domain!
The verifier's logic is currently:
- If Refused: Cover must be False/missing. Hulls must be missing.
- If Success: [No checks]

This means an attacker or a buggy implementation can emit:
```json
{"status": "SUCCESS", "terminal_domain_cover_certified": false}
```
And the verifier will say "Verified!". But it's not a success, the domain cover failed.
This is a BLOCKER. The verifier's logic is one-sided.

Let's frame this issue.
1. [BLOCKER] The verifier only enforces negative constraints on the `TRANSPORT_REFUSED` status and enforces zero positive constraints on `SUCCESS` or `FINAL_SYMBOLIC_DEPENDENCE_LOST`, allowing a buggy or malicious implementation to bypass the domain-cover requirement entirely.
   Location: `verifier.py:1-6` (implied)
   Why it matters: The verifier is meant to be an independent exact-rational check, but it unconditionally trusts the `status` string. If a receipt claims `status="SUCCESS"` but omits `terminal_domain_cover_certified` or sets it to `false`, the verifier accepts it. A partial or failed domain can simply be mislabeled to bypass the verifier.
   Fix: The verifier must explicitly check ALL valid states. e.g.:
   If `status == "SUCCESS"` or `"FINAL_SYMBOLIC_DEPENDENCE_LOST"`, `fail()` if `terminal_domain_cover_certified` is not `True`.

Let's look at the implementation snippet.
```python
if upward_returns and not unresolved:
    result.update(
        event_time=interval_json(hull([item.event_time for item in upward_returns])),
        event_derivative=interval_json(hull([item.derivative for item in upward_returns])),
        event_normal=interval_json(hull([item.normal for item in upward_returns])),
    )
```
What if `upward_returns` is empty, but `unresolved` is False?
Then the `if` block is skipped. That's fine.
But wait! What if `upward_returns` is NOT empty, and `unresolved` is True? The partial hulls are skipped.
But what if the implementation does:
`status = "TRANSPORT_REFUSED"` if `unresolved` else "SUCCESS".
Is that enforced by the verifier? No, as discussed.

Another issue: SHA-256 binding.
"every imported source plus the frozen prior receipt is bound by SHA-256 in the independently verified receipt."
If the verifier itself is not hashed, or if the environment can be tampered with? "Global hook replacements are each restored by `finally`". That's good practice but standard.
Wait, "every imported source plus the frozen prior receipt is bound by SHA-256".
Does this include the worker itself?
What about the Python executable and dynamically loaded libraries? "every imported source" usually implies pure-python modules. If numpy/scipy or C extensions are used, they aren't bound unless it's a fully reproducible Docker image with an attested hash. Since the prompt says "Can a third party reconstruct the analysis from the supplied files alone?", we can ask about reproducibility of the environment.

Wait, let's look at Question 3: "Is any remaining BLOCKER visible in this focused disposition? Do not invent clinical implications; this is a dynamical-systems falsifier and explicitly certifies no covering relation, recurrence, chaos, or open problem."
I will explicitly answer the questions at the end or integrate them into the response. No, the prompt says:
"Output format: Numbered list of issues... Do not include praise. Do not include 'overall the work is strong' preamble. Skip directly to issue 1."

I need to output the issues in the exact format requested.

Let's review the verifier logic again.
```python
if transport.get("status") == "TRANSPORT_REFUSED":
    if transport.get("terminal_domain_cover_certified") is True:
        fail("refused transport claims a terminal domain cover")
    for key in ("event_time", "event_derivative", "event_normal"):
        if key in transport:
            fail(f"refused transport exposes a partial {key} hull")
```
Is there any issue with `transport.get("terminal_domain_cover_certified") is True`?
If the receipt doesn't have this key, it returns `None`. `None is True` evaluates to `False`. So it passes.
If the implementation forgets to include `terminal_domain_cover_certified` in the receipt, the verifier allows it.
But for `SUCCESS` status, `terminal_domain_cover_certified` must be True. If it is missing, it is False. The verifier does not enforce this!
This is a major flaw. The verifier must validate the schema completely.
Let's call this BLOCKER 1.

Issue 2: The `status` string matching.
The verifier checks `if transport.get("status") == "TRANSPORT_REFUSED":`.
What if the implementation sets `status = "FINAL_SYMBOLIC_DEPENDENCE_LOST"` or `status = "SUCCESS"`?
The verifier only restricts what `TRANSPORT_REFUSED` can contain.
If the verifier's intent is to reject partial hulls on UNRESOLVED domains, relying on the self-reported `status` string is insufficient. The implementation could emit `status = "SUCCESS"` (or `FINAL_SYMBOLIC_DEPENDENCE_LOST`) on an unresolved domain, and the verifier would allow the partial hulls (or allow a missing domain cover).
Actually, the implementation code is:
`if upward_returns and not unresolved:` -> emits hulls.
If `unresolved` is True, the implementation does not emit the hulls.
BUT does it set `status = "TRANSPORT_REFUSED"`?
If there's a bug where `status = "TRANSPORT_REFUSED"` is NOT set, the verifier allows it. It won't have the hulls, but it might have `terminal_domain_cover_certified = False` and `status = "SUCCESS"` (if another bug exists).
Wait, the whole point of the verifier is to mathematically verify the receipt.
If the verifier does not know what `upward_returns` or `unresolved` are, it cannot verify the core logic. It only checks string equality!
A proper independent exact-rational verifier should verify the *math* (e.g., checking the `event_time` interval bounds, checking the `terminal_domain_cover_certified` against actual interval bounds).
Relying on `status == "TRANSPORT_REFUSED"` means the verifier is just checking Python string formatting, not doing exact-rational verification of the intervals.
Is this a fair critique? Yes, the prompt says "independent exact-rational verifier". If it only checks string keys based on a self-reported status, it is not mathematically verifying the transport properties.

Let's look at the mutation testing claim: "A dedicated negative mutation inserts a partial hull and must be rejected."
This only tests that inserting a key into a `TRANSPORT_REFUSED` receipt fails. It does not test what happens if you mutate the `status` from `TRANSPORT_REFUSED` to `SUCCESS` on a receipt that lacks a domain cover. The verifier would pass it.

What about `FINAL_SYMBOLIC_DEPENDENCE_LOST`?
The prompt asks: "Does this preserve useful hulls for complete returns and for the distinct `FINAL_SYMBOLIC_DEPENDENCE_LOST` result, which has a complete domain cover but fails only the six-variable acceptance gate?"
Answer: Yes, because the verifier only fails the presence of hulls when `status == "TRANSPORT_REFUSED"`.
However, if `FINAL_SYMBOLIC_DEPENDENCE_LOST` has a complete domain cover, its receipt should have `terminal_domain_cover_certified == True`. The verifier does NOT check this. So it would also accept a `FINAL_SYMBOLIC_DEPENDENCE_LOST` that FAILED the domain cover, which breaks the definition of the status!

Let's write out Issue 2:
2. [BLOCKER] The verifier relies entirely on the self-reported `status` string to gate constraints and enforces no positive schema or mathematical bounds for `SUCCESS` or `FINAL_SYMBOLIC_DEPENDENCE_LOST`.
<location: verifier snippet lines 1-6>
<why it matters: An "exact-rational verifier" should not validate system state via brittle string matching. Because it only checks `status == "TRANSPORT_REFUSED"`, a receipt claiming `status="SUCCESS"` or `FINAL_SYMBOLIC_DEPENDENCE_LOST` with `terminal_domain_cover_certified = False` (or missing entirely) will pass verification, completely bypassing the six-variable acceptance gate logic and domain cover requirements.>
<minimal fix: The verifier must explicitly assert positive constraints for all valid states: if `status in ("SUCCESS", "FINAL_SYMBOLIC_DEPENDENCE_LOST")`, `fail()` if `terminal_domain_cover_certified` is not `True`. Furthermore, reject any unrecognized `status` string.>

Let's review Issue 3: Hash binding and Reproducibility.
"every imported source plus the frozen prior receipt is bound by SHA-256 in the independently verified receipt."
Is this a strong guarantee? Python sources can import compiled C extensions or dynamically loaded shared objects (e.g., exact-rational libraries like GMP/MPFR if used). "every imported source" typically means `.py` files. If the environment relies on native binaries (like `gmpy2`, `numpy`, or a custom C++ interval arithmetic library), hashing the Python files is insufficient to guarantee reproducibility of the "exact-rational" arithmetic.
<location: Architecture description - SHA-256 binding>
<why it matters: If the exact-rational verifier or the worker relies on C-extensions or native libraries, hashing only Python "imported source" files does not prevent shared library hijacking or floating-point/native implementation drift, compromising the "exact-rational" and "fail-closed" guarantees.>
<minimal fix: Specify and bind the hash of the complete execution environment (e.g., via Nix, Docker image digest, or hardware-level attestation), or explicitly restrict the verifier to a pure Python exact-rational library (like `fractions.Fraction`).>

Let's review Issue 4: Mutation testing scope.
"A dedicated negative mutation inserts a partial hull and must be rejected."
The mutation only tests the negative case of a `TRANSPORT_REFUSED` receipt containing a hull.
<location: Test suite description>
<why it matters: This single mutation does not validate the logic against state misclassification. A critical mutation test would be to mutate a partial return's `status` from `TRANSPORT_REFUSED` to `SUCCESS`. Since the verifier lacks positive constraints on `SUCCESS` states, this mutation would incorrectly pass verification.>
<minimal fix: Add mutation tests for state transitions: 1) Mutate `status` to `SUCCESS` on an unresolved domain. 2) Mutate `terminal_domain_cover_certified` to `False` on a `FINAL_SYMBOLIC_DEPENDENCE_LOST` receipt.>

Let's refine the answers to the author's specific questions to ensure I address them either implicitly through the issues or in the format. Wait, the prompt requires me to output EXACTLY the format:
```
N. [SEVERITY] <one-sentence problem>
   <location: file:line or §section>
   <why it matters>
   <minimal fix>
```
And I must not include praise, preamble, etc.
I should just write the issues.

Wait, I should directly answer the user's questions:
1. Is the partial-hull under-approximation fixed for *every* refused result?
Answer: No, it is only fixed for receipts that *correctly report themselves* as `TRANSPORT_REFUSED`. (Covered in BLOCKER 2).
2. Does this preserve useful hulls for complete returns and for `FINAL_SYMBOLIC_DEPENDENCE_LOST`?
Answer: Yes, it preserves them syntactically, but fails to validate that `FINAL_SYMBOLIC_DEPENDENCE_LOST` actually has a certified terminal domain cover. (Covered in BLOCKER 1 / 2).
3. Is any remaining BLOCKER visible?
Answer: Yes, the verifier's one-sided positive/negative constraint enforcement and reliance on unverified string state. (Covered in BLOCKER 1, 2, 3).

Let's draft the BLOCKER 1 carefully:
```
1. [BLOCKER] The verifier enforces zero positive constraints on `SUCCESS` or `FINAL_SYMBOLIC_DEPENDENCE_LOST`, allowing incomplete or failed domain covers to bypass verification.
   <location: Verifier snippet lines 1-6>
   <why it matters: The verifier only rejects `terminal_domain_cover_certified = True` when `status == "TRANSPORT_REFUSED"`. It does not check if `SUCCESS` or `FINAL_SYMBOLIC_DEPENDENCE_LOST` actually *have* `terminal_domain_cover_certified = True`. An attacker or buggy implementation can emit `status="FINAL_SYMBOLIC_DEPENDENCE_LOST"` or `"SUCCESS"` with a missing or false `terminal_domain_cover_certified`, and the verifier will accept it, falsely certifying a complete return.>
   <minimal fix: Add strict positive constraints: `if transport.get("status") in ("SUCCESS", "FINAL_SYMBOLIC_DEPENDENCE_LOST"): assert transport.get("terminal_domain_cover_certified") is True`.>
```

Draft BLOCKER 2:
```
2. [BLOCKER] The verifier relies entirely on a self-reported `status` string rather than mathematical proof to prevent partial-hull under-approximations.
   <location: Verifier snippet lines 1-6 / Implementation snippet line 3>
   <why it matters: The prior hostile review found that partially refused domains leaked partial hulls. The implementation logic `if upward_returns and not unresolved:` stops this, but the independent verifier cannot know if `unresolved` was actually `True` or `False`; it blindly trusts the `status` field. If a bug causes the implementation to mislabel an unresolved domain as `SUCCESS`, the verifier will pass it, defeating the purpose of an "independent exact-rational verifier".>
   <minimal fix: The verifier must mathematically or cryptographically validate the `unresolved` state (e.g., by checking the domain partition sums) rather than matching on a fragile status string.>
```

Draft MAJOR 1 (Mutation test gap):
```
3. [MAJOR] The dedicated negative mutation test is insufficient to prove the verifier is fail-closed against state misclassification.
   <location: Test suite description - Negative mutation>
   <why it matters: Mutating a `TRANSPORT_REFUSED` receipt to include a partial hull only tests the `fail()` branch for string-key inclusion. It does not test whether a partial return can bypass the system by spoofing its `status` to `SUCCESS` or `FINAL_SYMBOLIC_DEPENDENCE_LOST`.>
   <minimal fix: Add mutation tests that flip a `TRANSPORT_REFUSED` receipt to `SUCCESS` or `FINAL_SYMBOLIC_DEPENDENCE_LOST` and ensure the verifier still rejects it based on missing domain covers or invalid interval bounds.>
```

Draft MAJOR 2 (Reproducibility / Environment binding):
```
4. [MAJOR] SHA-256 binding of "imported source" does not guarantee reproducibility of exact-rational arithmetic if native dependencies are used.
   <location: Architecture description - SHA-256 binding>
   <why it matters: If the dynamical-systems falsifier or exact-rational verifier relies on compiled C-extensions (e.g., `gmpy2`, `mpmath`, or external C++ interval libraries), hashing only Python source files leaves the exact-rational arithmetic vulnerable to shared library drift or floating-point implementation differences, compromising the "exact" guarantee.>
   <minimal fix: Explicitly enforce a reproducible execution environment (e.g., Nix derivation or Docker image digest) or explicitly restrict the worker to pure Python (e.g., standard `fractions` module).>
```

Let's double check the prompt for the answers to the questions. The prompt poses questions, but the system prompt says "Skip directly to issue 1." and "Output format: Numbered list of issues...". I don't need to write "Answers to your questions:", I will just embed the answers in the issues as requested by the strict system prompt constraints.

Wait, the prompt says: "If the artifact has no issues at the SEVERITY level requested (default: all), respond exactly: NO ISSUES FOUND AT REQUESTED SEVERITY."
Since I found issues, I will just list them.

Let's review the logic of the implementation.
```python
if upward_returns and not unresolved:
    result.update(
        event_time=interval_json(hull([item.event_time for item in upward_returns])),
        event_derivative=interval_json(hull([item.derivative for item in upward_returns])),
        event_normal=interval_json(hull([item.normal for item in upward_returns])),
    )
```
Is there a problem with `interval_json(hull(...))`?
If `upward_returns` is an array of objects with `.event_time`, etc.
Is there a risk of `AttributeError` if `item.event_time` is missing? The prompt mentions "All unexpected exceptions terminate the worker without a receipt." So an unhandled `AttributeError` would terminate the worker and emit no receipt.
But this implies that any data anomaly results in a hard crash with no diagnostic receipt. Is that bad? For a fail-closed falsifier, it's actually good design.

Is there a problem with `upward_returns and not unresolved`?
What if `upward_returns` contains elements, but `unresolved` is a list?
Wait, `not unresolved` - if `unresolved` is an empty list `[]`, `not unresolved` is True.
If `unresolved` is `[item]`, `not unresolved` is False.
This relies on Python's truthiness of lists. That's fine, standard Python.

Wait, the prompt says: "The prior hostile review found that a partially refused domain could still emit top-level hulls over only the successful branches."
The fix is: "Partial-branch hulls are not global enclosures of an unresolved cover."
And the code implements: `if upward_returns and not unresolved:`.
This is exactly the correct logic to prevent emitting hulls when there are unresolved elements.

But what about `FINAL_SYMBOLIC_DEPENDENCE_LOST`? The question states:
"Does this preserve useful hulls for complete returns and for the distinct `FINAL_SYMBOLIC_DEPENDENCE_LOST` result, which has a complete domain cover but fails only the six-variable acceptance gate?"
If it has a complete domain cover, `unresolved` is empty (`[]`). So `not unresolved` is True.
If `upward_returns` is not empty, it emits the hulls.
So yes, the implementation correctly preserves the hulls for `FINAL_SYMBOLIC_DEPENDENCE_LOST` *as long as* `unresolved` is correctly evaluated as empty for that case.

Wait, what if `upward_returns` is empty?
If `upward_returns` is empty, the `if` condition fails, and it emits NO hulls, even if `not unresolved` is True.
Is it possible to have a complete return (`SUCCESS`) or `FINAL_SYMBOLIC_DEPENDENCE_LOST` but `upward_returns` is empty?
If `upward_returns` is empty, there are no successful branches. If there are no successful branches, but `unresolved` is also empty, then the entire domain cover evaluated to "false" or "nothing"?
If so, emitting no hulls is correct.

Let's look at the verifier logic again.
```python
if transport.get("status") == "TRANSPORT_REFUSED":
    if transport.get("terminal_domain_cover_certified") is True:
        fail("refused transport claims a terminal domain cover")
    for key in ("event_time", "event_derivative", "event_normal"):
        if key in transport:
            fail(f"refused transport exposes a partial {key} hull")
```
If `status` is not "TRANSPORT_REFUSED", it allows the keys.
It allows `terminal_domain_cover_certified` to be missing or False.
This is still the most glaring issue.

Could there be an issue with exact rational arithmetic?
The prompt mentions: "independent exact-rational verifier".
Are the intervals `event_time`, `event_derivative`, `event_normal` exact rational?
The implementation uses `interval_json(hull(...))`.
`hull` typically refers to the convex hull of intervals.
If it's an exact-rational convex hull, the endpoints should be rationals.
Does the verifier check that the endpoints of the emitted intervals in `event_time` etc. are exact rationals?
The provided verifier snippet only checks for the *presence* or *absence* of the keys! It does not check the *format* or *values* of those keys!
If the implementation emits `event_time = "[1.5, 2.5]"`, the verifier checks if the key is present. It does NOT verify that `1.5` and `2.5` are exact rationals, or that they actually bound the branches, or that they are correctly formatted.
This is a massive omission for an "independent exact-rational verifier".

Let's write that as a BLOCKER.
```
1. [BLOCKER] The verifier only checks for the presence or absence of specific keys, completely omitting any exact-rational validation of the actual interval bounds.
   <location: Verifier snippet lines 1-6>
   <why it matters: An "independent exact-rational verifier" must mathematically validate the hulls. By only checking `if key in transport: fail(...)`, the verifier proves that the schema is respected, but does not verify that the `event_time`, `event_derivative`, or `event_normal` intervals actually enclose the successful branches, or that their endpoints are exact rationals rather than floating-point approximations. A forged receipt with correct keys but mathematically invalid intervals will pass.>
   <minimal fix: The verifier must parse the `interval_json` outputs and assert that the endpoints are exact rationals, and recursively verify that the top-level hull mathematically bounds the branch-level intervals.>
```

This is highly relevant and aligns perfectly with the prompt's focus on "formal verification", "mathematical soundness", and "does the soundness theorem actually entail the operational claim?".

Let's combine this with the status string issue, or keep them separate?
Separate is better.

Issue 1 (Status Trusting):
```
1. [BLOCKER] The verifier relies entirely on a self-reported `status` string to gate mathematical constraints and enforces zero positive validation on `SUCCESS` or `FINAL_SYMBOLIC_DEPENDENCE_LOST`.
   <location: Verifier snippet lines 1-6>
   <why it matters: The verifier only enforces constraints when `status == "TRANSPORT_REFUSED"`. It does not verify that a `FINAL_SYMBOLIC_DEPENDENCE_LOST` receipt actually has a complete domain cover (`terminal_domain_cover_certified = True`), nor does it validate the hulls mathematically. A partial failure can simply misreport its status as `SUCCESS` to bypass the verifier.>
   <minimal fix: Enforce strict positive constraints for all valid statuses: if `status` is `SUCCESS` or `FINAL_SYMBOLIC_DEPENDENCE_LOST`, `fail()` if `terminal_domain_cover_certified` is not `True` or if the hull bounds are mathematically invalid.>
```

Issue 2 (Mathematical validation omitted):
```
2. [BLOCKER] The independent verifier only checks for the presence or absence of JSON keys, omitting any exact-rational validation of the actual interval bounds.
   <location: Verifier snippet lines 4-6>
   <why it matters: The system claims to be an "independent exact-rational verifier", but the snippet proves it only validates schema string-key inclusion. It does not verify that the emitted `event_time`, `event_derivative`, or `event_normal` intervals actually enclose the successful branches, nor does it ensure their endpoints are exact rationals.>
   <minimal fix: The verifier must parse the `interval_json` outputs and recursively prove that the top-level hull mathematically bounds the branch-level intervals using exact arithmetic.>
```

Issue 3 (Global hook restorations):
"All unexpected exceptions terminate the worker without a receipt. The process is a single-threaded, one-shot Slurm job. Global hook replacements are each restored by `finally`..."
This is defensive programming. What happens if an exception occurs *inside* the `finally` block?
If a `finally` block that restores a global hook raises an exception (e.g., TypeError, MemoryError), the hook remains permanently replaced, but the program crashes. Since it's a one-shot Slurm job, a crash means it dies, so the state doesn't persist. That is actually safe.

What about "every imported source plus the frozen prior receipt is bound by SHA-256"?
If the verifier doesn't verify the SHA-256 binding, how is it "independently verified"?
The prompt says: "every imported source plus the frozen prior receipt is bound by SHA-256 in the independently verified receipt."
This means the receipt contains hashes of the sources.
If the verifier's job is to verify the receipt, does it check these hashes?
The prompt doesn't show the verifier checking the hashes.
But we can assume it does, or we can flag it. Let's flag it as MAJOR if it's missing, but the snippet provided doesn't show it. I will focus on the provided snippet.

Let's refine Issue 3 (Mutation testing gap):
```
3. [MAJOR] The dedicated negative mutation test only validates the `fail()` branch for string-key inclusion, failing to prove the verifier is fail-closed against state misclassification.
   <location: Test suite description>
   <why it matters: Mutating a `TRANSPORT_REFUSED` receipt to include a partial hull proves the `fail` statement works. It does not test what happens if a partial return is spoofed as `SUCCESS`.>
   <minimal fix: Add mutation tests that mutate a `TRANSPORT_REFUSED` receipt to `SUCCESS` or `FINAL_SYMBOLIC_DEPENDENCE_LOST` to prove the verifier rejects it via positive constraints.>
```

Issue 4 (Reproducibility binding):
```
4. [MAJOR] SHA-256 binding of "imported source" does not guarantee reproducibility of exact-rational arithmetic if native dependencies are used.
   <location: Architecture description - SHA-256 binding>
   <why it matters: If the exact-rational verifier or the worker relies on compiled C-extensions (e.g., `gmpy2`, `mpmath`, or external interval libraries), hashing only Python source files leaves the exact-rational arithmetic vulnerable to shared library drift or implementation differences.>
   <minimal fix: Enforce a reproducible execution environment (e.g., Nix derivation, Docker image digest) or explicitly restrict the worker to pure Python exact rationals (e.g., standard `fractions` module).>
```

Issue 5 (Schema validation):
```
5. [MAJOR] The verifier's use of `transport.get("terminal_domain_cover_certified") is True` fails to catch missing keys or schema violations for unrecognized statuses.
   <location: Verifier snippet line 2>
   <why it matters: If the `status` field is anything other than `"TRANSPORT_REFUSED"` (e.g., a typo like `"REFUSED"` or `None`), the verifier completely skips all checks. Furthermore, relying on `.get(...) is True` means an omitted boolean evaluates to `None` and bypasses the check.>
   <minimal fix: Explicitly define a closed set of allowed statuses. Fail immediately if `status` is not in the allowed set. Assert strict boolean equality (`== True`) rather than `is True` where appropriate, and reject missing keys.>
```

Wait, the system prompt strictly says: "Do not include praise. Do not include 'overall the work is strong' preamble. Skip directly to issue 1."

Let's do a final check of the questions posed by the user in the prompt to ensure they are definitively addressed.
1. "Is the partial-hull under-approximation fixed for every refused result?"
Yes, syntactically. As long as the receipt is honestly labeled `TRANSPORT_REFUSED`. I will note this.
2. "Does
