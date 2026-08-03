1. [BLOCKER] The hardcoded acceptance counts (662 selected attempts, 331 paired leaves) are asserted without any probabilistic or statistical justification, making the verification criteria arbitrary and unfalsifiable.
   - Location: verify() lines ~280-290
   - A hostile referee would immediately ask: why these specific numbers? What is the expected distribution under the null hypothesis of no adaptive epistemic cover?
   - Minimal fix: Provide a statistical model with confidence intervals, or remove the magic-number assertions and report empirical counts with uncertainty bounds.

2. [BLOCKER] The `legacy` variable in `expected_certificate()` uses `flags.get("CERTIFICATE_PASS", True) or flags.get("TERMINAL_CERTIFIED", True)`, which defaults to True when the flag is absent, silently classifying missing legacy certificates as passed.
   - Location: expected_certificate(), line ~160
   - This inverts the verification logic: any attempt lacking explicit legacy-certificate flags will be treated as legacy-passed, potentially masking genuine failures.
   - Minimal fix: Require explicit flag presence; fail if neither flag exists in the stdout.

3. [BLOCKER] The `JOINT_EQUALS_LIOUVILLE` check compares exact Fraction equality of binary64-derived rationals, but the joint interval is computed from a max/min over six intervals while Liouville is only one of them—this equality is structurally impossible unless all six intervals are identical to Liouville.
   - Location: expected_certificate(), lines 150-155
   - The "epistemic certificate" claims joint coverage but the verification demands exact coincidence with a single component, which is mathematically over-constrained and likely never true in practice.
   - Minimal fix: Define what "equals" means semantically (e.g., within tolerance) or remove this field entirely.

4. [MAJOR] The `certificate` boolean requires `not legacy`, but the verification then counts `LEGACY_CERTIFICATE_FALSE` as 662—this means all selected attempts must have legacy certificates explicitly false, yet the code defaults to True when flags are absent, creating a logical contradiction if any stdout is malformed.
   - Location: expected_certificate() lines 165-175, count assertions lines 280-290
   - This makes the verification brittle: a single missing flag in any of the 662 attempts would cause a hard failure, not a statistical anomaly.
   - Minimal fix: Separate "legacy certificate absent" from "legacy certificate false" and handle both cases explicitly.

5. [MAJOR] The `raw_determinants()` function requires exactly the six expected interval types and fails on any extra or missing entry, but the regex `INTERVAL_RE` is greedy and could match intervals from unrelated log lines, causing false parsing.
   - Location: raw_determinants(), lines 60-80
   - Any log line containing `NAME=[value,value]` pattern (e.g., from debugging output or error messages) will be misinterpreted as a determinant interval.
   - Minimal fix: Anchor the regex to line starts and validate the exact format including whitespace.

6. [MAJOR] The `d4_identity()` and `d5_identity()` functions construct file paths from parsed integer values without validating that the resulting path exists in the archive, relying on `archive_bytes()` to fail—but this failure occurs after potentially expensive processing.
   - Location: d4_identity(), d5_identity(), archive_bytes()
   - A maliciously crafted results.tsv could reference arbitrary paths, causing denial-of-service through excessive file lookups or path traversal if the archive contains unexpected members.
   - Minimal fix: Validate all identifiers against the archive's member list upfront, and constrain to expected patterns.

7. [MAJOR] The `compare_row()` function fails on the FIRST mismatch, providing no diagnostic about how many fields were checked or which ones passed, making debugging of large datasets nearly impossible.
   - Location: compare_row(), lines 200-205
   - For 662 certificates × 15 fields, a single typo in one field produces only a cryptic "field X: value != value" error without context.
   - Minimal fix: Collect all mismatches and report them together, or at least include the row identity in the error message.

8. [MAJOR] The `expected_summary` dictionary includes `"PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED": "false"` and `"ANALYSIS_MODE": "RETROSPECTIVE_RETAINED_RECEIPT_AUDIT"`, but the script name says "Independently verify"—this is an admission that the verification is retrospective and cannot validate the original computation's correctness.
   - Location: verify() lines 260-270
   - A hostile referee would ask: what does "independently" mean if you're replaying the same archives and checking against a pre-computed receipt? This is at best a consistency check, not independent verification.
   - Minimal fix: Either rename the script or explain what independent verification means in this context.

9. [MINOR] The `exact()` function converts binary64 hex to Fraction via `as_integer_ratio()`, but this loses the original precision information and may produce enormous fractions (e.g., 1e-300 becomes a fraction with ~1000-digit numerator/denominator), causing performance issues in the max/min comparisons.
   - Location: exact(), lines 50-55
   - For 662 attempts × 6 intervals × 2 endpoints, this creates 7944 Fraction objects with potentially huge integer components.
   - Minimal fix: Use decimal floating-point with sufficient precision, or compare using the original hex strings with a custom comparator.

10. [MINOR] The `parse_summary()` function requires exactly one "=" per line and rejects empty values, but the `expected_summary` includes boolean strings like "false"—if any value were empty (e.g., "KEY="), the verification would fail without explaining why.
    - Location: parse_summary(), lines 215-225
    - This is overly strict for a format that could reasonably contain empty values for optional fields.
    - Minimal fix: Allow empty values and validate against expected schema separately.

11. [NIT] The `archive_bytes()` function caches all `./results.tsv` and `*/stdout.txt` members on first access, but never validates that the archive contains exactly the expected number of members or that all paths are unique.
    - Location: archive_bytes(), lines 85-100
    - Duplicate member names in the tar would silently overwrite the cache, potentially verifying against the wrong stdout.
    - Minimal fix: Validate member count and uniqueness during cache population.

12. [NIT] The script uses `print()` for verification results rather than structured output, making it impossible to parse programmatically or integrate with CI/CD pipelines that expect machine-readable results.
    - Location: final print statements, lines 295-305
    - The verification outcome is only human-readable, not machine-verifiable.
    - Minimal fix: Output JSON or a key-value format with explicit success/failure status.
