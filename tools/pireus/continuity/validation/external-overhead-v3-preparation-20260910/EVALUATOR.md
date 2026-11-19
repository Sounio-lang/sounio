# V3 evaluator implementation

The pinned preparation is evaluated by ops/evaluate_external_overhead_v3.py. It delegates the unchanged acceptance bounds to the v2 numerical evaluator and adds v3 schema, process identity, boundary cadence, duration containment, RSS/high-water readability, and monotonic CPU checks. Evaluation does not authorize execution or grant custody acceptance.

All 56 external tests passed locally through the existing CI discovery pattern test*external*.py, which already includes the new test module. Seven new test methods include twelve adversarial subcases and resource-only replay of real CPU job 11982. No loaded-model run occurred.

The original proposal and preparation manifest are unchanged historical inputs. evaluator-validation.json records the implementation hashes and remaining execution gates. A runnable packet still needs explicit helper custody and launcher adaptation.
