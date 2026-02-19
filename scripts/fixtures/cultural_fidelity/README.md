# Cultural Fidelity Gate Fixtures

- `pass_user_output.txt`: should pass (no Rust terms).
- `fail_user_output.txt`: should fail (contains `cargo`).
- `dev_build.txt`: contains Rust terms but should pass only with `allowlist.tsv`.

Run:

```bash
python3 scripts/cultural_fidelity_gate.py --self-test
```
