#!/usr/bin/env bash
# Fixture gate for the self-falsifying compiler rung: always fails, so claims
# pointing at it must be reported CLAIM_FAIL and abort the compilation.
echo "self_falsifying_claim_gate_fail: falsified" >&2
exit 1
