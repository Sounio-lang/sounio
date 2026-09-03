#!/usr/bin/env bash
# Fixture gate for the self-falsifying compiler rung: sleeps past the
# executor's 30s wall-clock timeout so CLAIM_TIMEOUT can be exercised
# (optional gate clause F6, SFC_TEST_TIMEOUT=1).
sleep 120
exit 0
