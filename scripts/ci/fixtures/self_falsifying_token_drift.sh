#!/usr/bin/env bash
# Token-binding fixture: exits 0 but emits a DIFFERENT token than declared.
# This is the drift case — invisible to exit-code gating, caught by token binding.
echo "SFC_TOKEN_FIXTURE_VERDICT TOKEN_BETA"
exit 0
