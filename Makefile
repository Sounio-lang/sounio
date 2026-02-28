.PHONY: ops-guardrail-local ops-infra-up ops-strict-up ops-status

ifneq ("$(wildcard Makefile.verify)","")
include Makefile.verify
endif

ops-guardrail-local:
	@echo "→ Running local ops guardrail (strict lane)"
	@PLAN_BIG_OPS_SUITE_RUNNER_GATE_SCRIPT=scripts/plan_big_gate.sh \
		bash scripts/overnight_plan_big_ops_suite.sh --with-gate --burnin-duration-sec 120 --burnin-check-interval-sec 20

ops-infra-up:
	@bash scripts/tmux_big_ops_infra.sh up --reset

ops-strict-up:
	@bash scripts/tmux_big_ops_strict.sh up --reset

ops-status:
	@bash scripts/tmux_big_ops_default.sh status
