.PHONY: build check test test-stdlib clean fmt install help lint lint-fix \
         ops-guardrail-local ops-infra-up ops-strict-up ops-status \
         proof-check proof-regen

SOUC := ./bin/souc

ifneq ("$(wildcard Makefile.verify)","")
include Makefile.verify
endif

##@ Developer Targets

help:                ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build:               ## Bootstrap compile: gen1 → gen2 → gen3 (fixed-point verification)
	@echo "→ Stage 1: boot4.elf compiles lean_single → gen1.elf"
	./artifacts/bootstrap/boot4.elf self-hosted/compiler/lean_single.sio gen1.elf
	@echo "→ Stage 2: gen1.elf compiles lean_single → gen2.elf"
	./gen1.elf self-hosted/compiler/lean_single.sio gen2.elf
	@echo "→ Stage 3: gen2.elf compiles lean_single → gen3.elf"
	./gen2.elf self-hosted/compiler/lean_single.sio gen3.elf
	@echo "→ Verifying fixed-point..."
	@MD5_GEN2=$$(md5sum gen2.elf | awk '{print $$1}'); \
	 MD5_GEN3=$$(md5sum gen3.elf | awk '{print $$1}'); \
	 if [ "$$MD5_GEN2" = "$$MD5_GEN3" ]; then \
	   echo "✓ FIXED POINT OK ($$MD5_GEN2)"; \
	 else \
	   echo "✗ FIXED POINT BROKEN"; \
	   echo "  gen2: $$MD5_GEN2"; \
	   echo "  gen3: $$MD5_GEN3"; \
	   exit 1; \
	 fi

check:               ## Type-check self-hosted compiler and run lint gates
	@echo "→ Type-checking self-hosted/compiler/lean_single.sio"
	$(SOUC) check self-hosted/compiler/lean_single.sio
	@echo "→ Running lint gates..."
	@bash scripts/ci/full_gate.sh 2>&1 | tail -30

test:                ## Run full test suite (compile-fail + run-pass + stdlib)
	@echo "→ Running full test suite"
	@bash scripts/run_sio_test_suite.sh

test-stdlib:         ## Run stdlib integration tests (subset)
	@echo "→ Running stdlib tests"
	$(SOUC) run tests/stdlib/bayes/test_prior_e2e.sio
	$(SOUC) run tests/stdlib/complex/test_complex.sio

clean:               ## Remove generated ELF artifacts (gen1, gen2, gen3)
	rm -f gen1.elf gen2.elf gen3.elf
	@echo "✓ Cleaned generated artifacts"

fmt:                 ## Format .sio source code (not yet implemented)
	@echo "⚠ soufmt not yet implemented"

lint:                ## Lint .sio files for Rust hallucinations (LLM grammar enforcer)
	@find tests/stdlib examples -name "*.sio" -print0 | xargs -0 python3 scripts/dev/sounio-lint.py --errors-only 2>&1 | grep -v "^sounio-lint:.*OK$$" || true

lint-fix:            ## Apply automatic fixes to a file: make lint-fix FILE=path/to/file.sio
	@if [ -z "$(FILE)" ]; then echo "Usage: make lint-fix FILE=path/to/file.sio"; exit 1; fi
	@python3 scripts/dev/sounio-lint.py --fix $(FILE)

proof-check:         ## Verify EGC proof obligations via lake build (requires elan/lean)
	@echo "→ Building SounioGradedModal, SounioMeasConf, SounioProofObligation"
	cd formal/lean4 && lake build SounioGradedModal SounioMeasConf SounioProofObligation
	@echo "✓ EGC proof obligations verified"

proof-regen:         ## Regenerate SounioProofObligation.lean from compiler --emit-proof-obligations
	@if [ ! -f gen17.elf ]; then echo "Run 'make build' first to produce gen17.elf"; exit 1; fi
	@echo "→ Emitting PLATINUM proof goals from lean_single.sio self-compile"
	./gen17.elf self-hosted/compiler/lean_single.sio /tmp/t_proof.elf \
	    --emit-proof-obligations 2>/dev/null 1>formal/lean4/SounioProofObligation.lean
	@echo "→ Verifying generated obligations"
	cd formal/lean4 && lake build SounioProofObligation
	@echo "✓ Proof obligations regenerated and verified"

install:             ## Install souc compiler to ~/.local/bin/souc
	mkdir -p ~/.local/bin
	install -m755 bin/souc ~/.local/bin/souc
	@echo "✓ Installed souc to ~/.local/bin/souc"

##@ Operations Targets (long-running infrastructure)

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
