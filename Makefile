.PHONY: build check test test-stdlib clean fmt install help lint lint-fix lint-docs \
         docs-gen generated-ontology test-generated-ontology \
         test-generated-ontology-manifest test-generated-ontology-fresh \
         test-ontology-bundle-directive-native-scan \
         test-ontology-cache-frontend-composition \
         test-unit-types test-unit-types-derived test-ontology-unit-metadata \
         test-unit-types-clinical-current-source \
         test-knowledge-context-phase2 test-knowledge-unit-constraints \
         test-knowledge-numeric-constraints test-knowledge-composite \
         test-knowledge-static-values test-knowledge-runtime-obligations \
         test-knowledge-runtime-guards test-knowledge-runtime-guard-directive \
         test-knowledge-runtime-guard-lowering-plan \
         test-knowledge-runtime-guard-native-lowering \
         test-knowledge-context-static \
         test-semantic-knowledge-spine \
         test-madaros-identity test-real-language-runner test-project-spine \
         ops-guardrail-local ops-infra-up ops-strict-up ops-status \
         website-verified-snapshot

SOUC := ./bin/souc

ifneq ("$(wildcard Makefile.verify)","")
include Makefile.verify
endif

##@ Developer Targets

help:                ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build:               ## Bootstrap compile: JIT → gen1 → gen2 → gen3 (fixed-point verification)
	@echo "→ Stage 0: JIT (bin/souc-linux-x86_64) compiles lean_single → gen1.elf"
	./bin/souc-linux-x86_64 self-hosted/compiler/lean_single.sio gen1.elf
	chmod +x gen1.elf
	@echo "→ Stage 1: gen1.elf compiles lean_single → gen2.elf"
	./gen1.elf self-hosted/compiler/lean_single.sio gen2.elf
	chmod +x gen2.elf
	@echo "→ Stage 2: gen2.elf compiles lean_single → gen3.elf"
	./gen2.elf self-hosted/compiler/lean_single.sio gen3.elf
	chmod +x gen3.elf
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
	@echo "→ Regenerating stdlib API reference"
	@bash scripts/build/gen_stdlib_api_md.sh
	@echo "→ Running lint gates..."
	@bash -o pipefail -c 'bash scripts/dev/full_gate.sh 2>&1 | tail -40'

test:                ## Run full test suite (compile-fail + run-pass + stdlib)
	@echo "→ Running full test suite"
	@bash scripts/run_sio_test_suite.sh

test-stdlib:         ## Run stdlib integration tests (subset)
	@echo "→ Running stdlib tests"
	$(SOUC) run tests/stdlib/bayes/test_prior_e2e.sio
	$(SOUC) run tests/stdlib/complex/test_complex.sio

build-madaros:       ## Build the Stage1 modular compiler (Madaros)
	@echo "→ Building Madaros (Stage1 modular compiler)"
	bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros

test-madaros-identity: ## Verify Madaros identifies as the Stage1 modular Sounio compiler
	@bash scripts/gates/g6_madaros_identity.sh

test-real-language-runner: ## Verify public souc CLI + REPL + Madaros identity
	@bash scripts/ci/real_language_runner_gate.sh

test-project-spine: ## Verify sounio.toml project mode on souc and Madaros
	@bash scripts/ci/project_spine_gate.sh

madaros-full-gate: build-madaros ## Build Madaros, then run the Stage1 end-to-end gate
	@echo "→ Running Madaros full-functioning gate"
	bash scripts/ci/madaros_full_gate.sh

madaros-wide-int-gate: ## Run the wide-integer (i128/i256) experimental gate
	@echo "→ Running Madaros wide-integer gate"
	bash scripts/ci/madaros_wide_int_gate.sh

madaros-enum-gate:   ## Run the focused enum declaration + variant-use regression gate
	@echo "→ Running Madaros enum regression gate"
	bash scripts/ci/madaros_enum_gate.sh

madaros-loop-gate:   ## Run the focused for/while + break/continue regression gate
	@echo "→ Running Madaros loop regression gate"
	bash scripts/ci/madaros_loop_gate.sh

clean:               ## Remove generated ELF artifacts (gen1, gen2, gen3)
	rm -f gen1.elf gen2.elf gen3.elf gen4.elf
	@echo "✓ Cleaned generated artifacts"

fmt:                 ## Format .sio source code (not yet implemented)
	@echo "⚠ soufmt not yet implemented"

lint:                ## Lint .sio files for Rust hallucinations (LLM grammar enforcer)
	@find tests/stdlib examples -name "*.sio" -print0 | xargs -0 python3 scripts/dev/sounio-lint.py --errors-only 2>&1 | grep -v "^sounio-lint:.*OK$$" || true

lint-fix:            ## Apply automatic fixes to a file: make lint-fix FILE=path/to/file.sio
	@if [ -z "$(FILE)" ]; then echo "Usage: make lint-fix FILE=path/to/file.sio"; exit 1; fi
	@python3 scripts/dev/sounio-lint.py --fix $(FILE)

docs-gen:            ## Regenerate stdlib API reference from source
	@bash scripts/build/gen_stdlib_api_md.sh

generated-ontology:  ## Regenerate .dontology bundles and generated ontology .sio stubs
	@echo "→ Building .dontology bundles from stdlib ontology source slices"
	@python3 scripts/ontology/build_bundle.py \
		--source-dir stdlib/data/data/ontology/source \
		--output-dir stdlib/data/data/ontology/bundles
	@echo "→ Validating stable public .dontology bundles and SSSOM mapping shards"
	@python3 scripts/ontology/validate_bundle.py \
		--bundle stdlib/data/data/ontology/bundles/alg.dontology \
		--bundle stdlib/data/data/ontology/bundles/chebi.dontology \
		--bundle stdlib/data/data/ontology/bundles/go.dontology \
		--bundle stdlib/data/data/ontology/bundles/hpo.dontology \
		--bundle stdlib/data/data/ontology/bundles/loinc.dontology \
		--bundle stdlib/data/data/ontology/bundles/part.dontology \
		--bundle stdlib/data/data/ontology/bundles/phys.dontology \
		--bundle stdlib/data/data/ontology/bundles/qm.dontology \
		--bundle stdlib/data/data/ontology/bundles/snomed.dontology \
		--mapping-dir stdlib/data/data/ontology/bundles/mappings
	@echo "→ Generating stdlib ontology stubs via the C FFI importer"
	@bash scripts/ontology/generate_dontology_stubs.sh

test-generated-ontology: generated-ontology ## Regenerate ontology stubs and run generated ontology witnesses
	@bash scripts/run_sio_test_suite.sh ontology_generated --verbose

test-generated-ontology-manifest: generated-ontology ## Validate generated ontology manifest coverage
	@bash scripts/ci/generated_ontology_manifest_gate.sh

test-generated-ontology-fresh: ## Regenerate ontology artifacts and fail if generated outputs drift
	@bash scripts/ci/generated_ontology_gate.sh --check

test-ontology-bundle-directive: ## Expand //@ ontology-bundle directives through the C importer and test witnesses
	@bash scripts/ci/ontology_bundle_directive_gate.sh

test-ontology-bundle-directive-native-scan: ## Run compiler-side //@ ontology-bundle directive scanner gate
	@bash scripts/ci/ontology_bundle_directive_native_scan_gate.sh

test-ontology-cache-frontend-composition: ## Run focused .ontocache + Knowledge<T> frontend composition gate
	@bash scripts/ci/ontology_cache_frontend_composition_gate.sh

test-unit-types: ## Run focused unit/dimensional analysis Phase 1 gate
	@bash scripts/ci/unit_types_phase1_gate.sh

test-unit-types-derived: ## Run derived/current-source unit dimensional analysis gate
	@bash scripts/ci/unit_types_derived_gate.sh

test-unit-types-clinical-current-source: ## Run current-source internal label dimensional analysis gate
	@bash scripts/ci/unit_types_clinical_current_source_gate.sh

test-ontology-unit-metadata: ## Validate ontology-linked internal dimension-label metadata
	@bash scripts/ci/ontology_unit_metadata_gate.sh

test-knowledge-context-phase2: ## Run static ontology Knowledge<T> proof-context gate
	@bash scripts/ci/knowledge_context_phase2_ontology_gate.sh

test-knowledge-unit-constraints: ## Run static Knowledge<T> unit proof-context gate
	@bash scripts/ci/knowledge_context_unit_gate.sh

test-knowledge-numeric-constraints: ## Run static Knowledge<T> numeric proof-context gate
	@bash scripts/ci/knowledge_context_numeric_gate.sh

test-knowledge-composite: ## Run static composite ontology+unit+numeric Knowledge<T> gate
	@bash scripts/ci/knowledge_context_composite_gate.sh

test-knowledge-static-values: ## Run static Knowledge<T> value-initializer gate
	@bash scripts/ci/knowledge_context_static_value_gate.sh

test-knowledge-runtime-obligations: ## Run dynamic Knowledge<T> runtime obligation gate
	@bash scripts/ci/knowledge_context_runtime_obligation_gate.sh

test-knowledge-runtime-guards: ## Run pre-native dynamic Knowledge<T> runtime guard expansion and directive scanner gates
	@bash scripts/ci/knowledge_runtime_guard_expansion_gate.sh
	@bash scripts/ci/knowledge_runtime_guard_directive_native_scan_gate.sh
	@bash scripts/ci/knowledge_runtime_guard_lowering_plan_gate.sh

test-knowledge-runtime-guard-directive: ## Run compiler-side //@ knowledge-runtime-guards scanner gate
	@bash scripts/ci/knowledge_runtime_guard_directive_native_scan_gate.sh

test-knowledge-runtime-guard-lowering-plan: ## Run compiler-side Knowledge<T> runtime guard lowering-plan gate
	@bash scripts/ci/knowledge_runtime_guard_lowering_plan_gate.sh

test-knowledge-runtime-guard-native-lowering: ## Run Sounio-native expander → compile → run gate (14 cases, replaces bash expander)
	@bash scripts/ci/knowledge_runtime_guard_native_lowering_gate.sh

test-knowledge-context-static: ## Run static Knowledge<T> umbrella gate
	@bash scripts/ci/knowledge_context_static_gate.sh

test-semantic-knowledge-spine: ## Run the focused ontology -> units -> Knowledge<T> umbrella gate
	@bash scripts/ci/semantic_knowledge_spine_gate.sh

website-verified-snapshot: ## Run stdlib reliability gate + refresh website verified-snapshot.json (needs SOUC_BIN)
	@echo "→ SOUC_BIN must point to a working souc (e.g. export SOUC_BIN=/tmp/souc.elf)"
	@bash scripts/dev/stdlib_reliability_gate.sh
	@npm run gen:verified-snapshot --prefix website
	@echo "✓ Updated website/src/data/verified-snapshot.json (review and git add)"

lint-docs:           ## Extract and check code snippets from docs/**/*.md
	@bash scripts/ci/check_doc_snippets.sh

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
