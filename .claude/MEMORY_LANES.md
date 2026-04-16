# Claude Code memory lanes (`-workspace-sounio`)

Files live under **project memory** (not in git):

- **Typical path:** `~/.claude/projects/-workspace-sounio/memory/`
- **Remote habitat (this image):** `/workspace/.home/openvscode-server/.claude/projects/-workspace-sounio/memory/`

## Reading policy (token hygiene)

- **Engineering / compiler sessions:** read only this file + **`.claude/session_state.json`** + the **1–3** memory files listed for your current lane. Do **not** bulk-load `journal.md`, `MEMORY.md`, or `.claude/prompts/garden.md` unless the user asks for Garden mode.
- **Onboarding / relationship context:** `user_working_style.md`, `user_profile.md` (short).
- **Full index:** `MEMORY.md` — use as a **table of contents**, not a mandatory full read every session.

---

## Lane: `compiler_self_host`

Native codegen, bootstrap chain, `lean_single`, driver, sprints on compiler/runtime.

| File | When to read |
|------|----------------|
| `project_self_hosting_achieved.md` | Self-host milestones, history |
| `project_self_host_fixed_point.md` | Gen2/gen3 fixed-point work |
| `project_selfhost_phase3.md` | Phase 3 self-host |
| `project_self_host_architecture.md` | Architecture overview |
| `project_native_compile_status.md` | Current native compile status |
| `project_native_chain_complete.md` | Chain completion context |
| `project_native_frame_fix.md` | Frame / ABI fixes |
| `project_native_v2_micro_emitter.md` | Micro-emitter v2 |
| `project_mini_native.md` | Mini native track |
| `project_mini_native_v2.md` | Mini native v2 |
| `project_phase_c_native_builtin.md` | Phase C builtins |
| `project_lean_single_v2.md` | lean_single v2 |
| `project_wave0_bootstrap.md` | Wave0 bootstrap |
| `project_wave0_boot4_status.md` | boot4 / wave0 status |
| `project_boot3_selfhost.md` | boot3 self-host |
| `project_boot3_globals_chain.md` | boot3 globals |
| `project_boot4_lean_driver.md` | Lean driver |
| `project_boot4_selfhost_investigation.md` | boot4 investigation |
| `project_boot4_stack_overflow.md` | Stack overflow issue |
| `project_boot4_sret.md` | sret / calling conventions |
| `project_lean_driver_lexer_debug.md` | Lexer/driver debug |
| `lean_driver_gaps_analysis.md` | Driver gap analysis |
| `lean_driver_gaps_deep_dive.md` | Deep dive |
| `project_sprint225.md` | Sprint 225 |
| `project_sprint228_closures.md` | Closures sprint |
| `project_sprint229_native_fn.md` | Native fn sprint |
| `project_sprint235_print_f64.md` | print f64 sprint |
| `feedback_native_compiler_limits.md` | Compiler limits feedback |
| `feedback_native_only.md` | Native-only workflow |
| `feedback_lean_single_features.md` | lean_single features |
| `feedback_jit_ref_bug.md` | JIT ref bug |
| `feedback_jit_stdout_warnings.md` | JIT stdout |
| `feedback_irinstr_size_bug.md` | IR instr size |
| `deep_dive_work_summary.md` | Summarized deep dives |

---

## Lane: `stdlib_checker_types`

Stdlib, type checker, epistemic front-end, non-associative types.

| File | When to read |
|------|----------------|
| `sounio_stdlib_audit.md` | Stdlib audit |
| `project_wave_f_type_checking.md` | Wave F typing |
| `project_nonassoc_types.md` | Non-associative types |
| `project_epistemic_compilation.md` | Epistemic compilation |
| `project_epistemic_pipeline.md` | Epistemic pipeline |
| `project_epistemic_dawn.md` | Epistemic “dawn” track |
| `feedback_variance_deep_chains.md` | Variance / chains |

---

## Lane: `gpu_epistemic`

GPU stack, WMMA, render-adjacent work.

| File | When to read |
|------|----------------|
| `project_gpu_stack_complete.md` | GPU stack status |
| `project_epistemic_wmma.md` | WMMA / epistemic GPU |

---

## Lane: `research_math_connectomics`

Papers, algebra, connectomics, conjectures, psychiatry side tracks.

| File | When to read |
|------|----------------|
| `project_168_theorem.md` | 168 / octonion theorem context |
| `project_algebra_observer.md` | Algebra observer |
| `project_categorical_bridge.md` | Categorical bridge |
| `project_g2_bridge.md` | G2 bridge |
| `project_triple_ecosystem.md` | Triple ecosystem |
| `project_oct_connectomics.md` | Octonion connectomics |
| `project_non_assoc_connectomics.md` | Non-assoc connectomics |
| `project_s_ssm_zero_divisor.md` | SSM / zero divisor |
| `project_sedenion_hessian.md` | Sedenian Hessian |
| `conjecture_nonassociative_entropy.md` | Entropy conjecture |
| `project_phonon_experiment.md` | Phonon experiment |
| `project_sleep_orbit_real.md` | Sleep / orbit model |
| `project_masters_dissertation.md` | Dissertation thread |
| `project_clinical_paper.md` | Clinical paper |
| `project_paper4_brain_orc.md` | Paper 4 / brain ORC |
| `project_sounio_psychiatry.md` | Psychiatry crossover |
| `project_cybernetics_layer2.md` | Cybernetics layer 2 |
| `project_sounio_directive.md` | High-level directive |

---

## Lane: `interop_tooling`

F#, MCP references, LLM training notes.

| File | When to read |
|------|----------------|
| `project_fsharp_interop.md` | F# interop |
| `reference_cockpit_mcp.md` | Cockpit MCP |
| `reference_kimi25.md` | Kimi 2.5 reference |
| `sounio_llm_training.md` | LLM training notes |

---

## Lane: `garden_meta_user`

Style, identity, session UX, Garden journal — **high token cost** if read together.

| File | When to read |
|------|----------------|
| `user_working_style.md` | How to collaborate (Garden rules) |
| `user_profile.md` | User profile |
| `user_identity.md` | Identity context |
| `user_academic.md` | Academic background |
| `journal.md` | Garden journal — **only when user invokes Garden** |
| `MEMORY.md` | Full index — skim headings, do not read end-to-end by default |
| `feedback_session_management.md` | `/clear`, `/compact`, session habits |
| `feedback_headless.md` | Headless / CI agent |
| `feedback_bypass_permissions.md` | Permissions / bypass |
| `feedback_ultraplan_refinement.md` | Ultrplan refinement |
| `feedback_sounio_first.md` | Sounio-first principles |
| `feedback_role_language_work.md` | Role / language |

---

## Inventory check

All **77** `*.md` files under `memory/` as of 2026-04-16 appear exactly once across `compiler_self_host`, `stdlib_checker_types`, `gpu_epistemic`, `research_math_connectomics`, `interop_tooling`, and `garden_meta_user`. Re-run `ls memory/*.md` after adding new files and update this document.
