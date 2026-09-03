# 90-Second Talk Track — Sounio × PPCR Demo

> **Setup:** have `demo/fregni/run.sh` ready in a terminal. Speak while it runs.

---

**(0:00–0:10)**
"Felipe, the problem I want to show is simple: in a clinical trial, the *source* and *confidence* of a number are usually metadata stored in a PDF protocol. The programming language itself has no opinion about whether a value is measured, imputed, or simulated, or whether it is confident enough to enter the primary analysis."

**(0:10–0:20)**
"Sounio is a self-hosted systems language with epistemic types. Here is a dosing pipeline. It accepts a measured creatinine clearance only if its confidence is above a pre-specified threshold — 950 permille, i.e. 95 percent."

*Run the script. Point at Scenario 1 output.*

**(0:20–0:35)**
"Scenario one passes: 980 permille confidence, and the AUC-based dose is 32,000 milligrams per day. That matches the Python reference on the line above. Scenario two is a *simulated* clearance with only 700 permille confidence. The gate rejects it. Scenario three is an imputed model with acceptable confidence but a 30 percent coefficient of variation against a 25 percent protocol limit — also rejected."

**(0:35–0:50)**
"Now the part no production language does. If a programmer tries to pass a simulated value into a function that only accepts measured values, the compiler rejects it — not at runtime, at compile time."

*Point at the `bad_path.sio` failure: `expected MeasuredGUMI64, found SimulationI64`.*

**(0:50–1:05)**
"That is the analogue of a pre-specified analysis plan: the protocol says what data lineage is admissible, and the language enforces it. The runtime confidence gate is the analogue of a data-quality rule that would otherwise be checked manually."

**(1:05–1:20)**
"What is real today is the epistemic foundation you just saw, plus a machine-checked causal-inference module in Lean and ISO 17025-style calibration code. What is *not* yet real — and where PPCR collaboration comes in — are randomisation modules, blinding indices, validated survival analysis, and enforced 21 CFR Part 11 audit trails."

**(1:20–1:30)**
"The pitch is: let's build the PPCR-specific parts *inside* a language that already treats uncertainty and provenance as first-class, instead of bolting them on afterwards."

---

## Three hardest questions Fregni could ask

### 1. "Can it run a real clinical trial?"

**Grounded answer:** No, not today. The demo is a teaching illustration. What exists are the *language primitives* for confidence gates and provenance tracking. Randomisation, blinding indices, validated sample-size routines, and regulatory audit trails would need to be built — that is the collaboration. Evidence: `docs/ppcr/CLAIMS_LEDGER.md` tags these as `DO-NOT-CLAIM` or `PARTIAL`.

### 2. "Why not just do this in R or Python with a validation package?"

**Grounded answer:** You can enforce a rule in R or Python, but you cannot make the *language* reject a wrongly-sourced value at compile time. Sounio's path wrappers (`MeasuredGUMI64`, `SimulationI64`) are distinct types; a function expecting one cannot accidentally receive the other. That is a structural guarantee, not a runtime check. Evidence: `demo/fregni/bad_path.sio` fails `bin/souc check` with `expected MeasuredGUMI64, found SimulationI64`.

### 3. "Where is the peer-reviewed validation?"

**Grounded answer:** The only formally verified piece is the causal-inference Lean module (`formal/lean4/SounioCausality.lean`). The clinical demo is validated only against an internal Python reference (`demo/fregni/reference.py`). No independent peer-reviewed validation exists yet for the PPCR-specific code. We should not claim otherwise. Evidence: `docs/ppcr/CLAIMS_LEDGER.md` §2 and §8.
