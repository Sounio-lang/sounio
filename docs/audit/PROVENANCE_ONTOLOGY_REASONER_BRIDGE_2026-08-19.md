<!-- docs:meta
topic_id: repo.docs.audit.provenance-ontology-reasoner-bridge-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.provenance-ontology-reasoner-bridge-2026-08-19
-->

# Provenance ↔ ontology reasoner — Phase 1 measurement

**Date:** 2026-08-19  
**sha:** `f9b3147364` (`origin/main` at measure)  
**Round:** measure + cost classification only — **no code**, no bridge proposal until founder confirms Phase 1.  
**TSV:** `docs/audit/PROVENANCE_ONTOLOGY_REASONER_BRIDGE_2026-08-19.tsv`  
**Compile receipt:** Slurm `cpu-ops` — `ontology_reasoner_compile_gate.sh` → **PASS**

---

## Semantic lane declaration

```text
Semantic-Lane-ID: prov-ontology-bridge-measure-20260819
Owner: grok-cli4
Concept-IDs: SOUNIO-PROVENANCE (concept file provenance.md); related
  SOUNIO-JUSTIFICATION / admissibility surfaces named in dispatch (no Status change)
Intent-Preserved: types-as-ontology-terms and ontology-validates-claims (1+2);
  this lane only measures whether provenance can ride the existing reasoner
Transformation: none — observational
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced:
  - the live reasoner entry is stdlib/ontology/reasoner.sio (gate-followed)
  - that reasoner already implements a minimal class-assertion individual path
  - zero external callers of that path exist today
  - ProvEntity/ProvId live in stdlib/epistemic/prov.sio with no bridge into the reasoner
  - cost class for class-assertion linking is "bridge/wiring", not "invent ABox from zero"
  - cost class for full PROV relation graphs (wasGeneratedBy, used, …) remains OPEN /
    likely larger than class-assertion wiring
Claims-Forbidden:
  - that the link is known viable end-to-end (INDETERMINATE until Phase 2 approved)
  - that the reasoner is TBox-only (false: APIs exist; live use is TBox-only)
  - that stdlib/compiler/ontology/store.sio is an OWL ABox (it is a CURIE term store)
  - that founder TBox counts (931/…) are reproduced exactly here without stating path
Assumptions: "reasoner" means what the compile gate builds, not every string matching /reason/
Write-Set: docs/audit/PROVENANCE_ONTOLOGY_REASONER_BRIDGE_2026-08-19.{md,tsv}
Read-Set: stdlib/ontology/reasoner.sio, model.sio, query.sio, compiler/ontology/store.sio,
  epistemic/prov.sio, scripts/ci/ontology_reasoner_compile_gate.sh, ontology bundles
Positive-Witness: subclass_of present in stdlib/ontology; reasoner gate PASS on Slurm;
  test_instance_classification in reasoner.sio source
Negative-Witness: rg callers of reasoner_is_instance outside reasoner.sio = 0
Acceptance-Gate: this document; founder confirmation before Phase 2
Integration-Target: none this round
Authoritative-Only-If: Phase 1 answers 1–4 are re-runnable from the commands below
```

---

## Instrument discipline

| claim | positive control before trusting a zero |
|---|---|
| `subclass_of` counting | hits `deployment_validity_revocable_authority.sio` and generated CHEBI parents |
| reasoner identity | gate path → `stdlib/ontology/reasoner.sio`, not `acquisition_reason` |
| “no individual callers” | symbols `reasoner_init_individuals` etc. appear in **1** file only (`reasoner.sio`) |
| compile liveness | Slurm ran `scripts/ci/ontology_reasoner_compile_gate.sh` → PASS |

Founder TBox headline numbers (subclass_of 931, …) were **not** bit-reproduced on this tree with a simple `rg` over `stdlib/ontology` (measured subclass_of **888**). Difference is path/format (bundles are JSON `.dontology` with `"parents"` not the token `subclass_of`; generated `.sio` uses `subclass_of`). **Do not treat founder counts and this rg as the same instrument.** Bundles are real (chebi, go, hpo, loinc, snomed, alg, part, phys, qm).

---

## 1) What the reasoner does today

### Location (followed from the gate)

```text
scripts/ci/ontology_reasoner_compile_gate.sh
  → cat stdlib/ontology/reasoner.sio
       + scripts/ci/ontology_fixtures/reasoner_exercise_main.sio
  → compile + run; exit code is the verdict
```

Header of `reasoner.sio`: *“Basic OWL Reasoning — subsumption, transitive closure, classification.”*

### Questions it answers (API)

| family | API | meaning |
|---|---|---|
| **TBox** | `reasoner_add_superclass`, `reasoner_compute_closure`, `reasoner_is_subclass`, `reasoner_all_superclasses` / `_subclasses` | class hierarchy + transitive subsumption |
| **TBox similarity** | `reasoner_least_common_subsumer`, `reasoner_wu_palmer_similarity` | depth-based class similarity |
| **Individual / class-assertion** | `reasoner_init_individuals`, `reasoner_add_individual_type`, `reasoner_classify_individuals`, `reasoner_is_instance`, `reasoner_all_instances` | assert `ind : Class`, inherit superclasses |
| **Materialize** | `reasoner_materialize` | closure + depths + **classify individuals** |

In-source unit `test_instance_classification`: Fido asserted `Animal`, after materialize is instance of `Animal` and inferred `Thing`.

### What the **gate exercise** actually runs

`reasoner_exercise_main.sio` is **TBox-only**: five classes, subclass edges, LCS(Dog,Cat)=Mammal, Wu–Palmer > 0, Dog ⊑ Thing. **No individuals.**

So: **capability includes a minimal ABox (class assertions + type inheritance); the CI witness only proves the TBox path.**

### What it is **not**

- Not a full OWL 2 DL reasoner (no roles/restrictions tableau, no nominals story beyond int ids).
- Does **not** implement object/data **property** assertions on individuals (those constructors live on `model.sio`’s `Individual`, unused by the reasoner).
- Does **not** answer PROV graph questions (`wasGeneratedBy`, shared activity ancestor) unless encoded as classes or fed to another store.

---

## 2) Cost difference — main Phase 1 deliverable

### Finding (with evidence)

| layer | ABox-shaped support? | Live use? |
|---|---|---|
| `reasoner.sio` individual APIs | **Yes** (class assertions + inheritance) | **No callers outside the file** |
| `model.sio` `Individual` + property assertion axioms | **Yes** (structs + ctors) | **No callers of `individual_new` / `axiom_class_assertion` outside model** |
| Generated bundles / `.dontology` | Classes + parents (+ disjoint) | **No named individuals** in importer surface |
| `compiler/ontology/store.sio` | Term records | CURIE→label/parents; `provenance: string` = **shard label**, not `ProvEntity` |

### Cost class

```text
For mapping ProvEntity  ↦  reasoner individual + entity_type ↦ class assertion
+ query reasoner_is_instance / shared inferred types:

    → BRIDGE / WIRING cost
      (APIs and in-file tests already exist; nothing loads them from PROV or bundles)

NOT "extend the reasoner from a pure TBox blank slate".

For mapping full PROV relations (activities, agents, wasDerivedFrom chains,
"do these two measurements share a generation ancestor?" as graph reachability):

    → OPEN / LARGER
      reasoner has no property graph over individuals;
      query.sio TripleStore is a separate SPARQL-like toy with no measured
      production callers feeding ProvEntity into it (callers outside query.sio: 0
      for triple_store_new/sparql_execute in this sweep).
      That path may need store population + query design, or reasoner extension.
```

**INDETERMINATE (explicit):** whether the **bridge** is enough for SOUNIO-PROVENANCE’s “is this measurement that measurement?” depends on whether identity is class-level + id equality or full PROV graph. Phase 1 does **not** conclude viability of the product link — only the cost *class* of the class-assertion fragment vs a from-scratch ABox.

---

## 3) What `stdlib/compiler/ontology/store.sio` stores

Not an open triple ABox.

- Hard-coded / slice `ResolvedOntologyTerm` records: `curie`, `label`, `definition`, `parents[]`, `synonyms[]`, `iri`, `mapping_confidence`, **`provenance: string`** (e.g. `"local:snomed"`).
- Resolve-by-CURIE for a small clinical demo slice (SNOMED/LOINC/HPO examples).
- Comment: *“Phase 1 local biomedical slice derived from bundled .dontology shards.”*

**Place for facts about individuals?** Only insofar as a term is a class-like CURIE. No instance ids, no property facts, no ProvId.

Parallel: `stdlib/ontology/model.sio` **does** define `Ontology { individuals: Seq<Individual> }` and assertion axiom constructors — a **schema** for ABox, not the compiler store’s content.

---

## 4) Existing bridges provenance ↔ ontology?

| candidate | what it is | bridge to reasoner? |
|---|---|---|
| `stdlib/epistemic/prov.sio` | Hand W3C-PROV-DM: `ProvId`, `ProvEntity`, activities, agents, records | **No** import/use of `reasoner_*` |
| Ontology files `has_provenance` roles | **TBox** roles on epistemic artifact classes → `ProvenanceReceipt` / similar | Terminology about provenance, not instances of measurements |
| `compiler/ontology` `provenance` field | String tag on resolved terms | Homonym of PROV, not `ProvEntity` |
| `bindings.tsv` for zero-provenance / ordered-path-provenance | Points at epistemic/IR/tests — **not** `stdlib/ontology/reasoner.sio` | No |

**No pre-built ProvEntity→ABox bridge found.** (Checked before proposing one.)

Homonyms avoided: PK `Individual`, medlang `IndividualParameters`, effect “individual effect” — not OWL individuals.

---

## Implications for the three concepts (descriptive only)

| concept | why ABox-shaped | what Phase 1 allows saying |
|---|---|---|
| **SOUNIO-PROVENANCE** | “this measurement vs that” is about individuals | Class-assertion wiring is a **bridge-sized** step; graph identity may need more |
| **SOUNIO-JUSTIFICATION** | empirical justification attaches to an instance | Same split: type of justification vs chain of acts |
| **SOUNIO-ADMISSIBILITY** | world-state as assertions | Needs a fact store; reasoner individual types are a thin slice |

---

## Phase 2 (blocked on founder confirmation)

Not written as a design to implement — only the fork Phase 1 opens:

1. **If founder accepts class-assertion bridge scope:** design ProvEntity→`reasoner_add_individual_type`, entity_type taxonomy as classes, queries via `reasoner_is_instance` / shared supers; cost = wiring + tests + concept Status honesty.  
2. **If product needs full PROV graph queries:** design must include TripleStore (or extension) population and query semantics; cost class changes; may be research.  
3. **Do not start either until Phase 1 is confirmed.**

---

## Reproduce

```bash
# Reasoner gate (Slurm or Linux x86_64)
bash scripts/ci/ontology_reasoner_compile_gate.sh

# Individual API caller count (expect only reasoner.sio)
rg -l --glob '*.sio' reasoner_is_instance

# Prov module
sed -n '1,100p' stdlib/epistemic/prov.sio
```
