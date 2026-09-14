<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-graph-identity-composition
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-graph-identity-composition
-->

# Pireus: A Local Integer Is Not A Global IRI

> **Status**: Garden | **Date**: 2026-08-27 | **Authority**: founder direction

## Butterfly

Pireus already has independently frozen Sounio graphs for target profiles,
execution engines, x86 XED forms, Arm AARCHMRS forms, NVIDIA PTX forms, and
Apple Metal families. Those graphs use compact integer IRIs. An integer that
is unique inside one importer is not necessarily unique after graphs meet.

```text
local integer + producer scope -> graph identity
graph identity + declared namespace -> composable IRI
```

This Garden opens only the identity boundary required to compose existing
Pireus graphs. It does not renumber, rewrite, or supersede a frozen importer.

## Observed Collision

Read-only inspection of the current Sounio constant declarations found that
`execution_engine.sio` and `aarchmrs_import.sio` both allocate module terms in
the `703xxx` range. Eighteen declared integer values currently denote
different terms, including:

```text
703000 = PIREUS_CLASS_CLUSTER
703000 = PIREUS_ARM_CLASS_INSTRUCTION_FORM

703100 = PIREUS_CLUSTER_HAS_MACHINE
703100 = PIREUS_ARM_HAS_CORPUS

703200 = PIREUS_CLUSTER_DARWIN
703200 = PIREUS_ARM_CORPUS_AARCHMRS_FAT_2025_12

703220 = PIREUS_ENGINE_ISA_X86_64
703220 = PIREUS_ARM_VECTOR_IRI_FIXED_128

703231 = PIREUS_INTERFACE_METAL
703231 = PIREUS_ARM_TABLES_IRI_2
```

The complete overlap among those declarations is `703000..703004`,
`703100..703104`, `703200`, `703210..703211`, `703220..703221`,
`703230..703231`, and `703240`.

That declaration-only observation is not a complete store-level collision
census. The AARCHMRS projection also derives instruction-form subjects as
`703300 + i`, while the execution-engine graph declares individuals in that
region. The complete overlap therefore cannot be obtained by comparing
exported constants. This Garden deliberately freezes no replacement count:
the first Sounio executable must enumerate every admitted IRI-bearing triple
position, including derived subjects, and create the authoritative store-level
collision census itself. The number eighteen is evidence that composition is
necessary, not an expected collision result.

The import stores are transitive expansions rather than producer-local deltas.
Execution engine, XED, AARCHMRS, and PTX start from `pireus_v01_store()`;
Apple starts from `pireus_execution_engine_store()`. Their local witnesses
remain internally consistent, but a naive union would both identify unrelated
`703xxx` terms and repeat inherited base triples. Store isolation is therefore
containment, not a composition rule.

Producer scope is also not term ownership. The Apple importer deliberately
references `PIREUS_BLUEPRINT_APPLE_GPU` from the execution-engine module, and
the XED importer references access terms from the target-profile module. A
graph can contain shared terms and registered foreign terms alongside terms it
owns. Prefixing every integer with the graph producer would incorrectly split
those cross-module identities.

These observations select the Garden boundary. They are not a proposed global
numbering or a Sounio-produced expected result.

## Frozen Authority Boundary

The existing authority streams, source hashes, semantic hashes, receipts, and
local queries remain unchanged. In particular, the first executable must not:

- edit the frozen constants in an importer;
- silently renumber triples while they are still inside their local store;
- use one colliding importer's meaning to reinterpret another importer;
- declare that load order, file path, or current module name is the IRI;
- present a migrated graph as the original frozen authority graph.

Composition is a new provenance-carrying projection over local graphs.

## First Executable Contract

The first Sounio executable must create the authoritative identity registry and
composition result. It must:

1. admit a bounded declarative registry of graph producers and namespace
   owners before reading any graph, with graph production and term ownership
   represented separately;
2. identify every input graph by its Sounio producer concept and frozen source
   or semantics hash;
3. admit an acyclic parent-graph registry and verify each transitive input
   store against the exact frozen parent graph closure it declares;
4. distinguish shared Pireus core terms from producer-owned local terms by an
   explicit registry entry, never by numeric range inference alone;
5. resolve every subject, predicate, and IRI object through a term-owner entry,
   including explicit imports of terms owned by another producer and terms
   whose local integer is derived during graph construction;
6. make the lifted identity stable across graph order, process order,
   worktree, and machine;
7. distinguish a canonical lifted triple from each source occurrence of that
   triple, retaining graph, parent, producer, local integers, and source hash
   on the occurrence provenance;
8. resolve literal IDs through a registered literal-vocabulary owner, retain
   the exact source value bits and literal shape, and never reinterpret a
   literal as an IRI;
9. preserve the IRI-or-literal sort of every variable binding across composed
   query joins, rejecting incompatible reuse instead of comparing only the raw
   integer carried by the binding;
10. coalesce shared core terms and foreign term references only when their
   registry owner and identity agree;
11. coalesce identical lifted triples for composed query semantics without
   discarding or merging their distinct source occurrences;
12. preserve distinct identities for colliding producer-owned terms;
13. emit deterministic registry, dependency, lifted-graph, occurrence,
    collision, and provenance digests computed in Sounio;
14. create the first expected collision census, composed query results, and
    negative witnesses in Sounio.

The executable may use the ontology core's namespace-bearing `IRI` shape as an
implementation ingredient. This Garden does not freeze a numeric allocation
formula. The formula, bounds, overflow behavior, reserved namespaces, and
canonical serialization must be born in the first Sounio executable.

## Identity Classes

At minimum, the registry must distinguish:

```text
shared language term
shared Pireus core term
registered foreign term reference
registered parent graph
registered literal vocabulary
producer-owned class
producer-owned predicate
producer-owned individual
vendor corpus identity
evidence-role identity
legacy local integer
canonical lifted triple
source triple occurrence
canonical literal
```

An identical local integer in two producer scopes is evidence of a collision,
not evidence that the terms are equivalent. Conversely, duplicate copies of a
registered shared core term are candidates for coalescing only after exact
registry agreement.

## Required Negative Surface

At minimum, Sounio must deliberately reject:

- an input graph whose producer concept or frozen hash is absent;
- duplicate producer or namespace registrations;
- a namespace allocation that overlaps another allocation;
- a missing, cyclic, reordered, or hash-mismatched parent-graph declaration;
- a child store whose claimed inherited closure omits or changes a parent
  triple;
- an unregistered local IRI in any IRI-bearing triple position;
- a producer-owned term claimed as shared without an explicit registry edge;
- a producer-owned term presented as a foreign import, or a foreign reference
  rebound to the importing graph's namespace;
- two incompatible meanings assigned to one lifted identity;
- one local term mapped to two lifted identities;
- an IRI object treated as a literal, or a literal rewritten as an IRI;
- a query variable first bound as a literal and later reused in a subject,
  predicate, or IRI-object position, or the reverse reuse of an IRI-bound
  variable as a literal, even when the carried integers are equal;
- a literal ID used without a registered vocabulary owner;
- two literal occurrences coalesced solely by numeric equality despite
  different registered identities or value bits;
- integer overflow or capacity exhaustion during lifting;
- a result that changes when graph input order is reversed;
- duplicate inherited triples that inflate a composed query result;
- two equal canonical triples whose distinct source occurrences are lost;
- a collision census that omits any admitted colliding term;
- a collision census derived only from exported constant declarations;
- loss of producer, local-integer, source-hash, or triple provenance;
- promotion of a local-store query result into a composed-graph result without
  a composition receipt.

Python and Rust are prohibited. Node, Ruby, shell, `awk`, `bc`, a database
loader, RDF tool, or external model cannot create the registry, global
identities, expected collision census, composed graph, or expected queries.

## First Result Boundary

The authoritative result may establish only collision-free identity and graph
composition:

```text
ProducerScope -> FrozenAuthorityHash
ProducerScope -> NamespaceRegistration
ProducerGraph -> ParentGraph
GraphTermReference -> TermOwner
GraphLiteralReference -> LiteralVocabularyOwner
LocalTerm -> LiftedTerm
SourceLiteral -> CanonicalLiteral
SourceTripleOccurrence -> ProducerGraph
SourceTripleOccurrence -> CanonicalLiftedTriple
ComposedGraph -> CanonicalLiftedTriple
```

It emits zero instruction equivalences, zero processor observations, zero
capability inheritance, zero lowering choices, and zero performance claims.
No parity language may define or repair the registry retrospectively.

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This seed establishes only `GARDEN` for
`SOUNIO-PIREUS-GRAPH-IDENTITY-COMPOSITION`.
