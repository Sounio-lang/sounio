# stdlib/ontology

Ontology facade.

## Backend
SNOMED CT/ICD-10 terminology via compiler resolver

## Functions
- `onto_is_a`: Check hierarchy
- `search`: Search terms
- `ancestors`: Get ancestors
- `mapping`: Term mappings

## Modules
- `elplus.sio`: role-aware EL+ boolean closure (verified engine; dense
  64-concept variant and sparse 4096-class Anatomy-profile variant)
- `temporal.sio`: qualitative temporal reasoning — the pointisable
  Allen interval relations (forward before / meets / overlaps / starts
  / during / finishes / equality; inverses by argument swap) encoded as
  EL+ role composition over interval endpoints, with a
  path-consistency consistency oracle (`temporal_consistent`) over the
  closed dense matrix.  Demo:
  `examples/clinical/pathway_temporal_demo.sio`; gate:
  `scripts/clinical_pathway_temporal_gate.sh`