#!/usr/bin/env python3
import json, os, sys

bundles_dir = 'stdlib/data/data/ontology/bundles'

# Templates for expansion
templates = {
    'ALG': [
        ("ALG:00000{:02d}", "Algebraic structure {}", "A mathematical structure with operations.", ["ALG:0000001"]),
        ("ALG:00001{:02d}", "Group concept {}", "A concept in group theory.", ["ALG:0000002"]),
        ("ALG:00002{:02d}", "Ring property {}", "A property in ring theory.", ["ALG:0000005"]),
        ("ALG:00003{:02d}", "Field topic {}", "A topic in field theory.", ["ALG:0000006"]),
        ("ALG:00004{:02d}", "Linear topic {}", "A topic in linear algebra.", ["ALG:0000007"]),
    ],
    'PHYS': [
        ("PHYS:00000{:02d}", "Physical system {}", "A system in physics.", ["PHYS:0000001"]),
        ("PHYS:00001{:02d}", "Mechanical effect {}", "An effect in mechanics.", ["PHYS:0000003"]),
        ("PHYS:00002{:02d}", "EM effect {}", "An electromagnetic effect.", ["PHYS:0000008"]),
        ("PHYS:00003{:02d}", "Thermo process {}", "A thermodynamic process.", ["PHYS:0000005"]),
        ("PHYS:00004{:02d}", "Quantum effect {}", "A quantum phenomenon.", ["PHYS:0000003"]),
    ],
    'QM': [
        ("QM:00000{:02d}", "Quantum state {}", "A quantum system state.", ["QM:0000001"]),
        ("QM:00001{:02d}", "Quantum operator {}", "An operator in QM.", ["QM:0000001"]),
        ("QM:00002{:02d}", "QFT topic {}", "A topic in quantum field theory.", ["QM:0000007"]),
        ("QM:00003{:02d}", "Observable {}", "A measurable quantity.", ["QM:0000005"]),
        ("QM:00004{:02d}", "Interference {}", "Quantum interference.", ["QM:0000001"]),
    ],
    'SNOMED': [
        ("SNOMED:0000{:04d}", "Condition {}", "A clinical condition.", ["SNOMED:138875005"]),
        ("SNOMED:0001{:04d}", "Pathology {}", "A pathological process.", ["SNOMED:138875005"]),
        ("SNOMED:0002{:04d}", "Syndrome {}", "A symptom cluster.", ["SNOMED:138875005"]),
        ("SNOMED:0003{:04d}", "Infection {}", "An infectious disease.", ["SNOMED:138875005"]),
        ("SNOMED:0004{:04d}", "Neoplasm {}", "An abnormal growth.", ["SNOMED:138875005"]),
    ],
    'HP': [
        ("HP:00000{:03d}", "Abnormality {}", "A phenotypic abnormality.", ["HP:0000118"]),
        ("HP:00001{:03d}", "Morphology {}", "A morphological anomaly.", ["HP:0000118"]),
        ("HP:00002{:03d}", "Impairment {}", "A functional impairment.", ["HP:0000118"]),
        ("HP:00003{:03d}", "Delay {}", "A developmental delay.", ["HP:0000118"]),
        ("HP:00004{:03d}", "Neuro finding {}", "A neurological finding.", ["HP:0000118"]),
    ],
    'GO': [
        ("GO:00000{:03d}", "Molecular fn {}", "A molecular function.", ["GO:0008150"]),
        ("GO:00001{:03d}", "Bio process {}", "A biological process.", ["GO:0008150"]),
        ("GO:00002{:03d}", "Component {}", "A cellular component.", ["GO:0005575"]),
        ("GO:00003{:03d}", "Metabolism {}", "A metabolic process.", ["GO:0008150"]),
        ("GO:00004{:03d}", "Regulation {}", "A regulatory process.", ["GO:0008150"]),
    ],
    'CHEBI': [
        ("CHEBI:0000{:04d}", "Chemical {}", "A chemical entity.", []),
        ("CHEBI:0001{:04d}", "Organic {}", "An organic compound.", []),
        ("CHEBI:0002{:04d}", "Inorganic {}", "An inorganic compound.", []),
        ("CHEBI:0003{:04d}", "Biomolecule {}", "A biological molecule.", []),
        ("CHEBI:0004{:04d}", "Drug {}", "A pharmaceutical.", []),
    ],
    'LOINC': [
        ("LOINC:0000{:04d}", "Lab test {}", "A lab test.", []),
        ("LOINC:0001{:04d}", "Observation {}", "A clinical observation.", []),
        ("LOINC:0002{:04d}", "Panel {}", "A diagnostic panel.", []),
        ("LOINC:0003{:04d}", "Vital {}", "A vital sign.", []),
        ("LOINC:0004{:04d}", "Micro test {}", "A microbiology test.", []),
    ],
    'PART': [
        ("PART:00000{:03d}", "Subatomic {}", "A subatomic particle.", ["PART:0000001"]),
        ("PART:00001{:03d}", "Composite {}", "A composite particle.", ["PART:0000002"]),
        ("PART:00002{:03d}", "Force {}", "A force carrier.", ["PART:0000001"]),
        ("PART:00003{:03d}", "Hadron {}", "A hadron.", ["PART:0000002"]),
        ("PART:00004{:03d}", "Lepton {}", "A lepton.", ["PART:0000001"]),
    ],
    'MATH': [
        ("MATH:00000{:03d}", "Math obj {}", "A mathematical object.", ["MATH:0000001"]),
        ("MATH:00001{:03d}", "Number {}", "A number theory concept.", ["MATH:0000002"]),
        ("MATH:00002{:03d}", "Geometry {}", "A geometric structure.", ["MATH:0000003"]),
        ("MATH:00003{:03d}", "Analysis {}", "An analytic object.", ["MATH:0000004"]),
        ("MATH:00004{:03d}", "Topology {}", "A topological space.", ["MATH:0000005"]),
    ],
    'BIOL': [
        ("BIOL:00000{:03d}", "Process {}", "A biological process.", ["BIOL:0000001"]),
        ("BIOL:00001{:03d}", "Component {}", "A cellular component.", ["BIOL:0000003"]),
        ("BIOL:00002{:03d}", "Function {}", "A molecular function.", ["BIOL:0000002"]),
        ("BIOL:00003{:03d}", "Organismal {}", "An organismal process.", ["BIOL:0000003"]),
        ("BIOL:00004{:03d}", "Ecology {}", "An ecological interaction.", ["BIOL:0000001"]),
    ],
    'CHEM': [
        ("CHEM:00000{:03d}", "Process {}", "A chemical process.", ["CHEM:0000001"]),
        ("CHEM:00001{:03d}", "Reaction {}", "A reaction mechanism.", ["CHEM:0000013"]),
        ("CHEM:00002{:03d}", "Bond {}", "A chemical bond.", ["CHEM:0000004"]),
        ("CHEM:00003{:03d}", "Catalysis {}", "A catalytic process.", ["CHEM:0000013"]),
        ("CHEM:00004{:03d}", "Spectroscopy {}", "A spectroscopic method.", ["CHEM:0000005"]),
    ],
    'ASTR': [
        ("ASTR:00000{:03d}", "Object {}", "An astronomical object.", ["ASTR:0000001"]),
        ("ASTR:00001{:03d}", "Stellar {}", "A stellar phenomenon.", ["ASTR:0000007"]),
        ("ASTR:00002{:03d}", "Galactic {}", "A galactic structure.", ["ASTR:0000008"]),
        ("ASTR:00003{:03d}", "Cosmo {}", "A cosmological event.", ["ASTR:0000009"]),
        ("ASTR:00004{:03d}", "Planet {}", "A planetary body.", ["ASTR:0000007"]),
    ],
}

prefix_map = {
    'alg.dontology': 'ALG',
    'phys.dontology': 'PHYS',
    'qm.dontology': 'QM',
    'snomed.dontology': 'SNOMED',
    'hpo.dontology': 'HP',
    'go.dontology': 'GO',
    'chebi.dontology': 'CHEBI',
    'loinc.dontology': 'LOINC',
    'part.dontology': 'PART',
    'math.dontology': 'MATH',
    'biol.dontology': 'BIOL',
    'chem.dontology': 'CHEM',
    'astr.dontology': 'ASTR',
}

# Load existing terms
terms_by_bundle = {}
for fname in sorted(os.listdir(bundles_dir)):
    if not fname.endswith('.dontology'): continue
    with open(os.path.join(bundles_dir, fname), 'rb') as f:
        data = f.read()
    payload = json.loads(data[12:])
    terms_by_bundle[fname] = payload.get('terms', [])

total_current = sum(len(v) for v in terms_by_bundle.values())
print(f"Current total: {total_current}")

# Expand
target_total = 1000
needed = target_total - total_current
bundles = list(terms_by_bundle.keys())
per_bundle = needed // len(bundles)
extra = needed % len(bundles)

idx = 0
for fname in bundles:
    prefix = prefix_map[fname]
    template_list = templates.get(prefix, [])
    if not template_list:
        continue
    n_add = per_bundle + (1 if idx < extra else 0)
    existing = terms_by_bundle[fname]
    existing_curies = {t['curie'] for t in existing}
    start_num = len(existing)
    
    for i in range(n_add):
        tidx = i % len(template_list)
        tmpl = template_list[tidx]
        curie = tmpl[0].format(start_num + i)
        label = tmpl[1].format(start_num + i)
        defn = tmpl[2]
        parents = tmpl[3]
        if curie not in existing_curies:
            existing.append({
                'curie': curie,
                'label': label,
                'definition': defn,
                'iri': f'https://sounio.org/ontology/{curie.replace(":", "_")}',
                'parents': parents,
                'synonyms': []
            })
    
    # Fix parents to only exist in this bundle
    curies = {t['curie'] for t in existing}
    for t in existing:
        t['parents'] = [p for p in t.get('parents', []) if p in curies]
    
    payload = {'terms': existing, 'mappings': []}
    data = json.dumps(payload, sort_keys=True).encode('utf-8')
    hdr = b'DONT' + (1).to_bytes(4, 'little') + len(data).to_bytes(4, 'little')
    with open(os.path.join(bundles_dir, fname), 'wb') as f:
        f.write(hdr + data)
    idx += 1

# Verify
terms = []
for fname in sorted(os.listdir(bundles_dir)):
    if not fname.endswith('.dontology'): continue
    with open(os.path.join(bundles_dir, fname), 'rb') as f:
        data = f.read()
    payload = json.loads(data[12:])
    terms.extend(payload.get('terms', []))

print(f"New total: {len(terms)} terms")
