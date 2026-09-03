import { Fragment, useState } from 'react';

// Mock data to simulate the ontology database (stdlib modules exist; type-level integration planned for 2026)
const ONTOLOGY_DB = [
  {
    id: "SNOMED:22298006",
    name: "Myocardial infarction",
    type: "Disease",
    database: "SNOMED CT",
    definition: "Ischemic necrosis of the heart muscle. The condition is characterized by a reduction in blood supply to the myocardium.",
    parents: ["Ischemic heart disease", "Myocardial disease"],
  },
  {
    id: "LOINC:718-7",
    name: "Hemoglobin [Mass/volume] in Blood",
    type: "Measurement",
    database: "LOINC",
    definition: "A measurement of the concentration of hemoglobin in whole blood.",
    parents: ["Erythrocyte measurements", "Hematology"],
  },
  {
    id: "GO:0006915",
    name: "Apoptotic process",
    type: "Biological Process",
    database: "Gene Ontology",
    definition: "A programmed cell death process which begins when a cell receives an internal or external signal and proceeds through a series of biochemical events.",
    parents: ["Programmed cell death"],
  },
  {
    id: "HPO:HP:0001658",
    name: "Myocardial infarction",
    type: "Phenotype",
    database: "HPO",
    definition: "Necrosis of the myocardium caused by an obstruction of the blood supply to the heart.",
    parents: ["Abnormality of cardiovascular system physiology"],
  }
];

export function OntologyAutocomplete() {
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [selectedTerm, setSelectedTerm] = useState<typeof ONTOLOGY_DB[0] | null>(null);

  const results = ONTOLOGY_DB.filter(term => 
    term.name.toLowerCase().includes(query.toLowerCase()) || 
    term.id.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div className="relative w-full max-w-2xl mx-auto font-sans">
      <div className="flex flex-col gap-2">
        <label className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider">
          Ontology Term Resolution (Stdlib Modules)
        </label>
        <div className="relative">
          <input
            type="text"
            className="w-full bg-[var(--color-surface-secondary)] border border-[var(--glass-border)] text-[var(--color-text-primary)] rounded-lg px-4 py-3 focus:outline-none focus:border-[var(--color-accent-teal)] focus:ring-1 focus:ring-[var(--color-accent-teal)] transition-all"
            placeholder="Search concepts (e.g. 'Myocardial', 'LOINC:718-7')..."
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setIsOpen(true);
              if (selectedTerm && e.target.value !== selectedTerm.name) {
                setSelectedTerm(null);
              }
            }}
            onFocus={() => setIsOpen(true)}
          />
          <div className="absolute right-3 top-3 text-[var(--color-text-tertiary)]">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
        </div>
      </div>

      {isOpen && query.length > 0 && !selectedTerm && (
        <div className="absolute z-10 w-full mt-2 bg-[var(--color-surface)] border border-[var(--glass-border)] rounded-xl shadow-2xl overflow-hidden backdrop-blur-xl">
          {results.length > 0 ? (
            <ul className="max-h-80 overflow-y-auto p-2">
              {results.map((term) => (
                <li key={term.id}>
                  <button
                    className="w-full text-left px-3 py-3 rounded-lg hover:bg-[var(--color-surface-hover)] transition-colors flex items-start gap-3 group"
                    onClick={() => {
                      setSelectedTerm(term);
                      setQuery(term.name);
                      setIsOpen(false);
                    }}
                  >
                    <div className="mt-0.5 w-6 h-6 rounded bg-[var(--color-accent-teal)]/10 text-[var(--color-accent-teal)] flex items-center justify-center shrink-0">
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                      </svg>
                    </div>
                    <div>
                      <div className="text-[var(--color-text-primary)] font-medium flex items-center gap-2">
                        {term.name}
                        <span className="text-xs px-1.5 py-0.5 rounded bg-[var(--color-surface-secondary)] text-[var(--color-text-secondary)] border border-[var(--glass-border)]">
                          {term.id}
                        </span>
                      </div>
                      <div className="text-xs text-[var(--color-text-secondary)] mt-1 line-clamp-1">
                        {term.database} • {term.type}
                      </div>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <div className="p-6 text-center text-[var(--color-text-secondary)]">
              No entities found matching "{query}"
            </div>
          )}
        </div>
      )}

      {selectedTerm && (
        <div className="mt-4 p-5 rounded-xl border border-[var(--color-accent-teal)]/30 bg-[var(--color-accent-teal)]/5 backdrop-blur-sm animate-in fade-in slide-in-from-top-2">
          <div className="flex justify-between items-start mb-3">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-[var(--color-accent-teal)]/20 text-[var(--color-accent-teal)] border border-[var(--color-accent-teal)]/30">
                  {selectedTerm.database}
                </span>
                <span className="text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">
                  {selectedTerm.type}
                </span>
              </div>
              <h3 className="text-xl font-bold text-[var(--color-text-primary)]">
                {selectedTerm.name}
              </h3>
            </div>
            <div className="text-sm font-mono text-[var(--color-text-tertiary)] bg-[var(--color-surface)] px-2 py-1 rounded border border-[var(--glass-border)]">
              {selectedTerm.id}
            </div>
          </div>
          
          <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed mb-4">
            {selectedTerm.definition}
          </p>

          <div className="space-y-2">
            <div className="text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider">
              Hierarchy Path
            </div>
            <div className="flex flex-wrap gap-2 items-center">
              {selectedTerm.parents.map((parent, idx) => (
                <Fragment key={idx}>
                  <div className="text-xs px-2 py-1 rounded bg-[var(--color-surface)] text-[var(--color-text-secondary)] border border-[var(--glass-border)]">
                    {parent}
                  </div>
                  <svg className="w-3 h-3 text-[var(--color-text-tertiary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                </Fragment>
              ))}
              <div className="text-xs px-2 py-1 rounded bg-[var(--color-accent-teal)]/10 text-[var(--color-accent-teal)] border border-[var(--color-accent-teal)]/20 font-medium">
                {selectedTerm.name}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}