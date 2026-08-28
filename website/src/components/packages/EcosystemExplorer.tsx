import { useState } from 'react';

const mockPackages = [
  {
    name: 'sounio-causal-core',
    domain: 'Math',
    gumScore: '99.9%',
    effect: '! Mut'
  },
  {
    name: 'medlang-pkpd',
    domain: 'Domain',
    gumScore: '99.8%',
    effect: '! None'
  },
  {
    name: 'sounio-fmri',
    domain: 'Domain',
    gumScore: '99.5%',
    effect: '! Gpu Io'
  },
  {
    name: 'onn-core',
    domain: 'Infrastructure',
    gumScore: '100%',
    effect: '! Gpu'
  }
];

export default function EcosystemExplorer() {
  const [filter, setFilter] = useState('All');

  const categories = ['All', 'Domain', 'Math', 'Infrastructure'];

  const filteredPackages = filter === 'All' 
    ? mockPackages 
    : mockPackages.filter(p => p.domain === filter);

  return (
    <div className="w-full max-w-7xl mx-auto p-6">
      <div className="flex gap-4 mb-8">
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => setFilter(cat)}
            className={`px-4 py-2 rounded-lg border border-[var(--glass-border)] transition-colors ${
              filter === cat ? 'text-black' : 'glass glass-specular text-white hover:bg-white/10'
            }`}
            style={filter === cat ? { backgroundColor: 'var(--color-accent-green)' } : {}}
          >
            {cat}
          </button>
        ))}
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {filteredPackages.map(pkg => (
          <div key={pkg.name} className="glass glass-specular border border-[var(--glass-border)] rounded-xl p-6 flex flex-col gap-4">
            <h3 className="text-xl font-bold text-white">{pkg.name}</h3>
            
            <div className="flex flex-col gap-2 text-sm mt-4">
              <div className="flex justify-between items-center bg-black/20 rounded p-2">
                <span className="font-semibold text-gray-300">Domain:</span>
                <span className="text-white">{pkg.domain}</span>
              </div>

              <div className="flex justify-between items-center bg-black/20 rounded p-2">
                <span className="font-semibold text-gray-300">GUM Compliant:</span>
                <span className="text-[var(--color-accent-green)]">{pkg.gumScore}</span>
              </div>
              
              <div className="flex justify-between items-center bg-black/20 rounded p-2">
                <span className="font-semibold text-gray-300">Effect:</span>
                <span className="font-mono text-xs text-orange-400">{pkg.effect}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
