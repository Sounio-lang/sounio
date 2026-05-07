import { useState } from 'react';

export default function InteractiveRoadmap() {
  const [expandedNodes, setExpandedNodes] = useState<Record<number, boolean>>({});

  const toggleNode = (index: number) => {
    setExpandedNodes(prev => ({ ...prev, [index]: !prev[index] }));
  };

  const phases = [
    {
      title: 'Phase 1',
      items: ['Core Epistemics', 'MedLang DSL', 'fMRI Pipeline'],
      status: 'completed'
    },
    {
      title: 'Phase 2',
      items: ['Rustless Bootstrap', 'Poseidon VM', 'SOIR bytecode'],
      status: 'completed'
    },
    {
      title: 'Phase 3',
      items: ['Z3 SMT Verification', 'LLVM Backend', 'QNN-MNIST'],
      status: 'in-progress'
    },
    {
      title: 'Phase 4',
      items: ['Distributed Compilation', 'GLM-4.7 Optimization'],
      status: 'planned'
    }
  ];

  return (
    <div className="interactive-roadmap max-w-2xl mx-auto py-12">
      <div className="relative border-l-2 border-gray-200 dark:border-gray-800 ml-4 space-y-8">
        {phases.map((phase, i) => {
          const isCompleted = phase.status === 'completed';
          const isInProgress = phase.status === 'in-progress';
          const isPlanned = phase.status === 'planned';
          
          let colorClass = 'bg-gray-200 border-gray-300';
          let pulseClass = '';
          let textClass = 'text-gray-500';
          
          if (isCompleted) {
            colorClass = 'bg-green-500 border-green-600';
            textClass = 'text-green-600 dark:text-green-400';
          } else if (isInProgress) {
            colorClass = 'bg-[var(--color-accent-gold)] border-[var(--color-accent-gold)]';
            pulseClass = 'motion-safe:animate-pulse';
            textClass = 'text-[var(--color-accent-gold)]';
          } else if (isPlanned) {
            colorClass = 'bg-gray-300 border-gray-400 opacity-50';
            textClass = 'text-gray-400';
          }

          return (
            <div key={i} className="relative pl-8">
              <div 
                className={`absolute w-4 h-4 rounded-full -left-[9px] top-1.5 border-2 ${colorClass} ${pulseClass}`} 
              />
              
              <div 
                className="cursor-pointer select-none"
                onClick={() => toggleNode(i)}
              >
                <h3 className={`text-xl font-bold ${textClass}`}>
                  {phase.title}
                </h3>
              </div>
              
              {expandedNodes[i] && (
                <ul className="mt-3 list-disc pl-5 space-y-1 text-gray-700 dark:text-gray-300">
                  {phase.items.map((item, j) => (
                    <li key={j}>{item}</li>
                  ))}
                </ul>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
