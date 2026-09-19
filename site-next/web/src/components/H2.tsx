import { useState } from 'react';
import { Claim } from './Claim';
import { claim } from '../evidence/claim';
import { STEPS, widths, f3, type StepKey, type Ratios } from '../h2';
import './h2.css';

/** A cena de combustão. Extraída de benchmarks/chemistry/RESULTS.md. */
export const H2 = 'site/h2_scene.v1';

const SPECIES = ['H', 'O', 'OH', 'H2O', 'HO2', 'H2O2', 'H2', 'O2'];

const ratios = (side: 'quadrature' | 'native', s: string): Ratios => ({
  half1: claim(H2, `${side}.${s}.half1`).value,
  half2: claim(H2, `${side}.${s}.half2`).value,
  quarter: claim(H2, `${side}.${s}.quarter`).value,
});

// espécies cuja banda não acumula: dominada pela incerteza da condição inicial
const NON_ACC = new Set(['H2', 'O2']);

/**
 * Duas implementações, a mesma química, o mesmo mecanismo.
 *
 * A interação tem três posições porque a medição tem três. Um slider contínuo
 * insinuaria um continuum que ninguém mediu — e numa página sobre não afirmar
 * além da amostra isso seria o erro mais caro possível.
 */
export function H2Scene() {
  const [step, setStep] = useState<StepKey>('dt4');

  return (
    <>
      <div className="sweep" data-island="h2-sweep" data-step={step}>
        <div className="sweep-head">
          <p>1-σ band width, relative to its own value at the coarsest step</p>
          <div className="steps" role="group" aria-label="Integration step">
            {STEPS.map(s => (
              <button key={s.key} type="button" className="step" data-step={s.key}
                      aria-pressed={step === s.key} onClick={() => setStep(s.key)}>
                dt = {s.label}
              </button>
            ))}
          </div>
        </div>

        <div className="sweep-legend">
          <span data-side="quadrature">per-step quadrature</span>
          <span data-side="native">Sounio, coherent propagation</span>
        </div>

        <ul className="sp">
          {SPECIES.map(s => {
            const Q = widths(ratios('quadrature', s));
            const N = widths(ratios('native', s));
            // As três larguras vão para o DOM: a ilha da versão estática troca
            // de passo lendo estes atributos, sem recalcular nada.
            const attrs = (w: typeof Q) => ({
              'data-dt4': f3(w.dt4), 'data-dt2': f3(w.dt2), 'data-dt1': f3(w.dt1),
            });
            return (
              <li key={s} data-flat={NON_ACC.has(s) || undefined}>
                <span className="sp-name">{s}</span>
                <span className="sp-track">
                  <span className="sp-bar" data-side="quadrature" {...attrs(Q)}
                        style={{ width: `${Q[step] * 100}%` }} />
                  <span className="sp-bar" data-side="native" {...attrs(N)}
                        style={{ width: `${N[step] * 100}%` }} />
                </span>
                <span className="sp-n" data-side="quadrature" {...attrs(Q)}>{f3(Q[step])}</span>
                <span className="sp-n" data-side="native" {...attrs(N)}>{f3(N[step])}</span>
              </li>
            );
          })}
        </ul>

        <p className="sweep-foot">
          H2 and O2 do not move on either side. Their band is dominated by the
          uncertainty seeded in the initial mixture, which does not accumulate —
          so the rule that the band scales with the step is false for{' '}
          <Claim of={H2} metric="non_accumulating_count" /> of the{' '}
          <Claim of={H2} metric="species_reported" /> species reported. Saying it
          without that qualifier would be the same kind of overstatement this
          page is about.
        </p>
      </div>

      <p className="thesis">
        Refining the step made one band shrink and left the other alone.<br />
        <em>A band that tracks your solver is reporting the solver.</em>
      </p>
    </>
  );
}

/** O contraste em três implementações independentes, sob um fator quatro no passo. */
export function H2Contrast() {
  const rows = [
    { k: 'replica', name: 'Python replica', note: 'per-step independent quadrature' },
    { k: 'cpp', name: 'C++ cross-check', note: 'same formula, independently written from the published protocol rather than translated' },
    { k: 'native', name: 'Sounio native', note: 'coherent sensitivity propagation' },
  ];
  return (
    <div className="contrast">
      {rows.map(r => (
        <div className="cr" key={r.k} data-native={r.k === 'native' || undefined}>
          <span className="cr-name">{r.name}</span>
          <span className="cr-val">
            <Claim of={H2} metric={`${r.k}_lo`} /> – <Claim of={H2} metric={`${r.k}_hi`} />
          </span>
          <span className="cr-note">{r.note}</span>
        </div>
      ))}
    </div>
  );
}
