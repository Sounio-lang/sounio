import type { ReactNode } from 'react';
import { Claim } from './Claim';

/** O corpus de programas que o compilador recusa. */
export const REFUSALS = 'site/refusals.v1';

/**
 * Três recusas reais, escolhidas por serem erros que o leitor comete.
 *
 * Os nomes são os arquivos de `tests/compile-fail/`, sem edição. A descrição
 * diz o que o programa faz de errado e por que a linguagem dele deixaria
 * passar — que é a única parte que interessa a quem está avaliando adoção.
 */
const SHOWN: { id: string; what: ReactNode; why: string }[] = [
  {
    id: 'dsep_fork_unconditioned',
    what: <>Two quantities composed in quadrature as if independent, when a
          common cause makes them dependent.</>,
    why: 'The test\'s own comment states the consequence: it would emit a bound tighter than the truth. An error bar too narrow is the one kind this language will not let you write.',
  },
  {
    id: 'audit_posterior_weights_not_normalized',
    // os pesos são citados do arquivo de teste, por isso vão como código
    what: <>Bayesian model averaging where the posterior weights are <code>0.6</code>{' '}
          and <code>0.6</code>. Posterior weights must sum to one.</>,
    why: 'Every statistical language you know will average with those weights and return a number. It will look completely normal.',
  },
  {
    id: 'affine_double_use',
    what: 'A value that may be consumed once, consumed twice.',
    why: 'The second read is of something that is no longer there. In C it is a use-after-move; in a data pipeline it is a sample counted twice.',
  },
  {
    id: 'annotation_proof_mismatch',
    what: 'A function annotated as carrying a proof, whose proof does not match the claim it is attached to.',
    why: 'The annotation is the thing a reader trusts. Letting it drift from what was actually proved is how a citation becomes a lie.',
  },
];

export function Refusals() {
  return (
    <>
      <div className="refusals">
        {SHOWN.map(r => (
          <div className="ref" key={r.id}>
            <span className="ref-id">{r.id}</span>
            <div>
              <p>{r.what}</p>
              <p>{r.why}</p>
            </div>
          </div>
        ))}
      </div>
      <p className="note">
        The corpus holds <Claim of={REFUSALS} metric="live" /> live refusal tests
        across <Claim of={REFUSALS} metric="error_codes" /> declared error classes,
        plus <Claim of={REFUSALS} metric="ignored" /> currently disabled and marked
        as such in the source. The disabled ones are counted separately rather than
        folded into the headline, because a refusal that is switched off is not a
        refusal.
      </p>
    </>
  );
}
