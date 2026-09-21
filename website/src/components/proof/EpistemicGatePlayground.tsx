import { useState } from 'react';
import {
  EpistemicGateCard,
  type EpistemicGateCardProps,
  type GateVerdict,
} from '../epistemic/EpistemicGateCard';

type WitnessId = 'trustworthy' | 'unbounded' | 'refused';

type Witness = {
  id: WitnessId;
  label: string;
  card: EpistemicGateCardProps;
};

const WITNESSES: Witness[] = [
  {
    id: 'trustworthy',
    label: 'Trustworthy',
    card: {
      claim: 'gum_k95 finite-dof · Student-t, not silent 1.96',
      ceiling: 'E3',
      verdict: 'TRUSTWORTHY',
      reason:
        'Type-A-dominant budget (n=5) gives ν_eff≈4 and k95≈2.776. The witness integer is 2776. Wave10 closed the bitcast that printed 1.960 on this path. E4 is unreached: this is an executable gate, not a Lean theorem of the emitted factor. E5 is unreached.',
      sha: '6b76b700b0',
      href: 'https://github.com/Sounio-lang/sounio/blob/6b76b700b0/docs/audit/EPISTEMIC_TRUST_MAP_2026-07-14.md',
      hrefLabel: 'EPISTEMIC_TRUST_MAP_2026-07-14.md',
    },
  },
  {
    id: 'unbounded',
    label: 'Unbounded',
    card: {
      claim: 'check.sio implements well_typed_value_or_refuse',
      ceiling: 'E4',
      verdict: 'UNBOUNDED',
      reason:
        'SounioRefusalHonesty.lean proves the E219 fragment with no sorry. The file itself says that is a model, not a proof that check.sio implements the model. Citing the theorem as a compiler proof is empirical extrapolation. E5 is unreached.',
      sha: '31adf7b4bc',
      href: 'https://github.com/Sounio-lang/sounio/blob/31adf7b4bc/formal/lean4/SounioRefusalHonesty.lean',
      hrefLabel: 'SounioRefusalHonesty.lean',
    },
  },
  {
    id: 'refused',
    label: 'Refused',
    card: {
      claim: 'pre-TDM Cmin band ⊂ [10, 20] mg/L',
      ceiling: 'E3',
      verdict: 'REFUSED',
      reason:
        'Support [9.052178, 24.298861] overflows the therapeutic window. PRE_REFUSE. These are the #1797 landing witnesses for vancomycin_pbpk.sio main() (Fréchet corners, 6 d.p.). Variance is left uncalibrated in that source; this card does not print it as a zero. E4 and E5 are unreached because the band already fails the window.',
      sha: 'c5754c0c84',
      href: 'https://github.com/Sounio-lang/sounio/blob/c5754c0c84/website/src/components/home/KnowledgeRefusalPlayground.tsx',
      hrefLabel: 'KnowledgeRefusalPlayground.tsx',
    },
  },
];

export function EpistemicGatePlayground() {
  const [id, setId] = useState<WitnessId>('trustworthy');
  const witness = WITNESSES.find((w) => w.id === id) ?? WITNESSES[0];

  return (
    <div className="egc-play">
      <div className="egc-play-toggle" role="radiogroup" aria-label="Epistemic gate witness">
        {WITNESSES.map((item) => (
          <button
            key={item.id}
            type="button"
            role="radio"
            aria-checked={item.id === id}
            className="egc-play-btn"
            data-active={item.id === id}
            data-state={item.card.verdict.toLowerCase() as Lowercase<GateVerdict>}
            onClick={() => setId(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>
      <EpistemicGateCard {...witness.card} />
    </div>
  );
}
