import { useState } from 'react';
import {
  KnowledgeRefusalMeter,
  type KnowledgeRefusalMeterProps,
} from '../epistemic/KnowledgeRefusalMeter';

type WitnessId = 'refused' | 'verified' | 'uncertain';

type Witness = {
  id: WitnessId;
  label: string;
  source: string;
  meter: KnowledgeRefusalMeterProps;
};

const WITNESSES: Witness[] = [
  {
    id: 'refused',
    label: 'Refused',
    source: 'docs/audit/MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md · pre-TDM',
    meter: {
      signature: 'cmin: Knowledge<mg/L>',
      boundLow: 9.052178,
      boundHigh: 24.298861,
      variance: 0,
      varianceCalibrated: false,
      unit: 'mg/L',
      provenance: 'pre-TDM · 78.5 kg · CrCl 65 · 1000 mg q12h',
      state: 'refused',
      windowLow: 10,
      windowHigh: 20,
      guard: 'where band ⊂ [10, 20] mg/L',
      diagnostic:
        'PRE_REFUSE. Fréchet support [9.05, 24.30] is not inside the therapeutic window. The stdlib leaves variance uncalibrated (v = 0.0 in source); this control will not print var=0.000.',
    },
  },
  {
    id: 'verified',
    label: 'Verified',
    source: 'docs/audit/MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md · post-TDM',
    meter: {
      signature: 'cmin: Knowledge<mg/L>',
      boundLow: 12.820636,
      boundHigh: 17.358234,
      variance: 0,
      varianceCalibrated: false,
      unit: 'mg/L',
      provenance: 'post-TDM · 3 samples · same patient',
      state: 'verified',
      windowLow: 10,
      windowHigh: 20,
      guard: 'where band ⊂ [10, 20] mg/L',
      diagnostic:
        'POST_PRESCRIBE. Support [12.82, 17.36] ⊂ [10, 20]. Same uncalibrated variance field as the refused witness — still not rendered as a zero.',
    },
  },
  {
    id: 'uncertain',
    label: 'Uncertain',
    source: 'tests/run-pass/med/vancomycin_full_propagation.sio',
    meter: {
      signature: 'crcl: Knowledge<mL/min>',
      value: 65.0,
      epsilon: 0.72,
      unit: 'mL/min',
      provenance: 'Cockcroft_Gault_2025',
      state: 'uncertain',
      diagnostic:
        'The measurement stands. ε = 0.72 is a unit-interval confidence, not a bit pattern. This is not a dosing decision and not an error.',
    },
  },
];

export function KnowledgeRefusalPlayground() {
  const [id, setId] = useState<WitnessId>('refused');
  const witness = WITNESSES.find((w) => w.id === id) ?? WITNESSES[0];

  return (
    <div className="krm-play">
      <div className="krm-play-toggle" role="radiogroup" aria-label="Epistemic witness">
        {WITNESSES.map((w) => (
          <button
            key={w.id}
            type="button"
            role="radio"
            aria-checked={w.id === id}
            className="krm-play-btn"
            data-active={w.id === id}
            data-state={w.id}
            onClick={() => setId(w.id)}
          >
            {w.label}
          </button>
        ))}
      </div>
      <KnowledgeRefusalMeter {...witness.meter} />
      <p className="krm-play-source">{witness.source}</p>
    </div>
  );
}
