import { useState } from 'react';

const STAGES = [
  {
    id: 1,
    title: 'Source Code (.sio)',
    code: `// Patient dosing administration
let dose: Knowledge<f64> = input_dose
let volume: Knowledge<f64> = patient_volume

let conc: Knowledge<f64> = dose / volume`
  },
  {
    id: 2,
    title: 'Epistemic Type & Effect Checking',
    code: `[Type Check Passed]
- 'dose' inferred as Knowledge<f64> with valid provenance
- 'volume' inferred as Knowledge<f64>
- 'conc' resolved as Knowledge<f64>

[Effect Analysis]
- Requires 'Read<PatientData>' effect
- Epistemic uncertainty propagated: { conc.ε = f(dose.ε, vol.ε) }`
  },
  {
    id: 3,
    title: 'HIR / CPS Transform',
    code: `fn compute_conc_cps(dose: K[f64], volume: K[f64], k: Cont) -> ! {
    let _t1 = fdiv(dose.value, volume.value);
    let _t2 = propagate_epsilon(dose.ε, volume.ε);
    
    let conc = Knowledge(_t1, ε=_t2, prov="derived");
    k(conc)
}`
  },
  {
    id: 4,
    title: 'MIR Optimization (ML-guided)',
    code: `;; [ML-Guided Vectorization Applied]
;; Loop unrolling and SIMD alignment
%bb0:
  v1 = load %dose
  v2 = load %volume
  
  ;; Fused multiply-add replacement evaluated -> rejected
  v3 = fdiv fast v1, v2
  v4 = call @fast_variance_prop(v1, v2)
  
  ret { v3, v4 }`
  },
  {
    id: 5,
    title: 'Native CodeGen (PTX/x86 Assembly)',
    code: `; x86_64 AVX-512 target
vdivss %xmm1, %xmm0, %xmm2
vmovaps %xmm2, -8(%rbp)

; PTX TensorCore backend (Optional)
div.approx.ftz.f32 %f3, %f1, %f2;
mul.rn.f32 %f4, %f3, %f2;`
  }
];

export default function CompilerPipeline() {
  const [activeStage, setActiveStage] = useState(0);

  return (
    <div style={{
      display: 'flex', 
      flexDirection: 'row', 
      gap: '2rem', 
      width: '100%', 
      margin: '2rem 0',
      fontFamily: 'system-ui, sans-serif'
    }}>
      {/* Left: Stage List */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        {STAGES.map((stage, index) => {
          const isActive = index === activeStage;
          return (
            <button
              key={stage.id}
              onClick={() => setActiveStage(index)}
              style={{
                textAlign: 'left',
                padding: '1rem',
                borderRadius: '8px',
                border: `2px solid ${isActive ? 'var(--color-accent-green, #4ade80)' : 'transparent'}`,
                backgroundColor: isActive ? 'rgba(74, 222, 128, 0.1)' : 'rgba(255, 255, 255, 0.05)',
                color: isActive ? 'var(--color-accent-green, #4ade80)' : 'inherit',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                fontSize: '1rem',
                fontWeight: isActive ? 'bold' : 'normal'
              }}
            >
              <div style={{ fontSize: '0.8rem', opacity: 0.8, marginBottom: '0.2rem' }}>Stage {stage.id}</div>
              <div>{stage.title}</div>
            </button>
          );
        })}
      </div>

      {/* Right: Code Pane */}
      <div style={{
        flex: 1.5,
        backgroundColor: '#1e1e1e',
        color: '#d4d4d4',
        padding: '1.5rem',
        borderRadius: '12px',
        overflowX: 'auto',
        fontFamily: 'monospace',
        fontSize: '0.9rem',
        lineHeight: 1.5,
        boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
        transition: 'opacity 0.3s ease'
      }}>
        <pre style={{ margin: 0, transition: 'all 0.3s ease' }}>
          <code>
            {STAGES[activeStage].code}
          </code>
        </pre>
      </div>
    </div>
  );
}
