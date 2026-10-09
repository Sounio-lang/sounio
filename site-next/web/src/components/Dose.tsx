import { useState } from 'react';
import { claim } from '../evidence/claim';
import { propagate, scaleKind, pos, ticks, f1, type Scene } from '../dose';
import './dose.css';

/** A cena vem de um exemplo real do repositório, extraído — não digitado. */
export const DOSE = 'site/dose_scene.v1';

const v = (m: string) => claim(DOSE, m).value;

const scene: Scene = {
  weight: v('weight'), perKg: v('per_kg'), perKgVar: v('per_kg_var'),
  windowLo: v('window_lo'), windowHi: v('window_hi'), z: v('z'),
  axisLo: v('axis_lo'), axisHi: v('axis_hi'), tickStep: v('tick_step'),
};
const WEIGHT_SD = v('weight_sd');
const SIG = { min: v('sigma_min'), max: v('sigma_max'), step: v('sigma_step') };
const PER_KG_SD = v('per_kg_sd');

const n = (x: number) => x.toLocaleString('en-GB');

/**
 * Os dois painéis: a mesma aritmética, e o que cada linguagem guarda dela.
 *
 * O código à direita é sintaxe real e testada — `measure(_, uncertainty:)` com
 * `Knowledge<f64>`, e multiplicação Knowledge⊗Knowledge, ambos exercitados por
 * exemplos `run-pass` do repositório. Mostrar sintaxe que não compila seria
 * exatamente o que esta página acusa.
 */
export function DoseScene() {
  const r = propagate(scene, WEIGHT_SD);
  return (
    <>
      <div className="scene">
        <div>
          <h3>Any language you already use</h3>
          <pre>
{`weight = ${f1(scene.weight)}
per_kg = ${f1(scene.perKg)}
dose   = weight * per_kg
print(dose)`}
          </pre>
          <div className="out">
            <span>stdout</span>
            <b>{f1(r.mean)}</b>
          </div>
        </div>
        <div>
          <h3>Sounio</h3>
          <pre>
{`let weight: Knowledge<f64> = measure(${f1(scene.weight)}, uncertainty: ${f1(WEIGHT_SD)})
let per_kg: Knowledge<f64> = measure(${f1(scene.perKg)}, uncertainty: ${f1(PER_KG_SD)})
let dose = weight * per_kg`}
          </pre>
          <div className="out">
            <span>dose : Knowledge&lt;f64&gt;</span>
            <b className="band">{f1(r.mean)} ± {f1(r.sd)} mg</b>
          </div>
        </div>
      </div>
      <p className="note">
        Same arithmetic. The left column discarded what you knew about your own
        measurement on line three, and nothing complained. That number went into
        a table; the table went into a paper. The scale was{' '}
        <b>± {f1(WEIGHT_SD)} kg</b> and the guideline was a range, and neither
        survived the multiplication.
      </p>
    </>
  );
}

/* ------------------------------------------------------------------------ */

const TICKS = ticks(scene);
const at = (mg: number) => `${pos(scene, mg)}%`;

/**
 * O instrumento: arraste a precisão da balança.
 *
 * O ponto nunca sai do centro. A banda cresce até deixar a janela. É o
 * argumento inteiro num gesto — e é por isso que ele é interativo em vez de
 * ilustrado: ninguém acredita numa animação, todo mundo acredita no próprio dedo.
 */
export function DoseRig() {
  const [sd, setSd] = useState(WEIGHT_SD);
  const r = propagate(scene, sd);

  return (
    <>
      <div className="rig">
        <div className="rig-head">
          <p>dose = weight × {f1(scene.perKg)} mg/kg · GUM propagation · 95% coverage</p>
          {/* a ilha da versão estática atualiza este nó pela classe */}
          <p className="rig-var">Var = {Math.round(r.variance)}</p>
        </div>

        <div className="axis" data-refused={r.outside || undefined}>
          <div className="win" style={{ left: at(scene.windowLo),
                 width: `${pos(scene, scene.windowHi) - pos(scene, scene.windowLo)}%` }} />
          <span className="win-lab" style={{ left: at(scene.windowLo) }}>
            {n(scene.windowLo)} mg
          </span>
          <span className="win-lab" style={{ left: at(scene.windowHi) }}>
            {n(scene.windowHi)} mg
          </span>
          <div className="edge" style={{ left: at(scene.windowLo) }} />
          <div className="edge" style={{ left: at(scene.windowHi) }} />
          <div className="bandbar" style={{ left: at(r.lo),
                 width: `${pos(scene, r.hi) - pos(scene, r.lo)}%` }} />
          <div className="pointmk" style={{ left: at(r.mean) }} />
          <span className="pointlab" style={{ left: at(r.mean) }}>{f1(r.mean)}</span>
          <div className="ticks">
            {TICKS.map(t => <span key={t} style={{ left: at(t) }}>{n(t)}</span>)}
          </div>
        </div>

        <div className="ctl">
          <div>
            <label htmlFor="dose-sd">how well the patient was weighed</label>
            <input id="dose-sd" type="range"
                   min={SIG.min} max={SIG.max} step={SIG.step}
                   value={sd} onChange={e => setSd(+e.target.value)} />
            <div className="ctl-val">
              <span>σ = <b>{f1(sd)}</b> kg</span>
              <span>{scaleKind(sd)}</span>
            </div>
          </div>
          <div className="verdict" data-refused={r.outside || undefined}>
            <div className="verdict-k">what can still be asserted</div>
            <div className="verdict-v">
              {r.outside ? 'REFUSED' : `${f1(r.mean)} ± ${f1(r.sd)} mg`}
            </div>
            <div className="verdict-w">
              {r.outside
                ? <>The interval [{f1(r.lo)}, {f1(r.hi)}] leaves the window. The
                    bound is checked against the interval, so there is no longer
                    a safe dose to assert.</>
                : <>Interval [{f1(r.lo)}, {f1(r.hi)}] — inside the window</>}
            </div>
          </div>
        </div>
      </div>

      <p className="thesis">
        The point estimate never moved off {f1(r.mean)}.<br />
        <em>A point cannot cross a line. Only a band can.</em>
      </p>
    </>
  );
}
