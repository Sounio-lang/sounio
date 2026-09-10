import { Claim } from '../components/Claim';
import { DoseScene, DoseRig, DOSE } from '../components/Dose';
import { H2Scene, H2Contrast, H2 } from '../components/H2';
import { Refusals, REFUSALS } from '../components/Refusals';
import { CorpusBar, CORPUS } from '../components/Corpus';
import { EVIDENCE_STATS } from '../evidence/claim';

const P = EVIDENCE_STATS.proportions;

export function Home() {
  return (
    <div className="wrap">
      {/* 1. O MEDO — a cena é do leitor, não do projeto. Um site que abre
             pelo tamanho do próprio compilador está falando de si mesmo. */}
      <section className="band band-open" style={{ borderTop: 0 }}>
        <p className="kicker">a dose calculation</p>
        <h1>Your analysis returned a number.<br /><em>It was never that exact.</em></h1>
        <p className="lede lede-lg">
          A nurse weighs a patient: <b><Claim of={DOSE} metric="weight" /> kg</b>.
          The guideline says <b><Claim of={DOSE} metric="per_kg" /> mg/kg</b>.
          You multiply. Every language on earth gives you the same answer,
          instantly, with total confidence.
        </p>

        <DoseScene />
      </section>

      {/* 2. O GESTO — o argumento inteiro num movimento do dedo. */}
      <section className="band">
        <p className="kicker">the therapeutic window</p>
        <h2>
          The dose is safe between <Claim of={DOSE} metric="window_lo" /> and{' '}
          <Claim of={DOSE} metric="window_hi" /> mg.
        </h2>
        <p className="lede lede-lg">
          <b><Claim of={DOSE} metric="dose" /></b> sits dead centre. It looks like
          the safest possible answer — and it will keep looking that way no matter
          how bad the scale is, because a point estimate has nowhere to move.
          Drag the scale's precision and watch what does. The window is a bound
          your program declares; what changes is that Sounio checks it against the
          interval instead of against the point.
        </p>

        <DoseRig />
      </section>

      {/* 2b. A SEGUNDA CENA — outro campo, outra tese. Quem trabalha com
             combustão não se comove com dose de vancomicina. */}
      <section className="band">
        <p className="kicker">hydrogen ignition · GRI-Mech 3.0 H/O</p>
        <h2>Refine your time step and your uncertainty shrinks. That is not convergence.</h2>
        <p className="lede lede-lg">
          You integrated a hydrogen mechanism — <Claim of={H2} metric="species" />{' '}
          species, <Claim of={H2} metric="reactions" /> reactions — and reported a
          1-σ band on the radicals. Halve the step and the band shrinks by about
          √2. Halve it again and it shrinks again. Nothing about the chemistry
          changed.
        </p>

        <H2Scene />

        <p className="lede">
          Three implementations, measured under a factor four in step size. Two of
          them were written independently of each other; they agree, and they are
          both wrong in the same way.
        </p>

        <H2Contrast />

        <p className="lede">
          The per-step form adds an <em>independent</em> uncertainty at every
          step, when successive steps share the same rate parameters and are not
          independent. Quadrature over dependent terms emits a bound{' '}
          <em>tighter than the truth</em> — which is the one direction of error
          this language treats as a lie rather than an approximation.
        </p>
        <p className="note">
          Sounio refuses that composition at compile time, and the refusal is in
          the corpus below as <b>dsep_fork_unconditioned</b>. Wiring the same
          refusal into the chemistry module's quadrature path is a separate,
          still-open change — so the correctness above is measured, not enforced,
          and this page says which is which.
        </p>
      </section>

      {/* 3. A CREDENCIAL QUE IMPORTA — não o tamanho, a recusa. */}
      <section className="band">
        <p className="kicker">
          <Claim of={REFUSALS} metric="live" /> programs this compiler refuses
        </p>
        <h2>Not warnings. Not lint. It will not build.</h2>
        <p className="lede lede-lg">
          Sounio ships a corpus of programs that <em>must</em> fail to compile.
          Each one is a way of being wrong that your current language compiles
          without a word. Three of them, unedited:
        </p>

        <Refusals />
      </section>

      {/* 4. E NÃO É PROTÓTIPO — o tamanho entra aqui, depois do motivo. */}
      <section className="band">
        <h2>And it is not a prototype.</h2>
        <p className="lede">
          The compiler that enforces the above is written in Sounio and compiles
          itself: <b><Claim of={CORPUS} metric="compiler.lines" /> lines</b> of it,
          inside a written corpus of{' '}
          <b><Claim of={CORPUS} metric="written.lines" /></b> across{' '}
          <Claim of={CORPUS} metric="written.files" /> files, with the semantics
          formalised in Lean — <Claim of={CORPUS} metric="lean.lines" /> lines of it.
        </p>

        <CorpusBar />

        <p className="lede">
          Every numeral on this site, including all of the above, is read from{' '}
          <b>artifacts/**/*.json</b> at build time. A hand-written number fails the
          build. That is the same discipline the language applies to your data,
          applied to its own marketing.
        </p>

        <div className="readout">
          <div><b>{EVIDENCE_STATS.artifacts}</b><span>artifacts indexed</span></div>
          <div><b>{EVIDENCE_STATS.byKind.gate}</b><span>gates with a verdict</span></div>
          <div><b>{EVIDENCE_STATS.byLevel.refused}</b><span>gates that refuse</span></div>
          <div><b>{P.allPass}</b><span>reporting all-pass</span></div>
        </div>
      </section>

      {/* 5. A PORTA — a honestidade é o que faz o resto acreditável. */}
      <section className="band">
        <h2>Why you would believe any of this.</h2>
        <p className="lede lede-lg">
          The flagship uncertainty example in this repository documents an error
          bar half the width of the one its own arithmetic produces. It was found
          while building this page, and instead of being fixed quietly it is
          written up on <a href="#/honesty">the honesty page</a>, next to a formal
          model this project refuted and republished under a banner saying so.
        </p>
        <p className="note">
          That is worth more to you than any benchmark here. It is the one thing
          that tells you what happens in this project when something is found.
        </p>
        <div className="cta">
          <a className="btn btn-solid" href="#/honesty">Read the argument</a>
          <a className="btn" href="#/proof">Browse the corpus</a>
        </div>
      </section>
    </div>
  );
}
