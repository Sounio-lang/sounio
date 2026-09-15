import { Claim } from '../components/Claim';
import { Interval } from '../components/Interval';
import { Excluded, CORPUS } from '../components/Corpus';
import { DOSE } from '../components/Dose';
import { EVIDENCE, EVIDENCE_STATS } from '../evidence/claim';
import type { Artifact } from '../evidence/types';
import './honesty.css';

const RATCHET = 'gates/fn_type_effect_ratchet';
const CENSUS = 'research/cross_engine_runpass_census/metrics_check_latest';
const P = EVIDENCE_STATS.proportions;

const ratchet = EVIDENCE[RATCHET] as Artifact | undefined;

const ACTS = [
  {
    when: '08:00 · the model',
    title: 'A theorem can be false about your own language.',
    teaches:
      'A formal model is a claim about a system, and the system can refute it. When they disagree, one of them is wrong — and it is not automatically the implementation.',
    body: [
      'EpistemicEffects.lean lost subject reduction. A preservation counterexample was proved, and the ruling was recorded: collapsing a measurement payload to Real is a defect in the model, not a design choice in the language. The checker had been right all along — measure of a Nat is Knowledge<Nat>.',
      'The consequence is in the repository and not softened: do not cite that file as metatheory. It is a refuted model, and it stays published under a banner saying so. Deleting it would have been the dishonest option.',
    ],
  },
  {
    when: '14:00 · the instrument',
    title: 'A proof is more useful as a lamp than as a trophy.',
    teaches:
      'Once refusal can be stated formally, you can go looking for places that fail to refuse. The theorem stops being a credential and becomes a search procedure.',
    body: [
      'Two properties were proved in the E219 fragment with no sorry: that refusal is not zero, and that a well-typed program yields a value or refuses. That file is a model. It does not prove the checker implements the model — the gate stops at E4 and says so, which is the difference between this page and marketing.',
      'The same afternoon the model was turned against the compiler and found two sites still fabricating after a refusal. abs() + 1 was still typed i64 where it should have been an error type. An empty IR stub still read as a fall-through zero; it now emits a trap.',
    ],
  },
  {
    when: '01:00 · the same absence',
    title: 'Four lanes, one bug class: a plausible value where absence belonged.',
    teaches:
      'The interesting defects are not wrong numbers. They are plausible numbers standing in for something never measured — invisible until you have a word for them.',
    body: [
      'Four lanes sharing no write-set landed the same discovery on four surfaces. A pin counter read as zero live pins when the emitter never incremented it; its sentinel is now negative one, meaning unwired rather than empty. A call instruction was refused only as an unknown opcode; it now has a name and declares that it has no value. And a reclamation wall shipped its own failing witness before the fix existed.',
      'The class is not closed — one surface is still open on the dissertation. What changed that night is that the class became visible, and visible is the prerequisite for closed.',
    ],
  },
];

export function Honesty() {
  return (
    <div className="wrap">
      <section className="band" style={{ borderTop: 0 }}>
        <p className="kicker">17–18 august 2026 · one day</p>
        <h1>The model was wrong. The compiler was not.<br /><em>The silent zero was the lie.</em></h1>
        <p className="lede">
          This page is one day, not a catalogue of controls. In twenty-four hours a formal
          model lost a theorem, a proof was turned into a lamp and found two places still
          fabricating, and four independent lanes named the same absence on four surfaces.
        </p>
        <p className="lede">
          Everything else in this language — the types, the gates, the compiler — exists to
          make one sentence enforceable: <b>a well-typed program returns a value, or it refuses.</b>
        </p>
      </section>

      <ol className="acts">
        {ACTS.map(a => (
          <li className="act" key={a.when}>
            <p className="act-when">{a.when}</p>
            <div className="act-body">
              <h2>{a.title}</h2>
              <p className="act-teaches">{a.teaches}</p>
              {a.body.map((t, i) => <p className="lede" key={i}>{t}</p>)}
            </div>
          </li>
        ))}
      </ol>

      {/* O mesmo defeito da terceira acto, cometido por mim, nesta página. */}
      <section className="band">
        <h2>The same discipline, turned on this page</h2>
        <p className="lede">
          The third act names a bug class: <b>a plausible value standing in for something
          that was never what it claimed</b>. While measuring the corpus for the opening of
          this site, I produced one.
        </p>
        <p className="lede">
          An early pass weighted the field by file and swept everything unclassified into a
          bucket labelled <b>tooling</b>. That bucket held{' '}
          <Claim of={CORPUS} metric="excluded.lines" /> lines across{' '}
          <Claim of={CORPUS} metric="excluded.files" /> files — an average thick enough to
          give it away. Almost none of it was written. It is generated scale probes, the
          largest a single file of <Claim of={CORPUS} metric="biggest_generated" /> lines,
          and superseded bootstrap images kept for the record.
        </p>
        <p className="lede">
          Counted as written, the opening figure would have been{' '}
          <Claim of={CORPUS} metric="total_tracked" /> rather than{' '}
          <Claim of={CORPUS} metric="written.lines" />. Three trees are excluded from every
          figure on this site, and the exclusion is published at the same scale as the thing
          it removes:
        </p>

        <Excluded />

        <p className="note">
          Counting generated output as authored work is the first thing a project like this
          has to refuse, and the refusal is worth nothing unless the amount refused is
          stated. The written corpus is{' '}
          <Claim of={CORPUS} metric="written.lines" /> lines; what you see above is what
          that number would have been inflated by.
        </p>
      </section>

      {/* O achado no exemplo carro-chefe. Cuidado deliberado no veredito:
             o que está estabelecido é a DISCORDÂNCIA, não qual lado erra. */}
      <section className="band">
        <h2>The example on the front page understated its own uncertainty</h2>
        <p className="lede lede-lg">
          The clinical scene that opens this site is not a mock-up. Its numbers are
          extracted at build time from{' '}
          <b>examples/real_world/01_dose_uncertainty.sio</b>, the flagship
          demonstration of GUM uncertainty propagation in this repository.
        </p>
        <p className="lede">
          Its header documents the expected output as{' '}
          <b>σ = <Claim of={DOSE} metric="documented_sd" /> mg</b>, interval{' '}
          <b>[<Claim of={DOSE} metric="documented_ci_lo" />,{' '}
          <Claim of={DOSE} metric="documented_ci_hi" />]</b>. The arithmetic the
          file itself writes out gives a variance of{' '}
          <Claim of={DOSE} metric="variance" />, so{' '}
          <b>σ = <Claim of={DOSE} metric="sd" /> mg</b> and interval{' '}
          <b>[<Claim of={DOSE} metric="ci_lo" />,{' '}
          <Claim of={DOSE} metric="ci_hi" />]</b> — a band{' '}
          <Claim of={DOSE} metric="computed_width" /> mg wide where the header
          promises <Claim of={DOSE} metric="documented_width" />.
        </p>
        <p className="lede">
          <b>Which of the two is wrong is not yet settled, and this page will not
          pretend otherwise.</b> The header may simply be stale. Or the code may
          have gained an uncertainty on the dosing guideline that the header never
          intended — in which case the question is whether a clinical guideline
          range belongs in the propagation at all. That is a modelling decision,
          not a typo, and it is open.
        </p>
        <p className="note">
          What is settled is that they disagree, and that the disagreement runs in
          the direction this language exists to prevent: the documented figure
          claims to know more than the computation supports. The build now extracts
          these parameters from the example rather than restating them, so the
          scene on the front page cannot drift from the file it illustrates —
          which is how the disagreement surfaced in the first place.
        </p>
      </section>

      <section className="band">
        <h2>Where the argument is enforced</h2>
        <p className="lede">
          Not decoration. Three places you can check, with the numbers read from the corpus.
        </p>

        <div className="enforce">
          <div>
            <h3>Types</h3>
            <p className="lede">
              Measurement carries its payload type; a refusal is an error type, not a zero.
              The effect ratchet passes <Claim of={RATCHET} metric="passed" /> of{' '}
              <Claim of={RATCHET} metric="total" /> — which is this band, not certainty:
            </p>
            {ratchet && <Interval a={ratchet} showId={false} />}
          </div>

          <div>
            <h3>Two engines</h3>
            <p className="lede">
              Madaros and Lean over <Claim of={CENSUS} metric="total" /> cases, diverging
              explicitly on <Claim of={CENSUS} metric="diverge_explicit" />. The divergences
              are published, because a disagreement between your own engines is evidence.
            </p>
          </div>

          <div>
            <h3>The ladder</h3>
            <p className="lede">
              Every claim stops at a level, E0 through E5, stated rather than implied.
              Stopping at E4 is not a failure. Saying you reached E5 when you stopped at E4 is.
              Across the corpus, <b>{P.allPass}</b> gates report all-pass, and not one of them reaches certainty.
            </p>
          </div>
        </div>

        <p className="note">
          This page is the day, not a catalogue of controls. The instruments appear where the
          story needs them — and the story is the argument.
        </p>
        <div className="cta">
          <a className="btn btn-solid" href="#/proof">See the corpus</a>
        </div>
      </section>
    </div>
  );
}
