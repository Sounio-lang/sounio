import { useEffect, useRef, useState, type CSSProperties } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
gsap.registerPlugin(ScrollTrigger);

type ActId = 1 | 2 | 3 | 4;

interface ActData {
  id: ActId;
  numeral: string;
  title: string;
  subtitle: string;
  image: string;
  imageAlt: string;
  blendMode?: CSSProperties['mixBlendMode'];
  paragraphs: string[];
}

interface ActTheme {
  bg: string;
  accent: string;
  border: string;
  numeralColor: string;
}

const ACTS: ActData[] = [
  {
    id: 1,
    numeral: 'I',
    title: 'The Illusion of Exactness',
    subtitle: 'The Pre-Existence Assumption',
    image: '/assets/vision/act1-column.jpg',
    imageAlt: 'A Greek marble column shattering under red digital glitch distortions',
    paragraphs: [
      'For decades, programming languages have treated numbers as perfect. 5.23 is exactly that — no more, no less. But this is not how science works. This is not how measurement works. This is not how reality works.',
      'When a measurement of 5.23 mg/L flows between systems and databases, the ±0.15 disappears. Downstream analyses treat it as exact. Between 2011 and 2021, an estimated $28 billion was wasted on irreproducible preclinical research — not from malice, but from the silent erasure of uncertainty.',
      'The assumption that data exists in determinate form before we ask about it is so pervasive it escapes notice. Classical science inherited it from classical physics. Classical programming inherited it from mathematics.',
      'When the assumption fails, it fails silently.',
    ],
  },
  {
    id: 2,
    numeral: 'II',
    title: 'Productive Indeterminacy',
    subtitle: 'Measurement Produces. It Does Not Sample.',
    image: '/assets/vision/act2-poseidon.jpg',
    imageAlt: 'An ethereal Poseidon figure emerging from glowing data streams and digital waves',
    blendMode: 'screen',
    paragraphs: [
      'Social theory inherits from physics an assumption so pervasive it escapes notice: that beliefs exist before we ask about them, that preferences precede their expression, that the survey samples what was already there.',
      'Quantum cognition research has shown otherwise. Opinion-production exhibits specifically quantum features — order effects where response to question B depends on whether A was asked first, in ways that violate classical probability. The measurement does not reveal the state. The measurement constitutes it.',
      'This is not philosophical speculation. It is the structural condition of all complex systems where the observer cannot be extracted from the observation.',
      'For phenomena exhibiting constitutive indeterminacy, classical vocabulary does not simplify. It distorts.',
    ],
  },
  {
    id: 3,
    numeral: 'III',
    title: 'Entropic Coherence',
    subtitle: 'Order Is Not Given. Order Is Maintained.',
    image: '/assets/vision/act3-fractal.jpg',
    imageAlt: 'A recursive golden fractal gear structure with green energy currents against dark chaos',
    paragraphs: [
      'We experience unified selfhood as given. Continuous identity persisting through time. But dissociative disorders expose the work that healthy functioning conceals — coherence is not substance but maintained relationship.',
      'In quantum mechanics, coherence is not a thing. It is the preservation of phase correlations among the components of a superposition. And it is thermodynamically unstable. It degrades spontaneously. Maintaining it requires continuous work against entropic drift.',
      'Scientific truth works the same way. A measurement, a model, a result — these are not static facts. They are coherence structures, maintained by provenance chains, reproducibility protocols, and uncertainty bounds. Remove those structures and the result decoheres into noise.',
      'The Symbolic Manifolds framework formalizes this: knowledge is a recursive eigenform — what remains invariant under the transformative operation of scrutiny. Not what is simply there, but what survives being questioned.',
    ],
  },
  {
    id: 4,
    numeral: 'IV',
    title: 'The Sounio Solution',
    subtitle: 'Confidence-Aware Programming',
    image: '/assets/vision/act4-sounion.jpg',
    imageAlt: 'Cape Sounion with a holographic green Greek temple standing over the Aegean sea at sunset',
    paragraphs: [
      'Cape Sounion stands at the southernmost tip of Attica. For 2,500 years the Temple of Poseidon has watched the Aegean — a fixed point from which sailors navigated the uncertain sea.',
      "Sounio the language embodies this: computing at the boundary between what we know and what we don't. Every value carries not just data, but knowledge of its own uncertainty.",
      'Knowledge<T> is not an annotation. It is not a library. It is a first-class type that makes indeterminacy structurally visible — where measurement participates in producing what it claims to sample, and confidence gates decide what execution is authorized to proceed.',
      'The columns stand firm. The sea is unpredictable. Sounio helps you reason about both.',
    ],
  },
];

const THEMES: Record<ActId, ActTheme> = {
  1: { bg: '#080b10', accent: '#c0392b', border: 'rgba(192, 57, 43, 0.28)', numeralColor: 'rgba(192, 57, 43, 0.4)' },
  2: { bg: '#050d1a', accent: '#5dade2', border: 'rgba(93, 173, 226, 0.22)', numeralColor: 'rgba(93, 173, 226, 0.35)' },
  3: { bg: '#0c0902', accent: '#c8a436', border: 'rgba(200, 164, 54, 0.28)', numeralColor: 'rgba(200, 164, 54, 0.4)' },
  4: { bg: '#071320', accent: '#2ec27e', border: 'rgba(46, 194, 126, 0.28)', numeralColor: 'rgba(46, 194, 126, 0.4)' },
};

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduced(mq.matches);
    const handler = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);
  return reduced;
}

function AnimatedParagraph({
  text,
  index,
  isReduced,
}: {
  text: string;
  index: number;
  isReduced: boolean;
}) {
  const ref = useRef<HTMLParagraphElement>(null);
  const [visible, setVisible] = useState(isReduced);

  useEffect(() => {
    if (isReduced) {
      setVisible(true);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [isReduced]);

  return (
    <p
      ref={ref}
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(18px)',
        transition: isReduced
          ? 'none'
          : `opacity 0.65s ease ${index * 110}ms, transform 0.65s ease ${index * 110}ms`,
        color: 'rgba(255,255,255,0.72)',
        fontSize: 'clamp(0.97rem, 2vw, 1.1rem)',
        lineHeight: '1.8',
        margin: '0 0 1.3rem 0',
      }}
    >
      {text}
    </p>
  );
}

function ActSection({ act, isReduced }: { act: ActData; isReduced: boolean }) {
  const theme = THEMES[act.id];
  const titleRef = useRef<HTMLHeadingElement>(null);
  const [titleVisible, setTitleVisible] = useState(isReduced);

  useEffect(() => {
    if (isReduced) {
      setTitleVisible(true);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setTitleVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.2 }
    );
    if (titleRef.current) observer.observe(titleRef.current);
    return () => observer.disconnect();
  }, [isReduced]);

  return (
    <div
      className="vision-act"
      data-act-id={act.id}
      style={{ borderTop: `1px solid ${theme.border}` }}
    >
      <div className="vision-act-image-col">
        <div className="vision-act-image-sticky">
          <img
            src={act.image}
            alt={act.imageAlt}
            loading="lazy"
            decoding="async"
            style={{
              mixBlendMode: act.blendMode || 'normal',
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              objectPosition: 'center',
              display: 'block',
            }}
          />
          {act.blendMode === 'screen' && (
            <div
              style={{
                position: 'absolute',
                inset: 0,
                background: `radial-gradient(ellipse at center, ${theme.bg}00 40%, ${theme.bg}cc 100%)`,
                pointerEvents: 'none',
              }}
            />
          )}
        </div>
      </div>

      <div className="vision-act-text-col">
        <div className="vision-act-text-inner">
          <span
            className="vision-numeral"
            style={{ color: theme.numeralColor }}
          >
            {act.numeral}
          </span>

          <h2
            ref={titleRef}
            className="vision-act-title"
            style={{
              color: '#fff',
              opacity: titleVisible ? 1 : 0,
              transform: titleVisible ? 'translateY(0)' : 'translateY(16px)',
              transition: isReduced ? 'none' : 'opacity 0.6s ease, transform 0.6s ease',
            }}
          >
            {act.title}
          </h2>

          <p
            className="vision-act-subtitle"
            style={{
              color: theme.accent,
              opacity: titleVisible ? 1 : 0,
              transition: isReduced ? 'none' : 'opacity 0.6s ease 0.1s',
            }}
          >
            {act.subtitle}
          </p>

          <div className="vision-act-body">
            {act.paragraphs.map((p, i) => (
              <AnimatedParagraph key={i} text={p} index={i} isReduced={isReduced} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export function VisionScroll() {
  const [activeAct, setActiveAct] = useState<ActId>(1);
  const isReduced = usePrefersReducedMotion();
  const theme = THEMES[activeAct];
  const ctaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const sentinels = document.querySelectorAll<HTMLElement>('[data-act-id]');
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const id = parseInt(entry.target.getAttribute('data-act-id') || '1', 10) as ActId;
            setActiveAct(id);
          }
        });
      },
      { rootMargin: '-25% 0px -55% 0px' }
    );
    sentinels.forEach((s) => observer.observe(s));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const el = ctaRef.current;
    if (!el) return;
    const mm = gsap.matchMedia();
    mm.add('(prefers-reduced-motion: no-preference)', () => {
      const children = el.querySelectorAll('.vision-quote, .vision-quote-cite, .vision-cta-heading, .vision-cta-btn');
      gsap.from(children, {
        y: 28, opacity: 0, stagger: 0.14, duration: 0.75, ease: 'power2.out',
        scrollTrigger: { trigger: el, start: 'top 78%', once: true },
      });
    });
    return () => mm.revert();
  }, []);

  return (
    <div
      className="vision-scroll-root"
      style={{
        backgroundColor: theme.bg,
        transition: isReduced ? 'none' : 'background-color 0.9s ease',
        width: '100%',
      }}
    >
      {ACTS.map((act) => (
        <ActSection key={act.id} act={act} isReduced={isReduced} />
      ))}

      <div
        ref={ctaRef}
        className="vision-cta-section"
        style={{
          borderTop: `1px solid ${THEMES[4].border}`,
          background: `linear-gradient(to bottom, ${THEMES[4].bg}, #0b1e3a 60%)`,
        }}
      >
        <div className="vision-cta-inner">
          <blockquote className="vision-quote">
            "Place me on Sunium's marbled steep,
            <br />
            Where nothing, save the waves and I,
            <br />
            May hear our mutual murmurs sweep."
          </blockquote>
          <cite className="vision-quote-cite">— Lord Byron, Don Juan (1819)</cite>
          <h2 className="vision-cta-heading">Compute at the horizon of certainty.</h2>
          <a href="/language" className="hero-cta primary vision-cta-btn">
            Explore the Language →
          </a>
        </div>
      </div>
    </div>
  );
}
