/**
 * motion.ts — single motion authority for the site.
 *
 * One GSAP + ScrollTrigger + Lenis singleton, mounted from BaseLayout.
 * Components register reveals via data attributes or the helpers below;
 * everything respects prefers-reduced-motion and Astro view transitions.
 */
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import Lenis from 'lenis';

gsap.registerPlugin(ScrollTrigger);

export const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

let lenis: Lenis | null = null;

export function getLenis() {
  return lenis;
}

/** Mount once from BaseLayout. Safe to call on every astro:page-load. */
export function initMotion() {
  if (typeof window === 'undefined') return;
  if ((window as any).__sounioMotion) return;
  (window as any).__sounioMotion = true;

  if (!prefersReducedMotion()) {
    lenis = new Lenis({
      duration: 1.1,
      easing: (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
    });
    lenis.on('scroll', ScrollTrigger.update);
    gsap.ticker.add((time) => lenis?.raf(time * 1000));
    gsap.ticker.lagSmoothing(0);

    // Lenis + Astro view transitions: stop smoothing mid-swap to avoid jumps.
    document.addEventListener('astro:before-swap', () => lenis?.stop());
    document.addEventListener('astro:page-load', () => {
      lenis?.start();
      lenis?.scrollTo(window.scrollY, { immediate: true });
      ScrollTrigger.refresh();
    });
  }
}

/** Kill all ScrollTriggers created for the current page (called before swap). */
export function cleanupPageTriggers(scope?: Element | Document) {
  ScrollTrigger.getAll().forEach((st) => {
    const trigger = st.trigger;
    if (!scope || !trigger || (scope as Document).contains?.(trigger)) st.kill();
  });
}

/**
 * Standard editorial reveal: fade + rise with stagger.
 * Usage in an Astro component script: `revealOnScroll('.my-section [data-reveal]')`.
 */
export function revealOnScroll(targets: gsap.TweenTarget, opts: gsap.TweenVars = {}) {
  if (prefersReducedMotion()) return;
  return gsap.from(targets, {
    opacity: 0,
    y: 24,
    duration: 0.7,
    ease: 'power2.out',
    stagger: 0.08,
    clearProps: 'opacity,transform',
    scrollTrigger: { trigger: targets as gsap.DOMTarget, start: 'top 85%' },
    ...opts,
  });
}

/** Count-up for instrument numerals. `end` parsed from el text or passed. */
export function countUp(el: HTMLElement, end: number, opts: { duration?: number; suffix?: string } = {}) {
  if (prefersReducedMotion()) {
    el.textContent = `${end}${opts.suffix ?? ''}`;
    return;
  }
  const state = { v: 0 };
  return gsap.to(state, {
    v: end,
    duration: opts.duration ?? 1.6,
    ease: 'power2.out',
    scrollTrigger: { trigger: el, start: 'top 88%' },
    onUpdate: () => {
      el.textContent = `${Math.round(state.v)}${opts.suffix ?? ''}`;
    },
  });
}

export { gsap, ScrollTrigger };
