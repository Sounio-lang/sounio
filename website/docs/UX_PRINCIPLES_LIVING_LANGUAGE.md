# Sounio Living Language UX Principles

This document is the working design contract for the Sounio website.

Sounio is not a static product brochure. It is a living language and scientific
system that can change by tens of thousands of lines per day. The website must
turn that velocity into orientation, evidence into trust, and interaction into
competence.

The design stance is inspired by Positive Technology and Transformative
Experience Design: the interface should improve the visitor's ability to
understand, act, and make meaning. Beauty is welcome when it clarifies. Motion is
welcome when it reveals structure. Evidence is mandatory when a claim matters.

## Core Thesis

Sounio UX is transformation without disorientation.

Every important page should change the user's model of the world while leaving
them more oriented than when they arrived.

## Four Pillars

### 1. Orientation

The interface reduces vertigo in the face of speed.

- Show what is current, what is experimental, and what is proven.
- Make the next useful action obvious without flattening the project's depth.
- Prefer progressive disclosure over dense walls of claims.
- Give each page a clear role in the user's journey.
- Treat multilingual navigation as a first-class orientation system.

### 2. Evidence

The interface makes truth inspectable.

- Pair major claims with status, provenance, and a path to evidence.
- Distinguish demos, gates, audits, artifacts, and aspirational roadmap items.
- Avoid implying semantic maturity from visual polish alone.
- Keep uncertainty visible without making the experience feel broken.
- When a feature is moving fast, show its freshness and acceptance level.

### 3. Transformation

The interface should not merely inform; it should produce insight.

- Each page should have a cognitive arc: problem, tension, demonstration,
  insight, action.
- Let visitors feel why Sounio needs to exist, not just learn that it exists.
- Use interactive examples to shift understanding, not to decorate.
- Design moments where epistemic types, uncertainty, compiler evidence, or
  clinical reasoning become legible.
- Preserve seriousness while allowing wonder.

### 4. Agency

The user should leave more capable.

- Build paths for different intents: understand, try, verify, contribute,
  evaluate scientifically.
- Translate complexity into navigable choices.
- Let technical users inspect real artifacts quickly.
- Let non-specialist scientific users understand the stakes without needing to
  read compiler internals first.
- Treat the site as an instrument panel, not a billboard.

## Living Language Patterns

### Daily Pulse

Surface recent movement as a curated signal, not a raw git dump.

Good pulse items:

- compiler gate changed
- stdlib coverage changed
- docs/i18n sync changed
- important example promoted
- known blocker resolved or reclassified

Bad pulse items:

- unaudited churn
- raw line counts without meaning
- vague "rapid progress" claims
- status badges not backed by scripts or artifacts

### Maturity Badges

Use consistent states across pages:

- `Exploratory`: idea, prototype, or research lane.
- `Implemented`: code exists, but acceptance scope is narrow.
- `Checked`: local compiler or website checks passed.
- `Gate-Passed`: named gate passed with reproducible evidence.
- `Claim-Ready`: the wording is backed enough for public-facing claims.
- `Blocked`: next action is known, but evidence is incomplete.

### Evidence Cards

Evidence cards should answer:

- What is the claim?
- What proves it?
- When was it checked?
- What path, command, artifact, or gate backs it?
- What remains uncertain?

### Transformative Interactions

Interactions should change comprehension.

Strong examples:

- uncertainty propagation that visibly changes a clinical decision band
- compiler diagnostics that teach the effect or epistemic rule
- a proof/status timeline that shows maturity increasing through gates
- multilingual content freshness that shows sync state by locale

Weak examples:

- decorative counters with no provenance
- animation that delays reading
- simulation-like visuals with no model boundary
- "AI-looking" visuals that obscure evidence

## Motion Rules

- Motion must reveal structure, causality, progression, or state.
- Scroll reveals should support sequence, not hide ordinary content.
- Avoid motion that competes with code, data, or evidence.
- When motion and evidence legibility conflict, evidence legibility wins.
- Respect reduced-motion preferences.
- Do not use visual spectacle to compensate for missing proof.

## Multilingual Rules

Translation is cultural presence, not string mirroring.

- Each locale must preserve technical accuracy and human tone.
- Locale freshness should be visible when content changes quickly.
- Fallbacks must be honest and non-jarring.
- Avoid half-localized journeys where navigation is translated but core evidence
  reverts silently.
- Prefer local clarity over literal phrasing when the meaning is preserved.

## Page-Level Heuristics

### Homepage

The homepage should answer:

- What is Sounio?
- Why does it matter?
- What is real today?
- How fast is it evolving?
- How can I enter safely?

The first viewport should communicate living-language momentum without creating
the feeling of chaos.

### Docs

Docs should create competence.

- Start with the smallest useful mental model.
- Keep status and language maturity visible.
- Link concepts to examples and compiler behavior.
- Avoid burying effect, uncertainty, and provenance semantics as advanced
  curiosities.

### Showcases

Showcases should create belief through inspectable demonstrations.

- Pair each showcase with its evidence level.
- Explain what is modeled and what is not.
- Make claims proportional to gates.
- Prefer fewer, stronger demonstrations over broad decorative galleries.

### Status And Releases

Status pages are the trust layer.

- They should be scannable under uncertainty.
- They should distinguish release state, gate state, and research state.
- They should make blockers feel actionable, not embarrassing.

## Review Checklist

Before merging meaningful UX/UI work, ask:

- Does this page orient the visitor faster than before?
- Does it make at least one important truth more inspectable?
- Does it increase user agency?
- Does every major claim have an evidence path or honest status boundary?
- Does motion reveal something structural?
- Does the page still work when translated?
- Does the experience reduce confusion caused by project velocity?
- Would a skeptical scientist know where to click next?
- Would a curious newcomer feel invited rather than overwhelmed?

## Anti-Patterns

- Brochure voice for a living research system.
- Vague "revolutionary language" claims without evidence.
- One-note visual spectacle detached from compiler or scientific truth.
- Locale coverage that hides stale or fallback content.
- Changelog dumps presented as user orientation.
- Animated sections that look mature but carry unverified claims.
- Treating uncertainty as a defect instead of a navigable state.

## North Star

A visitor should leave thinking:

"This is moving fast, but it is not drifting. I can see the evidence, choose a
path, and understand why this language needs to exist."
