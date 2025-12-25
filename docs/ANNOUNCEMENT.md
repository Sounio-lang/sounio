# Sounio Launch Announcement

Social media posts for the Sounio language launch.

---

## Twitter/X Thread

### Tweet 1 (Main)
```
Introducing Sounio — a systems programming language for epistemic computing.

Every measurement has uncertainty. Every model has confidence bounds. Every prediction has error bars.

Why do our programming languages pretend otherwise?

🏛️ sounio-lang.org

🧵 1/6
```

### Tweet 2
```
The problem: $28 billion wasted on irreproducible research in the US alone (2011-2021).

One major cause? Loss of uncertainty information as data flows through systems.

Sounio makes uncertainty *infectious*. You can't accidentally drop it.

2/6
```

### Tweet 3
```
Meet Knowledge<T> — values that know their own uncertainty:

let dose = Knowledge::new(
    value: 500.0 mg,
    uncertainty: 2.5 mg,
    confidence: 0.95,
    source: "clinical_trial"
)

Uncertainty propagates automatically through all operations (GUM-compliant).

3/6
```

### Tweet 4
```
Confidence gates execution:

if measurement.confidence > 0.95 {
    proceed(measurement)
} else {
    require_confirmation()
}

The system knows what it doesn't know.

4/6
```

### Tweet 5
```
151,000+ lines of stdlib:

• Epistemic types with provenance
• MedLang DSL for PK/PD modeling
• fMRI preprocessing pipeline
• Causal inference
• GPU-accelerated computing
• Network analysis with uncertainty

5/6
```

### Tweet 6
```
Named after Cape Sounion 🇬🇷 — where the Temple of Poseidon has watched over the Aegean for 2,500 years.

Computing at the horizon of certainty.

GitHub: github.com/sounio-lang/sounio
Manifesto: sounio-lang.org/MANIFESTO

MIT License. Contributions welcome.

🏛️🌊 6/6
```

---

## LinkedIn Post

```
I'm excited to announce the release of Sounio — a new systems programming language designed for epistemic computing.

THE PROBLEM

Between 2011 and 2021, an estimated $28 billion was wasted on irreproducible preclinical research in the United States alone. One major cause: the silent loss of uncertainty information as data flows between systems.

When a measurement of "5.23 mg/L" moves through databases and calculations, the "± 0.15" often disappears. Downstream analyses treat approximate values as exact. Conclusions are drawn that the original uncertainty would have precluded.

THE SOLUTION

Sounio treats uncertainty as a first-class citizen. The Knowledge<T> type wraps values with:

• Uncertainty (following GUM standards)
• Confidence levels
• Provenance tracking
• Source metadata

Uncertainty propagates automatically through mathematical operations. You write the science; the compiler handles the statistics.

Confidence gates can halt execution when certainty drops below thresholds. The system knows what it doesn't know.

WHAT'S INCLUDED

The standard library (151,000+ lines) includes:

• Epistemic type system with automatic propagation
• MedLang DSL for PK/PD and PBPK modeling
• fMRI preprocessing pipeline with atlas support
• Causal inference and discovery algorithms
• GPU-accelerated computing kernels
• Network analysis with uncertainty bounds

WHY "SOUNIO"?

Cape Sounion (Σούνιο) stands at the southernmost tip of Attica, Greece, where the Temple of Poseidon has watched over the Aegean for 2,500 years. At sunset, its Doric columns catch the last light—the horizon where certainty meets the unknown sea.

Sounio the language embodies this metaphor: computing at the boundary between what we know and what we don't.

GET INVOLVED

🔗 GitHub: github.com/sounio-lang/sounio
🔗 Website: sounio-lang.org
📄 License: MIT

I'd love to hear from researchers, data scientists, and scientific computing professionals. If reproducibility matters to you, if uncertainty should be computed rather than ignored, Sounio might be worth a look.

#Programming #ScientificComputing #DataScience #Research #OpenSource #Reproducibility
```

---

## Reddit Post (r/ProgrammingLanguages)

**Title:** Sounio: A systems language where uncertainty is first-class

```
I've been working on a programming language that treats uncertainty as a fundamental type, not an afterthought.

**The core idea:**

Most languages treat `5.23` as exactly 5.23. But in science, that number might really be "5.23 ± 0.15 with 95% confidence, measured by Lab X on 2024-03-15."

Sounio's `Knowledge<T>` type carries all this metadata:

    let concentration = Knowledge::new(
        value: 5.23 mg/L,
        uncertainty: 0.15 mg/L,
        confidence: 0.95,
        source: Source {
            origin: "Phase III Trial NCT04123456",
            method: "Population PK analysis"
        }
    )

**Automatic propagation:**

When you do `mass / volume`, the uncertainty propagates automatically following GUM (Guide to the Expression of Uncertainty in Measurement). No manual error propagation.

**Confidence gates:**

    if result.confidence < 0.90 {
        return Action::RequestMoreData
    }

The type system prevents you from accidentally treating uncertain values as certain.

**Stdlib:**

151K+ lines including:
- PK/PD modeling DSL (pharmacokinetics)
- fMRI preprocessing pipeline
- Causal inference
- GPU kernels
- Network analysis

**Links:**
- GitHub: https://github.com/sounio-lang/sounio
- Manifesto: (read this for the philosophy)

Would love feedback, especially from anyone working on scientific computing or dealing with measurement uncertainty.
```

---

## Hacker News Post

**Title:** Sounio – A programming language for epistemic computing

```
Sounio is a systems programming language built around the idea that uncertainty should be computed, not ignored.

Every value can carry its uncertainty, confidence level, and provenance. When you perform calculations, uncertainty propagates automatically following the GUM (Guide to Uncertainty in Measurement).

Example:

    let mass = Knowledge::new(100.0 g, uncertainty: 0.5 g)
    let volume = Knowledge::new(50.0 mL, uncertainty: 0.2 mL)
    
    let density = mass / volume
    // density.uncertainty is computed via:
    // δρ/ρ = sqrt((δm/m)² + (δV/V)²)

The type system prevents accidentally dropping uncertainty. You can't convert Knowledge<T> to T without explicit acknowledgment.

Named after Cape Sounion in Greece, where the Temple of Poseidon marks the horizon between land and sea—certainty and uncertainty.

MIT licensed, 151K lines of stdlib for scientific computing.

https://github.com/sounio-lang/sounio
```

---

*Last updated: December 2024*
