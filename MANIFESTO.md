# The Sounio Manifesto

## Epistemic Computing: A New Paradigm

> *"The measure of intelligence is the ability to change."*  
> — Albert Einstein

For decades, programming languages have treated numbers as perfect: `3.14159` is exactly that, no more, no less. But science doesn't work this way. Every measurement has error. Every model has uncertainty. Every prediction has confidence bounds.

**Sounio** is built on a radical premise: **uncertainty is not a bug—it's a feature**.

---

## The Five Principles of Epistemic Computing

### 1. All Knowledge is Uncertain

In the physical world, there is no such thing as a perfect measurement. The Heisenberg uncertainty principle is not a limitation of our instruments—it's a fundamental property of reality. Even macroscopic measurements carry noise, calibration error, and finite precision.

```sio
// Wrong: pretending we know exactly
let concentration = 5.23  // mg/L... but really?

// Right: acknowledging uncertainty
let concentration = Knowledge::new(5.23 mg/L, uncertainty: 0.15 mg/L)
```

Sounio makes this explicit. When you declare a value, you must consider: *how well do I actually know this?*

### 2. Provenance is Non-Negotiable

Data without origin is data without trust. When a regulatory agency asks "where did this number come from?", you should have an answer that traces back to primary sources.

```sio
let clearance = Knowledge::new(
    value: 10.5 L/h,
    uncertainty: 1.2 L/h,
    source: Source {
        origin: "Phase III Trial NCT04123456",
        timestamp: 2024-03-15,
        method: "Population PK analysis",
        confidence: 0.95
    }
)
```

Every `Knowledge<T>` carries its provenance. The lineage of your data is as important as the data itself.

### 3. Uncertainty Propagates Automatically

Manual uncertainty propagation is tedious and error-prone. The GUM (Guide to the Expression of Uncertainty in Measurement) defines how uncertainties combine through mathematical operations. Sounio implements this automatically.

```sio
let mass = Knowledge::new(100.0 g, uncertainty: 0.5 g)
let volume = Knowledge::new(50.0 mL, uncertainty: 0.2 mL)

// Density calculation with automatic propagation
let density = mass / volume
// density.uncertainty is computed via GUM: 
// δρ/ρ = sqrt((δm/m)² + (δV/V)²)
```

You write the physics. The compiler handles the statistics.

### 4. Confidence Gates Execution

Not all computations should proceed blindly. When confidence drops below a threshold, execution should pause, warn, or take alternative paths.

```sio
fn critical_decision(data: Knowledge<f64>) -> Action {
    if data.confidence < 0.90 {
        return Action::RequestMoreData
    }
    
    if data.confidence < 0.95 {
        return Action::ProceedWithCaution(data)
    }
    
    Action::Proceed(data)
}
```

This is not defensive programming—it's *epistemic programming*. The system knows what it doesn't know.

### 5. Standards Compliance by Design

Science has standards for a reason. Sounio is built to comply with:

- **GUM** — ISO Guide to the Expression of Uncertainty in Measurement
- **ISO 17025** — Competence of testing and calibration laboratories
- **21 CFR Part 11** — Electronic records and signatures (FDA)
- **FAIR Principles** — Findable, Accessible, Interoperable, Reusable data

These aren't afterthoughts—they're architectural foundations.

---

## The Problem We're Solving

### The Reproducibility Crisis

Between 2011 and 2021, an estimated $28 billion was wasted on irreproducible preclinical research in the United States alone. The causes are many, but one stands out: **loss of uncertainty information**.

When a measurement of `5.23 mg/L` is passed between systems, stored in databases, and used in calculations—the `±0.15` often disappears. Downstream analyses treat it as exact. Conclusions are drawn that the original uncertainty would have precluded.

### The Solution

Sounio makes uncertainty *infectious*. You cannot accidentally drop it. The type system won't let you convert `Knowledge<T>` to bare `T` without explicit acknowledgment.

```sio
let safe_value = measurement.value  // Compiler error!

let safe_value = measurement.unwrap_certain()  // Requires confidence > 0.99

let safe_value = measurement.acknowledge_uncertainty()  // Explicit opt-out, logged
```

---

## Why "Sounio"?

Cape Sounion, at the tip of Attica, is where ancient Greek sailors watched the horizon. The Temple of Poseidon there was both a landmark and a prayer—a fixed point from which to navigate the uncertain sea.

Sounio the language serves the same purpose: a stable foundation for navigating uncertain data. The columns are your type system. The sea is your scientific domain. The horizon is where certainty ends and exploration begins.

Lord Byron visited in 1810 and carved his name in the marble (please don't do this). He wrote:

> *"Place me on Sunium's marbled steep,*  
> *Where nothing, save the waves and I,*  
> *May hear our mutual murmurs sweep;*  
> *There, swan-like, let me sing and die."*

We're not quite that dramatic. But we are building something that, like those columns, might last.

---

## The Road Ahead

Sounio is not finished. It may never be. But the principles are set:

1. **Uncertainty is first-class** — Not a library, not an annotation, but a fundamental type.

2. **Propagation is correct** — GUM-compliant, tested, verified.

3. **Provenance is preserved** — From source to result, the chain is unbroken.

4. **Confidence is actionable** — The system responds to what it knows and doesn't know.

5. **Standards are built-in** — Compliance is not optional.

If you believe that science deserves better tools—that uncertainty should be computed, not ignored—that reproducibility is a feature, not an accident—then Sounio is for you.

---

*Join us at the horizon.*

**🏛️ SOUNIO 🌊**

---

*Last updated: December 2024*
