---
title: "Epistemic Computing Manifesto"
description: "Computing at the boundary between what we know and what we don't"
layout: "manifesto"
subtitle: "Five Principles for Scientific Computing"
---

> *"La medida de la inteligencia es la capacidad para cambiar."*
> — Albert Einstein

Durante décadas, los lenguajes de programación han tratado los números como perfectos: `3.14159` es exactamente eso, ni más ni menos. Pero la ciencia no funciona de esta manera. Toda medición tiene error. Todo modelo tiene incertidumbre. Toda predicción tiene límites de confianza.

**Sounio** se basa en una premisa radical: **la incertidumbre no es un error—es una característica**.

---

## Los Cinco Principios de Epistemic Computing

### 1. Todo Conocimiento es Incierto

En el mundo físico, no existe tal cosa como una medición perfecta. El principio de incertidumbre de Heisenberg no es una limitación de nuestros instrumentos—es una propiedad fundamental de la realidad. Incluso las mediciones macroscópicas llevan ruido, error de calibración y precisión finita.

```sio
// Wrong: pretending we know exactly
let concentration = 5.23  // mg/L... but really?

// Right: acknowledging uncertainty
let concentration = Knowledge::new(5.23 mg/L, uncertainty: 0.15 mg/L)
```

Sounio hace esto explícito. Cuando declaras un valor, debes considerar: *¿qué tan bien conozco realmente esto?*

### 2. Provenance es No Negociable

Datos sin origen son datos sin confianza. Cuando una agencia reguladora pregunta "¿de dónde viene este número?", deberías tener una respuesta que se remonta a las fuentes primarias.

```sio
let clearance = Knowledge::new(
    value: 10.5 L/h,
    uncertainty: 1.2 L/h,
    source: Source {
        origin: "Phase III Trial NCT04123456",
        timestamp: 2025-03-15,
        method: "Population PK analysis",
        confidence: 0.95
    }
)
```

Cada `Knowledge<T>` lleva su provenance. El linaje de tus datos es tan importante como los datos mismos.

### 3. La Incertidumbre se Propaga Automáticamente

La propagación manual de incertidumbre es tediosa y propensa a errores. El GUM (Guide to the Expression of Uncertainty in Measurement) define cómo se combinan las incertidumbres a través de operaciones matemáticas. Sounio implementa esto automáticamente.

```sio
let mass = Knowledge::new(100.0 g, uncertainty: 0.5 g)
let volume = Knowledge::new(50.0 mL, uncertainty: 0.2 mL)

// Density calculation with automatic propagation
let density = mass / volume
// density.uncertainty is computed via GUM:
// δρ/ρ = sqrt((δm/m)² + (δV/V)²)
```

Tú escribes la física. El compilador maneja la estadística.

### 4. La Confianza Controla la Ejecución

No todas las computaciones deberían proceder a ciegas. Cuando la confianza cae por debajo de un umbral, la ejecución debería pausarse, advertir o tomar caminos alternativos.

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

Esto no es programación defensiva—es *epistemic programming*. El sistema sabe lo que no sabe.

### 5. Cumplimiento de Estándares por Diseño

La ciencia tiene estándares por una razón. Sounio está construido para cumplir con:

- **GUM** — ISO Guide to the Expression of Uncertainty in Measurement
- **ISO 17025** — Competence of testing and calibration laboratories
- **21 CFR Part 11** — Electronic records and signatures (FDA)
- **FAIR Principles** — Findable, Accessible, Interoperable, Reusable data

Estos no son pensamientos posteriores—son fundamentos arquitectónicos.

---

## El Problema que Estamos Resolviendo

### La Crisis de Reproducibilidad

Entre 2011 y 2021, se estima que $28 mil millones fueron desperdiciados en investigación preclínica irreproducible en Estados Unidos solamente. Las causas son muchas, pero una destaca: **pérdida de información de incertidumbre**.

Cuando una medición de `5.23 mg/L` se pasa entre sistemas, se almacena en bases de datos y se usa en cálculos—el `±0.15` a menudo desaparece. Los análisis downstream la tratan como exacta. Se sacan conclusiones que la incertidumbre original habría precluido.

### La Solución

Sounio hace que la incertidumbre sea *infectiosa*. No puedes dejarla caer accidentalmente. El sistema de tipos no te permite convertir `Knowledge<T>` a un `T` desnudo sin un reconocimiento explícito.

```sio
let safe_value = measurement.value  // Compiler error!

let safe_value = measurement.unwrap_certain()  // Requires confidence > 0.99

let safe_value = measurement.acknowledge_uncertainty()  // Explicit opt-out, logged
```

---

## ¿Por Qué "Sounio"?

El Cabo Sounion, en la punta de Ática, es donde los antiguos marineros griegos observaban el horizonte. El Templo de Poseidón allí era tanto un hito como una oración—un punto fijo desde el cual navegar el mar incierto.

Sounio el lenguaje sirve el mismo propósito: una base estable para navegar datos inciertos. Las columnas son tu sistema de tipos. El mar es tu dominio científico. El horizonte es donde termina la certeza y comienza la exploración.

Lord Byron visitó en 1810 y talló su nombre en el mármol (por favor, no hagas esto). Escribió:

> *"Place me on Sunium's marbled steep,*
> *Where nothing, save the waves and I,*
> *May hear our mutual murmurs sweep;*
> *There, swan-like, let me sing and die."*

(Colóquenme en el escarpado mármol de Sunión,
donde nada, salvo las olas y yo,
pueda oír el barrido de nuestros murmullos mutuos;
allí, como un cisne, déjenme cantar y morir.)

No somos tan dramáticos. Pero estamos construyendo algo que, como esas columnas, podría durar.

---

## El Camino por Delante

Sounio no está terminado. Puede que nunca lo esté. Pero los principios están establecidos:

1. **La incertidumbre es de primera clase** — No una biblioteca, no una anotación, sino un tipo fundamental.

2. **La propagación es correcta** — Cumplimiento con GUM, probado, verificado.

3. **Provenance se preserva** — Desde la fuente hasta el resultado, la cadena es inquebrantable.

4. **La confianza es accionable** — El sistema responde a lo que sabe y no sabe.

5. **Los estándares están integrados** — El cumplimiento no es opcional.

Si crees que la ciencia merece mejores herramientas—que la incertidumbre debería computarse, no ignorarse—que la reproducibilidad es una característica, no un accidente—entonces Sounio es para ti.

---

*Únete a nosotros en el horizonte.*

**🏛️ SOUNIO 🌊**
