# Dosing Protocols in Sounio

Sounio's MedLang provides comprehensive support for specifying dosing regimens, from simple single doses to complex adaptive protocols. The `medlang::dose` module offers type-safe, unit-checked dosing specification.

## Dosing Regimen Specification

### Basic Dose Types

```sio
use medlang::dose::*

// IV bolus
let iv_dose = Dose::iv_bolus(
    amt: 500.0 mg,
    time: 0.0 h
)

// IV infusion
let infusion = Dose::iv_infusion(
    amt: 1000.0 mg,
    duration: 1.0 h,
    time: 0.0 h
)

// Oral dose
let oral_dose = Dose::oral(
    amt: 500.0 mg,
    time: 0.0 h
)

// Subcutaneous
let sc_dose = Dose::subcutaneous(
    amt: 40.0 mg,
    time: 0.0 h
)

// Intramuscular
let im_dose = Dose::intramuscular(
    amt: 100.0 mg,
    time: 0.0 h
)
```

### Routes of Administration

```sio
enum Route {
    IV,             // Intravenous (bolus or infusion)
    Oral,           // Per os
    SC,             // Subcutaneous
    IM,             // Intramuscular
    Topical,        // Dermal
    Inhalation,     // Pulmonary
    Intranasal,     // Nasal
    Transdermal,    // Patches
}
```

## Single Dose

### IV Bolus

```sio
use medlang::*

/// Single IV bolus administration
fn single_iv_bolus(dose: f64@mg, time: f64@h) -> DosingEvent {
    return DosingEvent {
        amt: dose,
        route: Route::IV,
        time: time,
        rate: 0.0 mg/h,      // Instantaneous
        duration: 0.0 h,
        bioavailability: 1.0,
        lag_time: 0.0 h
    }
}

// Usage
let dose_event = single_iv_bolus(500.0 mg, 0.0 h)
```

### IV Infusion

```sio
/// IV infusion at constant rate
fn iv_infusion(total_dose: f64@mg, duration: f64@h, start_time: f64@h) -> DosingEvent {
    let rate = total_dose / duration
    return DosingEvent {
        amt: total_dose,
        route: Route::IV,
        time: start_time,
        rate: rate,
        duration: duration,
        bioavailability: 1.0,
        lag_time: 0.0 h
    }
}

// 500mg over 30 minutes
let infusion_event = iv_infusion(500.0 mg, 0.5 h, 0.0 h)
```

### Oral with Bioavailability

```sio
/// Oral dose with F and lag time
fn oral_dose(dose: f64@mg, time: f64@h, f: f64, lag: f64@h) -> DosingEvent {
    return DosingEvent {
        amt: dose,
        route: Route::Oral,
        time: time,
        rate: 0.0 mg/h,
        duration: 0.0 h,
        bioavailability: f,
        lag_time: lag
    }
}

// 500mg oral with F=0.8 and 15-min lag
let oral_event = oral_dose(500.0 mg, 0.0 h, 0.8, 0.25 h)
```

## Multiple Dose Regimens

### Standard Frequencies

```sio
enum DosingFrequency {
    Once,           // Single dose
    BID,            // Twice daily (every 12h)
    TID,            // Three times daily (every 8h)
    QID,            // Four times daily (every 6h)
    QD,             // Once daily (every 24h)
    QOD,            // Every other day (every 48h)
    Weekly,         // Once weekly (every 168h)
    Every21Days,    // Q3W (every 504h)
    Q4W,            // Every 4 weeks
    Monthly,        // Once monthly (~720h)
}

/// Convert frequency to interval
fn frequency_to_interval(freq: DosingFrequency) -> f64@h {
    match freq {
        DosingFrequency::Once => 0.0 h,
        DosingFrequency::BID => 12.0 h,
        DosingFrequency::TID => 8.0 h,
        DosingFrequency::QID => 6.0 h,
        DosingFrequency::QD => 24.0 h,
        DosingFrequency::QOD => 48.0 h,
        DosingFrequency::Weekly => 168.0 h,
        DosingFrequency::Every21Days => 504.0 h,
        DosingFrequency::Q4W => 672.0 h,
        DosingFrequency::Monthly => 720.0 h,
    }
}
```

### Creating Multiple Dose Regimens

```sio
use medlang::dose::*

/// Multiple dosing regimen
struct DosingRegimen {
    doses: Vec<DosingEvent>,
    route: Route,
    frequency: DosingFrequency,
    n_doses: i32,
    total_duration: f64@h,
}

/// Create BID regimen
fn create_bid_regimen(
    dose_amt: f64@mg,
    n_days: i32,
    route: Route,
    bioavailability: f64
) -> DosingRegimen {
    var doses: Vec<DosingEvent> = vec![]
    let interval = 12.0 h

    for day in 0..n_days {
        for dose_num in 0..2 {
            let time = (day as f64) * 24.0 h + (dose_num as f64) * 12.0 h
            doses.push(DosingEvent {
                amt: dose_amt,
                route: route,
                time: time,
                rate: 0.0 mg/h,
                duration: 0.0 h,
                bioavailability: bioavailability,
                lag_time: 0.0 h
            })
        }
    }

    return DosingRegimen {
        doses: doses,
        route: route,
        frequency: DosingFrequency::BID,
        n_doses: n_days * 2,
        total_duration: (n_days as f64) * 24.0 h
    }
}
```

### QD Oral Regimen

```sio
/// Once-daily oral regimen
fn create_qd_oral(
    dose: f64@mg,
    n_days: i32,
    f_oral: f64
) -> DosingRegimen {
    var doses: Vec<DosingEvent> = vec![]

    for day in 0..n_days {
        let time = (day as f64) * 24.0 h
        doses.push(DosingEvent {
            amt: dose,
            route: Route::Oral,
            time: time,
            rate: 0.0 mg/h,
            duration: 0.0 h,
            bioavailability: f_oral,
            lag_time: 0.25 h  // 15-min lag
        })
    }

    return DosingRegimen {
        doses: doses,
        route: Route::Oral,
        frequency: DosingFrequency::QD,
        n_doses: n_days,
        total_duration: (n_days as f64) * 24.0 h
    }
}
```

### Custom Intervals

```sio
/// Regimen with custom dosing times
fn create_custom_regimen(
    dose: f64@mg,
    times: &Vec<f64@h>,
    route: Route,
    bioavailability: f64
) -> DosingRegimen {
    var doses: Vec<DosingEvent> = vec![]

    for time in times {
        doses.push(DosingEvent {
            amt: dose,
            route: route,
            time: time,
            rate: 0.0 mg/h,
            duration: 0.0 h,
            bioavailability: bioavailability,
            lag_time: 0.0 h
        })
    }

    let n = times.len() as i32
    let duration = times[n - 1]

    return DosingRegimen {
        doses: doses,
        route: route,
        frequency: DosingFrequency::Once,  // Custom
        n_doses: n,
        total_duration: duration
    }
}

// Example: TID schedule at 8am, 2pm, 8pm
let times = vec![8.0 h, 14.0 h, 20.0 h]
let regimen = create_custom_regimen(250.0 mg, &times, Route::Oral, 0.8)
```

## Loading and Maintenance Doses

### Loading Dose Strategy

```sio
/// Loading + maintenance dosing
struct LoadingMaintenance {
    loading_dose: DosingEvent,
    maintenance_regimen: DosingRegimen
}

fn create_loading_maintenance(
    loading_amt: f64@mg,
    maintenance_amt: f64@mg,
    frequency: DosingFrequency,
    n_maintenance: i32,
    route: Route,
    f: f64
) -> LoadingMaintenance {
    // Loading dose at t=0
    let loading = DosingEvent {
        amt: loading_amt,
        route: route,
        time: 0.0 h,
        rate: 0.0 mg/h,
        duration: 0.0 h,
        bioavailability: f,
        lag_time: 0.0 h
    }

    // Maintenance starting at first interval
    let interval = frequency_to_interval(frequency)
    var maintenance_doses: Vec<DosingEvent> = vec![]

    for i in 1..=n_maintenance {
        maintenance_doses.push(DosingEvent {
            amt: maintenance_amt,
            route: route,
            time: (i as f64) * interval,
            rate: 0.0 mg/h,
            duration: 0.0 h,
            bioavailability: f,
            lag_time: 0.0 h
        })
    }

    return LoadingMaintenance {
        loading_dose: loading,
        maintenance_regimen: DosingRegimen {
            doses: maintenance_doses,
            route: route,
            frequency: frequency,
            n_doses: n_maintenance,
            total_duration: (n_maintenance as f64) * interval
        }
    }
}

// Example: 1000mg loading, then 500mg QD x 6 days
let lm = create_loading_maintenance(
    loading_amt: 1000.0 mg,
    maintenance_amt: 500.0 mg,
    frequency: DosingFrequency::QD,
    n_maintenance: 6,
    route: Route::Oral,
    f: 0.8
)
```

### IV Loading + Oral Maintenance

```sio
fn create_iv_load_oral_maintenance(
    iv_load: f64@mg,
    oral_maintenance: f64@mg,
    frequency: DosingFrequency,
    n_oral: i32,
    f_oral: f64
) -> Vec<DosingEvent> {
    var events: Vec<DosingEvent> = vec![]

    // IV loading (100% bioavailable)
    events.push(DosingEvent {
        amt: iv_load,
        route: Route::IV,
        time: 0.0 h,
        rate: 0.0 mg/h,
        duration: 0.0 h,
        bioavailability: 1.0,
        lag_time: 0.0 h
    })

    // Oral maintenance
    let interval = frequency_to_interval(frequency)
    for i in 1..=n_oral {
        events.push(DosingEvent {
            amt: oral_maintenance,
            route: Route::Oral,
            time: (i as f64) * interval,
            rate: 0.0 mg/h,
            duration: 0.0 h,
            bioavailability: f_oral,
            lag_time: 0.25 h
        })
    }

    return events
}
```

## Infusion Protocols

### Continuous Infusion

```sio
/// Continuous IV infusion
fn continuous_infusion(
    rate: f64@mg_per_h,
    start_time: f64@h,
    end_time: f64@h
) -> DosingEvent {
    let duration = end_time - start_time
    let total_amt = rate * duration

    return DosingEvent {
        amt: total_amt,
        route: Route::IV,
        time: start_time,
        rate: rate,
        duration: duration,
        bioavailability: 1.0,
        lag_time: 0.0 h
    }
}

// 10 mg/h for 24 hours
let infusion = continuous_infusion(10.0 mg/h, 0.0 h, 24.0 h)
```

### Intermittent Infusions

```sio
/// Repeated intermittent infusions
fn intermittent_infusions(
    dose: f64@mg,
    infusion_duration: f64@h,
    interval: f64@h,
    n_doses: i32
) -> Vec<DosingEvent> {
    var events: Vec<DosingEvent> = vec![]
    let rate = dose / infusion_duration

    for i in 0..n_doses {
        let time = (i as f64) * interval
        events.push(DosingEvent {
            amt: dose,
            route: Route::IV,
            time: time,
            rate: rate,
            duration: infusion_duration,
            bioavailability: 1.0,
            lag_time: 0.0 h
        })
    }

    return events
}

// 500mg over 1h, every 8h, for 7 days
let infusions = intermittent_infusions(500.0 mg, 1.0 h, 8.0 h, 21)
```

## Adaptive Dosing Based on Uncertainty

Sounio's epistemic features enable uncertainty-aware dosing decisions:

```sio
use epistemic::*
use medlang::*

/// Adaptive dose selection based on AUC confidence
fn adaptive_dose(
    target_auc: f64@mg_h_per_L,
    current_auc: Knowledge[f64@mg_h_per_L],
    current_dose: f64@mg,
    dose_options: &Vec<f64@mg>
) -> (f64@mg, bool) {
    // If current AUC has low confidence, recommend TDM
    if current_auc.confidence < 0.70 {
        return (current_dose, true)  // Keep dose, request TDM
    }

    // Calculate ratio
    let ratio = target_auc / current_auc.value

    // Select closest dose from options
    let adjusted = current_dose * ratio
    var best_dose = dose_options[0]
    var min_diff = abs(best_dose - adjusted)

    for dose in dose_options {
        let diff = abs(dose - adjusted)
        if diff < min_diff {
            min_diff = diff
            best_dose = dose
        }
    }

    // Check if adjustment significant
    let dose_change = abs(best_dose - current_dose) / current_dose
    let need_adjustment = dose_change > 0.2  // >20% change

    return (best_dose, false)  // New dose, no TDM needed
}
```

### Confidence-Gated Dosing

```sio
/// Dose only if prediction confidence is sufficient
fn confidence_gated_dose(
    predicted_cmax: Knowledge[f64@mg_per_L],
    safety_threshold: f64@mg_per_L,
    confidence_threshold: f64
) -> DosingDecision {
    // High confidence that Cmax is safe
    if predicted_cmax.confidence >= confidence_threshold {
        if predicted_cmax.value < safety_threshold {
            return DosingDecision::Administer
        } else {
            return DosingDecision::Reduce
        }
    }

    // Low confidence - need more information
    if predicted_cmax.value < safety_threshold * 0.7 {
        // Likely safe even with uncertainty
        return DosingDecision::AdministerwithMonitoring
    }

    return DosingDecision::Defer  // Wait for TDM results
}
```

## Protocol DSL

MedLang provides a declarative protocol DSL:

### Weekly Dosing Protocol

```sio
protocol WeeklyDose {
    // Reference population model
    population_model: OneCompartmentIV_Population

    // Treatment arms
    arms {
        Dose50 = arm(
            name: "50 mg weekly",
            dose: 50.0 mg,
            route: IV,
            frequency: Weekly
        )
        Dose100 = arm(
            name: "100 mg weekly",
            dose: 100.0 mg,
            route: IV,
            frequency: Weekly
        )
        Dose200 = arm(
            name: "200 mg weekly",
            dose: 200.0 mg,
            route: IV,
            frequency: Weekly
        )
    }

    // Study visits
    visits {
        Baseline = visit(day: 0.0)
        Week4 = visit(day: 28.0)
        Week8 = visit(day: 56.0)
        Week12 = visit(day: 84.0)
    }

    // Inclusion criteria
    inclusion {
        age: 18..75
        ECOG: [0, 1]
    }

    // Endpoints
    endpoints {
        Response = binary_endpoint(
            observable: "TumourVolume",
            threshold: 0.30,  // 30% shrinkage
            window: 28.0..84.0 days
        )
    }
}
```

### Q3W Oncology Protocol

```sio
protocol Q3WDose {
    population_model: TwoCompartmentIV_Population

    arms {
        Standard = arm(
            name: "Standard Q3W",
            dose: 200.0 mg,
            route: IV,
            frequency: Every21Days
        )
        High = arm(
            name: "High Q3W",
            dose: 400.0 mg,
            route: IV,
            frequency: Every21Days
        )
    }

    visits {
        Baseline = visit(day: 0.0)
        Cycle2 = visit(day: 21.0)
        Cycle4 = visit(day: 63.0)
        Cycle6 = visit(day: 105.0)
    }

    inclusion {
        age: 18..80
        ECOG: [0, 1, 2]
    }

    endpoints {
        ORR = binary_endpoint(
            observable: "TumourVolume",
            threshold: 0.30,
            window: 21.0..105.0 days
        )
    }
}
```

### Daily Oral Protocol

```sio
protocol DailyOral {
    population_model: OneCompartmentOral_Population

    arms {
        Low = arm(name: "50 mg daily", dose: 50.0 mg, route: Oral, frequency: Daily)
        Medium = arm(name: "100 mg daily", dose: 100.0 mg, route: Oral, frequency: Daily)
        High = arm(name: "200 mg daily", dose: 200.0 mg, route: Oral, frequency: Daily)
    }

    visits {
        Baseline = visit(day: 0.0)
        Week2 = visit(day: 14.0)
        Week4 = visit(day: 28.0)
        Week8 = visit(day: 56.0)
    }

    inclusion {
        age: 18..75
        ECOG: [0, 1]
    }

    endpoints {
        Response = binary_endpoint(
            observable: "TumourVolume",
            threshold: 0.30,
            window: 14.0..56.0 days
        )
    }
}
```

## Time-Dependent Dosing

### Time-Varying Infusion Rate

```sio
/// Step-wise infusion (e.g., desensitization protocols)
fn stepwise_infusion(
    steps: &Vec<(f64@mg_per_h, f64@h)>  // (rate, duration) pairs
) -> Vec<DosingEvent> {
    var events: Vec<DosingEvent> = vec![]
    var current_time = 0.0 h

    for (rate, duration) in steps {
        let amt = rate * duration
        events.push(DosingEvent {
            amt: amt,
            route: Route::IV,
            time: current_time,
            rate: rate,
            duration: duration,
            bioavailability: 1.0,
            lag_time: 0.0 h
        })
        current_time = current_time + duration
    }

    return events
}

// Desensitization: gradually increasing rates
let steps = vec![
    (1.0 mg/h, 0.5 h),    // Step 1
    (5.0 mg/h, 0.5 h),    // Step 2
    (10.0 mg/h, 0.5 h),   // Step 3
    (50.0 mg/h, 1.0 h),   // Step 4
    (100.0 mg/h, 2.0 h)   // Final maintenance
]
let desensitization = stepwise_infusion(&steps)
```

## Simulating Dosing Regimens

```sio
use medlang::*

fn simulate_regimen(
    model: &Model,
    params: &ModelParams,
    regimen: &DosingRegimen,
    t_end: f64@h
) -> SimulationResult {
    // Initialize state
    var state = model.initial_state()
    var time = 0.0 h
    var dose_idx = 0

    // Output storage
    var time_points: Vec<f64> = vec![]
    var concentrations: Vec<f64> = vec![]

    // Simulation loop
    let dt = 0.1 h
    while time <= t_end {
        // Check for dose events at this time
        while dose_idx < regimen.doses.len() &&
              regimen.doses[dose_idx].time <= time {
            let dose = &regimen.doses[dose_idx]
            state = apply_dose(state, dose)
            dose_idx = dose_idx + 1
        }

        // Advance state
        state = model.step(state, params, time, dt)

        // Record
        time_points.push(time as f64)
        concentrations.push(state.cp)

        time = time + dt
    }

    return SimulationResult {
        time: time_points,
        cp: concentrations,
        cmax: max(&concentrations),
        tmax: time_of_max(&time_points, &concentrations),
        auc: trapz(&time_points, &concentrations)
    }
}
```

## Next Steps

- [Regulatory Compliance](regulatory-compliance.md) - FDA/EMA submission requirements
