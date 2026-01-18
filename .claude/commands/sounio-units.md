# Units of measure reference

Reference and utilities for Sounio's dimensional analysis and units of measure system.

## Arguments
- `--list` - List all available units
- `--category <category>` - Filter by category: si, pkpd, time, derived, all
- `--convert <value> <from> <to>` - Convert between units
- `--info <unit>` - Show detailed unit information
- `--check <expr>` - Check dimensional correctness of expression

## Examples
- `/sounio-units --list` - List all units
- `/sounio-units --category pkpd` - Pharmacokinetic units
- `/sounio-units --convert 500 mg g` - Convert 500mg to grams
- `/sounio-units --info mg/L` - Info about concentration unit

$ARGUMENTS

Execute from the `compiler/` directory for conversions:

```bash
cd /home/demetrios/sounio-1/compiler && cargo run -- units <subcommand>
```

## Unit Categories

**SI Base Units:**
- `m` - meter (length)
- `kg` - kilogram (mass)
- `s` - second (time)
- `A` - ampere (electric current)
- `K` - kelvin (temperature)
- `mol` - mole (amount of substance)
- `cd` - candela (luminous intensity)

**Mass Units:**
- `kg`, `g`, `mg`, `μg`, `ng`, `pg`
- Conversions: 1 kg = 1000 g = 1,000,000 mg

**Volume Units:**
- `L`, `mL`, `μL`, `nL`
- `m³`, `cm³`, `mm³`

**Time Units:**
- `s`, `ms`, `μs`, `ns`
- `min`, `h`, `day`

**PKPD (Pharmacokinetic) Units:**
- `mg/L` - concentration
- `mg/kg` - dose per body weight
- `L/h` - clearance rate
- `h⁻¹` - rate constant
- `mg·h/L` - AUC (area under curve)

**Derived Units:**
- `m/s` - velocity
- `m/s²` - acceleration
- `N` (kg·m/s²) - force
- `J` (N·m) - energy
- `W` (J/s) - power

## Usage in Sounio

```sio
// Declare with units
let dose: mg = 500.0
let volume: L = 0.5
let concentration: mg/L = dose / volume  // automatic unit derivation

// Unit conversion
let dose_g: g = dose.to(g)  // 0.5 g

// Compound units
let clearance: L/h = 2.5
let half_life: h = 0.693 / (clearance / volume)

// Dimensional analysis catches errors at compile time
// let invalid: mg = dose + volume  // ERROR: incompatible dimensions
```

## Unit Arithmetic

- Multiplication: `mg * L = mg·L`
- Division: `mg / L = mg/L`
- Addition: Only same dimensions allowed
- Powers: `m² = m * m`
