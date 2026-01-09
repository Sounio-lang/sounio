# Data Loading Recipes

Practical recipes for loading and processing data in Sounio.

## Loading CSV Files

### Problem

You need to load data from a CSV file and process it.

### Solution

```sio
use std::io::*
use std::str::*
use std::json::*

struct DataRow {
    id: i64,
    time: f64,
    value: f64,
    uncertainty: f64,
}

fn load_csv(path: string) -> Vec<DataRow> with IO, Panic {
    let content = read_file(path)

    var rows: Vec<DataRow> = vec![]
    var line_num = 0

    for line in content.lines() {
        line_num = line_num + 1

        // Skip header row
        if line_num == 1 {
            continue
        }

        // Skip empty lines
        let trimmed = line.trim()
        if trimmed.is_empty() {
            continue
        }

        // Parse CSV line
        match parse_csv_row(trimmed) {
            Some(row) => rows.push(row),
            None => {
                // Log warning and skip malformed rows
                eprintln("Warning: Could not parse line " ++ line_num.to_string())
            }
        }
    }

    return rows
}

fn parse_csv_row(line: string) -> Option<DataRow> {
    let parts: Vec<&str> = line.split(',').collect()

    if parts.len() < 4 {
        return None
    }

    let id = parts[0].trim().parse::<i64>()
    let time = parts[1].trim().parse::<f64>()
    let value = parts[2].trim().parse::<f64>()
    let uncertainty = parts[3].trim().parse::<f64>()

    match (id, time, value, uncertainty) {
        (Ok(i), Ok(t), Ok(v), Ok(u)) => {
            Some(DataRow {
                id: i,
                time: t,
                value: v,
                uncertainty: u,
            })
        }
        _ => None
    }
}
```

### Discussion

Key considerations for CSV parsing:
- Always handle header rows
- Trim whitespace from fields
- Handle missing or malformed data gracefully
- Consider using a proper CSV library for complex cases (quoted fields, escaped commas)

---

## Loading JSON Data

### Problem

You need to load structured data from a JSON file.

### Solution

```sio
use std::io::*
use std::json::*

struct Measurement {
    timestamp: i64,
    sensor_id: string,
    value: f64,
    unit: string,
    uncertainty: f64,
}

fn load_json_measurements(path: string) -> Result<Vec<Measurement>, string> with IO {
    let content = read_file(path)

    match parse_json(content) {
        Ok(json) => {
            if !json.is_array() {
                return Err("Expected JSON array")
            }

            var measurements: Vec<Measurement> = vec![]

            for i in 0..json.len() {
                let item = json[i]

                if !item.is_object() {
                    continue
                }

                let measurement = Measurement {
                    timestamp: item["timestamp"].as_i64().unwrap_or(0),
                    sensor_id: item["sensor_id"].as_str().unwrap_or("unknown").to_string(),
                    value: item["value"].as_f64().unwrap_or(0.0),
                    unit: item["unit"].as_str().unwrap_or("").to_string(),
                    uncertainty: item["uncertainty"].as_f64().unwrap_or(0.0),
                }

                measurements.push(measurement)
            }

            return Ok(measurements)
        }
        Err(e) => {
            return Err("JSON parse error: " ++ e.message())
        }
    }
}
```

### Nested JSON

For nested JSON structures:

```sio
fn load_nested_json(path: string) -> Result<ExperimentData, string> with IO {
    let content = read_file(path)

    match parse_json(content) {
        Ok(json) => {
            // Navigate nested structure
            let experiment_name = json["metadata"]["name"].as_str().unwrap_or("unnamed")
            let date = json["metadata"]["date"].as_str().unwrap_or("")

            // Access array of results
            let results_json = json["results"]
            if !results_json.is_array() {
                return Err("Expected results array")
            }

            var results: Vec<Result> = vec![]
            for i in 0..results_json.len() {
                let r = results_json[i]
                results.push(Result {
                    sample_id: r["sample_id"].as_str().unwrap_or("").to_string(),
                    concentration: r["concentration"].as_f64().unwrap_or(0.0),
                    response: r["response"].as_f64().unwrap_or(0.0),
                })
            }

            return Ok(ExperimentData {
                name: experiment_name.to_string(),
                date: date.to_string(),
                results: results,
            })
        }
        Err(e) => Err("Parse error: " ++ e.message())
    }
}
```

---

## Handling Missing Values with Uncertainty

### Problem

Your data has missing values that should be represented with appropriate uncertainty.

### Solution

```sio
use epistemic::core::*

struct RawDataRow {
    id: i64,
    value: Option<f64>,
    uncertainty: Option<f64>,
}

fn convert_to_epistemic(row: RawDataRow) -> EpistemicValue {
    match (row.value, row.uncertainty) {
        // Complete data
        (Some(v), Some(u)) => {
            epistemic_std(v, u, 0.95)
        }

        // Value present, uncertainty missing - use default uncertainty
        (Some(v), None) => {
            // Assume 10% relative uncertainty as default
            let default_u = abs_f64(v) * 0.10
            epistemic_std(v, default_u, 0.80)  // Lower confidence
        }

        // Value missing - use wide interval
        (None, _) => {
            // Return a "missing" value with maximum uncertainty
            epistemic_interval(-1.0e100, 1.0e100, 0.0)  // Zero confidence
        }
    }
}

fn load_with_missing(path: string) -> Vec<EpistemicValue> with IO {
    var result: Vec<EpistemicValue> = vec![]

    let content = read_file(path)

    for line in content.lines().skip(1) {  // Skip header
        let parts: Vec<&str> = line.split(',').collect()
        if parts.len() < 3 { continue }

        let id = parts[0].trim().parse::<i64>().ok()

        // Handle "NA", "NULL", empty strings
        let value = parse_maybe_missing(parts[1].trim())
        let uncertainty = parse_maybe_missing(parts[2].trim())

        let row = RawDataRow {
            id: id.unwrap_or(0),
            value: value,
            uncertainty: uncertainty,
        }

        result.push(convert_to_epistemic(row))
    }

    return result
}

fn parse_maybe_missing(s: &str) -> Option<f64> {
    if s.is_empty() || s == "NA" || s == "NULL" || s == "." {
        return None
    }
    match s.parse::<f64>() {
        Ok(v) => Some(v),
        Err(_) => None,
    }
}
```

### Discussion

Strategies for missing data:

1. **Missing completely at random (MCAR)**: Use wide uncertainty interval
2. **Missing with pattern**: Impute using domain knowledge or neighboring values
3. **Below detection limit**: Use interval from 0 to detection limit

The key insight is that missing data = maximum uncertainty. The epistemic type system makes this explicit.

---

## Converting to Knowledge Types

### Problem

You have raw numeric data and want to create properly typed `Knowledge<T>` values with appropriate uncertainty and provenance.

### Solution

```sio
use epistemic::knowledge::*

// Convert raw measurement to Knowledge
fn measurement_to_knowledge(
    value: f64,
    uncertainty: f64,
    instrument: string
) -> Knowledge<f64> {
    Knowledge::new(
        value,
        uncertainty * uncertainty,  // variance = std^2
        BetaConfidence::uniform(),  // Start with uniform prior on confidence
        Source::Measurement {
            instrument: instrument,
            timestamp: current_timestamp(),
        }
    )
}

// Convert from literature value
fn literature_to_knowledge(
    value: f64,
    ci_lower: f64,
    ci_upper: f64,
    doi: string
) -> Knowledge<f64> {
    // Estimate std from 95% CI width
    let std = (ci_upper - ci_lower) / (2.0 * 1.96)
    let variance = std * std

    // Literature values have established confidence
    let confidence = BetaConfidence::strong(0.95, 100.0)

    Knowledge::new(
        value,
        variance,
        confidence,
        Source::External {
            source: "literature",
            url: "https://doi.org/" ++ doi,
        }
    )
}

// Convert from expert assertion
fn assertion_to_knowledge(
    value: f64,
    uncertainty: f64,
    author: string,
    confidence_level: f64
) -> Knowledge<f64> {
    let confidence = if confidence_level > 0.5 {
        BetaConfidence::strong(confidence_level, 10.0)
    } else {
        BetaConfidence::weak(confidence_level)
    }

    Knowledge::new(
        value,
        uncertainty * uncertainty,
        confidence,
        Source::Assertion { author: author }
    )
}
```

### Bulk Conversion

```sio
fn convert_dataset(
    data: Vec<DataRow>,
    instrument: string
) -> Vec<Knowledge<f64>> {
    var result: Vec<Knowledge<f64>> = vec![]

    for row in data {
        let k = measurement_to_knowledge(
            row.value,
            row.uncertainty,
            instrument.clone()
        )
        result.push(k)
    }

    return result
}
```

---

## Reading Data with Units

### Problem

You want to load data that includes units of measure and have them type-checked.

### Solution

```sio
use units::*
use std::io::*

struct DoseRecord {
    subject_id: i64,
    dose: mg,           // Typed with units
    volume: mL,
    concentration: mg/mL,
}

fn parse_dose_record(line: string) -> Option<DoseRecord> {
    let parts: Vec<&str> = line.split(',').collect()
    if parts.len() < 4 { return None }

    // Parse and attach units
    let id = parts[0].trim().parse::<i64>().ok()?
    let dose_val = parts[1].trim().parse::<f64>().ok()?
    let vol_val = parts[2].trim().parse::<f64>().ok()?
    let conc_val = parts[3].trim().parse::<f64>().ok()?

    // Attach units - this is type-checked!
    let dose: mg = dose_val  // Implicit unit attachment
    let volume: mL = vol_val
    let concentration: mg/mL = conc_val

    // Verify consistency (will fail at compile time if units don't match)
    let computed_conc: mg/mL = dose / volume

    return Some(DoseRecord {
        subject_id: id,
        dose: dose,
        volume: volume,
        concentration: concentration,
    })
}
```

### Units from String

When units are specified in the data file:

```sio
fn parse_with_unit(value_str: string, unit_str: string) -> Option<f64> {
    let value = value_str.parse::<f64>().ok()?

    // Convert to standard units based on unit string
    let converted = match unit_str.trim().to_lowercase().as_str() {
        "mg" => value,
        "g" => value * 1000.0,
        "ug" | "mcg" => value / 1000.0,
        "ml" => value,
        "l" => value * 1000.0,
        "h" | "hr" => value,
        "min" => value / 60.0,
        _ => {
            eprintln("Unknown unit: " ++ unit_str)
            return None
        }
    }

    return Some(converted)
}
```

---

## Streaming Large Files

### Problem

The data file is too large to load entirely into memory.

### Solution

```sio
use std::io::*

// Process file line by line
fn process_large_file<F>(
    path: string,
    processor: F
) with IO
where F: fn(DataRow) -> ()
{
    // Note: This is conceptual - Sounio's actual streaming API may differ
    let file = open_file(path)
    var line_num = 0

    while let Some(line) = file.read_line() {
        line_num = line_num + 1

        if line_num == 1 { continue }  // Skip header

        match parse_csv_row(line) {
            Some(row) => processor(row),
            None => { }  // Skip invalid rows
        }
    }
}

// Example usage: compute running statistics
fn compute_stats_streaming(path: string) -> (f64, f64) with IO {
    var count: i64 = 0
    var sum: f64 = 0.0
    var sum_sq: f64 = 0.0

    process_large_file(path, |row: DataRow| {
        count = count + 1
        sum = sum + row.value
        sum_sq = sum_sq + row.value * row.value
    })

    let mean = sum / (count as f64)
    let variance = sum_sq / (count as f64) - mean * mean
    let std = sqrt_f64(variance)

    return (mean, std)
}
```

---

## Validating Data on Load

### Problem

You want to validate data as it's loaded and collect all errors.

### Solution

```sio
enum ValidationError {
    MissingField { row: i64, field: string },
    InvalidValue { row: i64, field: string, value: string },
    OutOfRange { row: i64, field: string, value: f64, min: f64, max: f64 },
}

struct ValidationResult<T> {
    data: Vec<T>,
    errors: Vec<ValidationError>,
    warnings: Vec<string>,
}

fn load_and_validate(path: string) -> ValidationResult<DataRow> with IO {
    var data: Vec<DataRow> = vec![]
    var errors: Vec<ValidationError> = vec![]
    var warnings: Vec<string> = vec![]

    let content = read_file(path)
    var line_num: i64 = 0

    for line in content.lines() {
        line_num = line_num + 1
        if line_num == 1 { continue }  // Skip header

        let parts: Vec<&str> = line.split(',').collect()

        // Check required fields
        if parts.len() < 3 {
            errors.push(ValidationError::MissingField {
                row: line_num,
                field: "required columns",
            })
            continue
        }

        // Parse with validation
        let id = match parts[0].trim().parse::<i64>() {
            Ok(v) => v,
            Err(_) => {
                errors.push(ValidationError::InvalidValue {
                    row: line_num,
                    field: "id",
                    value: parts[0].to_string(),
                })
                continue
            }
        }

        let value = match parts[1].trim().parse::<f64>() {
            Ok(v) => v,
            Err(_) => {
                errors.push(ValidationError::InvalidValue {
                    row: line_num,
                    field: "value",
                    value: parts[1].to_string(),
                })
                continue
            }
        }

        // Range validation
        if value < 0.0 || value > 1000.0 {
            errors.push(ValidationError::OutOfRange {
                row: line_num,
                field: "value",
                value: value,
                min: 0.0,
                max: 1000.0,
            })
            continue
        }

        // Uncertainty validation
        let uncertainty = parts[2].trim().parse::<f64>().unwrap_or(0.0)
        if uncertainty < 0.0 {
            warnings.push("Row " ++ line_num.to_string() ++ ": negative uncertainty, using 0")
        }

        data.push(DataRow {
            id: id,
            time: 0.0,
            value: value,
            uncertainty: if uncertainty < 0.0 { 0.0 } else { uncertainty },
        })
    }

    return ValidationResult {
        data: data,
        errors: errors,
        warnings: warnings,
    }
}
```

---

## See Also

- [Error Handling](error-handling.md) for robust error patterns
- [Uncertainty Recipes](uncertainty-recipes.md) for working with loaded epistemic data
- [Standard Library I/O](../stdlib/io.md) for file operations
