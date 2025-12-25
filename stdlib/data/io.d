// stdlib/data/io.d
// CSV I/O with Epistemic Integration
//
// Supports reading and writing DataFrames from/to CSV format.
// Special support for uncertainty notation:
// - "value±uncertainty" format: 100.0±2.0
// - Paired columns: value_col, value_col_u (uncertainty), value_col_conf (confidence)

extern "C" {
    fn sqrt(x: f64) -> f64;
}

// ============================================================================
// COLUMN TYPE (inline)
// ============================================================================

struct Column {
    name: String,
    dtype: i32,
    float_data: [f64],
    int_data: [i64],
    string_data: [String],
    bool_data: [bool],
    uncert_data: [f64],
    conf_data: [f64],
}

fn column_float(name: String, data: [f64]) -> Column {
    var empty_i: [i64] = []
    var empty_s: [String] = []
    var empty_b: [bool] = []
    var empty_u: [f64] = []
    var empty_c: [f64] = []
    Column {
        name: name,
        dtype: 0,
        float_data: data,
        int_data: empty_i,
        string_data: empty_s,
        bool_data: empty_b,
        uncert_data: empty_u,
        conf_data: empty_c,
    }
}

fn column_string(name: String, data: [String]) -> Column {
    var empty_f: [f64] = []
    var empty_i: [i64] = []
    var empty_b: [bool] = []
    var empty_u: [f64] = []
    var empty_c: [f64] = []
    Column {
        name: name,
        dtype: 2,
        float_data: empty_f,
        int_data: empty_i,
        string_data: data,
        bool_data: empty_b,
        uncert_data: empty_u,
        conf_data: empty_c,
    }
}

fn column_epistemic(name: String, values: [f64], uncerts: [f64], confs: [f64]) -> Column {
    var empty_i: [i64] = []
    var empty_s: [String] = []
    var empty_b: [bool] = []
    Column {
        name: name,
        dtype: 4,
        float_data: values,
        int_data: empty_i,
        string_data: empty_s,
        bool_data: empty_b,
        uncert_data: uncerts,
        conf_data: confs,
    }
}

fn column_len(col: Column) -> usize {
    if col.dtype == 0 { return col.float_data.len() }
    if col.dtype == 1 { return col.int_data.len() }
    if col.dtype == 2 { return col.string_data.len() }
    if col.dtype == 3 { return col.bool_data.len() }
    if col.dtype == 4 { return col.float_data.len() }
    0
}

struct DataFrame {
    columns: [Column],
    index_name: String,
}

fn dataframe_new() -> DataFrame {
    var empty: [Column] = []
    DataFrame { columns: empty, index_name: "" }
}

fn dataframe_add_column(df: DataFrame, col: Column) -> DataFrame {
    var cols = df.columns
    cols.push(col)
    DataFrame { columns: cols, index_name: df.index_name }
}

fn dataframe_nrows(df: DataFrame) -> usize {
    if df.columns.len() == 0 { return 0 }
    column_len(df.columns[0])
}

fn dataframe_ncols(df: DataFrame) -> usize {
    df.columns.len()
}

// ============================================================================
// STRING PARSING UTILITIES
// ============================================================================

fn split_by_comma(s: String) -> [String] {
    var result: [String] = []
    var current: String = ""
    let len = s.len()
    var i: usize = 0

    while i < len {
        let c = s.slice(i, i + 1)
        if c == "," {
            result.push(current)
            current = ""
        } else if c == "\r" {
            // skip
        } else if c == "\n" {
            break
        } else {
            current = current ++ c
        }
        i = i + 1
    }
    result.push(current)
    result
}

fn split_by_newline(s: String) -> [String] {
    var result: [String] = []
    var current: String = ""
    let len = s.len()
    var i: usize = 0

    while i < len {
        let c = s.slice(i, i + 1)
        if c == "\n" {
            if current.len() > 0 {
                let last = current.slice(current.len() - 1, current.len())
                if last == "\r" {
                    current = current.slice(0, current.len() - 1)
                }
            }
            result.push(current)
            current = ""
        } else {
            current = current ++ c
        }
        i = i + 1
    }
    if current.len() > 0 {
        result.push(current)
    }
    result
}

// Trim whitespace from string
fn trim(s: String) -> String {
    var start: usize = 0
    var end = s.len()

    while start < end {
        let c = s.slice(start, start + 1)
        if c == " " || c == "\t" || c == "\r" || c == "\n" {
            start = start + 1
        } else {
            break
        }
    }

    while end > start {
        let c = s.slice(end - 1, end)
        if c == " " || c == "\t" || c == "\r" || c == "\n" {
            end = end - 1
        } else {
            break
        }
    }

    if start >= end {
        return ""
    }
    s.slice(start, end)
}

// Check if string is numeric
fn is_numeric(s: String) -> bool {
    let trimmed = trim(s)
    if trimmed.len() == 0 { return false }

    var has_digit = false
    var has_dot = false
    var i: usize = 0

    while i < trimmed.len() {
        let c = trimmed.slice(i, i + 1)
        if c == "-" || c == "+" {
            if i != 0 { return false }
        } else if c == "." {
            if has_dot { return false }
            has_dot = true
        } else if c >= "0" && c <= "9" {
            has_digit = true
        } else {
            return false
        }
        i = i + 1
    }
    has_digit
}

// Parse string to float (simple version)
fn parse_float(s: String) -> f64 {
    let trimmed = trim(s)
    if trimmed.len() == 0 { return 0.0 }

    var result = 0.0
    var decimal_part = 0.0
    var decimal_divisor = 1.0
    var is_negative = false
    var in_decimal = false
    var i: usize = 0

    while i < trimmed.len() {
        let c = trimmed.slice(i, i + 1)
        if c == "-" && i == 0 {
            is_negative = true
        } else if c == "+" && i == 0 {
            // skip
        } else if c == "." {
            in_decimal = true
        } else if c >= "0" && c <= "9" {
            let digit = (c.byte_at(0) - 48) as f64
            if in_decimal {
                decimal_divisor = decimal_divisor * 10.0
                decimal_part = decimal_part + digit / decimal_divisor
            } else {
                result = result * 10.0 + digit
            }
        }
        i = i + 1
    }

    result = result + decimal_part
    if is_negative {
        result = 0.0 - result
    }
    result
}

// Check if string contains ± (plus-minus sign for uncertainty)
fn contains_plusminus(s: String) -> bool {
    var i: usize = 0
    while i < s.len() {
        // Check for UTF-8 ± or ASCII +/-
        if i + 1 < s.len() {
            let two_char = s.slice(i, i + 2)
            if two_char == "+-" || two_char == "-+" {
                return true
            }
        }
        // Also check for literal ± (3-byte UTF-8)
        if i + 2 < s.len() {
            let three_char = s.slice(i, i + 3)
            // ± is C2B1 in UTF-8
            if three_char == "±" {
                return true
            }
        }
        i = i + 1
    }
    false
}

// Parse value±uncertainty format
fn parse_uncertain(s: String) -> (f64, f64) {
    let trimmed = trim(s)

    // Find separator position
    var sep_pos: usize = 0
    var sep_len: usize = 0
    var i: usize = 0

    while i < trimmed.len() {
        if i + 1 < trimmed.len() {
            let two_char = trimmed.slice(i, i + 2)
            if two_char == "+-" || two_char == "-+" {
                sep_pos = i
                sep_len = 2
                break
            }
        }
        if i + 2 < trimmed.len() {
            let three_char = trimmed.slice(i, i + 3)
            if three_char == "±" {
                sep_pos = i
                sep_len = 3
                break
            }
        }
        i = i + 1
    }

    if sep_len == 0 {
        // No uncertainty found, return (value, 0)
        return (parse_float(trimmed), 0.0)
    }

    let value_str = trimmed.slice(0, sep_pos)
    let uncert_str = trimmed.slice(sep_pos + sep_len, trimmed.len())

    (parse_float(value_str), parse_float(uncert_str))
}

// ============================================================================
// CSV READING OPTIONS
// ============================================================================

pub struct CsvOptions {
    pub has_header: bool,
    pub delimiter: String,
    pub default_uncertainty: f64,
    pub default_confidence: f64,
    pub parse_uncertainty_notation: bool,  // Parse value±uncertainty
    pub uncertainty_suffix: String,         // e.g., "_u" for paired columns
    pub confidence_suffix: String,          // e.g., "_conf" for paired columns
}

pub fn csv_options_default() -> CsvOptions {
    CsvOptions {
        has_header: true,
        delimiter: ",",
        default_uncertainty: 0.0,
        default_confidence: 1.0,
        parse_uncertainty_notation: true,
        uncertainty_suffix: "_u",
        confidence_suffix: "_conf",
    }
}

// ============================================================================
// CSV READING
// ============================================================================

pub fn read_csv(text: String, options: CsvOptions) -> DataFrame {
    let lines = split_by_newline(text)
    if lines.len() == 0 {
        return dataframe_new()
    }

    var header: [String] = []
    var start_row: usize = 0

    if options.has_header && lines.len() > 0 {
        header = split_by_comma(lines[0])
        start_row = 1
    } else {
        // Generate column names: col0, col1, ...
        let first_row = split_by_comma(lines[0])
        var i: usize = 0
        while i < first_row.len() {
            header.push("col" ++ (i as String))
            i = i + 1
        }
    }

    // Collect all values per column
    let ncols = header.len()
    var column_data: [[String]] = []
    var j: usize = 0
    while j < ncols {
        var empty: [String] = []
        column_data.push(empty)
        j = j + 1
    }

    var row: usize = start_row
    while row < lines.len() {
        let fields = split_by_comma(lines[row])
        var col: usize = 0
        while col < ncols {
            if col < fields.len() {
                column_data[col].push(trim(fields[col]))
            } else {
                column_data[col].push("")
            }
            col = col + 1
        }
        row = row + 1
    }

    // Convert columns to appropriate types
    var df = dataframe_new()
    var col_idx: usize = 0

    while col_idx < ncols {
        let col_name = trim(header[col_idx])
        let data = column_data[col_idx]

        // Skip uncertainty and confidence suffix columns (they'll be paired)
        if col_name.len() >= options.uncertainty_suffix.len() {
            let suffix = col_name.slice(col_name.len() - options.uncertainty_suffix.len(), col_name.len())
            if suffix == options.uncertainty_suffix {
                col_idx = col_idx + 1
                continue
            }
        }
        if col_name.len() >= options.confidence_suffix.len() {
            let suffix = col_name.slice(col_name.len() - options.confidence_suffix.len(), col_name.len())
            if suffix == options.confidence_suffix {
                col_idx = col_idx + 1
                continue
            }
        }

        // Check if all values are numeric
        var all_numeric = true
        var has_uncertainty = false
        var i: usize = 0
        while i < data.len() {
            if data[i].len() > 0 {
                if contains_plusminus(data[i]) {
                    has_uncertainty = true
                } else if !is_numeric(data[i]) {
                    all_numeric = false
                    break
                }
            }
            i = i + 1
        }

        if all_numeric || has_uncertainty {
            // Check for paired uncertainty column
            let u_col_name = col_name ++ options.uncertainty_suffix
            let c_col_name = col_name ++ options.confidence_suffix
            var u_col_idx: usize = ncols  // Invalid index
            var c_col_idx: usize = ncols

            var k: usize = 0
            while k < ncols {
                if trim(header[k]) == u_col_name {
                    u_col_idx = k
                }
                if trim(header[k]) == c_col_name {
                    c_col_idx = k
                }
                k = k + 1
            }

            // Parse as epistemic if we have uncertainty info
            if has_uncertainty || u_col_idx < ncols {
                var values: [f64] = []
                var uncerts: [f64] = []
                var confs: [f64] = []

                i = 0
                while i < data.len() {
                    if has_uncertainty && options.parse_uncertainty_notation {
                        let parsed = parse_uncertain(data[i])
                        values.push(parsed.0)
                        uncerts.push(parsed.1)
                    } else {
                        values.push(parse_float(data[i]))
                        if u_col_idx < ncols && i < column_data[u_col_idx].len() {
                            uncerts.push(parse_float(column_data[u_col_idx][i]))
                        } else {
                            uncerts.push(options.default_uncertainty)
                        }
                    }

                    if c_col_idx < ncols && i < column_data[c_col_idx].len() {
                        confs.push(parse_float(column_data[c_col_idx][i]))
                    } else {
                        confs.push(options.default_confidence)
                    }
                    i = i + 1
                }

                df = dataframe_add_column(df, column_epistemic(col_name, values, uncerts, confs))
            } else {
                // Parse as plain float
                var values: [f64] = []
                i = 0
                while i < data.len() {
                    values.push(parse_float(data[i]))
                    i = i + 1
                }
                df = dataframe_add_column(df, column_float(col_name, values))
            }
        } else {
            // Keep as string
            df = dataframe_add_column(df, column_string(col_name, data))
        }

        col_idx = col_idx + 1
    }

    df
}

pub fn read_csv_simple(text: String) -> DataFrame {
    read_csv(text, csv_options_default())
}

// ============================================================================
// CSV WRITING
// ============================================================================

fn float_to_string(f: f64) -> String {
    // Simple float formatting
    var result = ""
    var value = f

    if value < 0.0 {
        result = result ++ "-"
        value = 0.0 - value
    }

    let int_part = value as i64
    result = result ++ (int_part as String)

    let frac = value - (int_part as f64)
    if frac > 0.0001 {
        result = result ++ "."
        var frac_scaled = (frac * 10000.0 + 0.5) as i64
        // Pad with zeros if needed
        if frac_scaled < 1000 { result = result ++ "0" }
        if frac_scaled < 100 { result = result ++ "0" }
        if frac_scaled < 10 { result = result ++ "0" }
        result = result ++ (frac_scaled as String)
    }

    result
}

pub struct CsvWriteOptions {
    pub write_uncertainty_notation: bool,  // Write as value±uncertainty
    pub write_paired_columns: bool,        // Write separate _u and _conf columns
    pub uncertainty_suffix: String,
    pub confidence_suffix: String,
}

pub fn csv_write_options_default() -> CsvWriteOptions {
    CsvWriteOptions {
        write_uncertainty_notation: false,
        write_paired_columns: true,
        uncertainty_suffix: "_u",
        confidence_suffix: "_conf",
    }
}

pub fn write_csv(df: DataFrame, options: CsvWriteOptions) -> String {
    var result = ""
    let nrows = dataframe_nrows(df)
    let ncols = dataframe_ncols(df)

    // Write header
    var col: usize = 0
    while col < ncols {
        if col > 0 { result = result ++ "," }
        result = result ++ df.columns[col].name

        // Add uncertainty/confidence headers for epistemic columns
        if df.columns[col].dtype == 4 && options.write_paired_columns {
            result = result ++ "," ++ df.columns[col].name ++ options.uncertainty_suffix
            result = result ++ "," ++ df.columns[col].name ++ options.confidence_suffix
        }
        col = col + 1
    }
    result = result ++ "\n"

    // Write data rows
    var row: usize = 0
    while row < nrows {
        col = 0
        while col < ncols {
            if col > 0 { result = result ++ "," }

            let c = df.columns[col]
            if c.dtype == 0 {
                if row < c.float_data.len() {
                    result = result ++ float_to_string(c.float_data[row])
                }
            } else if c.dtype == 1 {
                if row < c.int_data.len() {
                    result = result ++ (c.int_data[row] as String)
                }
            } else if c.dtype == 2 {
                if row < c.string_data.len() {
                    result = result ++ c.string_data[row]
                }
            } else if c.dtype == 3 {
                if row < c.bool_data.len() {
                    if c.bool_data[row] {
                        result = result ++ "true"
                    } else {
                        result = result ++ "false"
                    }
                }
            } else if c.dtype == 4 {
                if row < c.float_data.len() {
                    if options.write_uncertainty_notation && c.uncert_data[row] > 0.0 {
                        result = result ++ float_to_string(c.float_data[row])
                        result = result ++ "+-"
                        result = result ++ float_to_string(c.uncert_data[row])
                    } else {
                        result = result ++ float_to_string(c.float_data[row])
                    }

                    if options.write_paired_columns {
                        result = result ++ "," ++ float_to_string(c.uncert_data[row])
                        result = result ++ "," ++ float_to_string(c.conf_data[row])
                    }
                }
            }

            col = col + 1
        }
        result = result ++ "\n"
        row = row + 1
    }

    result
}

pub fn write_csv_simple(df: DataFrame) -> String {
    write_csv(df, csv_write_options_default())
}

// ============================================================================
// TESTS
// ============================================================================

fn main() -> i32 {
    print("Testing CSV I/O module...\n")

    // Test basic CSV parsing
    let csv1 = "name,age,weight\nAlice,25,65.5\nBob,30,78.2\nCarol,35,55.0"
    let df1 = read_csv_simple(csv1)

    if dataframe_nrows(df1) != 3 { return 1 }
    if dataframe_ncols(df1) != 3 { return 2 }

    print("Basic parsing: PASS\n")

    // Test numeric type detection
    let age_col = df1.columns[1]
    if age_col.dtype != 0 { return 3 }  // Should be float

    print("Type detection: PASS\n")

    // Test uncertainty notation parsing
    let csv2 = "measurement\n100.0+-2.0\n102.0+-2.5\n98.0+-1.5"
    let df2 = read_csv_simple(csv2)

    if dataframe_ncols(df2) != 1 { return 4 }
    let meas_col = df2.columns[0]
    if meas_col.dtype != 4 { return 5 }  // Should be epistemic

    // Check values parsed correctly
    if meas_col.float_data[0] < 99.9 || meas_col.float_data[0] > 100.1 { return 6 }
    if meas_col.uncert_data[0] < 1.9 || meas_col.uncert_data[0] > 2.1 { return 7 }

    print("Uncertainty notation: PASS\n")

    // Test paired columns
    let csv3 = "value,value_u,value_conf\n100.0,2.0,0.95\n102.0,2.5,0.90"
    let df3 = read_csv_simple(csv3)

    if dataframe_ncols(df3) != 1 { return 8 }  // Should combine into 1 epistemic column
    let val_col = df3.columns[0]
    if val_col.dtype != 4 { return 9 }
    if val_col.conf_data[0] < 0.94 || val_col.conf_data[0] > 0.96 { return 10 }

    print("Paired columns: PASS\n")

    // Test CSV writing
    var df4 = dataframe_new()
    let simple_vals: [f64] = [10.0, 20.0]
    df4 = dataframe_add_column(df4, column_float("data", simple_vals))

    let csv4 = write_csv_simple(df4)
    if csv4.len() == 0 { return 11 }
    print("CSV writing (float): PASS\n")

    // Test writing epistemic columns
    var df5 = dataframe_new()
    let epi_vals: [f64] = [100.0, 102.0]
    let epi_uncerts: [f64] = [2.0, 2.5]
    let epi_confs: [f64] = [0.95, 0.90]
    df5 = dataframe_add_column(df5, column_epistemic("measurement", epi_vals, epi_uncerts, epi_confs))

    let csv5 = write_csv_simple(df5)
    if csv5.len() == 0 { return 12 }
    print("CSV writing (epistemic): PASS\n")

    print("All CSV I/O tests PASSED\n")
    0
}
