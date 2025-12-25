// stdlib/data/frame.d
// DataFrame - Tabular Data with Epistemic Integration
//
// A DataFrame holds named columns of different types. This implementation
// uses a type-erased column approach where each column is stored separately.

extern "C" {
    fn sqrt(x: f64) -> f64;
}

fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    return sqrt(x)
}

// ============================================================================
// COLUMN TYPE - Type-erased column storage
// ============================================================================

// Column types: 0=Float, 1=Int, 2=String, 3=Bool, 4=Epistemic
pub struct Column {
    pub name: String,
    pub dtype: i32,           // Type discriminator
    pub float_data: [f64],
    pub int_data: [i64],
    pub string_data: [String],
    pub bool_data: [bool],
    // Epistemic data
    pub uncert_data: [f64],   // Uncertainties (for epistemic)
    pub conf_data: [f64],     // Confidences (for epistemic)
}

pub fn column_float(name: String, data: [f64]) -> Column {
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

pub fn column_int(name: String, data: [i64]) -> Column {
    var empty_f: [f64] = []
    var empty_s: [String] = []
    var empty_b: [bool] = []
    var empty_u: [f64] = []
    var empty_c: [f64] = []
    Column {
        name: name,
        dtype: 1,
        float_data: empty_f,
        int_data: data,
        string_data: empty_s,
        bool_data: empty_b,
        uncert_data: empty_u,
        conf_data: empty_c,
    }
}

pub fn column_string(name: String, data: [String]) -> Column {
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

pub fn column_bool(name: String, data: [bool]) -> Column {
    var empty_f: [f64] = []
    var empty_i: [i64] = []
    var empty_s: [String] = []
    var empty_u: [f64] = []
    var empty_c: [f64] = []
    Column {
        name: name,
        dtype: 3,
        float_data: empty_f,
        int_data: empty_i,
        string_data: empty_s,
        bool_data: data,
        uncert_data: empty_u,
        conf_data: empty_c,
    }
}

pub fn column_epistemic(name: String, values: [f64], uncerts: [f64], confs: [f64]) -> Column {
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

pub fn column_len(col: Column) -> usize {
    if col.dtype == 0 { return col.float_data.len() }
    if col.dtype == 1 { return col.int_data.len() }
    if col.dtype == 2 { return col.string_data.len() }
    if col.dtype == 3 { return col.bool_data.len() }
    if col.dtype == 4 { return col.float_data.len() }
    0
}

pub fn column_dtype_str(col: Column) -> String {
    if col.dtype == 0 { return "float" }
    if col.dtype == 1 { return "int" }
    if col.dtype == 2 { return "string" }
    if col.dtype == 3 { return "bool" }
    if col.dtype == 4 { return "epistemic" }
    "unknown"
}

pub fn column_is_numeric(col: Column) -> bool {
    col.dtype == 0 || col.dtype == 1 || col.dtype == 4
}

// ============================================================================
// DATAFRAME TYPE
// ============================================================================

pub struct DataFrame {
    pub columns: [Column],
    pub index_name: String,
}

pub fn dataframe_new() -> DataFrame {
    var empty: [Column] = []
    DataFrame { columns: empty, index_name: "" }
}

pub fn dataframe_ncols(df: DataFrame) -> usize {
    df.columns.len()
}

pub fn dataframe_nrows(df: DataFrame) -> usize {
    if df.columns.len() == 0 { return 0 }
    column_len(df.columns[0])
}

pub fn dataframe_shape(df: DataFrame) -> (usize, usize) {
    (dataframe_nrows(df), dataframe_ncols(df))
}

pub fn dataframe_add_column(df: DataFrame, col: Column) -> DataFrame {
    var cols = df.columns
    cols.push(col)
    DataFrame { columns: cols, index_name: df.index_name }
}

pub fn dataframe_column_names(df: DataFrame) -> [String] {
    var names: [String] = []
    var i: usize = 0
    while i < df.columns.len() {
        names.push(df.columns[i].name)
        i = i + 1
    }
    names
}

// Get column by name
pub fn dataframe_get_column(df: DataFrame, name: String) -> Column {
    var i: usize = 0
    while i < df.columns.len() {
        if df.columns[i].name == name {
            return df.columns[i]
        }
        i = i + 1
    }
    // Return empty column if not found
    var empty: [f64] = []
    column_float("", empty)
}

// Get column by index
pub fn dataframe_get_column_idx(df: DataFrame, idx: usize) -> Column {
    if idx < df.columns.len() {
        return df.columns[idx]
    }
    var empty: [f64] = []
    column_float("", empty)
}

// Check if column exists
pub fn dataframe_has_column(df: DataFrame, name: String) -> bool {
    var i: usize = 0
    while i < df.columns.len() {
        if df.columns[i].name == name {
            return true
        }
        i = i + 1
    }
    false
}

// Drop column by name
pub fn dataframe_drop_column(df: DataFrame, name: String) -> DataFrame {
    var new_cols: [Column] = []
    var i: usize = 0
    while i < df.columns.len() {
        if df.columns[i].name != name {
            new_cols.push(df.columns[i])
        }
        i = i + 1
    }
    DataFrame { columns: new_cols, index_name: df.index_name }
}

// ============================================================================
// ROW ACCESS
// ============================================================================

pub struct Row {
    pub column_names: [String],
    pub float_values: [f64],
    pub int_values: [i64],
    pub string_values: [String],
    pub bool_values: [bool],
    pub dtypes: [i32],
}

pub fn dataframe_get_row(df: DataFrame, row_idx: usize) -> Row {
    var names: [String] = []
    var floats: [f64] = []
    var ints: [i64] = []
    var strings: [String] = []
    var bools: [bool] = []
    var types: [i32] = []

    var i: usize = 0
    while i < df.columns.len() {
        let col = df.columns[i]
        names.push(col.name)
        types.push(col.dtype)

        if col.dtype == 0 {
            if row_idx < col.float_data.len() {
                floats.push(col.float_data[row_idx])
            } else {
                floats.push(0.0)
            }
            ints.push(0)
            strings.push("")
            bools.push(false)
        } else if col.dtype == 1 {
            floats.push(0.0)
            if row_idx < col.int_data.len() {
                ints.push(col.int_data[row_idx])
            } else {
                ints.push(0)
            }
            strings.push("")
            bools.push(false)
        } else if col.dtype == 2 {
            floats.push(0.0)
            ints.push(0)
            if row_idx < col.string_data.len() {
                strings.push(col.string_data[row_idx])
            } else {
                strings.push("")
            }
            bools.push(false)
        } else if col.dtype == 3 {
            floats.push(0.0)
            ints.push(0)
            strings.push("")
            if row_idx < col.bool_data.len() {
                bools.push(col.bool_data[row_idx])
            } else {
                bools.push(false)
            }
        } else if col.dtype == 4 {
            // Epistemic: return value as float
            if row_idx < col.float_data.len() {
                floats.push(col.float_data[row_idx])
            } else {
                floats.push(0.0)
            }
            ints.push(0)
            strings.push("")
            bools.push(false)
        }

        i = i + 1
    }

    Row {
        column_names: names,
        float_values: floats,
        int_values: ints,
        string_values: strings,
        bool_values: bools,
        dtypes: types,
    }
}

// Get float value from row by column name
pub fn row_get_float(row: Row, col_name: String) -> f64 {
    var i: usize = 0
    while i < row.column_names.len() {
        if row.column_names[i] == col_name {
            return row.float_values[i]
        }
        i = i + 1
    }
    0.0
}

// Get string value from row by column name
pub fn row_get_string(row: Row, col_name: String) -> String {
    var i: usize = 0
    while i < row.column_names.len() {
        if row.column_names[i] == col_name {
            return row.string_values[i]
        }
        i = i + 1
    }
    ""
}

// ============================================================================
// SLICING / FILTERING
// ============================================================================

// Get rows by range [start, end)
pub fn dataframe_slice(df: DataFrame, start: usize, end: usize) -> DataFrame {
    var new_cols: [Column] = []
    var i: usize = 0

    while i < df.columns.len() {
        let col = df.columns[i]

        if col.dtype == 0 {
            var new_data: [f64] = []
            var j = start
            while j < end && j < col.float_data.len() {
                new_data.push(col.float_data[j])
                j = j + 1
            }
            new_cols.push(column_float(col.name, new_data))
        } else if col.dtype == 1 {
            var new_data: [i64] = []
            var j = start
            while j < end && j < col.int_data.len() {
                new_data.push(col.int_data[j])
                j = j + 1
            }
            new_cols.push(column_int(col.name, new_data))
        } else if col.dtype == 2 {
            var new_data: [String] = []
            var j = start
            while j < end && j < col.string_data.len() {
                new_data.push(col.string_data[j])
                j = j + 1
            }
            new_cols.push(column_string(col.name, new_data))
        } else if col.dtype == 3 {
            var new_data: [bool] = []
            var j = start
            while j < end && j < col.bool_data.len() {
                new_data.push(col.bool_data[j])
                j = j + 1
            }
            new_cols.push(column_bool(col.name, new_data))
        } else if col.dtype == 4 {
            var vals: [f64] = []
            var uncerts: [f64] = []
            var confs: [f64] = []
            var j = start
            while j < end && j < col.float_data.len() {
                vals.push(col.float_data[j])
                uncerts.push(col.uncert_data[j])
                confs.push(col.conf_data[j])
                j = j + 1
            }
            new_cols.push(column_epistemic(col.name, vals, uncerts, confs))
        }

        i = i + 1
    }

    DataFrame { columns: new_cols, index_name: df.index_name }
}

// Head - first n rows
pub fn dataframe_head(df: DataFrame, n: usize) -> DataFrame {
    dataframe_slice(df, 0, n)
}

// Tail - last n rows
pub fn dataframe_tail(df: DataFrame, n: usize) -> DataFrame {
    let nrows = dataframe_nrows(df)
    if n >= nrows {
        return df
    }
    dataframe_slice(df, nrows - n, nrows)
}

// Filter by boolean mask
pub fn dataframe_filter(df: DataFrame, mask: [bool]) -> DataFrame {
    var new_cols: [Column] = []
    var i: usize = 0

    while i < df.columns.len() {
        let col = df.columns[i]

        if col.dtype == 0 {
            var new_data: [f64] = []
            var j: usize = 0
            while j < col.float_data.len() && j < mask.len() {
                if mask[j] {
                    new_data.push(col.float_data[j])
                }
                j = j + 1
            }
            new_cols.push(column_float(col.name, new_data))
        } else if col.dtype == 1 {
            var new_data: [i64] = []
            var j: usize = 0
            while j < col.int_data.len() && j < mask.len() {
                if mask[j] {
                    new_data.push(col.int_data[j])
                }
                j = j + 1
            }
            new_cols.push(column_int(col.name, new_data))
        } else if col.dtype == 2 {
            var new_data: [String] = []
            var j: usize = 0
            while j < col.string_data.len() && j < mask.len() {
                if mask[j] {
                    new_data.push(col.string_data[j])
                }
                j = j + 1
            }
            new_cols.push(column_string(col.name, new_data))
        } else if col.dtype == 3 {
            var new_data: [bool] = []
            var j: usize = 0
            while j < col.bool_data.len() && j < mask.len() {
                if mask[j] {
                    new_data.push(col.bool_data[j])
                }
                j = j + 1
            }
            new_cols.push(column_bool(col.name, new_data))
        } else if col.dtype == 4 {
            var vals: [f64] = []
            var uncerts: [f64] = []
            var confs: [f64] = []
            var j: usize = 0
            while j < col.float_data.len() && j < mask.len() {
                if mask[j] {
                    vals.push(col.float_data[j])
                    uncerts.push(col.uncert_data[j])
                    confs.push(col.conf_data[j])
                }
                j = j + 1
            }
            new_cols.push(column_epistemic(col.name, vals, uncerts, confs))
        }

        i = i + 1
    }

    DataFrame { columns: new_cols, index_name: df.index_name }
}

// ============================================================================
// COLUMN OPERATIONS
// ============================================================================

// Sum of numeric column
pub fn dataframe_col_sum(df: DataFrame, col_name: String) -> f64 {
    let col = dataframe_get_column(df, col_name)
    if col.dtype == 0 {
        var total = 0.0
        var i: usize = 0
        while i < col.float_data.len() {
            total = total + col.float_data[i]
            i = i + 1
        }
        return total
    }
    if col.dtype == 1 {
        var total: i64 = 0
        var i: usize = 0
        while i < col.int_data.len() {
            total = total + col.int_data[i]
            i = i + 1
        }
        return total as f64
    }
    if col.dtype == 4 {
        var total = 0.0
        var i: usize = 0
        while i < col.float_data.len() {
            total = total + col.float_data[i]
            i = i + 1
        }
        return total
    }
    0.0
}

// Mean of numeric column
pub fn dataframe_col_mean(df: DataFrame, col_name: String) -> f64 {
    let col = dataframe_get_column(df, col_name)
    let n = column_len(col)
    if n == 0 { return 0.0 }
    dataframe_col_sum(df, col_name) / (n as f64)
}

// Epistemic mean of epistemic column (with uncertainty propagation)
pub struct EpistemicResult {
    pub value: f64,
    pub uncertainty: f64,
    pub confidence: f64,
}

pub fn dataframe_col_mean_epistemic(df: DataFrame, col_name: String) -> EpistemicResult {
    let col = dataframe_get_column(df, col_name)
    let n = col.float_data.len()

    if n == 0 || col.dtype != 4 {
        return EpistemicResult { value: 0.0, uncertainty: 0.0, confidence: 0.0 }
    }

    // Calculate mean
    var sum_val = 0.0
    var i: usize = 0
    while i < n {
        sum_val = sum_val + col.float_data[i]
        i = i + 1
    }
    let mean_val = sum_val / (n as f64)

    // GUM: u(mean)^2 = (1/n^2) * sum(u_i^2)
    var sum_u_sq = 0.0
    i = 0
    while i < n {
        let ui = col.uncert_data[i]
        sum_u_sq = sum_u_sq + ui * ui
        i = i + 1
    }
    let mean_u = sqrt_f64(sum_u_sq) / (n as f64)

    // Minimum confidence
    var min_conf = 1.0
    i = 0
    while i < n {
        if col.conf_data[i] < min_conf {
            min_conf = col.conf_data[i]
        }
        i = i + 1
    }

    EpistemicResult { value: mean_val, uncertainty: mean_u, confidence: min_conf }
}

// ============================================================================
// DISPLAY
// ============================================================================

pub fn dataframe_info(df: DataFrame) -> String {
    let shape = dataframe_shape(df)
    var result = "DataFrame: " ++ (shape.0 as String) ++ " rows x " ++ (shape.1 as String) ++ " columns\n"
    result = result ++ "Columns:\n"

    var i: usize = 0
    while i < df.columns.len() {
        let col = df.columns[i]
        result = result ++ "  " ++ col.name ++ " (" ++ column_dtype_str(col) ++ ")\n"
        i = i + 1
    }
    result
}

// ============================================================================
// TESTS
// ============================================================================

fn main() -> i32 {
    print("Testing DataFrame module...\n")

    // Create DataFrame with columns
    var df = dataframe_new()

    let ages: [i64] = [25, 30, 35, 40]
    let names: [String] = ["Alice", "Bob", "Carol", "Dave"]
    let weights: [f64] = [65.0, 78.5, 55.0, 82.0]

    df = dataframe_add_column(df, column_int("age", ages))
    df = dataframe_add_column(df, column_string("name", names))
    df = dataframe_add_column(df, column_float("weight", weights))

    let shape = dataframe_shape(df)
    if shape.0 != 4 { return 1 }
    if shape.1 != 3 { return 2 }

    print("Shape test: PASS\n")

    // Column access
    let age_col = dataframe_get_column(df, "age")
    if age_col.int_data[0] != 25 { return 3 }

    print("Column access: PASS\n")

    // Row access
    let row = dataframe_get_row(df, 1)
    let name_val = row_get_string(row, "name")
    if name_val != "Bob" { return 4 }

    print("Row access: PASS\n")

    // Aggregations
    let sum_weight = dataframe_col_sum(df, "weight")
    if sum_weight < 280.0 || sum_weight > 281.0 { return 5 }

    let mean_weight = dataframe_col_mean(df, "weight")
    if mean_weight < 70.0 || mean_weight > 70.2 { return 6 }

    print("Aggregations: PASS\n")

    // Head/tail
    let head_df = dataframe_head(df, 2)
    if dataframe_nrows(head_df) != 2 { return 7 }

    let tail_df = dataframe_tail(df, 2)
    if dataframe_nrows(tail_df) != 2 { return 8 }

    print("Head/Tail: PASS\n")

    // Filter
    let mask: [bool] = [true, false, true, false]
    let filtered = dataframe_filter(df, mask)
    if dataframe_nrows(filtered) != 2 { return 9 }

    print("Filter: PASS\n")

    // Epistemic column
    let ep_vals: [f64] = [100.0, 102.0, 98.0]
    let ep_uncerts: [f64] = [2.0, 2.0, 2.0]
    let ep_confs: [f64] = [0.95, 0.90, 0.92]

    var df2 = dataframe_new()
    df2 = dataframe_add_column(df2, column_epistemic("measurement", ep_vals, ep_uncerts, ep_confs))

    let ep_mean = dataframe_col_mean_epistemic(df2, "measurement")
    if ep_mean.value < 99.0 || ep_mean.value > 101.0 { return 10 }
    if ep_mean.uncertainty < 1.0 || ep_mean.uncertainty > 1.5 { return 11 }
    if ep_mean.confidence > 0.91 { return 12 }  // Should be min (0.90)

    print("Epistemic column: PASS\n")

    // Drop column
    let df3 = dataframe_drop_column(df, "weight")
    if dataframe_ncols(df3) != 2 { return 13 }

    print("Drop column: PASS\n")

    print("All DataFrame tests PASSED\n")
    0
}
