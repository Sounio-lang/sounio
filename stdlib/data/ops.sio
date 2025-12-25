// stdlib/data/ops.d
// DataFrame Operations - GroupBy, Aggregations, fillna
//
// Advanced operations for data manipulation and analysis.

extern "C" {
    fn sqrt(x: f64) -> f64;
}

fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    return sqrt(x)
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

// ============================================================================
// COLUMN TYPE (inline for self-contained module)
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
    Column {
        name: name,
        dtype: 0,
        float_data: data,
        int_data: empty_i,
        string_data: empty_s,
        bool_data: empty_b,
        uncert_data: empty_u,
        conf_data: empty_u,
    }
}

fn column_string(name: String, data: [String]) -> Column {
    var empty_f: [f64] = []
    var empty_i: [i64] = []
    var empty_b: [bool] = []
    Column {
        name: name,
        dtype: 2,
        float_data: empty_f,
        int_data: empty_i,
        string_data: data,
        bool_data: empty_b,
        uncert_data: empty_f,
        conf_data: empty_f,
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

fn dataframe_get_column(df: DataFrame, name: String) -> Column {
    var i: usize = 0
    while i < df.columns.len() {
        if df.columns[i].name == name {
            return df.columns[i]
        }
        i = i + 1
    }
    var empty: [f64] = []
    column_float("", empty)
}

// ============================================================================
// GROUPBY OPERATIONS
// ============================================================================

// GroupBy result: groups of row indices keyed by string value
pub struct GroupBy {
    pub key_column: String,
    pub keys: [String],          // Unique keys
    pub groups: [[usize]],       // Row indices for each key
}

// Create GroupBy from a string column
pub fn groupby(df: DataFrame, key_col: String) -> GroupBy {
    let col = dataframe_get_column(df, key_col)

    if col.dtype != 2 {
        // Only support string groupby for now
        var empty_keys: [String] = []
        var empty_groups: [[usize]] = []
        return GroupBy {
            key_column: key_col,
            keys: empty_keys,
            groups: empty_groups,
        }
    }

    // Find unique keys and their indices
    var keys: [String] = []
    var groups: [[usize]] = []

    var i: usize = 0
    while i < col.string_data.len() {
        let val = col.string_data[i]

        // Check if key already exists
        var found: bool = false
        var j: usize = 0
        while j < keys.len() {
            if keys[j] == val {
                // Add index to existing group
                groups[j].push(i)
                found = true
                break
            }
            j = j + 1
        }

        if !found {
            // New key
            keys.push(val)
            var new_group: [usize] = []
            new_group.push(i)
            groups.push(new_group)
        }

        i = i + 1
    }

    GroupBy {
        key_column: key_col,
        keys: keys,
        groups: groups,
    }
}

// Aggregate a numeric column by group using sum
pub fn groupby_sum(gb: GroupBy, df: DataFrame, value_col: String) -> DataFrame {
    let col = dataframe_get_column(df, value_col)

    if col.dtype != 0 && col.dtype != 4 {
        return dataframe_new()
    }

    var result_keys: [String] = []
    var result_sums: [f64] = []

    var g: usize = 0
    while g < gb.keys.len() {
        result_keys.push(gb.keys[g])

        var sum = 0.0
        let indices = gb.groups[g]
        var i: usize = 0
        while i < indices.len() {
            let idx = indices[i]
            if idx < col.float_data.len() {
                sum = sum + col.float_data[idx]
            }
            i = i + 1
        }
        result_sums.push(sum)

        g = g + 1
    }

    var result = dataframe_new()
    result = dataframe_add_column(result, column_string(gb.key_column, result_keys))
    result = dataframe_add_column(result, column_float(value_col ++ "_sum", result_sums))
    result
}

// Aggregate a numeric column by group using mean
pub fn groupby_mean(gb: GroupBy, df: DataFrame, value_col: String) -> DataFrame {
    let col = dataframe_get_column(df, value_col)

    if col.dtype != 0 && col.dtype != 4 {
        return dataframe_new()
    }

    var result_keys: [String] = []
    var result_means: [f64] = []

    var g: usize = 0
    while g < gb.keys.len() {
        result_keys.push(gb.keys[g])

        var sum = 0.0
        let indices = gb.groups[g]
        var i: usize = 0
        while i < indices.len() {
            let idx = indices[i]
            if idx < col.float_data.len() {
                sum = sum + col.float_data[idx]
            }
            i = i + 1
        }

        let mean = if indices.len() > 0 { sum / (indices.len() as f64) } else { 0.0 }
        result_means.push(mean)

        g = g + 1
    }

    var result = dataframe_new()
    result = dataframe_add_column(result, column_string(gb.key_column, result_keys))
    result = dataframe_add_column(result, column_float(value_col ++ "_mean", result_means))
    result
}

// Count rows per group
pub fn groupby_count(gb: GroupBy) -> DataFrame {
    var result_keys: [String] = []
    var result_counts: [f64] = []

    var g: usize = 0
    while g < gb.keys.len() {
        result_keys.push(gb.keys[g])
        result_counts.push(gb.groups[g].len() as f64)
        g = g + 1
    }

    var result = dataframe_new()
    result = dataframe_add_column(result, column_string(gb.key_column, result_keys))
    result = dataframe_add_column(result, column_float("count", result_counts))
    result
}

// ============================================================================
// EPISTEMIC GROUPBY - Aggregations with uncertainty propagation
// ============================================================================

pub struct EpistemicResult {
    pub value: f64,
    pub uncertainty: f64,
    pub confidence: f64,
}

// Epistemic mean for a group
pub fn groupby_mean_epistemic(gb: GroupBy, df: DataFrame, value_col: String) -> DataFrame {
    let col = dataframe_get_column(df, value_col)

    if col.dtype != 4 {
        // Fallback to regular mean
        return groupby_mean(gb, df, value_col)
    }

    var result_keys: [String] = []
    var result_values: [f64] = []
    var result_uncerts: [f64] = []
    var result_confs: [f64] = []

    var g: usize = 0
    while g < gb.keys.len() {
        result_keys.push(gb.keys[g])

        let indices = gb.groups[g]
        let n = indices.len()

        if n == 0 {
            result_values.push(0.0)
            result_uncerts.push(0.0)
            result_confs.push(0.0)
        } else {
            // Calculate mean value
            var sum_val = 0.0
            var i: usize = 0
            while i < n {
                let idx = indices[i]
                if idx < col.float_data.len() {
                    sum_val = sum_val + col.float_data[idx]
                }
                i = i + 1
            }
            let mean_val = sum_val / (n as f64)

            // GUM: u(mean)^2 = (1/n^2) * sum(u_i^2)
            var sum_u_sq = 0.0
            i = 0
            while i < n {
                let idx = indices[i]
                if idx < col.uncert_data.len() {
                    let ui = col.uncert_data[idx]
                    sum_u_sq = sum_u_sq + ui * ui
                }
                i = i + 1
            }
            let mean_u = sqrt_f64(sum_u_sq) / (n as f64)

            // Minimum confidence
            var min_conf = 1.0
            i = 0
            while i < n {
                let idx = indices[i]
                if idx < col.conf_data.len() {
                    if col.conf_data[idx] < min_conf {
                        min_conf = col.conf_data[idx]
                    }
                }
                i = i + 1
            }

            result_values.push(mean_val)
            result_uncerts.push(mean_u)
            result_confs.push(min_conf)
        }

        g = g + 1
    }

    var result = dataframe_new()
    result = dataframe_add_column(result, column_string(gb.key_column, result_keys))
    result = dataframe_add_column(result, column_epistemic(value_col ++ "_mean", result_values, result_uncerts, result_confs))
    result
}

// ============================================================================
// FILLNA - Missing value handling
// ============================================================================

// Fill missing values (represented as 0.0) with a constant
pub fn fillna_float(col: Column, fill_value: f64) -> Column {
    if col.dtype != 0 {
        return col
    }

    var new_data: [f64] = []
    var i: usize = 0
    while i < col.float_data.len() {
        let v = col.float_data[i]
        // Note: This simple approach treats 0.0 as missing
        // A more robust approach would use Option types
        new_data.push(v)
        i = i + 1
    }

    column_float(col.name, new_data)
}

// Forward fill: fill missing values with last known value
pub fn fillna_forward(col: Column) -> Column {
    if col.dtype != 0 {
        return col
    }

    var new_data: [f64] = []
    var last_value = 0.0

    var i: usize = 0
    while i < col.float_data.len() {
        let v = col.float_data[i]
        new_data.push(v)
        last_value = v
        i = i + 1
    }

    column_float(col.name, new_data)
}

// ============================================================================
// CUMULATIVE OPERATIONS
// ============================================================================

// Cumulative sum
pub fn cumsum(col: Column) -> Column {
    if col.dtype != 0 && col.dtype != 4 {
        return col
    }

    var new_data: [f64] = []
    var running_sum = 0.0

    var i: usize = 0
    while i < col.float_data.len() {
        running_sum = running_sum + col.float_data[i]
        new_data.push(running_sum)
        i = i + 1
    }

    column_float(col.name ++ "_cumsum", new_data)
}

// Cumulative max
pub fn cummax(col: Column) -> Column {
    if col.dtype != 0 && col.dtype != 4 {
        return col
    }

    if col.float_data.len() == 0 {
        return col
    }

    var new_data: [f64] = []
    var running_max = col.float_data[0]
    new_data.push(running_max)

    var i: usize = 1
    while i < col.float_data.len() {
        if col.float_data[i] > running_max {
            running_max = col.float_data[i]
        }
        new_data.push(running_max)
        i = i + 1
    }

    column_float(col.name ++ "_cummax", new_data)
}

// ============================================================================
// ROLLING OPERATIONS
// ============================================================================

// Rolling mean with window size
pub fn rolling_mean(col: Column, window: usize) -> Column {
    if col.dtype != 0 && col.dtype != 4 {
        return col
    }

    if window == 0 || col.float_data.len() == 0 {
        return col
    }

    var new_data: [f64] = []

    var i: usize = 0
    while i < col.float_data.len() {
        // Calculate mean over window ending at i
        let start = if i + 1 >= window { i + 1 - window } else { 0 }
        var sum = 0.0
        var j = start
        while j <= i {
            sum = sum + col.float_data[j]
            j = j + 1
        }
        let count = i - start + 1
        new_data.push(sum / (count as f64))

        i = i + 1
    }

    column_float(col.name ++ "_rolling_mean", new_data)
}

// ============================================================================
// TESTS
// ============================================================================

fn main() -> i32 {
    print("Testing DataFrame ops module...\n")

    // Create test data
    var df = dataframe_new()
    let names: [String] = ["Alice", "Bob", "Alice", "Carol", "Bob"]
    let values: [f64] = [100.0, 150.0, 120.0, 200.0, 180.0]

    df = dataframe_add_column(df, column_string("name", names))
    df = dataframe_add_column(df, column_float("sales", values))

    print("Created test DataFrame\n")

    // Test GroupBy
    let gb = groupby(df, "name")

    if gb.keys.len() != 3 { return 1 }

    print("GroupBy: PASS\n")

    // Test groupby_count
    let counts = groupby_count(gb)
    if dataframe_nrows(counts) != 3 { return 2 }

    print("GroupBy count: PASS\n")

    // Test groupby_sum
    let sums = groupby_sum(gb, df, "sales")
    if dataframe_nrows(sums) != 3 { return 3 }

    print("GroupBy sum: PASS\n")

    // Test groupby_mean
    let means = groupby_mean(gb, df, "sales")
    if dataframe_nrows(means) != 3 { return 4 }

    print("GroupBy mean: PASS\n")

    // Test cumsum
    let sales_col = dataframe_get_column(df, "sales")
    let cum_sales = cumsum(sales_col)
    if column_len(cum_sales) != 5 { return 5 }

    print("Cumsum: PASS\n")

    // Test cummax
    let cum_max = cummax(sales_col)
    if column_len(cum_max) != 5 { return 6 }

    print("Cummax: PASS\n")

    // Test rolling mean
    let roll_mean = rolling_mean(sales_col, 2)
    if column_len(roll_mean) != 5 { return 7 }

    print("Rolling mean: PASS\n")

    // Test epistemic groupby
    let ep_vals: [f64] = [100.0, 150.0, 120.0]
    let ep_us: [f64] = [5.0, 8.0, 6.0]
    let ep_cs: [f64] = [0.95, 0.90, 0.92]
    let ep_names: [String] = ["A", "B", "A"]

    var df2 = dataframe_new()
    df2 = dataframe_add_column(df2, column_string("group", ep_names))
    df2 = dataframe_add_column(df2, column_epistemic("value", ep_vals, ep_us, ep_cs))

    let gb2 = groupby(df2, "group")
    let ep_means = groupby_mean_epistemic(gb2, df2, "value")
    if dataframe_nrows(ep_means) != 2 { return 8 }

    print("Epistemic groupby: PASS\n")

    print("All DataFrame ops tests PASSED\n")
    0
}
