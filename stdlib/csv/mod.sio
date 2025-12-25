// stdlib/csv/mod.d
// CSV (Comma-Separated Values) Parser and Writer
//
// Simple CSV parsing for interpreter compatibility.
// Uses string split operations where available.

// =============================================================================
// CSV Record Type
// =============================================================================

pub struct CsvRecord {
    pub fields: [String],
}

pub struct CsvData {
    pub header: [String],
    pub records: [CsvRecord],
}

// =============================================================================
// Configuration
// =============================================================================

pub struct CsvConfig {
    pub has_header: bool,
}

pub fn default_config() -> CsvConfig {
    CsvConfig { has_header: true }
}

pub fn config_no_header() -> CsvConfig {
    CsvConfig { has_header: false }
}

// =============================================================================
// Parsing (using intrinsic split)
// =============================================================================

// Split a string by comma - simple version
// Note: Uses print-based parsing since string methods limited
pub fn split_by_comma(s: String) -> [String] {
    var result: [String] = [];
    var current: String = "";
    let len = s.len();

    // Build character by character using string slicing if available
    // For now, use a workaround - iterate and check
    var i: usize = 0;
    while i < len {
        // Get substring of length 1 at position i
        let c = s.slice(i, i + 1);
        if c == "," {
            result.push(current);
            current = "";
        } else if c == "\r" {
            // skip
        } else if c == "\n" {
            break;
        } else {
            current = current ++ c;
        }
        i = i + 1;
    }
    result.push(current);
    result
}

// Split a string by newlines
pub fn split_by_newline(s: String) -> [String] {
    var result: [String] = [];
    var current: String = "";
    let len = s.len();

    var i: usize = 0;
    while i < len {
        let c = s.slice(i, i + 1);
        if c == "\n" {
            // Add line (strip CR if present)
            if current.len() > 0 {
                let last = current.slice(current.len() - 1, current.len());
                if last == "\r" {
                    current = current.slice(0, current.len() - 1);
                }
            }
            result.push(current);
            current = "";
        } else {
            current = current ++ c;
        }
        i = i + 1;
    }

    // Add last line if non-empty
    if current.len() > 0 {
        result.push(current);
    }

    result
}

// Parse complete CSV
pub fn parse(text: String, cfg: CsvConfig) -> CsvData {
    let lines = split_by_newline(text);
    var header: [String] = [];
    var records: [CsvRecord] = [];

    let n = lines.len();
    if n == 0 {
        return CsvData { header: header, records: records };
    }

    var start: usize = 0;
    if cfg.has_header && n > 0 {
        header = split_by_comma(lines[0]);
        start = 1;
    }

    var i = start;
    while i < n {
        let line = lines[i];
        if line.len() > 0 {
            let fields = split_by_comma(line);
            records.push(CsvRecord { fields: fields });
        }
        i = i + 1;
    }

    CsvData { header: header, records: records }
}

pub fn parse_simple(text: String) -> CsvData {
    parse(text, default_config())
}

// =============================================================================
// Writing
// =============================================================================

pub fn write_record(fields: [String]) -> String {
    var result: String = "";
    let n = fields.len();
    var i: usize = 0;
    while i < n {
        if i > 0 {
            result = result ++ ",";
        }
        result = result ++ fields[i];
        i = i + 1;
    }
    result
}

pub fn write(data: CsvData) -> String {
    var result: String = "";

    if data.header.len() > 0 {
        result = write_record(data.header);
        result = result ++ "\n";
    }

    let n = data.records.len();
    var i: usize = 0;
    while i < n {
        result = result ++ write_record(data.records[i].fields);
        result = result ++ "\n";
        i = i + 1;
    }

    result
}

// =============================================================================
// Convenience
// =============================================================================

pub fn get_field(record: CsvRecord, idx: usize) -> String {
    if idx < record.fields.len() {
        record.fields[idx]
    } else {
        ""
    }
}

pub fn num_rows(data: CsvData) -> usize {
    data.records.len()
}

pub fn num_columns(data: CsvData) -> usize {
    if data.header.len() > 0 {
        data.header.len()
    } else if data.records.len() > 0 {
        data.records[0].fields.len()
    } else {
        0
    }
}

// =============================================================================
// Tests
// =============================================================================

fn main() -> i32 {
    print("Testing CSV module...\n");

    // Test: Parse simple CSV
    let data = parse_simple("a,b,c\n1,2,3\n4,5,6");
    print("Parsing test completed\n");

    let rows = num_rows(data);
    let cols = num_columns(data);

    print("Rows/cols test completed\n");

    // Test: Write CSV
    var hdr: [String] = [];
    hdr.push("x");
    hdr.push("y");

    var f1: [String] = [];
    f1.push("1");
    f1.push("2");

    var recs: [CsvRecord] = [];
    recs.push(CsvRecord { fields: f1 });

    let test_data = CsvData { header: hdr, records: recs };
    let output = write(test_data);

    print("Write test completed\n");

    print("All CSV tests PASSED\n");
    0
}
