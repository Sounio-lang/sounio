// stdlib/io/argparse.d
// Command-line argument parsing
//
// Simple argument parsing without string slicing (interpreter compatible).
// Uses whole-string comparisons only.

// =============================================================================
// Argument Types
// =============================================================================

pub struct ArgValue {
    pub key: String,
    pub value: String,
    pub is_flag: bool,
}

pub struct ParsedArgs {
    pub program: String,
    pub args: [String],
    pub count: usize,
}

// =============================================================================
// Parsing
// =============================================================================

// Parse command line arguments (simple version)
// Just stores raw arguments for application to process
pub fn parse(args: [String]) -> ParsedArgs {
    var result = ParsedArgs {
        program: "",
        args: [],
        count: 0,
    };

    let n = args.len();
    if n > 0 {
        result.program = args[0];

        var i: usize = 1;
        while i < n {
            result.args.push(args[i]);
            i = i + 1;
        }
        result.count = result.args.len();
    }

    result
}

// =============================================================================
// Query Functions (using whole-string comparisons)
// =============================================================================

// Check if an exact argument exists
pub fn has_arg(args: ParsedArgs, name: String) -> bool {
    let n = args.args.len();
    var i: usize = 0;
    while i < n {
        if args.args[i] == name {
            return true;
        }
        i = i + 1;
    }
    false
}

// Check if --verbose flag exists
pub fn is_verbose(args: ParsedArgs) -> bool {
    has_arg(args, "--verbose") || has_arg(args, "-v")
}

// Check if --help flag exists
pub fn is_help(args: ParsedArgs) -> bool {
    has_arg(args, "--help") || has_arg(args, "-h")
}

// Check if --quiet flag exists
pub fn is_quiet(args: ParsedArgs) -> bool {
    has_arg(args, "--quiet") || has_arg(args, "-q")
}

// Get argument by index
pub fn get_arg(args: ParsedArgs, idx: usize) -> String {
    if idx < args.args.len() {
        args.args[idx]
    } else {
        ""
    }
}

// Get first non-flag argument (doesn't start with -)
// Useful for getting input file
pub fn get_first_positional(args: ParsedArgs) -> String {
    let n = args.args.len();
    var i: usize = 0;
    while i < n {
        let arg = args.args[i];
        // Check common flags to skip
        if arg != "--verbose" && arg != "-v" &&
           arg != "--help" && arg != "-h" &&
           arg != "--quiet" && arg != "-q" &&
           arg != "--debug" && arg != "-d" {
            // This might be a positional (simplified check)
            // If it starts with -, it's still a flag we don't know about
            // This simplified version just returns first non-common-flag
            return arg;
        }
        i = i + 1;
    }
    ""
}

// Get number of arguments (excluding program name)
pub fn num_args(args: ParsedArgs) -> usize {
    args.count
}

// =============================================================================
// Tests
// =============================================================================

fn main() -> i32 {
    print("Testing argparse module...\n");

    // Test 1: Parse simple args
    var test_args: [String] = [];
    test_args.push("program");
    test_args.push("file.txt");
    test_args.push("--verbose");

    let parsed = parse(test_args);
    if parsed.program != "program" {
        print("FAIL: program name\n");
        return 1;
    }
    if parsed.count != 2 {
        print("FAIL: arg count\n");
        return 1;
    }
    print("Parse args: PASS\n");

    // Test 2: Check verbose flag
    let verbose = is_verbose(parsed);
    if !verbose {
        print("FAIL: verbose flag\n");
        return 1;
    }
    print("Verbose flag: PASS\n");

    // Test 3: Check has_arg
    let has_file = has_arg(parsed, "file.txt");
    if !has_file {
        print("FAIL: has_arg\n");
        return 1;
    }
    print("Has arg: PASS\n");

    // Test 4: Get arg by index
    let arg0 = get_arg(parsed, 0);
    if arg0 != "file.txt" {
        print("FAIL: get_arg\n");
        return 1;
    }
    print("Get arg: PASS\n");

    // Test 5: Help flag
    var test_args2: [String] = [];
    test_args2.push("cmd");
    test_args2.push("--help");

    let parsed2 = parse(test_args2);
    let help = is_help(parsed2);
    if !help {
        print("FAIL: help flag\n");
        return 1;
    }
    print("Help flag: PASS\n");

    // Test 6: Num args
    if num_args(parsed) != 2 {
        print("FAIL: num_args\n");
        return 1;
    }
    print("Num args: PASS\n");

    print("All argparse tests PASSED\n");
    0
}
