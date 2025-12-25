// medlang::parser — Recursive Descent Parser for MedLang eDSL
//
// Parses tokenized MedLang source into AST nodes.
//
// Grammar (simplified EBNF):
//   program      ::= model_def* fit_spec* simulate_spec*
//   model_def    ::= 'model' IDENT '{' model_body '}'
//   model_body   ::= (param_def | compartment_def | flow_def | dose_def | observe_def)*
//   param_def    ::= 'param' IDENT (':' type_annot)? '~' distribution
//   distribution ::= 'LogNormal' '(' expr (',' 'omega' ':' expr)? ')'
//                  | 'Normal' '(' expr ',' expr ')'
//                  | 'Fixed' '(' expr ')'

// ============================================================================
// TOKEN TYPES (from lexer.d)
// ============================================================================

fn TT_EOF() -> i32 { 0 }
fn TT_ERROR() -> i32 { 1 }
fn TT_KW_MODEL() -> i32 { 10 }
fn TT_KW_PARAM() -> i32 { 11 }
fn TT_KW_COMPARTMENT() -> i32 { 12 }
fn TT_KW_FLOW() -> i32 { 13 }
fn TT_KW_DOSE() -> i32 { 14 }
fn TT_KW_OBSERVE() -> i32 { 15 }
fn TT_KW_FIT() -> i32 { 16 }
fn TT_KW_SIMULATE() -> i32 { 17 }
fn TT_KW_TO() -> i32 { 18 }
fn TT_KW_WITH() -> i32 { 19 }
fn TT_KW_VOLUME() -> i32 { 21 }
fn TT_KW_LOGNORMAL() -> i32 { 30 }
fn TT_KW_NORMAL() -> i32 { 31 }
fn TT_KW_FIXED() -> i32 { 33 }
fn TT_KW_OMEGA() -> i32 { 34 }
fn TT_KW_IV() -> i32 { 40 }
fn TT_KW_ORAL() -> i32 { 41 }
fn TT_KW_INFUSION() -> i32 { 42 }
fn TT_KW_ADDITIVE() -> i32 { 50 }
fn TT_KW_PROPORTIONAL() -> i32 { 51 }
fn TT_KW_COMBINED() -> i32 { 52 }
fn TT_KW_METHOD() -> i32 { 60 }
fn TT_KW_UNCERTAINTY() -> i32 { 61 }
fn TT_KW_LM() -> i32 { 62 }
fn TT_KW_GUM() -> i32 { 63 }
fn TT_KW_BOOTSTRAP() -> i32 { 64 }
fn TT_KW_TIME() -> i32 { 70 }
fn TT_KW_SUBJECTS() -> i32 { 71 }
fn TT_IDENTIFIER() -> i32 { 80 }
fn TT_INTEGER() -> i32 { 81 }
fn TT_FLOAT() -> i32 { 82 }
fn TT_STRING() -> i32 { 83 }
fn TT_UNIT_MG() -> i32 { 90 }
fn TT_UNIT_L() -> i32 { 91 }
fn TT_UNIT_H() -> i32 { 92 }
fn TT_UNIT_L_PER_H() -> i32 { 93 }
fn TT_UNIT_PER_H() -> i32 { 95 }
fn TT_OP_EQ() -> i32 { 105 }
fn TT_OP_TILDE() -> i32 { 106 }
fn TT_OP_ARROW() -> i32 { 107 }
fn TT_OP_DOT() -> i32 { 108 }
fn TT_OP_DOTDOT() -> i32 { 109 }
fn TT_LPAREN() -> i32 { 120 }
fn TT_RPAREN() -> i32 { 121 }
fn TT_LBRACE() -> i32 { 122 }
fn TT_RBRACE() -> i32 { 123 }
fn TT_LBRACKET() -> i32 { 124 }
fn TT_RBRACKET() -> i32 { 125 }
fn TT_COMMA() -> i32 { 126 }
fn TT_COLON() -> i32 { 127 }
fn TT_NEWLINE() -> i32 { 129 }

// ============================================================================
// TOKEN STRUCTURE (from lexer.d)
// ============================================================================

struct Token {
    token_type: i32,
    text: [i8; 64],
    text_len: i64,
    int_value: i64,
    float_value: f64,
    line: i64,
    column: i64,
}

fn token_new() -> Token {
    Token {
        token_type: TT_EOF(),
        text: [0; 64],
        text_len: 0,
        int_value: 0,
        float_value: 0.0,
        line: 1,
        column: 1,
    }
}

// ============================================================================
// DISTRIBUTION TYPES
// ============================================================================

fn DIST_FIXED() -> i32 { 0 }
fn DIST_NORMAL() -> i32 { 1 }
fn DIST_LOGNORMAL() -> i32 { 2 }
fn DIST_UNIFORM() -> i32 { 3 }

struct DistributionSpec {
    dist_type: i32,
    mu: f64,
    sigma: f64,
    omega: f64,
    lower: f64,
    upper: f64,
}

fn dist_spec_new() -> DistributionSpec {
    DistributionSpec {
        dist_type: DIST_FIXED(),
        mu: 0.0,
        sigma: 0.0,
        omega: 0.0,
        lower: 0.0,
        upper: 1e308,
    }
}

fn dist_fixed(value: f64) -> DistributionSpec {
    DistributionSpec {
        dist_type: DIST_FIXED(),
        mu: value,
        sigma: 0.0,
        omega: 0.0,
        lower: value,
        upper: value,
    }
}

fn dist_lognormal(mu: f64, omega: f64) -> DistributionSpec {
    DistributionSpec {
        dist_type: DIST_LOGNORMAL(),
        mu: mu,
        sigma: omega,
        omega: omega,
        lower: 0.0,
        upper: 1e308,
    }
}

fn dist_normal(mu: f64, sigma: f64) -> DistributionSpec {
    DistributionSpec {
        dist_type: DIST_NORMAL(),
        mu: mu,
        sigma: sigma,
        omega: sigma,
        lower: -1e308,
        upper: 1e308,
    }
}

// ============================================================================
// UNIT TYPES
// ============================================================================

fn UNIT_NONE() -> i32 { 0 }
fn UNIT_MG() -> i32 { 1 }
fn UNIT_L() -> i32 { 2 }
fn UNIT_H() -> i32 { 3 }
fn UNIT_L_PER_H() -> i32 { 4 }
fn UNIT_MG_PER_L() -> i32 { 5 }
fn UNIT_PER_H() -> i32 { 6 }

// ============================================================================
// AST NODE TYPES
// ============================================================================

/// Parameter node
struct ParamNode {
    name: [i8; 64],
    typical_value: f64,
    unit_type: i32,
    distribution: DistributionSpec,
    lower_bound: f64,
    upper_bound: f64,
    is_fixed: bool,
    index: i64,
}

fn param_node_new() -> ParamNode {
    ParamNode {
        name: [0; 64],
        typical_value: 0.0,
        unit_type: UNIT_NONE(),
        distribution: dist_spec_new(),
        lower_bound: 0.0,
        upper_bound: 1e308,
        is_fixed: false,
        index: -1,
    }
}

/// Compartment node
struct CompartmentNode {
    name: [i8; 64],
    volume_param: [i8; 64],
    initial_amount: f64,
    is_depot: bool,
    is_central: bool,
    state_index: i64,
}

fn compartment_node_new() -> CompartmentNode {
    CompartmentNode {
        name: [0; 64],
        volume_param: [0; 64],
        initial_amount: 0.0,
        is_depot: false,
        is_central: false,
        state_index: -1,
    }
}

/// Flow types
fn FLOW_ELIMINATION() -> i32 { 0 }
fn FLOW_TRANSFER() -> i32 { 1 }
fn FLOW_ABSORPTION() -> i32 { 2 }

/// Flow node
struct FlowNode {
    name: [i8; 64],
    flow_type: i32,
    from_compartment: [i8; 64],
    to_compartment: [i8; 64],
    rate_param: [i8; 64],
    expr_text: [i8; 256],
}

fn flow_node_new() -> FlowNode {
    FlowNode {
        name: [0; 64],
        flow_type: FLOW_ELIMINATION(),
        from_compartment: [0; 64],
        to_compartment: [0; 64],
        rate_param: [0; 64],
        expr_text: [0; 256],
    }
}

/// Dosing routes
fn ROUTE_IV_BOLUS() -> i32 { 0 }
fn ROUTE_IV_INFUSION() -> i32 { 1 }
fn ROUTE_ORAL() -> i32 { 2 }

/// Dosing node
struct DosingNode {
    route: i32,
    target_compartment: [i8; 64],
    amount: f64,
    time: f64,
    duration: f64,
    bioavailability: f64,
    times: [f64; 50],
    n_times: i64,
}

fn dosing_node_new() -> DosingNode {
    DosingNode {
        route: ROUTE_IV_BOLUS(),
        target_compartment: [0; 64],
        amount: 0.0,
        time: 0.0,
        duration: 0.0,
        bioavailability: 1.0,
        times: [0.0; 50],
        n_times: 0,
    }
}

/// Error model types
fn ERROR_ADDITIVE() -> i32 { 0 }
fn ERROR_PROPORTIONAL() -> i32 { 1 }
fn ERROR_COMBINED() -> i32 { 2 }

/// Observation node
struct ObservationNode {
    name: [i8; 64],
    source_compartment: [i8; 64],
    is_concentration: bool,
    error_type: i32,
    sigma_add: f64,
    sigma_prop: f64,
    lloq: f64,
}

fn observation_node_new() -> ObservationNode {
    ObservationNode {
        name: [0; 64],
        source_compartment: [0; 64],
        is_concentration: true,
        error_type: ERROR_COMBINED(),
        sigma_add: 0.0,
        sigma_prop: 0.1,
        lloq: 0.0,
    }
}

/// Fit methods
fn FIT_LM() -> i32 { 0 }
fn FIT_BFGS() -> i32 { 1 }

/// Uncertainty methods
fn UNC_NONE() -> i32 { 0 }
fn UNC_GUM() -> i32 { 1 }
fn UNC_BOOTSTRAP() -> i32 { 2 }

/// Fit specification
struct FitSpec {
    model_name: [i8; 64],
    data_file: [i8; 256],
    method: i32,
    uncertainty_method: i32,
    max_iterations: i64,
    coverage_factor: f64,
    n_bootstrap: i64,
}

fn fit_spec_new() -> FitSpec {
    FitSpec {
        model_name: [0; 64],
        data_file: [0; 256],
        method: FIT_LM(),
        uncertainty_method: UNC_GUM(),
        max_iterations: 1000,
        coverage_factor: 2.0,
        n_bootstrap: 1000,
    }
}

/// Simulate specification
struct SimulateSpec {
    model_name: [i8; 64],
    t_start: f64,
    t_end: f64,
    dt: f64,
    n_subjects: i64,
    include_iiv: bool,
    include_error: bool,
    seed: i64,
    output_file: [i8; 256],
}

fn simulate_spec_new() -> SimulateSpec {
    SimulateSpec {
        model_name: [0; 64],
        t_start: 0.0,
        t_end: 24.0,
        dt: 0.1,
        n_subjects: 1,
        include_iiv: false,
        include_error: false,
        seed: 42,
        output_file: [0; 256],
    }
}

/// Model node
struct ModelNode {
    name: [i8; 64],
    params: [ParamNode; 30],
    n_params: i64,
    compartments: [CompartmentNode; 10],
    n_compartments: i64,
    flows: [FlowNode; 20],
    n_flows: i64,
    doses: [DosingNode; 50],
    n_doses: i64,
    observations: [ObservationNode; 5],
    n_observations: i64,
}

fn model_node_new() -> ModelNode {
    ModelNode {
        name: [0; 64],
        params: [param_node_new(); 30],
        n_params: 0,
        compartments: [compartment_node_new(); 10],
        n_compartments: 0,
        flows: [flow_node_new(); 20],
        n_flows: 0,
        doses: [dosing_node_new(); 50],
        n_doses: 0,
        observations: [observation_node_new(); 5],
        n_observations: 0,
    }
}

/// Complete program
struct MedLangProgram {
    models: [ModelNode; 10],
    n_models: i64,
    fit_specs: [FitSpec; 10],
    n_fits: i64,
    sim_specs: [SimulateSpec; 10],
    n_sims: i64,
}

fn medlang_program_new() -> MedLangProgram {
    MedLangProgram {
        models: [model_node_new(); 10],
        n_models: 0,
        fit_specs: [fit_spec_new(); 10],
        n_fits: 0,
        sim_specs: [simulate_spec_new(); 10],
        n_sims: 0,
    }
}

// ============================================================================
// PARSE ERROR
// ============================================================================

struct ParseError {
    message: [i8; 256],
    line: i64,
    column: i64,
    token_text: [i8; 64],
}

fn parse_error_new() -> ParseError {
    ParseError {
        message: [0; 256],
        line: 0,
        column: 0,
        token_text: [0; 64],
    }
}

// ============================================================================
// PARSER STATE
// ============================================================================

struct Parser {
    tokens: [Token; 4096],
    n_tokens: i64,
    pos: i64,
    has_error: bool,
    errors: [ParseError; 32],
    n_errors: i64,
}

fn parser_new() -> Parser {
    Parser {
        tokens: [token_new(); 4096],
        n_tokens: 0,
        pos: 0,
        has_error: false,
        errors: [parse_error_new(); 32],
        n_errors: 0,
    }
}

// ============================================================================
// PARSER HELPERS
// ============================================================================

fn parser_at_end(p: Parser) -> bool {
    p.pos >= p.n_tokens || p.tokens[p.pos as usize].token_type == TT_EOF()
}

fn parser_current_type(p: Parser) -> i32 {
    if p.pos < p.n_tokens {
        p.tokens[p.pos as usize].token_type
    } else {
        TT_EOF()
    }
}

fn parser_check(p: Parser, t: i32) -> bool {
    p.pos < p.n_tokens && p.tokens[p.pos as usize].token_type == t
}

fn parser_advance(p: Parser) -> Parser {
    var new_p = p;
    if new_p.pos < new_p.n_tokens {
        new_p.pos = new_p.pos + 1;
    }
    // Skip newlines
    while new_p.pos < new_p.n_tokens && new_p.tokens[new_p.pos as usize].token_type == TT_NEWLINE() {
        new_p.pos = new_p.pos + 1;
    }
    new_p
}

fn parser_match(p: Parser, t: i32) -> ParseMatchResult {
    if parser_check(p, t) {
        ParseMatchResult {
            parser: parser_advance(p),
            matched: true,
        }
    } else {
        ParseMatchResult {
            parser: p,
            matched: false,
        }
    }
}

struct ParseMatchResult {
    parser: Parser,
    matched: bool,
}

fn parser_expect(p: Parser, t: i32) -> ParseMatchResult {
    if parser_check(p, t) {
        ParseMatchResult {
            parser: parser_advance(p),
            matched: true,
        }
    } else {
        var new_p = p;
        new_p.has_error = true;
        ParseMatchResult {
            parser: new_p,
            matched: false,
        }
    }
}

fn copy_token_text(tok: Token, dest: [i8; 64]) -> [i8; 64] {
    var result = dest;
    var i: i64 = 0;
    while i < tok.text_len && i < 63 {
        result[i as usize] = tok.text[i as usize];
        i = i + 1;
    }
    result[i as usize] = 0;
    result
}

// ============================================================================
// EXPRESSION PARSER
// ============================================================================

struct ParseValueResult {
    parser: Parser,
    value: f64,
    unit: i32,
}

fn parse_value(p: Parser) -> ParseValueResult {
    var value = 0.0;
    var unit = UNIT_NONE();
    var new_p = p;

    if parser_check(new_p, TT_INTEGER()) {
        value = new_p.tokens[new_p.pos as usize].int_value as f64;
        new_p = parser_advance(new_p);
    } else if parser_check(new_p, TT_FLOAT()) {
        value = new_p.tokens[new_p.pos as usize].float_value;
        new_p = parser_advance(new_p);
    }

    // Check for unit
    if parser_check(new_p, TT_UNIT_MG()) {
        unit = UNIT_MG();
        new_p = parser_advance(new_p);
    } else if parser_check(new_p, TT_UNIT_L()) {
        unit = UNIT_L();
        new_p = parser_advance(new_p);
    } else if parser_check(new_p, TT_UNIT_H()) {
        unit = UNIT_H();
        new_p = parser_advance(new_p);
    } else if parser_check(new_p, TT_UNIT_L_PER_H()) {
        unit = UNIT_L_PER_H();
        new_p = parser_advance(new_p);
    } else if parser_check(new_p, TT_UNIT_PER_H()) {
        unit = UNIT_PER_H();
        new_p = parser_advance(new_p);
    }

    ParseValueResult {
        parser: new_p,
        value: value,
        unit: unit,
    }
}

// ============================================================================
// DISTRIBUTION PARSER
// ============================================================================

struct ParseDistResult {
    parser: Parser,
    dist: DistributionSpec,
}

fn parse_distribution(p: Parser) -> ParseDistResult {
    var dist = dist_spec_new();
    var new_p = p;

    if parser_check(new_p, TT_KW_LOGNORMAL()) {
        new_p = parser_advance(new_p);
        let m1 = parser_expect(new_p, TT_LPAREN());
        new_p = m1.parser;

        let val = parse_value(new_p);
        new_p = val.parser;
        dist.mu = val.value;
        dist.dist_type = DIST_LOGNORMAL();

        // Optional omega
        let m2 = parser_match(new_p, TT_COMMA());
        if m2.matched {
            new_p = m2.parser;
            if parser_check(new_p, TT_KW_OMEGA()) {
                new_p = parser_advance(new_p);
                let m3 = parser_expect(new_p, TT_COLON());
                new_p = m3.parser;
            }
            let omega = parse_value(new_p);
            new_p = omega.parser;
            dist.omega = omega.value;
            dist.sigma = omega.value;
        }

        let m4 = parser_expect(new_p, TT_RPAREN());
        new_p = m4.parser;

    } else if parser_check(new_p, TT_KW_NORMAL()) {
        new_p = parser_advance(new_p);
        let m1 = parser_expect(new_p, TT_LPAREN());
        new_p = m1.parser;

        let mu = parse_value(new_p);
        new_p = mu.parser;
        dist.mu = mu.value;
        dist.dist_type = DIST_NORMAL();

        let m2 = parser_expect(new_p, TT_COMMA());
        new_p = m2.parser;
        let sigma = parse_value(new_p);
        new_p = sigma.parser;
        dist.sigma = sigma.value;
        dist.omega = sigma.value;

        let m3 = parser_expect(new_p, TT_RPAREN());
        new_p = m3.parser;

    } else if parser_check(new_p, TT_KW_FIXED()) {
        new_p = parser_advance(new_p);
        let m1 = parser_expect(new_p, TT_LPAREN());
        new_p = m1.parser;

        let val = parse_value(new_p);
        new_p = val.parser;
        dist = dist_fixed(val.value);

        let m2 = parser_expect(new_p, TT_RPAREN());
        new_p = m2.parser;

    } else {
        // Default: treat as fixed value
        let val = parse_value(new_p);
        new_p = val.parser;
        dist = dist_fixed(val.value);
    }

    ParseDistResult {
        parser: new_p,
        dist: dist,
    }
}

// ============================================================================
// MODEL ELEMENT PARSERS
// ============================================================================

struct ParseParamResult {
    parser: Parser,
    param: ParamNode,
}

fn parse_param(p: Parser) -> ParseParamResult {
    var new_p = p;
    var param = param_node_new();

    // Skip 'param' keyword
    let m1 = parser_expect(new_p, TT_KW_PARAM());
    new_p = m1.parser;

    // Parameter name
    if parser_check(new_p, TT_IDENTIFIER()) {
        param.name = copy_token_text(new_p.tokens[new_p.pos as usize], param.name);
        new_p = parser_advance(new_p);
    }

    // Optional type annotation
    let m2 = parser_match(new_p, TT_COLON());
    if m2.matched {
        new_p = m2.parser;
        // Skip type keyword
        if parser_check(new_p, TT_KW_VOLUME()) || parser_check(new_p, TT_IDENTIFIER()) {
            new_p = parser_advance(new_p);
        }
    }

    // Distribution
    let m3 = parser_expect(new_p, TT_OP_TILDE());
    new_p = m3.parser;

    let dist_result = parse_distribution(new_p);
    new_p = dist_result.parser;
    param.distribution = dist_result.dist;
    param.typical_value = dist_result.dist.mu;

    ParseParamResult {
        parser: new_p,
        param: param,
    }
}

struct ParseCompartmentResult {
    parser: Parser,
    comp: CompartmentNode,
}

fn parse_compartment(p: Parser) -> ParseCompartmentResult {
    var new_p = p;
    var comp = compartment_node_new();

    // Skip 'compartment' keyword
    let m1 = parser_expect(new_p, TT_KW_COMPARTMENT());
    new_p = m1.parser;

    // Compartment name
    if parser_check(new_p, TT_IDENTIFIER()) {
        comp.name = copy_token_text(new_p.tokens[new_p.pos as usize], comp.name);

        // Check for depot/central by name (D=68, e=101, C=67)
        if comp.name[0] == 68 && comp.name[1] == 101 {
            comp.is_depot = true;
        }
        if comp.name[0] == 67 && comp.name[1] == 101 {
            comp.is_central = true;
        }

        new_p = parser_advance(new_p);
    }

    // Optional arguments
    let m2 = parser_match(new_p, TT_LPAREN());
    if m2.matched {
        new_p = m2.parser;

        // Parse volume: param_name
        if parser_check(new_p, TT_KW_VOLUME()) || parser_check(new_p, TT_IDENTIFIER()) {
            new_p = parser_advance(new_p);

            let m3 = parser_match(new_p, TT_COLON());
            if m3.matched {
                new_p = m3.parser;
                if parser_check(new_p, TT_IDENTIFIER()) {
                    comp.volume_param = copy_token_text(new_p.tokens[new_p.pos as usize], comp.volume_param);
                    new_p = parser_advance(new_p);
                }
            }
        }

        let m4 = parser_expect(new_p, TT_RPAREN());
        new_p = m4.parser;
    }

    ParseCompartmentResult {
        parser: new_p,
        comp: comp,
    }
}

struct ParseFlowResult {
    parser: Parser,
    flow: FlowNode,
}

fn parse_flow(p: Parser) -> ParseFlowResult {
    var new_p = p;
    var flow = flow_node_new();

    // Skip 'flow' keyword
    let m1 = parser_expect(new_p, TT_KW_FLOW());
    new_p = m1.parser;

    // Flow name
    if parser_check(new_p, TT_IDENTIFIER()) {
        flow.name = copy_token_text(new_p.tokens[new_p.pos as usize], flow.name);

        // Determine flow type from name (E=69, l=108, A=65, b=98)
        if flow.name[0] == 69 && flow.name[1] == 108 {
            flow.flow_type = FLOW_ELIMINATION();
        } else if flow.name[0] == 65 && flow.name[1] == 98 {
            flow.flow_type = FLOW_ABSORPTION();
        } else {
            flow.flow_type = FLOW_TRANSFER();
        }

        new_p = parser_advance(new_p);
    }

    // Expression (skip until arrow or newline or brace)
    let m2 = parser_expect(new_p, TT_OP_EQ());
    new_p = m2.parser;

    // Collect expression tokens (simplified - just skip until -> or } or newline)
    var expr_len: i64 = 0;
    while !parser_at_end(new_p) {
        let tt = parser_current_type(new_p);
        if tt == TT_OP_ARROW() || tt == TT_RBRACE() || tt == TT_NEWLINE() {
            break;
        }
        new_p = parser_advance(new_p);
        expr_len = expr_len + 1;
    }

    // Optional target compartment
    let m3 = parser_match(new_p, TT_OP_ARROW());
    if m3.matched {
        new_p = m3.parser;
        if parser_check(new_p, TT_IDENTIFIER()) {
            flow.to_compartment = copy_token_text(new_p.tokens[new_p.pos as usize], flow.to_compartment);
            new_p = parser_advance(new_p);
        }
    }

    ParseFlowResult {
        parser: new_p,
        flow: flow,
    }
}

struct ParseDoseResult {
    parser: Parser,
    dose: DosingNode,
}

fn parse_dose(p: Parser) -> ParseDoseResult {
    var new_p = p;
    var dose = dosing_node_new();

    // Skip 'dose' keyword
    let m1 = parser_expect(new_p, TT_KW_DOSE());
    new_p = m1.parser;

    // Route
    if parser_check(new_p, TT_KW_IV()) {
        dose.route = ROUTE_IV_BOLUS();
        new_p = parser_advance(new_p);
    } else if parser_check(new_p, TT_KW_ORAL()) {
        dose.route = ROUTE_ORAL();
        new_p = parser_advance(new_p);
    } else if parser_check(new_p, TT_KW_INFUSION()) {
        dose.route = ROUTE_IV_INFUSION();
        new_p = parser_advance(new_p);
    } else if parser_check(new_p, TT_IDENTIFIER()) {
        new_p = parser_advance(new_p);
    }

    // Arguments
    let m2 = parser_expect(new_p, TT_LPAREN());
    new_p = m2.parser;

    // Amount
    let amount = parse_value(new_p);
    new_p = amount.parser;
    dose.amount = amount.value;

    // Optional time
    let m3 = parser_match(new_p, TT_COMMA());
    if m3.matched {
        new_p = m3.parser;
        if parser_check(new_p, TT_KW_TIME()) {
            new_p = parser_advance(new_p);
            let m4 = parser_expect(new_p, TT_COLON());
            new_p = m4.parser;
            let time_val = parse_value(new_p);
            new_p = time_val.parser;
            dose.time = time_val.value;
        }
    }

    let m5 = parser_expect(new_p, TT_RPAREN());
    new_p = m5.parser;

    // Target compartment
    let m6 = parser_expect(new_p, TT_OP_ARROW());
    new_p = m6.parser;
    if parser_check(new_p, TT_IDENTIFIER()) {
        dose.target_compartment = copy_token_text(new_p.tokens[new_p.pos as usize], dose.target_compartment);
        new_p = parser_advance(new_p);
    }

    ParseDoseResult {
        parser: new_p,
        dose: dose,
    }
}

struct ParseObserveResult {
    parser: Parser,
    obs: ObservationNode,
}

fn parse_observe(p: Parser) -> ParseObserveResult {
    var new_p = p;
    var obs = observation_node_new();

    // Skip 'observe' keyword
    let m1 = parser_expect(new_p, TT_KW_OBSERVE());
    new_p = m1.parser;

    // Observation name
    if parser_check(new_p, TT_IDENTIFIER()) {
        obs.name = copy_token_text(new_p.tokens[new_p.pos as usize], obs.name);
        new_p = parser_advance(new_p);
    }

    // Expression (source)
    let m2 = parser_expect(new_p, TT_OP_EQ());
    new_p = m2.parser;

    // Parse compartment.C
    if parser_check(new_p, TT_IDENTIFIER()) {
        obs.source_compartment = copy_token_text(new_p.tokens[new_p.pos as usize], obs.source_compartment);
        new_p = parser_advance(new_p);

        // Skip .C or .amount
        let m3 = parser_match(new_p, TT_OP_DOT());
        if m3.matched {
            new_p = m3.parser;
            if parser_check(new_p, TT_IDENTIFIER()) {
                new_p = parser_advance(new_p);
            }
        }
    }

    // Error model
    let m4 = parser_expect(new_p, TT_KW_WITH());
    new_p = m4.parser;

    if parser_check(new_p, TT_KW_COMBINED()) {
        obs.error_type = ERROR_COMBINED();
        new_p = parser_advance(new_p);
    } else if parser_check(new_p, TT_KW_ADDITIVE()) {
        obs.error_type = ERROR_ADDITIVE();
        new_p = parser_advance(new_p);
    } else if parser_check(new_p, TT_KW_PROPORTIONAL()) {
        obs.error_type = ERROR_PROPORTIONAL();
        new_p = parser_advance(new_p);
    } else if parser_check(new_p, TT_IDENTIFIER()) {
        new_p = parser_advance(new_p);
    }

    // Error model arguments
    let m5 = parser_expect(new_p, TT_LPAREN());
    new_p = m5.parser;

    let sigma1 = parse_value(new_p);
    new_p = sigma1.parser;
    obs.sigma_add = sigma1.value;

    let m6 = parser_match(new_p, TT_COMMA());
    if m6.matched {
        new_p = m6.parser;
        let sigma2 = parse_value(new_p);
        new_p = sigma2.parser;
        obs.sigma_prop = sigma2.value;
    }

    let m7 = parser_expect(new_p, TT_RPAREN());
    new_p = m7.parser;

    ParseObserveResult {
        parser: new_p,
        obs: obs,
    }
}

// ============================================================================
// MODEL PARSER
// ============================================================================

struct ParseModelResult {
    parser: Parser,
    model: ModelNode,
}

fn parse_model(p: Parser) -> ParseModelResult {
    var new_p = p;
    var model = model_node_new();

    // Skip 'model' keyword
    let m1 = parser_expect(new_p, TT_KW_MODEL());
    new_p = m1.parser;

    // Model name
    if parser_check(new_p, TT_IDENTIFIER()) {
        model.name = copy_token_text(new_p.tokens[new_p.pos as usize], model.name);
        new_p = parser_advance(new_p);
    }

    // Model body
    let m2 = parser_expect(new_p, TT_LBRACE());
    new_p = m2.parser;

    while !parser_check(new_p, TT_RBRACE()) && !parser_at_end(new_p) && !new_p.has_error {
        if parser_check(new_p, TT_KW_PARAM()) {
            let result = parse_param(new_p);
            new_p = result.parser;
            if model.n_params < 30 {
                model.params[model.n_params as usize] = result.param;
                model.n_params = model.n_params + 1;
            }
        } else if parser_check(new_p, TT_KW_COMPARTMENT()) {
            let result = parse_compartment(new_p);
            new_p = result.parser;
            if model.n_compartments < 10 {
                model.compartments[model.n_compartments as usize] = result.comp;
                model.n_compartments = model.n_compartments + 1;
            }
        } else if parser_check(new_p, TT_KW_FLOW()) {
            let result = parse_flow(new_p);
            new_p = result.parser;
            if model.n_flows < 20 {
                model.flows[model.n_flows as usize] = result.flow;
                model.n_flows = model.n_flows + 1;
            }
        } else if parser_check(new_p, TT_KW_DOSE()) {
            let result = parse_dose(new_p);
            new_p = result.parser;
            if model.n_doses < 50 {
                model.doses[model.n_doses as usize] = result.dose;
                model.n_doses = model.n_doses + 1;
            }
        } else if parser_check(new_p, TT_KW_OBSERVE()) {
            let result = parse_observe(new_p);
            new_p = result.parser;
            if model.n_observations < 5 {
                model.observations[model.n_observations as usize] = result.obs;
                model.n_observations = model.n_observations + 1;
            }
        } else {
            new_p = parser_advance(new_p);
        }
    }

    let m3 = parser_expect(new_p, TT_RBRACE());
    new_p = m3.parser;

    ParseModelResult {
        parser: new_p,
        model: model,
    }
}

// ============================================================================
// FIT/SIMULATE PARSERS
// ============================================================================

struct ParseFitResult {
    parser: Parser,
    spec: FitSpec,
}

fn parse_fit(p: Parser) -> ParseFitResult {
    var new_p = p;
    var spec = fit_spec_new();

    // Skip 'fit' keyword
    let m1 = parser_expect(new_p, TT_KW_FIT());
    new_p = m1.parser;

    // Model name
    if parser_check(new_p, TT_IDENTIFIER()) {
        spec.model_name = copy_token_text(new_p.tokens[new_p.pos as usize], spec.model_name);
        new_p = parser_advance(new_p);
    }

    // 'to' data file
    let m2 = parser_expect(new_p, TT_KW_TO());
    new_p = m2.parser;

    if parser_check(new_p, TT_STRING()) {
        var i: i64 = 0;
        let tok = new_p.tokens[new_p.pos as usize];
        while i < tok.text_len && i < 255 {
            spec.data_file[i as usize] = tok.text[i as usize];
            i = i + 1;
        }
        new_p = parser_advance(new_p);
    }

    // Options block
    let m3 = parser_expect(new_p, TT_LBRACE());
    new_p = m3.parser;

    while !parser_check(new_p, TT_RBRACE()) && !parser_at_end(new_p) && !new_p.has_error {
        if parser_check(new_p, TT_KW_METHOD()) {
            new_p = parser_advance(new_p);
            let m4 = parser_expect(new_p, TT_COLON());
            new_p = m4.parser;

            if parser_check(new_p, TT_KW_LM()) {
                spec.method = FIT_LM();
                new_p = parser_advance(new_p);
            } else if parser_check(new_p, TT_IDENTIFIER()) {
                new_p = parser_advance(new_p);
            }
        } else if parser_check(new_p, TT_KW_UNCERTAINTY()) {
            new_p = parser_advance(new_p);
            let m5 = parser_expect(new_p, TT_COLON());
            new_p = m5.parser;

            if parser_check(new_p, TT_KW_GUM()) {
                spec.uncertainty_method = UNC_GUM();
                new_p = parser_advance(new_p);
            } else if parser_check(new_p, TT_KW_BOOTSTRAP()) {
                spec.uncertainty_method = UNC_BOOTSTRAP();
                new_p = parser_advance(new_p);
            } else if parser_check(new_p, TT_IDENTIFIER()) {
                new_p = parser_advance(new_p);
            }
        } else {
            new_p = parser_advance(new_p);
        }
    }

    let m6 = parser_expect(new_p, TT_RBRACE());
    new_p = m6.parser;

    ParseFitResult {
        parser: new_p,
        spec: spec,
    }
}

struct ParseSimResult {
    parser: Parser,
    spec: SimulateSpec,
}

fn parse_simulate(p: Parser) -> ParseSimResult {
    var new_p = p;
    var spec = simulate_spec_new();

    // Skip 'simulate' keyword
    let m1 = parser_expect(new_p, TT_KW_SIMULATE());
    new_p = m1.parser;

    // Model name
    if parser_check(new_p, TT_IDENTIFIER()) {
        spec.model_name = copy_token_text(new_p.tokens[new_p.pos as usize], spec.model_name);
        new_p = parser_advance(new_p);
    }

    // Options block
    let m2 = parser_expect(new_p, TT_LBRACE());
    new_p = m2.parser;

    while !parser_check(new_p, TT_RBRACE()) && !parser_at_end(new_p) && !new_p.has_error {
        if parser_check(new_p, TT_KW_TIME()) {
            new_p = parser_advance(new_p);
            let m3 = parser_expect(new_p, TT_COLON());
            new_p = m3.parser;

            let t1 = parse_value(new_p);
            new_p = t1.parser;
            spec.t_start = t1.value;

            let m4 = parser_match(new_p, TT_OP_DOTDOT());
            if m4.matched {
                new_p = m4.parser;
                let t2 = parse_value(new_p);
                new_p = t2.parser;
                spec.t_end = t2.value;
            }
        } else if parser_check(new_p, TT_KW_SUBJECTS()) {
            new_p = parser_advance(new_p);
            let m5 = parser_expect(new_p, TT_COLON());
            new_p = m5.parser;

            if parser_check(new_p, TT_INTEGER()) {
                spec.n_subjects = new_p.tokens[new_p.pos as usize].int_value;
                new_p = parser_advance(new_p);
            }
        } else {
            new_p = parser_advance(new_p);
        }
    }

    let m6 = parser_expect(new_p, TT_RBRACE());
    new_p = m6.parser;

    ParseSimResult {
        parser: new_p,
        spec: spec,
    }
}

// ============================================================================
// TOP-LEVEL PARSER
// ============================================================================

struct ParseProgramResult {
    parser: Parser,
    program: MedLangProgram,
}

fn parse_program(p: Parser) -> ParseProgramResult {
    var new_p = p;
    var program = medlang_program_new();

    while !parser_at_end(new_p) && !new_p.has_error {
        if parser_check(new_p, TT_KW_MODEL()) {
            let result = parse_model(new_p);
            new_p = result.parser;
            if program.n_models < 10 {
                program.models[program.n_models as usize] = result.model;
                program.n_models = program.n_models + 1;
            }
        } else if parser_check(new_p, TT_KW_FIT()) {
            let result = parse_fit(new_p);
            new_p = result.parser;
            if program.n_fits < 10 {
                program.fit_specs[program.n_fits as usize] = result.spec;
                program.n_fits = program.n_fits + 1;
            }
        } else if parser_check(new_p, TT_KW_SIMULATE()) {
            let result = parse_simulate(new_p);
            new_p = result.parser;
            if program.n_sims < 10 {
                program.sim_specs[program.n_sims as usize] = result.spec;
                program.n_sims = program.n_sims + 1;
            }
        } else {
            new_p = parser_advance(new_p);
        }
    }

    ParseProgramResult {
        parser: new_p,
        program: program,
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_parse_dist() -> bool {
    // Test distribution parsing constructs
    let d1 = dist_fixed(10.0);
    if d1.dist_type != DIST_FIXED() { return false; }
    if d1.mu != 10.0 { return false; }

    let d2 = dist_lognormal(10.0, 0.3);
    if d2.dist_type != DIST_LOGNORMAL() { return false; }
    if d2.omega != 0.3 { return false; }

    let d3 = dist_normal(0.0, 1.0);
    if d3.dist_type != DIST_NORMAL() { return false; }

    true
}

fn test_parse_nodes() -> bool {
    // Test AST node construction
    let p = param_node_new();
    if p.index != -1 { return false; }

    let c = compartment_node_new();
    if c.is_depot { return false; }

    let f = flow_node_new();
    if f.flow_type != FLOW_ELIMINATION() { return false; }

    let d = dosing_node_new();
    if d.bioavailability != 1.0 { return false; }

    let o = observation_node_new();
    if o.error_type != ERROR_COMBINED() { return false; }

    true
}

fn test_parse_model_struct() -> bool {
    let m = model_node_new();
    if m.n_params != 0 { return false; }
    if m.n_compartments != 0 { return false; }
    if m.n_flows != 0 { return false; }
    true
}

fn test_parse_program_struct() -> bool {
    let prog = medlang_program_new();
    if prog.n_models != 0 { return false; }
    if prog.n_fits != 0 { return false; }
    if prog.n_sims != 0 { return false; }
    true
}

fn main() -> i32 {
    print("Testing medlang::parser module...\n");

    if !test_parse_dist() {
        print("FAIL: parse_dist\n");
        return 1;
    }
    print("PASS: parse_dist\n");

    if !test_parse_nodes() {
        print("FAIL: parse_nodes\n");
        return 2;
    }
    print("PASS: parse_nodes\n");

    if !test_parse_model_struct() {
        print("FAIL: parse_model_struct\n");
        return 3;
    }
    print("PASS: parse_model_struct\n");

    if !test_parse_program_struct() {
        print("FAIL: parse_program_struct\n");
        return 4;
    }
    print("PASS: parse_program_struct\n");

    print("All medlang::parser tests PASSED\n");
    0
}
