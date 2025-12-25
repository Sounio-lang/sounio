// medlang::parser_ext — Extended Parser for PBPK and Quantum Grammar
//
// Extends the base MedLang parser with:
// - tissue {...} blocks for PBPK models
// - binding X ~ Quantum(...) specifications
// - elimination kinetics (MichaelisMenten, Hill)
// - protein binding specifications
// - covariate effects
//
// Grammar extensions (EBNF):
//
//   tissue_def    ::= 'tissue' IDENT '{' tissue_body '}'
//   binding_def   ::= 'binding' IDENT '~' quantum_spec
//   elimination   ::= 'elimination' IDENT '{' elim_body '}'
//   covariate_def ::= 'covariate' IDENT 'on' IDENT ':' covariate_expr

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn fabs(x: f64) -> f64;
}

// ============================================================================
// TOKEN TYPE EXTENSIONS (additional keywords for PBPK/Quantum)
// ============================================================================

// Extended keywords (starting at 200 to avoid collision with base lexer)
fn TT_KW_TISSUE() -> i32 { 200 }
fn TT_KW_BINDING() -> i32 { 201 }
fn TT_KW_COVARIATE() -> i32 { 202 }
fn TT_KW_QUANTUM() -> i32 { 203 }

// Tissue types
fn TT_KW_PLASMA() -> i32 { 210 }
fn TT_KW_LIVER() -> i32 { 211 }
fn TT_KW_KIDNEY() -> i32 { 212 }
fn TT_KW_BRAIN() -> i32 { 213 }
fn TT_KW_ADIPOSE() -> i32 { 214 }
fn TT_KW_MUSCLE() -> i32 { 215 }
fn TT_KW_GUT() -> i32 { 216 }
fn TT_KW_LUNG() -> i32 { 217 }
fn TT_KW_HEART() -> i32 { 218 }
fn TT_KW_SPLEEN() -> i32 { 219 }
fn TT_KW_SKIN() -> i32 { 220 }
fn TT_KW_BONE() -> i32 { 221 }
fn TT_KW_REST() -> i32 { 222 }
fn TT_KW_VENOUS() -> i32 { 223 }
fn TT_KW_ARTERIAL() -> i32 { 224 }

// Kinetics keywords
fn TT_KW_MICHAELIS_MENTEN() -> i32 { 230 }
fn TT_KW_HILL() -> i32 { 231 }
fn TT_KW_VMAX() -> i32 { 232 }
fn TT_KW_KM() -> i32 { 233 }
fn TT_KW_CLINT() -> i32 { 234 }
fn TT_KW_METABOLISM() -> i32 { 235 }
fn TT_KW_BLOOD_FLOW() -> i32 { 236 }
fn TT_KW_PERMEABILITY() -> i32 { 237 }
fn TT_KW_PS() -> i32 { 238 }
fn TT_KW_BBB() -> i32 { 239 }
fn TT_KW_KP() -> i32 { 240 }

// QM method keywords
fn TT_KW_HF() -> i32 { 250 }
fn TT_KW_DFT() -> i32 { 251 }
fn TT_KW_MP2() -> i32 { 252 }
fn TT_KW_CCSD() -> i32 { 253 }
fn TT_KW_DLPNO() -> i32 { 254 }

// DFT functional keywords
fn TT_KW_B3LYP() -> i32 { 260 }
fn TT_KW_PBE0() -> i32 { 261 }
fn TT_KW_M062X() -> i32 { 262 }
fn TT_KW_WB97XD() -> i32 { 263 }
fn TT_KW_TPSS() -> i32 { 264 }

// Basis set keywords
fn TT_KW_CC_PVDZ() -> i32 { 270 }
fn TT_KW_CC_PVTZ() -> i32 { 271 }
fn TT_KW_DEF2_SVP() -> i32 { 272 }
fn TT_KW_DEF2_TZVP() -> i32 { 273 }

// Solvent model keywords
fn TT_KW_SMD() -> i32 { 280 }
fn TT_KW_CPCM() -> i32 { 281 }

// Dispersion keywords
fn TT_KW_D3() -> i32 { 285 }
fn TT_KW_D3BJ() -> i32 { 286 }
fn TT_KW_D4() -> i32 { 287 }

// Other QM keywords
fn TT_KW_FUNCTIONAL() -> i32 { 290 }
fn TT_KW_BASIS() -> i32 { 291 }
fn TT_KW_SOLVENT() -> i32 { 292 }
fn TT_KW_DISPERSION() -> i32 { 293 }
fn TT_KW_RECEPTOR() -> i32 { 294 }
fn TT_KW_LIGAND() -> i32 { 295 }
fn TT_KW_COMPLEX() -> i32 { 296 }
fn TT_KW_BSSE() -> i32 { 297 }
fn TT_KW_THERMAL() -> i32 { 298 }

// Covariate keywords
fn TT_KW_POWER() -> i32 { 300 }
fn TT_KW_EXPONENTIAL() -> i32 { 301 }
fn TT_KW_LINEAR_COV() -> i32 { 302 }
fn TT_KW_CATEGORICAL() -> i32 { 303 }

// ============================================================================
// UNIT TYPES
// ============================================================================

/// Unit type for dimensional analysis
enum UnitType {
    Dimensionless,
    L,
    L_per_h,
    Mg,
    Mg_per_L,
    Per_h,
    H,
}

// ============================================================================
// DISTRIBUTION TYPES
// ============================================================================

/// Distribution type for stochastic parameters
enum DistType {
    Normal,
    LogNormal,
    Uniform,
    Beta,
    Gamma,
}

/// Distribution specification
struct DistributionSpec {
    dist_type: DistType,
    param1: f64,
    param2: f64,
    has_omega: bool,
    omega: f64,
    lower_bound: f64,
    upper_bound: f64,
    has_bounds: bool,
}

fn distribution_spec_new() -> DistributionSpec {
    DistributionSpec {
        dist_type: DistType::Normal,
        param1: 0.0,
        param2: 1.0,
        has_omega: false,
        omega: 0.0,
        lower_bound: 0.0,
        upper_bound: 0.0,
        has_bounds: false,
    }
}

// ============================================================================
// EXTENDED AST NODES
// ============================================================================

/// Tissue type enumeration
enum TissueTypeAST {
    Plasma,
    Venous,
    Arterial,
    Lung,
    Heart,
    Brain,
    Adipose,
    Bone,
    Gut,
    Spleen,
    Liver,
    Kidney,
    Muscle,
    Skin,
    Rest,
    Custom,
}

/// Kinetics type for elimination
enum KineticsType {
    Linear,
    MichaelisMenten,
    Hill,
    Biphasic,
}

/// Tissue definition AST node
struct TissueDefNode {
    name: [i8; 64],
    tissue_type: TissueTypeAST,

    // Volume specification
    has_volume: bool,
    volume_value: f64,
    volume_unit: UnitType,
    volume_is_stochastic: bool,
    volume_dist: DistributionSpec,

    // Blood flow specification
    has_blood_flow: bool,
    blood_flow_value: f64,
    blood_flow_unit: UnitType,
    blood_flow_is_stochastic: bool,
    blood_flow_dist: DistributionSpec,

    // Partition coefficient
    has_kp: bool,
    kp_value: f64,
    kp_is_stochastic: bool,
    kp_dist: DistributionSpec,
    kp_method: i32,     // 0=fixed, 1=Poulin-Theil, 2=Rodgers-Rowland

    // Permeability
    has_permeability: bool,
    ps_value: f64,
    is_bbb: bool,

    // Elimination in this tissue
    has_elimination: bool,
    kinetics_type: KineticsType,
    vmax: f64,
    km: f64,
    hill_n: f64,
    cl_int: f64,

    // Renal-specific
    is_renal: bool,
    gfr_fraction: f64,
    secretion_cl: f64,
    reabsorption: f64,

    // Line info
    line: i64,
    col: i64,
}

fn tissue_def_node_new() -> TissueDefNode {
    TissueDefNode {
        name: [0i8; 64],
        tissue_type: TissueTypeAST::Custom,
        has_volume: false,
        volume_value: 1.0,
        volume_unit: UnitType::L,
        volume_is_stochastic: false,
        volume_dist: distribution_spec_new(),
        has_blood_flow: false,
        blood_flow_value: 1.0,
        blood_flow_unit: UnitType::L_per_h,
        blood_flow_is_stochastic: false,
        blood_flow_dist: distribution_spec_new(),
        has_kp: false,
        kp_value: 1.0,
        kp_is_stochastic: false,
        kp_dist: distribution_spec_new(),
        kp_method: 0,
        has_permeability: false,
        ps_value: 0.0,
        is_bbb: false,
        has_elimination: false,
        kinetics_type: KineticsType::Linear,
        vmax: 0.0,
        km: 1.0,
        hill_n: 1.0,
        cl_int: 0.0,
        is_renal: false,
        gfr_fraction: 0.0,
        secretion_cl: 0.0,
        reabsorption: 0.0,
        line: 0,
        col: 0,
    }
}

/// QM method for binding
enum QMMethodAST {
    HF,
    DFT,
    MP2,
    CCSD,
    DLPNO_CCSD,
    DLPNO_CCSD_T,
    SemiEmpirical,
}

/// DFT functional
enum FunctionalAST {
    B3LYP,
    PBE0,
    M062X,
    wB97XD,
    TPSS,
}

/// Basis set
enum BasisSetAST {
    Basis_6_31G_d,
    cc_pVDZ,
    cc_pVTZ,
    def2_SVP,
    def2_TZVP,
}

/// Solvent model
enum SolventModelAST {
    None,
    CPCM,
    SMD,
}

/// Quantum specification AST node
struct QuantumSpecNode {
    method: QMMethodAST,
    functional: FunctionalAST,
    basis: BasisSetAST,
    solvent_model: SolventModelAST,
    solvent_name: [i8; 32],

    // Dispersion
    dispersion: i32,        // 0=none, 1=D3, 2=D3BJ, 3=D4

    // Files
    receptor_file: [i8; 256],
    ligand_file: [i8; 256],
    complex_file: [i8; 256],

    // Options
    include_bsse: bool,
    include_thermal: bool,
    temperature: f64,

    // Line info
    line: i64,
    col: i64,
}

fn quantum_spec_node_new() -> QuantumSpecNode {
    QuantumSpecNode {
        method: QMMethodAST::DFT,
        functional: FunctionalAST::M062X,
        basis: BasisSetAST::cc_pVTZ,
        solvent_model: SolventModelAST::SMD,
        solvent_name: [0i8; 32],
        dispersion: 2,
        receptor_file: [0i8; 256],
        ligand_file: [0i8; 256],
        complex_file: [0i8; 256],
        include_bsse: true,
        include_thermal: false,
        temperature: 298.15,
        line: 0,
        col: 0,
    }
}

/// Binding definition AST node
struct BindingDefNode {
    name: [i8; 64],         // e.g., "Kd", "Ki", "IC50"
    qm_spec: QuantumSpecNode,
    line: i64,
    col: i64,
}

fn binding_def_node_new() -> BindingDefNode {
    BindingDefNode {
        name: [0i8; 64],
        qm_spec: quantum_spec_node_new(),
        line: 0,
        col: 0,
    }
}

/// Protein binding specification
struct ProteinBindingNode {
    has_fu_plasma: bool,
    fu_plasma: f64,
    fu_plasma_is_stochastic: bool,
    fu_plasma_dist: DistributionSpec,

    has_fu_tissue: bool,
    fu_tissue: f64,

    albumin_conc: f64,
    agp_conc: f64,

    is_saturable: bool,
    bmax: f64,
    kd: f64,

    line: i64,
    col: i64,
}

fn protein_binding_node_new() -> ProteinBindingNode {
    ProteinBindingNode {
        has_fu_plasma: false,
        fu_plasma: 1.0,
        fu_plasma_is_stochastic: false,
        fu_plasma_dist: distribution_spec_new(),
        has_fu_tissue: false,
        fu_tissue: 1.0,
        albumin_conc: 40.0,
        agp_conc: 0.7,
        is_saturable: false,
        bmax: 0.0,
        kd: 0.0,
        line: 0,
        col: 0,
    }
}

/// Covariate effect type
enum CovariateEffectType {
    Power,          // CL * (COV/ref)^exp
    Exponential,    // CL * exp(theta * COV)
    Linear,         // CL * (1 + theta * (COV - ref))
    Categorical,    // CL * theta[category]
}

/// Covariate definition
struct CovariateDefNode {
    covariate_name: [i8; 64],   // e.g., "WT", "AGE", "SEX"
    param_name: [i8; 64],       // Parameter affected
    effect_type: CovariateEffectType,
    reference_value: f64,
    exponent: f64,              // For power model
    theta: f64,                 // Effect coefficient
    theta_is_stochastic: bool,
    theta_dist: DistributionSpec,
    line: i64,
    col: i64,
}

fn covariate_def_node_new() -> CovariateDefNode {
    CovariateDefNode {
        covariate_name: [0i8; 64],
        param_name: [0i8; 64],
        effect_type: CovariateEffectType::Power,
        reference_value: 70.0,
        exponent: 0.75,
        theta: 0.0,
        theta_is_stochastic: false,
        theta_dist: distribution_spec_new(),
        line: 0,
        col: 0,
    }
}

// ============================================================================
// EXTENDED TOKEN
// ============================================================================

/// Extended token for parser_ext
struct ExtToken {
    token_type: i32,
    text: [i8; 64],
    value: f64,
    line: i64,
    col: i64,
}

fn ext_token_new() -> ExtToken {
    ExtToken {
        token_type: 0,
        text: [0i8; 64],
        value: 0.0,
        line: 0,
        col: 0,
    }
}

// ============================================================================
// EXTENDED MODEL NODE
// ============================================================================

/// Extended model with PBPK and quantum binding
struct ExtendedModelNode {
    // Base model info
    name: [i8; 64],

    // PBPK tissues
    tissues: [TissueDefNode; 20],
    n_tissues: i64,

    // Quantum bindings
    bindings: [BindingDefNode; 5],
    n_bindings: i64,

    // Protein binding
    has_protein_binding: bool,
    protein_binding: ProteinBindingNode,

    // Covariates
    covariates: [CovariateDefNode; 20],
    n_covariates: i64,

    // Special features
    has_enterohepatic: bool,
    ehc_fraction: f64,
    ehc_delay: f64,

    // Model type
    is_pbpk: bool,
    is_popPK: bool,

    // Line info
    line: i64,
    col: i64,
}

fn extended_model_node_new() -> ExtendedModelNode {
    ExtendedModelNode {
        name: [0i8; 64],
        tissues: [tissue_def_node_new(); 20],
        n_tissues: 0,
        bindings: [binding_def_node_new(); 5],
        n_bindings: 0,
        has_protein_binding: false,
        protein_binding: protein_binding_node_new(),
        covariates: [covariate_def_node_new(); 20],
        n_covariates: 0,
        has_enterohepatic: false,
        ehc_fraction: 0.0,
        ehc_delay: 0.0,
        is_pbpk: false,
        is_popPK: true,
        line: 0,
        col: 0,
    }
}

// ============================================================================
// PARSER STATE
// ============================================================================

/// Parser state for extended grammar
struct ExtendedParser {
    tokens: [ExtToken; 10000],
    n_tokens: i64,
    pos: i64,

    // Current model being built
    current_model: ExtendedModelNode,

    // Error handling
    has_error: bool,
    error_msg: [i8; 256],
    error_line: i64,
    error_col: i64,
}

fn extended_parser_new() -> ExtendedParser {
    ExtendedParser {
        tokens: [ext_token_new(); 10000],
        n_tokens: 0,
        pos: 0,
        current_model: extended_model_node_new(),
        has_error: false,
        error_msg: [0i8; 256],
        error_line: 0,
        error_col: 0,
    }
}

fn parser_error(parser: &!ExtendedParser, msg: &[u8], line: i64, col: i64) {
    parser.has_error = true;
    parser.error_line = line;
    parser.error_col = col;

    var i: i64 = 0;
    while i < 256 && i < msg.len() as i64 && msg[i as usize] != 0 {
        parser.error_msg[i as usize] = msg[i as usize] as i8;
        i = i + 1;
    }
}

// ============================================================================
// TOKEN HELPERS
// ============================================================================

fn current_token(parser: &ExtendedParser) -> &ExtToken {
    if parser.pos < parser.n_tokens {
        &parser.tokens[parser.pos as usize]
    } else {
        &parser.tokens[0]
    }
}

fn advance(parser: &!ExtendedParser) {
    if parser.pos < parser.n_tokens {
        parser.pos = parser.pos + 1;
    }
}

fn expect(parser: &!ExtendedParser, tt: i32) -> bool {
    let tok = current_token(parser);
    if tok.token_type == tt {
        advance(parser);
        true
    } else {
        let msg = "Unexpected token\0";
        parser_error(parser, msg.as_bytes(), tok.line, tok.col);
        false
    }
}

fn match_token(parser: &ExtendedParser, tt: i32) -> bool {
    current_token(parser).token_type == tt
}

fn peek_token(parser: &ExtendedParser, offset: i64) -> &ExtToken {
    let idx = parser.pos + offset;
    if idx < parser.n_tokens && idx >= 0 {
        &parser.tokens[idx as usize]
    } else {
        &parser.tokens[0]
    }
}

// ============================================================================
// TISSUE PARSING
// ============================================================================

/// Parse tissue type from keyword
fn parse_tissue_type(parser: &!ExtendedParser) -> TissueTypeAST {
    let tok = current_token(parser);

    if tok.token_type == TT_KW_PLASMA() {
        advance(parser);
        TissueTypeAST::Plasma
    } else if tok.token_type == TT_KW_LIVER() {
        advance(parser);
        TissueTypeAST::Liver
    } else if tok.token_type == TT_KW_KIDNEY() {
        advance(parser);
        TissueTypeAST::Kidney
    } else if tok.token_type == TT_KW_BRAIN() {
        advance(parser);
        TissueTypeAST::Brain
    } else if tok.token_type == TT_KW_ADIPOSE() {
        advance(parser);
        TissueTypeAST::Adipose
    } else if tok.token_type == TT_KW_MUSCLE() {
        advance(parser);
        TissueTypeAST::Muscle
    } else if tok.token_type == TT_KW_GUT() {
        advance(parser);
        TissueTypeAST::Gut
    } else if tok.token_type == TT_KW_LUNG() {
        advance(parser);
        TissueTypeAST::Lung
    } else if tok.token_type == TT_KW_HEART() {
        advance(parser);
        TissueTypeAST::Heart
    } else if tok.token_type == TT_KW_SPLEEN() {
        advance(parser);
        TissueTypeAST::Spleen
    } else if tok.token_type == TT_KW_SKIN() {
        advance(parser);
        TissueTypeAST::Skin
    } else if tok.token_type == TT_KW_BONE() {
        advance(parser);
        TissueTypeAST::Bone
    } else if tok.token_type == TT_KW_REST() {
        advance(parser);
        TissueTypeAST::Rest
    } else {
        TissueTypeAST::Custom
    }
}

/// Parse kinetics specification
fn parse_kinetics(parser: &!ExtendedParser, tissue: &!TissueDefNode) {
    let tok = current_token(parser);

    if tok.token_type == TT_KW_MICHAELIS_MENTEN() {
        advance(parser);
        tissue.kinetics_type = KineticsType::MichaelisMenten;

        // Expect (Vmax: value, Km: value)
        if !expect(parser, 120) { return; }  // LPAREN

        // Parse Vmax
        if match_token(parser, TT_KW_VMAX()) {
            advance(parser);
            if !expect(parser, 127) { return; }  // COLON
            let vmax_tok = current_token(parser);
            if vmax_tok.token_type == 82 {  // FLOAT
                tissue.vmax = vmax_tok.value;
                advance(parser);
            }
        }

        // Comma
        if match_token(parser, 126) {  // COMMA
            advance(parser);
        }

        // Parse Km
        if match_token(parser, TT_KW_KM()) {
            advance(parser);
            if !expect(parser, 127) { return; }  // COLON
            let km_tok = current_token(parser);
            if km_tok.token_type == 82 {  // FLOAT
                tissue.km = km_tok.value;
                advance(parser);
            }
        }

        let _ = expect(parser, 121);  // RPAREN

    } else if tok.token_type == TT_KW_HILL() {
        advance(parser);
        tissue.kinetics_type = KineticsType::Hill;

        // Expect (Vmax: value, Km: value, n: value)
        if !expect(parser, 120) { return; }  // LPAREN

        // Parse Vmax
        if match_token(parser, TT_KW_VMAX()) {
            advance(parser);
            if !expect(parser, 127) { return; }  // COLON
            let vmax_tok = current_token(parser);
            if vmax_tok.token_type == 82 {  // FLOAT
                tissue.vmax = vmax_tok.value;
                advance(parser);
            }
        }

        if match_token(parser, 126) { advance(parser); }  // COMMA

        // Parse Km
        if match_token(parser, TT_KW_KM()) {
            advance(parser);
            if !expect(parser, 127) { return; }  // COLON
            let km_tok = current_token(parser);
            if km_tok.token_type == 82 {  // FLOAT
                tissue.km = km_tok.value;
                advance(parser);
            }
        }

        if match_token(parser, 126) { advance(parser); }  // COMMA

        // Parse n (Hill coefficient)
        let n_tok = current_token(parser);
        if n_tok.text[0] == 110 {  // 'n'
            advance(parser);
            if !expect(parser, 127) { return; }  // COLON
            let val_tok = current_token(parser);
            if val_tok.token_type == 82 {  // FLOAT
                tissue.hill_n = val_tok.value;
                advance(parser);
            }
        }

        let _ = expect(parser, 121);  // RPAREN

    } else {
        // Linear: CLint: value
        tissue.kinetics_type = KineticsType::Linear;
        if match_token(parser, TT_KW_CLINT()) {
            advance(parser);
            if !expect(parser, 127) { return; }  // COLON
            let cl_tok = current_token(parser);
            if cl_tok.token_type == 82 {  // FLOAT
                tissue.cl_int = cl_tok.value;
                advance(parser);
            }
        }
    }
}

/// Parse tissue body
fn parse_tissue_body(parser: &!ExtendedParser, tissue: &!TissueDefNode) {
    // Expect {
    if !expect(parser, 122) { return; }  // LBRACE

    while !match_token(parser, 123) && !parser.has_error {  // RBRACE
        let tok = current_token(parser);

        // volume: value unit
        if tok.text[0] == 118 && tok.text[1] == 111 {  // "vo" for volume
            advance(parser);
            if !expect(parser, 127) { return; }  // COLON

            let val_tok = current_token(parser);
            if val_tok.token_type == 82 {  // FLOAT
                tissue.volume_value = val_tok.value;
                tissue.has_volume = true;
                advance(parser);

                // Check for unit
                let unit_tok = current_token(parser);
                if unit_tok.token_type == 91 {  // UNIT_L
                    tissue.volume_unit = UnitType::L;
                    advance(parser);
                }
            }
        }

        // blood_flow: value unit
        else if tok.token_type == TT_KW_BLOOD_FLOW() {
            advance(parser);
            if !expect(parser, 127) { return; }  // COLON

            let val_tok = current_token(parser);
            if val_tok.token_type == 82 {  // FLOAT
                tissue.blood_flow_value = val_tok.value;
                tissue.has_blood_flow = true;
                advance(parser);

                // Check for unit
                let unit_tok = current_token(parser);
                if unit_tok.token_type == 93 {  // UNIT_L_PER_H
                    tissue.blood_flow_unit = UnitType::L_per_h;
                    advance(parser);
                }
            }
        }

        // Kp: value or Kp ~ Distribution
        else if tok.token_type == TT_KW_KP() {
            advance(parser);
            tissue.has_kp = true;

            let next = current_token(parser);
            if next.token_type == 127 {  // COLON
                advance(parser);
                let val_tok = current_token(parser);
                if val_tok.token_type == 82 {  // FLOAT
                    tissue.kp_value = val_tok.value;
                    advance(parser);
                }
            } else if next.token_type == 106 {  // TILDE
                advance(parser);
                tissue.kp_is_stochastic = true;
                tissue.kp_dist = parse_distribution(parser);
            }
        }

        // metabolism: kinetics_spec
        else if tok.token_type == TT_KW_METABOLISM() {
            advance(parser);
            if !expect(parser, 127) { return; }  // COLON
            tissue.has_elimination = true;
            parse_kinetics(parser, tissue);
        }

        // permeability: PS(value)
        else if tok.token_type == TT_KW_PERMEABILITY() {
            advance(parser);
            if !expect(parser, 127) { return; }  // COLON
            tissue.has_permeability = true;

            if match_token(parser, TT_KW_PS()) {
                advance(parser);
                if !expect(parser, 120) { return; }  // LPAREN
                let val_tok = current_token(parser);
                if val_tok.token_type == 82 {  // FLOAT
                    tissue.ps_value = val_tok.value;
                    advance(parser);
                }
                let _ = expect(parser, 121);  // RPAREN
            }
        }

        // BBB marker
        else if tok.token_type == TT_KW_BBB() {
            advance(parser);
            tissue.is_bbb = true;
        }

        // Unknown token - skip
        else {
            advance(parser);
        }

        // Optional newline
        while match_token(parser, 129) {  // NEWLINE
            advance(parser);
        }
    }

    let _ = expect(parser, 123);  // RBRACE
}

/// Parse tissue definition
fn parse_tissue_def(parser: &!ExtendedParser) -> TissueDefNode {
    var tissue = tissue_def_node_new();

    let start = current_token(parser);
    tissue.line = start.line;
    tissue.col = start.col;

    // Expect 'tissue'
    if !expect(parser, TT_KW_TISSUE()) {
        return tissue;
    }

    // Get tissue name (identifier or predefined type)
    let name_tok = current_token(parser);

    // Check if it's a predefined tissue type
    tissue.tissue_type = parse_tissue_type(parser);

    // If custom, get identifier name
    if tissue.tissue_type == TissueTypeAST::Custom {
        if name_tok.token_type == 80 {  // IDENTIFIER
            var i: i64 = 0;
            while i < 64 && name_tok.text[i as usize] != 0 {
                tissue.name[i as usize] = name_tok.text[i as usize];
                i = i + 1;
            }
            advance(parser);
        }
    } else {
        // Copy predefined name
        var i: i64 = 0;
        while i < 64 && name_tok.text[i as usize] != 0 {
            tissue.name[i as usize] = name_tok.text[i as usize];
            i = i + 1;
        }
    }

    // Parse body
    parse_tissue_body(parser, &!tissue);

    tissue
}

// ============================================================================
// QUANTUM BINDING PARSING
// ============================================================================

/// Parse QM method
fn parse_qm_method(parser: &!ExtendedParser) -> QMMethodAST {
    let tok = current_token(parser);

    if tok.token_type == TT_KW_HF() {
        advance(parser);
        QMMethodAST::HF
    } else if tok.token_type == TT_KW_DFT() {
        advance(parser);
        QMMethodAST::DFT
    } else if tok.token_type == TT_KW_MP2() {
        advance(parser);
        QMMethodAST::MP2
    } else if tok.token_type == TT_KW_CCSD() {
        advance(parser);
        QMMethodAST::CCSD
    } else if tok.token_type == TT_KW_DLPNO() {
        advance(parser);
        QMMethodAST::DLPNO_CCSD
    } else {
        QMMethodAST::DFT
    }
}

/// Parse DFT functional
fn parse_functional(parser: &!ExtendedParser) -> FunctionalAST {
    let tok = current_token(parser);

    if tok.token_type == TT_KW_B3LYP() {
        advance(parser);
        FunctionalAST::B3LYP
    } else if tok.token_type == TT_KW_PBE0() {
        advance(parser);
        FunctionalAST::PBE0
    } else if tok.token_type == TT_KW_M062X() {
        advance(parser);
        FunctionalAST::M062X
    } else if tok.token_type == TT_KW_WB97XD() {
        advance(parser);
        FunctionalAST::wB97XD
    } else if tok.token_type == TT_KW_TPSS() {
        advance(parser);
        FunctionalAST::TPSS
    } else {
        FunctionalAST::B3LYP
    }
}

/// Parse basis set
fn parse_basis(parser: &!ExtendedParser) -> BasisSetAST {
    let tok = current_token(parser);

    if tok.token_type == TT_KW_CC_PVDZ() {
        advance(parser);
        BasisSetAST::cc_pVDZ
    } else if tok.token_type == TT_KW_CC_PVTZ() {
        advance(parser);
        BasisSetAST::cc_pVTZ
    } else if tok.token_type == TT_KW_DEF2_SVP() {
        advance(parser);
        BasisSetAST::def2_SVP
    } else if tok.token_type == TT_KW_DEF2_TZVP() {
        advance(parser);
        BasisSetAST::def2_TZVP
    } else {
        advance(parser);
        BasisSetAST::Basis_6_31G_d
    }
}

/// Parse quantum specification
fn parse_quantum_spec(parser: &!ExtendedParser) -> QuantumSpecNode {
    var spec = quantum_spec_node_new();

    let start = current_token(parser);
    spec.line = start.line;
    spec.col = start.col;

    // Expect 'Quantum'
    if !expect(parser, TT_KW_QUANTUM()) {
        return spec;
    }

    // Expect (
    if !expect(parser, 120) {  // LPAREN
        return spec;
    }

    // Parse options
    while !match_token(parser, 121) && !parser.has_error {  // RPAREN
        let tok = current_token(parser);

        // method: QMMethod
        if tok.text[0] == 109 && tok.text[1] == 101 {  // "me" for method
            advance(parser);
            let _ = expect(parser, 127);  // COLON
            spec.method = parse_qm_method(parser);
        }

        // functional: Functional
        else if tok.token_type == TT_KW_FUNCTIONAL() {
            advance(parser);
            let _ = expect(parser, 127);  // COLON
            spec.functional = parse_functional(parser);
        }

        // basis: BasisSet
        else if tok.token_type == TT_KW_BASIS() {
            advance(parser);
            let _ = expect(parser, 127);  // COLON
            spec.basis = parse_basis(parser);
        }

        // solvent: Model(solvent)
        else if tok.token_type == TT_KW_SOLVENT() {
            advance(parser);
            let _ = expect(parser, 127);  // COLON

            let model_tok = current_token(parser);
            if model_tok.token_type == TT_KW_SMD() {
                spec.solvent_model = SolventModelAST::SMD;
                advance(parser);
            } else if model_tok.token_type == TT_KW_CPCM() {
                spec.solvent_model = SolventModelAST::CPCM;
                advance(parser);
            }

            // Parse (solvent_name)
            if match_token(parser, 120) {  // LPAREN
                advance(parser);
                let name_tok = current_token(parser);
                var i: i64 = 0;
                while i < 32 && name_tok.text[i as usize] != 0 {
                    spec.solvent_name[i as usize] = name_tok.text[i as usize];
                    i = i + 1;
                }
                advance(parser);
                let _ = expect(parser, 121);  // RPAREN
            }
        }

        // dispersion: D3/D3BJ/D4
        else if tok.token_type == TT_KW_DISPERSION() {
            advance(parser);
            let _ = expect(parser, 127);  // COLON

            let disp_tok = current_token(parser);
            if disp_tok.token_type == TT_KW_D3() {
                spec.dispersion = 1;
                advance(parser);
            } else if disp_tok.token_type == TT_KW_D3BJ() {
                spec.dispersion = 2;
                advance(parser);
            } else if disp_tok.token_type == TT_KW_D4() {
                spec.dispersion = 3;
                advance(parser);
            }
        }

        // receptor: "file.pdb"
        else if tok.token_type == TT_KW_RECEPTOR() {
            advance(parser);
            let _ = expect(parser, 127);  // COLON
            let file_tok = current_token(parser);
            if file_tok.token_type == 83 {  // STRING
                var i: i64 = 0;
                while i < 256 && file_tok.text[i as usize] != 0 {
                    spec.receptor_file[i as usize] = file_tok.text[i as usize];
                    i = i + 1;
                }
                advance(parser);
            }
        }

        // ligand: "file.xyz"
        else if tok.token_type == TT_KW_LIGAND() {
            advance(parser);
            let _ = expect(parser, 127);  // COLON
            let file_tok = current_token(parser);
            if file_tok.token_type == 83 {  // STRING
                var i: i64 = 0;
                while i < 256 && file_tok.text[i as usize] != 0 {
                    spec.ligand_file[i as usize] = file_tok.text[i as usize];
                    i = i + 1;
                }
                advance(parser);
            }
        }

        // complex: "file.pdb"
        else if tok.token_type == TT_KW_COMPLEX() {
            advance(parser);
            let _ = expect(parser, 127);  // COLON
            let file_tok = current_token(parser);
            if file_tok.token_type == 83 {  // STRING
                var i: i64 = 0;
                while i < 256 && file_tok.text[i as usize] != 0 {
                    spec.complex_file[i as usize] = file_tok.text[i as usize];
                    i = i + 1;
                }
                advance(parser);
            }
        }

        // bsse: true/false
        else if tok.token_type == TT_KW_BSSE() {
            advance(parser);
            let _ = expect(parser, 127);  // COLON
            let val_tok = current_token(parser);
            spec.include_bsse = val_tok.text[0] == 116;  // 't' for true
            advance(parser);
        }

        // thermal: true/false
        else if tok.token_type == TT_KW_THERMAL() {
            advance(parser);
            let _ = expect(parser, 127);  // COLON
            let val_tok = current_token(parser);
            spec.include_thermal = val_tok.text[0] == 116;
            advance(parser);
        }

        // temperature: value
        else if tok.text[0] == 116 && tok.text[1] == 101 && tok.text[2] == 109 {  // "tem"
            advance(parser);
            let _ = expect(parser, 127);  // COLON
            let val_tok = current_token(parser);
            if val_tok.token_type == 82 {  // FLOAT
                spec.temperature = val_tok.value;
                advance(parser);
            }
        }

        // Skip comma
        else if match_token(parser, 126) {  // COMMA
            advance(parser);
        }

        // Unknown - skip
        else {
            advance(parser);
        }
    }

    let _ = expect(parser, 121);  // RPAREN

    spec
}

/// Parse binding definition
fn parse_binding_def(parser: &!ExtendedParser) -> BindingDefNode {
    var binding = binding_def_node_new();

    let start = current_token(parser);
    binding.line = start.line;
    binding.col = start.col;

    // Expect 'binding'
    if !expect(parser, TT_KW_BINDING()) {
        return binding;
    }

    // Get binding name (Kd, Ki, IC50, etc.)
    let name_tok = current_token(parser);
    if name_tok.token_type == 80 {  // IDENTIFIER
        var i: i64 = 0;
        while i < 64 && name_tok.text[i as usize] != 0 {
            binding.name[i as usize] = name_tok.text[i as usize];
            i = i + 1;
        }
        advance(parser);
    }

    // Expect ~
    if !expect(parser, 106) {  // TILDE
        return binding;
    }

    // Parse quantum spec
    binding.qm_spec = parse_quantum_spec(parser);

    binding
}

// ============================================================================
// COVARIATE PARSING
// ============================================================================

/// Parse covariate definition
fn parse_covariate_def(parser: &!ExtendedParser) -> CovariateDefNode {
    var cov = covariate_def_node_new();

    let start = current_token(parser);
    cov.line = start.line;
    cov.col = start.col;

    // Expect 'covariate'
    if !expect(parser, TT_KW_COVARIATE()) {
        return cov;
    }

    // Get covariate name
    let cov_name = current_token(parser);
    if cov_name.token_type == 80 {  // IDENTIFIER
        var i: i64 = 0;
        while i < 64 && cov_name.text[i as usize] != 0 {
            cov.covariate_name[i as usize] = cov_name.text[i as usize];
            i = i + 1;
        }
        advance(parser);
    }

    // Expect 'on'
    let on_tok = current_token(parser);
    if on_tok.text[0] == 111 && on_tok.text[1] == 110 {  // "on"
        advance(parser);
    }

    // Get parameter name
    let param_name = current_token(parser);
    if param_name.token_type == 80 {  // IDENTIFIER
        var i: i64 = 0;
        while i < 64 && param_name.text[i as usize] != 0 {
            cov.param_name[i as usize] = param_name.text[i as usize];
            i = i + 1;
        }
        advance(parser);
    }

    // Expect :
    let _ = expect(parser, 127);  // COLON

    // Parse effect type and parameters
    let effect_tok = current_token(parser);

    if effect_tok.token_type == TT_KW_POWER() {
        cov.effect_type = CovariateEffectType::Power;
        advance(parser);

        // Expect (ref: value, exp: value)
        if match_token(parser, 120) {  // LPAREN
            advance(parser);

            // Parse reference
            let ref_tok = current_token(parser);
            if ref_tok.text[0] == 114 && ref_tok.text[1] == 101 && ref_tok.text[2] == 102 {  // "ref"
                advance(parser);
                let _ = expect(parser, 127);  // COLON
                let val = current_token(parser);
                if val.token_type == 82 {  // FLOAT
                    cov.reference_value = val.value;
                    advance(parser);
                }
            }

            if match_token(parser, 126) { advance(parser); }  // COMMA

            // Parse exponent
            let exp_tok = current_token(parser);
            if exp_tok.text[0] == 101 && exp_tok.text[1] == 120 && exp_tok.text[2] == 112 {  // "exp"
                advance(parser);
                let _ = expect(parser, 127);  // COLON
                let val = current_token(parser);
                if val.token_type == 82 {  // FLOAT
                    cov.exponent = val.value;
                    advance(parser);
                }
            }

            let _ = expect(parser, 121);  // RPAREN
        }
    }

    cov
}

// ============================================================================
// DISTRIBUTION PARSING
// ============================================================================

fn parse_distribution(parser: &!ExtendedParser) -> DistributionSpec {
    var dist = distribution_spec_new();

    let tok = current_token(parser);

    // LogNormal, Normal, Uniform, etc.
    if tok.token_type == 30 {  // KW_LOGNORMAL
        dist.dist_type = DistType::LogNormal;
        advance(parser);
    } else if tok.token_type == 31 {  // KW_NORMAL
        dist.dist_type = DistType::Normal;
        advance(parser);
    } else if tok.token_type == 32 {  // KW_UNIFORM
        dist.dist_type = DistType::Uniform;
        advance(parser);
    } else {
        // Default to Normal
        advance(parser);
    }

    // Parse (param1, omega: value) or (param1, param2)
    if match_token(parser, 120) {  // LPAREN
        advance(parser);

        let p1 = current_token(parser);
        if p1.token_type == 82 {  // FLOAT
            dist.param1 = p1.value;
            advance(parser);
        }

        if match_token(parser, 126) {  // COMMA
            advance(parser);
        }

        // Check for omega:
        let next = current_token(parser);
        if next.token_type == 34 {  // KW_OMEGA
            advance(parser);
            let _ = expect(parser, 127);  // COLON
            let omega_val = current_token(parser);
            if omega_val.token_type == 82 {  // FLOAT
                dist.omega = omega_val.value;
                dist.has_omega = true;
                advance(parser);
            }
        } else if next.token_type == 82 {  // FLOAT
            dist.param2 = next.value;
            advance(parser);
        }

        let _ = expect(parser, 121);  // RPAREN
    }

    dist
}

// ============================================================================
// MAIN PARSING
// ============================================================================

/// Parse extended model
fn parse_extended_model(parser: &!ExtendedParser) -> ExtendedModelNode {
    var model = extended_model_node_new();

    let start = current_token(parser);
    model.line = start.line;
    model.col = start.col;

    // Expect 'model'
    if !expect(parser, 10) {  // KW_MODEL
        return model;
    }

    // Get model name
    let name_tok = current_token(parser);
    if name_tok.token_type == 80 {  // IDENTIFIER
        var i: i64 = 0;
        while i < 64 && name_tok.text[i as usize] != 0 {
            model.name[i as usize] = name_tok.text[i as usize];
            i = i + 1;
        }
        advance(parser);
    }

    // Expect {
    if !expect(parser, 122) {  // LBRACE
        return model;
    }

    // Parse model body
    while !match_token(parser, 123) && !parser.has_error {  // RBRACE
        let tok = current_token(parser);

        // tissue definition
        if tok.token_type == TT_KW_TISSUE() {
            let tissue = parse_tissue_def(parser);
            if model.n_tissues < 20 {
                model.tissues[model.n_tissues as usize] = tissue;
                model.n_tissues = model.n_tissues + 1;
                model.is_pbpk = true;
            }
        }

        // binding definition
        else if tok.token_type == TT_KW_BINDING() {
            let binding = parse_binding_def(parser);
            if model.n_bindings < 5 {
                model.bindings[model.n_bindings as usize] = binding;
                model.n_bindings = model.n_bindings + 1;
            }
        }

        // covariate definition
        else if tok.token_type == TT_KW_COVARIATE() {
            let cov = parse_covariate_def(parser);
            if model.n_covariates < 20 {
                model.covariates[model.n_covariates as usize] = cov;
                model.n_covariates = model.n_covariates + 1;
            }
        }

        // param definition (from base parser - just skip for now)
        else if tok.token_type == 11 {  // KW_PARAM
            advance(parser);
        }

        // Skip newlines
        else if match_token(parser, 129) {  // NEWLINE
            advance(parser);
        }

        // Unknown - skip
        else {
            advance(parser);
        }
    }

    let _ = expect(parser, 123);  // RBRACE

    model
}

// ============================================================================
// TESTS
// ============================================================================

fn test_tissue_parsing() -> bool {
    // Test tissue node creation
    let tissue = tissue_def_node_new();
    tissue.volume_value == 1.0 && tissue.km == 1.0
}

fn test_quantum_parsing() -> bool {
    // Test quantum spec creation
    let spec = quantum_spec_node_new();
    spec.temperature > 298.0 && spec.temperature < 299.0
}

fn test_parser_creation() -> bool {
    let parser = extended_parser_new();
    parser.pos == 0 && parser.n_tokens == 0
}

fn main() -> i32 {
    print("Testing medlang::parser_ext module...\n");

    if !test_tissue_parsing() {
        print("FAIL: tissue_parsing\n");
        return 1;
    }
    print("PASS: tissue_parsing\n");

    if !test_quantum_parsing() {
        print("FAIL: quantum_parsing\n");
        return 2;
    }
    print("PASS: quantum_parsing\n");

    if !test_parser_creation() {
        print("FAIL: parser_creation\n");
        return 3;
    }
    print("PASS: parser_creation\n");

    print("All medlang::parser_ext tests PASSED\n");
    0
}
