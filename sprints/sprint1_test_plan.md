# Sprint 1: Frontend Stabilization Test Plan

## Goal
90% test coverage for lexer and parser with comprehensive edge case handling

## Timeline
**Duration:** 3-5 days
**Start:** Immediately
**End:** When all tests pass and coverage metrics met

## Test Categories

### 1. Lexer Tests (Day 1-2)
**Coverage Target:** 100% of TokenKind variants

#### Unit Tests:
- [x] `test_int_literals` - All integer formats (decimal, hex, binary, octal)
- [x] `test_float_literals` - Float with scientific notation
- [x] `test_string_literals` - Strings with escapes and unicode
- [x] `test_bool_literals` - true/false keywords
- [x] `test_keywords` - All language keywords
- [x] `test_operators` - All operators (single and compound)
- [x] `test_delimiters` - All punctuation/delimiters
- [x] `test_identifiers` - Valid identifier patterns
- [x] `test_edge_cases` - Empty input, whitespace, comments
- [x] `test_error_handling` - Invalid characters recovery

#### Integration Tests:
- [x] `test_empty_input` - Empty source → Eof token
- [x] `test_whitespace` - Whitespace tokenization
- [x] `test_simple_identifiers` - Basic identifier recognition
- [x] `test_keywords` - Keyword vs identifier distinction
- [x] `test_numbers` - Number literal parsing
- [x] `test_operators` - Operator recognition
- [x] `test_delimiters` - Delimiter recognition

### 2. Parser Tests (Day 2-3)
**Coverage Target:** 90% of AST node types

#### Golden Tests:
- [ ] `test_parse_expressions` - Arithmetic, logical, comparison
- [ ] `test_parse_statements` - Let, var, if, while, for, return
- [ ] `test_parse_functions` - Function definitions and calls
- [ ] `test_parse_structs` - Struct definitions and literals
- [ ] `test_parse_enums` - Enum definitions and matching
- [ ] `test_parse_imports` - Use and import statements
- [ ] `test_parse_types` - Type annotations and generics
- [ ] `test_parse_effects` - Effect annotations and handlers

#### Roundtrip Tests:
- [ ] `test_parse_print_roundtrip` - Parse → Print → Parse equivalence
- [ ] `test_comment_preservation` - Comments preserved in AST

### 3. Integration Tests (Day 3-4)
**Coverage Target:** End-to-end lex+parse pipeline

#### Pipeline Tests:
- [ ] `test_lex_parse_integration` - Full pipeline on example files
- [ ] `test_error_recovery` - Syntax error handling
- [ ] `test_incremental_parsing` - Partial input handling

#### Performance Tests:
- [ ] `benchmark_lexer` - Throughput (tokens/sec)
- [ ] `benchmark_parser` - Throughput (AST nodes/sec)
- [ ] `memory_usage` - Peak memory during parsing

### 4. Fuzz Tests (Day 4-5)
**Coverage Target:** Crash-free on random input

#### Random Input:
- [ ] `fuzz_lexer` - 10,000 random strings, no crashes
- [ ] `fuzz_parser` - 10,000 random token sequences
- [ ] `property_based` - Lex/parse invariants

## Success Metrics

### Quantitative:
1. **Test Coverage:** ≥90% line coverage for lexer/parser
2. **Performance:** Lexing <1ms per 1KB, parsing <10ms per 1KB
3. **Stability:** Zero crashes on fuzz tests
4. **Correctness:** All golden tests pass

### Qualitative:
1. **Error Messages:** Clear, actionable syntax errors
2. **Recovery:** Graceful handling of malformed input
3. **Documentation:** All public APIs documented
4. **Maintainability:** Tests are clear and maintainable

## Test Infrastructure

### Files Created:
1. `tests/frontend/lexer_unit.sio` - Unit tests (stubs)
2. `tests/frontend/lexer_integration.sio` - Integration tests
3. `tests/frontend/parser_golden.sio` - Parser golden tests
4. `tests/frontend/fuzz.sio` - Fuzz tests
5. `tests/frontend/benchmark.sio` - Performance tests
6. `scripts/run_sprint1_tests.sh` - Test runner

### CI Integration:
- Runs as part of `selfhost-regression` workflow
- Fail-fast on critical tests
- Performance regression detection
- Coverage reporting

## Execution Plan

### Day 1: Lexer Foundation
1. Run existing lexer tests
2. Fix any failures
3. Add missing test cases
4. Verify 100% TokenKind coverage

### Day 2: Parser Foundation  
1. Create parser golden tests
2. Test all AST node types
3. Add error recovery tests
4. Verify 90% parser coverage

### Day 3: Integration & Performance
1. Create pipeline integration tests
2. Add performance benchmarks
3. Test with real-world examples
4. Optimize hot paths

### Day 4: Fuzzing & Edge Cases
1. Implement fuzz tests
2. Test extreme inputs
3. Verify crash resistance
4. Add property-based tests

### Day 5: Polish & Documentation
1. Improve error messages
2. Document public APIs
3. Create usage examples
4. Final verification

## Risk Mitigation

### Technical Risks:
1. **Performance regression** - Continuous benchmarking
2. **Memory leaks** - Memory usage monitoring
3. **Test flakiness** - Deterministic test data
4. **Coverage gaps** - Automated coverage reporting

### Process Risks:
1. **Scope creep** - Strict adherence to test plan
2. **Time overrun** - Daily progress tracking
3. **Quality issues** - Code review for all tests
4. **Integration failures** - Continuous integration

## Deliverables

1. **Code:** Complete test suite for lexer and parser
2. **Metrics:** Coverage reports and performance benchmarks
3. **Documentation:** Test plan and results
4. **CI:** Integrated test execution in workflows

## Exit Criteria

Sprint 1 is complete when:
1. ✅ All tests pass
2. ✅ Coverage targets met
3. ✅ Performance targets met
4. ✅ No crashes on fuzz tests
5. ✅ CI integration working

## Next Sprint

**Sprint 2:** Type Checker Stabilization
- Type inference tests
- Error message quality
- Performance optimization
