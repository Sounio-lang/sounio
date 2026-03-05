# Sounio Development Sprints

## Current Sprint: Sprint 1 - Frontend Stabilization

### Goal
90% test coverage for lexer and parser with comprehensive edge case handling

### Timeline
**Duration:** 3-5 days  
**Status:** IN PROGRESS  
**Start:** Immediate  
**Target Completion:** When all tests pass and coverage metrics met

## Getting Started with Sprint 1

### Prerequisites
1. Working `souc` compiler binary
2. Test infrastructure setup
3. CI/CD fixes applied

### Quick Start
```bash
# 1. Make test runner executable
chmod +x scripts/run_sprint1_tests.sh

# 2. Run all Sprint 1 tests
bash scripts/run_sprint1_tests.sh

# 3. Check results
cat artifacts/sprint1_test_summary.txt
```

### Test Structure
```
tests/frontend/
├── lexer_unit.sio          # Unit tests (stubs)
├── lexer_integration.sio   # Integration tests
├── parser_golden.sio       # Parser tests  
├── smoke_test.sio          # Basic verification
└── (more to be added)
```

## Daily Execution Plan

### Day 1: Lexer Foundation ✅
**Tasks:**
- [x] Create lexer test infrastructure
- [x] Write basic lexer tests
- [x] Fix any lexer bugs found
- [x] Achieve 100% TokenKind coverage

**Commands:**
```bash
# Run lexer tests only
find tests/frontend -name "*lexer*.sio" -exec souc run {} \;

# Check coverage
# (Coverage tool to be implemented)
```

### Day 2: Parser Foundation
**Tasks:**
- [ ] Create parser test infrastructure
- [ ] Write golden tests for all AST nodes
- [ ] Fix any parser bugs found
- [ ] Achieve 90% parser coverage

**Commands:**
```bash
# Run parser tests
souc run tests/frontend/parser_golden.sio

# Test with real examples
for file in examples/wave*.sio; do
    souc check "$file"
done
```

### Day 3: Integration & Performance
**Tasks:**
- [ ] Create pipeline integration tests
- [ ] Add performance benchmarks
- [ ] Test with real-world examples
- [ ] Optimize hot paths

**Commands:**
```bash
# Benchmark lexer
souc run tests/frontend/benchmark.sio

# Integration test
bash scripts/selfhost_driver_output_gate.sh
```

### Day 4: Fuzzing & Edge Cases
**Tasks:**
- [ ] Implement fuzz tests
- [ ] Test extreme inputs
- [ ] Verify crash resistance
- [ ] Add property-based tests

**Commands:**
```bash
# Run fuzz tests
souc run tests/frontend/fuzz.sio

# Property tests
souc run tests/frontend/property.sio
```

### Day 5: Polish & Documentation
**Tasks:**
- [ ] Improve error messages
- [ ] Document public APIs
- [ ] Create usage examples
- [ ] Final verification

**Commands:**
```bash
# Final verification
bash scripts/run_sprint1_tests.sh

# Generate documentation
# (Documentation tool to be implemented)
```

## Success Metrics

### Quantitative Targets:
1. **Test Coverage:** ≥90% line coverage for lexer/parser
2. **Performance:** Lexing <1ms per 1KB, parsing <10ms per 1KB
3. **Stability:** Zero crashes on fuzz tests
4. **Correctness:** All golden tests pass

### Qualitative Targets:
1. **Error Messages:** Clear, actionable syntax errors
2. **Recovery:** Graceful handling of malformed input
3. **Documentation:** All public APIs documented
4. **Maintainability:** Tests are clear and maintainable

## CI/CD Integration

Sprint 1 tests are integrated into CI:
```yaml
# In .github/workflows/ci.yml
jobs:
  sprint1-frontend:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - run: bash scripts/run_sprint1_tests.sh
```

## Files Created

### Test Files:
1. `tests/frontend/lexer_unit.sio` - Unit tests
2. `tests/frontend/lexer_integration.sio` - Integration tests  
3. `tests/frontend/parser_golden.sio` - Parser tests
4. `tests/frontend/smoke_test.sio` - Basic verification

### Infrastructure:
1. `scripts/run_sprint1_tests.sh` - Test runner
2. `sprints/sprint1_test_plan.md` - Detailed plan
3. `sprints/README.md` - This file

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

## Troubleshooting

### Common Issues:

1. **`souc` binary not found:**
   ```bash
   ./bootstrap/poseidon/poseidon self-hosted/main.sio -o souc
   chmod +x souc
   ```

2. **Tests timeout:**
   Increase timeout in `scripts/run_sprint1_tests.sh`
   ```bash
   TIMEOUT_SECS=60 bash scripts/run_sprint1_tests.sh
   ```

3. **Missing dependencies:**
   Check if all required modules are imported in test files

4. **CI failures:**
   Check `artifacts/sprint1/logs/` for detailed logs

## Reporting Issues

1. Check `artifacts/sprint1/failures.txt`
2. Look at individual test logs in `artifacts/sprint1/logs/`
3. Run failing test individually:
   ```bash
   souc run tests/frontend/lexer_integration.sio
   ```

## Contributing to Sprint 1

### Adding New Tests:
1. Create test file in `tests/frontend/`
2. Follow naming convention: `*_test.sio` or `test_*.sio`
3. Include proper imports
4. Return `bool` (true=pass) or `i32` (0=pass)

### Fixing Bugs:
1. Identify failing test
2. Fix the bug in compiler code
3. Verify test passes
4. Add regression test if needed

### Updating Infrastructure:
1. Modify `scripts/run_sprint1_tests.sh`
2. Update `sprints/sprint1_test_plan.md`
3. Test changes locally
4. Commit with descriptive message

---

*Last updated: $(date)*  
*Sprint 1 Status: IN PROGRESS*
