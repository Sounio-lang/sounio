<!-- docs:meta
topic_id: repo.docs.ai-prompt-templates
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ai-prompt-templates
-->

# AI Prompt Templates for Sounio Development

## Based on the Methodology That Built Sounio in 60 Days

### Core Philosophy
**Good prompts = Good code.** The quality of your specification determines the quality of the AI's output.

## Template Categories

### 1. Function Implementation Prompts

#### Basic Function Template
```markdown
Implement a function with the following specification:

**Function Name:** [NAME]
**Signature:** [SIGNATURE]
**Purpose:** [WHAT IT DOES]

**Inputs:**
- [PARAM1]: [TYPE] - [DESCRIPTION]
- [PARAM2]: [TYPE] - [DESCRIPTION]

**Output:**
- [RETURN TYPE] - [DESCRIPTION]

**Algorithm:**
1. [STEP 1]
2. [STEP 2]
3. [STEP 3]

**Edge Cases:**
- [EDGE CASE 1]: [HOW TO HANDLE]
- [EDGE CASE 2]: [HOW TO HANDLE]

**Error Conditions:**
- [ERROR 1]: [WHAT TO DO]
- [ERROR 2]: [WHAT TO DO]

**Examples:**
```sio
// Example usage
let result = [NAME]([EXAMPLE_INPUTS])
// Expected: [EXAMPLE_OUTPUT]
```

**Performance Requirements:**
- Time complexity: [O(...)]
- Space complexity: [O(...)]
- Memory usage: [LIMITS]

**Testing Requirements:**
- [TEST CASE 1]
- [TEST CASE 2]

Generate the implementation in Sounio.
```

#### Epistemic Function Template
```markdown
Implement an epistemic function with uncertainty propagation:

**Function:** [NAME]
**Purpose:** [OPERATION] on epistemic values

**Mathematical Foundation:**
- Based on GUM (Guide to Uncertainty in Measurement)
- Uncertainty propagation formula: [FORMULA]
- Confidence combination rule: [RULE]

**Properties to Maintain:**
1. **Uncertainty Propagation:** Correct according to GUM
2. **Confidence Monotonicity:** Confidence never increases
3. **Provenance Tracking:** Source information preserved
4. **Numerical Stability:** Handle extreme values gracefully

**Special Cases:**
- Zero uncertainty
- Perfect confidence (1.0)
- Correlated uncertainties (if applicable)
- NaN/Infinity inputs

**Verification Tests:**
- Compare with manual GUM calculation
- Check uncertainty growth property
- Verify confidence bounds

Generate the implementation with full error handling.
```

### 2. Module/Class Prompts

#### Module Template
```markdown
Create a module for [DOMAIN] with the following structure:

**Module Name:** [NAME]
**Purpose:** [BRIEF DESCRIPTION]

**Public Interface:**
- Types:
  - [TYPE1]: [DESCRIPTION]
  - [TYPE2]: [DESCRIPTION]
- Functions:
  - [FUNC1]: [PURPOSE]
  - [FUNC2]: [PURPOSE]

**Internal Implementation:**
- Data structures: [LIST]
- Helper functions: [LIST]
- Constants: [LIST]

**Dependencies:**
- Imports from: [MODULES]
- Exports to: [MODULES]

**Error Handling Strategy:**
- [STRATEGY]

**Performance Characteristics:**
- [CHARACTERISTICS]

**Example Usage:**
```sio
import [MODULE]

// Typical usage pattern
[EXAMPLE_CODE]
```

Generate the complete module implementation.
```

#### Compiler Component Template
```markdown
Implement a compiler component:

**Component:** [NAME] (e.g., Lexer, Parser, Type Checker)
**Stage:** [COMPILATION_STAGE]
**Input:** [INPUT_FORMAT]
**Output:** [OUTPUT_FORMAT]

**Responsibilities:**
1. [RESPONSIBILITY 1]
2. [RESPONSIBILITY 2]
3. [RESPONSIBILITY 3]

**Algorithms:**
- [ALGORITHM 1]: [DESCRIPTION]
- [ALGORITHM 2]: [DESCRIPTION]

**Error Reporting:**
- Error types: [LIST]
- Recovery strategy: [STRATEGY]
- User messages: [FORMAT]

**Performance Targets:**
- Throughput: [N] tokens/lines per second
- Memory: [M] MB maximum

**Integration Points:**
- Previous stage: [STAGE]
- Next stage: [STAGE]
- Data formats: [FORMATS]

**Testing Strategy:**
- Unit tests: [COVERAGE]
- Integration tests: [WITH OTHER COMPONENTS]
- Fuzz tests: [RANDOM INPUTS]

Generate the implementation with proper abstraction boundaries.
```

### 3. Test Generation Prompts

#### Unit Test Template
```markdown
Generate comprehensive tests for [FUNCTION/MODULE]:

**Test Subject:** [NAME]

**Test Categories:**
1. **Normal Cases:**
   - [CASE 1]: [INPUT] → [EXPECTED]
   - [CASE 2]: [INPUT] → [EXPECTED]

2. **Edge Cases:**
   - [EDGE 1]: [INPUT] → [EXPECTED/BEHAVIOR]
   - [EDGE 2]: [INPUT] → [EXPECTED/BEHAVIOR]

3. **Error Cases:**
   - [ERROR 1]: [INPUT] → [ERROR_TYPE]
   - [ERROR 2]: [INPUT] → [ERROR_TYPE]

4. **Property Tests:**
   - [PROPERTY 1]: Should hold for all inputs
   - [PROPERTY 2]: Should hold for all inputs

**Test Framework:**
- Use Sounio's test framework
- Each test returns bool (true = pass)
- Include descriptive failure messages

**Coverage Goals:**
- Line coverage: [X]%
- Branch coverage: [Y]%
- Path coverage: [Z]%

**Performance Tests:**
- Benchmark: [OPERATION] should take < [TIME]
- Memory: Should use < [MEMORY]

Generate the test suite.
```

#### Integration Test Template
```markdown
Create integration tests for [COMPONENT_INTERACTION]:

**Integration:** [COMPONENT_A] + [COMPONENT_B]

**Workflows to Test:**
1. **Happy Path:**
   - [INPUT] → [PROCESSING STEPS] → [OUTPUT]

2. **Error Propagation:**
   - [ERROR_INPUT] → [ERROR_HANDLING] → [RECOVERY]

3. **Performance Integration:**
   - [LOAD] → [THROUGHPUT_MEASUREMENT]

**Setup Requirements:**
- Test data: [SOURCE]
- Environment: [REQUIREMENTS]
- Cleanup: [PROCEDURE]

**Assertions:**
- [ASSERTION 1]: [CONDITION]
- [ASSERTION 2]: [CONDITION]

**Non-Functional Requirements:**
- Latency: < [TIME]
- Throughput: > [RATE]
- Memory: < [USAGE]

Generate the integration tests.
```

### 4. Documentation Prompts

#### API Documentation Template
```markdown
Generate API documentation for [MODULE/FUNCTION]:

**Format:**
```markdown
# [NAME]

## Overview
[OVERVIEW]

## Signature
```sio
[SIGNATURE]
```

## Parameters
- `[PARAM1]`: [TYPE] - [DESCRIPTION]
  - **Constraints:** [CONSTRAINTS]
  - **Example:** [EXAMPLE]

## Returns
- [TYPE] - [DESCRIPTION]
  - **Properties:** [PROPERTIES]
  - **Edge Cases:** [EDGE_CASES]

## Examples
### Example 1: Basic Usage
```sio
[CODE]
```

### Example 2: Advanced Usage
```sio
[CODE]
```

## Errors
- [ERROR_TYPE1]: [CONDITION] - [RESOLUTION]
- [ERROR_TYPE2]: [CONDITION] - [RESOLUTION]

## Performance
- Time: [COMPLEXITY]
- Space: [COMPLEXITY]

## See Also
- [RELATED_FUNCTION1]
- [RELATED_FUNCTION2]
```

Generate the documentation.
```

#### Tutorial Template
```markdown
Create a tutorial for [TOPIC]:

**Audience:** [BEGINNER/INTERMEDIATE/EXPERT]
**Prerequisites:** [KNOWLEDGE REQUIRED]
**Time:** [DURATION]

**Learning Objectives:**
1. [OBJECTIVE 1]
2. [OBJECTIVE 2]
3. [OBJECTIVE 3]

**Outline:**
1. **Introduction:** [OVERVIEW]
2. **Basic Concepts:** [CONCEPTS]
3. **Hands-on Exercise:** [EXERCISE]
4. **Common Patterns:** [PATTERNS]
5. **Advanced Topics:** [TOPICS]
6. **Summary:** [RECAP]

**Code Examples:**
- Start simple, build complexity
- Include comments explaining each step
- Show common mistakes and fixes

**Exercises:**
- [EXERCISE 1]: [DESCRIPTION]
- [EXERCISE 2]: [DESCRIPTION]

**Further Reading:**
- [RESOURCE 1]
- [RESOURCE 2]

Generate the tutorial.
```

### 5. Refactoring Prompts

#### Code Improvement Template
```markdown
Refactor the following code to improve [QUALITY_ASPECT]:

**Current Code:**
```sio
[EXISTING_CODE]
```

**Issues to Address:**
1. [ISSUE 1]: [DESCRIPTION]
2. [ISSUE 2]: [DESCRIPTION]
3. [ISSUE 3]: [DESCRIPTION]

**Improvement Goals:**
- **Readability:** [GOALS]
- **Performance:** [GOALS]
- **Maintainability:** [GOALS]
- **Safety:** [GOALS]

**Constraints:**
- Must maintain backward compatibility
- Cannot change public API
- Performance cannot degrade

**Specific Changes Requested:**
1. [CHANGE 1]
2. [CHANGE 2]
3. [CHANGE 3]

**Testing Strategy:**
- All existing tests must pass
- Add tests for new edge cases
- Benchmark performance

Generate the refactored code.
```

### 6. Architecture Prompts

#### System Design Template
```markdown
Design a system for [PURPOSE]:

**Requirements:**
1. **Functional:**
   - [REQ 1]
   - [REQ 2]
   
2. **Non-Functional:**
   - Performance: [METRICS]
   - Scalability: [TARGETS]
   - Reliability: [UPTIME]
   
3. **Constraints:**
   - [CONSTRAINT 1]
   - [CONSTRAINT 2]

**High-Level Architecture:**
```
[COMPONENT_DIAGRAM_DESCRIPTION]
```

**Components:**
1. **[COMPONENT_A]:**
   - Responsibility: [DESCRIPTION]
   - Interfaces: [APIS]
   - Dependencies: [DEPENDENCIES]

2. **[COMPONENT_B]:**
   - Responsibility: [DESCRIPTION]
   - Interfaces: [APIS]
   - Dependencies: [DEPENDENCIES]

**Data Flow:**
1. [STEP 1]: [DATA] → [PROCESSING]
2. [STEP 2]: [DATA] → [PROCESSING]

**Error Handling:**
- [STRATEGY]

**Scalability Considerations:**
- [CONSIDERATIONS]

**Implementation Plan:**
1. Phase 1: [COMPONENTS]
2. Phase 2: [COMPONENTS]
3. Phase 3: [COMPONENTS]

Generate the detailed design document.
```

## Prompt Engineering Tips

### 1. Be Specific
**Bad:** "Write a function"
**Good:** "Write a function that adds two epistemic values with GUM-compliant uncertainty propagation"

### 2. Provide Examples
**Bad:** "Handle errors"
**Good:** "Return an error if input is NaN, with message 'Input must be a finite number'"

### 3. Define Constraints
**Bad:** "Make it fast"
**Good:** "Time complexity must be O(n), memory O(1)"

### 4. Include Verification
**Bad:** "It should work"
**Good:** "Include tests that verify: (1) result = a + b, (2) uncertainty = sqrt(a² + b²), (3) confidence decreases"

### 5. Specify Format
**Bad:** "Write code"
**Good:** "Write Sounio code with function signature: fn add_epistemic(a: Epistemic<f64>, b: Epistemic<f64>) -> Epistemic<f64>"

## Common Patterns in Sounio Development

### Epistemic Operations Pattern
```markdown
When implementing epistemic operations:
1. **Value Calculation:** Standard arithmetic
2. **Uncertainty Propagation:** GUM formulas
3. **Confidence Combination:** Minimum confidence * decay factor
4. **Provenance Tracking:** Combine source information
5. **Error Checking:** Validate inputs, check bounds
```

### Compiler Pass Pattern
```markdown
When implementing compiler passes:
1. **Input Validation:** Check IR well-formedness
2. **Transformation:** Apply optimization/analysis
3. **Output Validation:** Ensure correctness preserved
4. **Metrics Collection:** Track improvements
5. **Error Reporting:** Location-aware messages
```

## Quality Checklist for Generated Code

Before accepting AI-generated code, verify:
1. [ ] Matches specification exactly
2. [ ] Handles all edge cases
3. [ ] Includes error handling
4. [ ] Has appropriate performance
5. [ ] Includes basic documentation
6. [ ] Follows Sounio style conventions
7. [ ] Integrates with existing code
8. [ ] Passes basic sanity tests

## Iterative Refinement Process

1. **First Draft:** Generate initial implementation
2. **Review:** Check against specification
3. **Test:** Run basic tests
4. **Refine:** Adjust prompt based on issues
5. **Regenerate:** Create improved version
6. **Repeat:** Until quality meets standards

## Template Repository

Store successful prompts for reuse:
```
prompts/
├── functions/
│   ├── epistemic_operations.md
│   ├── compiler_passes.md
│   └── stdlib_functions.md
├── modules/
│   ├── epistemic_core.md
│   ├── compiler_frontend.md
│   └── gpu_backend.md
├── tests/
│   ├── unit_tests.md
│   ├── integration_tests.md
│   └── property_tests.md
└── documentation/
    ├── api_docs.md
    ├── tutorials.md
    └── architecture.md
```

## Continuous Improvement

1. **Track prompt effectiveness:** Which prompts produce best results?
2. **Refine templates:** Update based on experience
3. **Share knowledge:** Document what works
4. **Automate:** Create prompt generation tools

---

*These templates are based on the methodology that successfully built the complete Sounio language and compiler in 60 days.*
