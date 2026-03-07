<!-- docs:meta
topic_id: repo.docs.architects-guide
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architects-guide
-->

# The Sounio Architect's Guide

## How to Build Complex Systems Without Writing Code

*Based on the methodology used to build Sounio in 60 days*

### The Core Philosophy

**You don't need to write code to build software.** You need to:
1. Understand the problem domain
2. Design the architecture
3. Specify requirements clearly
4. Verify correctness

### The 5-Step Methodology

#### Step 1: Domain Understanding
**Before any code:**
- What problem are you solving?
- What are the core concepts?
- What are the constraints?
- What does "correct" mean?

**Example (Sounio):**
- Problem: Scientific computing loses uncertainty information
- Core concepts: Measurements, uncertainty, confidence, propagation
- Constraints: Must be GUM-compliant, performant
- Correctness: Uncertainty calculations must match established standards

#### Step 2: Architectural Design
**Think in components, not code:**
```
[User Code]
    ↓
[Epistemic Type System]
    ↓
[Uncertainty Propagator]
    ↓
[Confidence Checker]
    ↓
[Output]
```

**Key questions:**
- What are the modules?
- How do they interact?
- What are the interfaces?
- What are the data flows?

#### Step 3: AI Specification
**How to talk to AI tools:**

**Bad:** "Write a compiler"
**Good:** "Create a type system that tracks uncertainty with the following properties:
1. Every value has: value, uncertainty, confidence
2. Operations propagate uncertainty using GUM rules
3. Confidence decreases with each operation
4. Provide these specific functions: add_epistemic, mul_epistemic, fuse_measurements"

**Template for AI prompts:**
```
I need a [MODULE] that:
1. Purpose: [WHAT IT DOES]
2. Inputs: [DATA STRUCTURES]
3. Outputs: [RESULTS]
4. Algorithms: [HOW IT WORKS]
5. Edge cases: [SPECIAL CASES]
6. Tests: [VERIFICATION]

Example usage:
[CODE EXAMPLE]
```

#### Step 4: Verification & Integration
**How to check AI output:**
1. **Conceptual correctness:** Does it match the design?
2. **Technical correctness:** Are there obvious bugs?
3. **Integration:** Does it work with other components?
4. **Edge cases:** What happens with unusual inputs?

**Verification checklist:**
- [ ] Matches specification
- [ ] No syntax errors
- [ ] Handles edge cases
- [ ] Integrates with existing code
- [ ] Performance is acceptable

#### Step 5: Iteration
**The feedback loop:**
1. Test the component
2. Identify issues
3. Refine specification
4. Regenerate with AI
5. Repeat until correct

### Tools & Workflow

#### Essential Tools:
1. **AI Assistants:** Claude, Codex, etc.
2. **Version Control:** Git (for tracking changes)
3. **Testing Framework:** Built-in or custom
4. **Documentation:** Always write docs first

#### Daily Workflow:
**Morning (Design):**
- Review yesterday's work
- Design today's components
- Write specifications

**Afternoon (Implementation):**
- Generate code with AI
- Review and test
- Fix issues

**Evening (Integration):**
- Integrate with system
- Run full test suite
- Update documentation

### Case Study: Building Sounio's Epistemic Types

#### Day 1-5: Research & Design
- Read GUM documentation
- Design Knowledge<T> type
- Specify propagation rules
- Create test cases

#### Day 6-10: Core Implementation
- Generate basic type definition
- Implement add/mul operations
- Add confidence tracking
- Create verification tests

#### Day 11-15: Advanced Features
- Interval arithmetic
- Measurement fusion
- Provenance tracking
- Performance optimization

#### Day 16-20: Integration
- Integrate with compiler
- Add to standard library
- Create examples
- Write documentation

### Common Pitfalls & Solutions

#### Pitfall 1: Vague Specifications
**Problem:** AI produces wrong or incomplete code
**Solution:** Be extremely specific. Include:
- Exact function signatures
- Algorithm descriptions
- Error conditions
- Performance requirements

#### Pitfall 2: Integration Issues
**Problem:** Components don't work together
**Solution:** Design interfaces first. Create:
- API contracts
- Data format specifications
- Integration tests

#### Pitfall 3: Quality Control
**Problem:** Bugs slip through
**Solution:** Rigorous testing:
- Unit tests for each function
- Integration tests for components
- Property-based testing
- Fuzz testing

### Scaling the Methodology

#### For Larger Teams:
1. **Divide by domain:** Different experts handle different parts
2. **Standardize specifications:** Common format for all AI prompts
3. **Centralize integration:** One person ensures components work together
4. **Automate testing:** CI/CD pipeline catches issues early

#### For Different Domains:
The same methodology works for:
- **Scientific software** (like Sounio)
- **Business applications**
- **Embedded systems**
- **Machine learning pipelines**

### The Future of This Approach

#### Short-term (1-2 years):
- Better AI tools for specification
- Standardized specification languages
- Automated verification

#### Long-term (3-5 years):
- AI that understands domain expertise
- Automatic architecture generation
- Self-verifying systems

### Getting Started Today

1. **Pick a small project** (1-2 week scope)
2. **Follow the 5-step methodology**
3. **Document everything**
4. **Share your results**

### Resources

- **Sounio Codebase:** Study the architecture
- **GUM Documentation:** ISO Guide 98-3
- **AI Prompt Engineering Guides**
- **Software Architecture Books**

---

*"The best way to predict the future is to invent it."* - Alan Kay

*This guide documents the methodology used to build Sounio, a complete scientific computing language and compiler, in 60 days by a solo architect using AI assistance.*
