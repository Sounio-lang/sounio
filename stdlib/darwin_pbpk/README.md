# Darwin PBPK Platform - Demetrios Implementation

Complete physiologically-based pharmacokinetic (PBPK) modeling system implemented in the Demetrios language.

## Architecture Overview

Total: 10 modules, ~150KB of Demetrios code

## Module: simulation.d (THIS MODULE)

**Purpose:** Top-level PBPK simulation orchestrator that integrates all components.

**Location:** /mnt/e/workspace/demetrios/stdlib/darwin_pbpk/simulation.d

**Size:** 23KB, 680 lines, 19 functions, 3 structs

### Key Components

#### 1. Configuration Structures

- SimulationConfig: User-facing configuration
- SimulationResult: Complete PK metrics output

#### 2. Main Functions

- run_pbpk_simulation() - Main orchestrator
- scale_params_for_patient() - Allometric scaling
- calculate_drug_params() - Rodgers-Rowland Kp
- simulate_iv_bolus() - IV bolus wrapper
- simulate_oral() - Oral dose wrapper
- example_midazolam_simulation() - Test case
- example_metformin_simulation() - Test case

#### 3. PK Calculations

- calculate_half_life_from_curve()
- calculate_clearance_from_auc()
- calculate_vdss_from_cl_thalf()
- validate_against_observed()

## Usage

The module provides wrapper functions for common scenarios and detailed example implementations for Midazolam (IV) and Metformin (oral).

## File Statistics

- Lines of code: 680
- Functions: 19
- Structs: 3
- Size: 23KB
- Language: Demetrios with unit annotations

## Created

December 8, 2025 - Version 1.0.0
