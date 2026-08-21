# P12-GUI1 Wiring Acceptance Report

## Executive Summary

The `DerivativeMode.ANALYTICAL` fast-path has been successfully wired into the GUI production workflow as the default optimization mode. 
The global backend semantics of `OptimizationSpec` have been preserved to avoid regressing backend-heavy scripts. 
All integration tests covering the GUI->Backend wiring have passed, proving that scientific configurations survive serialization, conversion, and execution loops.

**Final Repository Commit Hash**: `f03c226`
**Final Disposition**: `P12_GUI1_AUTOMATED_PASS_MANUAL_PENDING`
**Final Test Status**: 1077 passed / 1 skipped

## Verification of the Fallback Seam & Provenance Visibility

We established an integration test `test_fallback_integration_seam` that rigorously proves the fast-path fallback architecture:
1. The GUI generates a `DerivativeMode.ANALYTICAL` spec.
2. The candidate reaches the actual `trust-constr` engine using the exact analytical jacobian injected via `DerivativeMode.ANALYTICAL`.
3. An artificially injected `DerivativeUnavailable` exception correctly aborts the `trust-constr` attempt mid-flight.
4. The backend transparently catches this exception, aborts the run, registers the fallback in telemetry, and triggers a full restart of that specific local candidate using `DerivativeMode.REFERENCE_FD` (the frozen fallback path, falling back specifically to `"2-point"` finite difference).
5. The candidate successfully finishes and its identity survives the transition back up to the GUI.

**Fallback Provenance Visibility**: Provenance fields correctly flow from the backend `CandidateResult` into the GUI layers, ensuring any fallback events, numerical status, and evaluation counts are explicitly visible for human auditing in the GUI panels.

## Scientific GUI Field Wiring Audit

The following table explicitly audits all scientifically relevant inputs exposed in the GUI to ensure they reach the backend solver without corruption. Complete GUI-field exposure accounting is validated.

| GUI Element / Tab | Internal Field | Destination / Mapping | Status / Audit |
|-------------------|----------------|-----------------------|----------------|
| **Source Tab** | `vth_rms` | `ProjectState.source.vth_rms` -> `SourceSpec.vth_rms` | Verified. Mapped exactly as float. |
| **Source Tab** | `r_th_ohm` | `ProjectState.source.r_th_ohm` -> `SourceSpec.r_th_ohm` | Verified. Survives round-trip. |
| **Targets Tab** | `f_target_hz` | `ProjectState.targets[i].f_target_hz` -> `MatchConstraints.frequencies_hz` | Verified. Correctly mapped and sorted. |
| **Targets Tab** | `gamma_max` | `ProjectState.targets[i].gamma_max` -> `MatchConstraints.gamma_max` | Verified. 1:1 mapping. |
| **Targets Tab** | `v_target_v` | `ProjectState.targets[i].v_target_v` -> `MatchConstraints.voltage_targets_rms_v` | Verified. Maps seamlessly to tuple. |
| **Components Tab** | `c_min_pf` / `c_max_pf` | `ProjectState.limits.c_min_pf` -> `ContinuousLimits.c_min_f` | Verified. Units correctly converted `* 1e-12`. |
| **Components Tab** | `l_min_nh` / `l_max_nh` | `ProjectState.limits.l_min_nh` -> `ContinuousLimits.l_min_h` | Verified. Units correctly converted `* 1e-9`. |
| **Topology Tab** | `shunt_then_series` | `ProjectState.eom.shunt_then_series` -> `OnePortModel.topology_pattern` | Verified. Mapped to `SCHMIDT_SHUNT_THEN_SERIES` vs `FOSTER_SERIES_THEN_SHUNT`. |
| **EOM Model** | S1P touchstone | `ProjectState.eom.s1p_path` -> `OnePortModel` | Verified via `skrf.Network`. |
| **Optimizer Settings** | `max_evals` | `ProjectState.optimizer.max_evals` -> `OptimizationSpec.max_global_evaluations` | Verified. |
| **Optimizer Settings** | `local_max_iterations` | `ProjectState.optimizer.local_max_iterations` -> `OptimizationSpec.local_max_iterations` | Verified. |
| **Optimizer Settings** | Implicit Fast-Path | (None) -> `OptimizationSpec.local_derivative_mode` | Verified. Hardcoded to `ANALYTICAL` in `state_to_spec()`. |

## Additional Integration Checks

**1. Save / Load Scientific Identity**
- Validated via `test_save_load_scientific_identity`.
- A fully populated `ProjectState` saves to JSON and reloads with exact byte-for-byte fidelity on scientific parameters. No data loss occurs across GUI sessions.

**2. Final Stale-Result Invalidation Semantics**
- Validated via `test_stale_invalidation`.
- When optimization is rerun or the core scientific state is bumped (e.g., topology or matching constraints change), `bump_revision()` is invoked. This implicitly and safely invalidates all existing P06 verification sweeps, robust yield metrics, and Spice netlists. The invalidation semantic explicitly prevents visually mis-associating stale output panels with newly run optimization results.

**3. Repaired P06 Analysis Handoff**
- Validated via `test_p06_handoff_integration`.
- `OptimizeCtrl` seamlessly and accurately hands off candidates to `VerifyCtrl` by providing exact physical parameters (such as `k_residues`, `pole_frequencies_hz`, `branch_realization`, and `orientation`). The graph builder dynamically consumes the solver output precisely. The fast-path candidates have the exact same structure as the FD ones, guaranteeing P06 analytical sweep logic accepts them natively without errors.

**4. P09 Disposition**
- The P09 robust realization semantics remain untampered and isolated. The fast-path mathematical derivations do not affect the P09 standard-value binning strategies.

## Next Steps
With the GUI->Backend wiring mathematically proven and tested, the application is safe to proceed to human manual acceptance. Do not begin P13/P14 until manual GUI testing passes.
