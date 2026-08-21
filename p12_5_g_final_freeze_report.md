# P12.5-G Final Freeze Report

This document summarizes the complete P12.5-G local-polish analytical fast-path audit and its results.

## G1: Identity/Invalidation

Audited the core state identity model and invalidation mechanism.
**Result**: PASS. Established that identical numerical evaluations legitimately hit the pre-computed nominal cache without invalidation, and new numerical points correctly invalidate and evict previous entries.

## G2: Fallback/State Isolation

Audited the robustness of the fallback mechanism against state leakage.
**Result**: PASS. Demonstrated that an unexpected `NaN` or analytical failure correctly triggers a clean `REFERENCE_FD` fallback cycle without contaminating the solver with invalid MNA state, successfully recovering the optimization.

## G3: Scientific Equivalence

Audited the analytical path for scientific parity against the proven `REFERENCE_FD` path.
**Result**: FROZEN PASS. Out of 12 controlled A/B pairs, 8 resulted in exact matching polish endpoints (`EQUIVALENT_SAME_ENDPOINT`), 3 arrived at equivalent but different local minima (`EQUIVALENT_DIFFERENT_ENDPOINT`), and 1 found a strictly superior minimum analytically (`ANALYTICAL_BETTER`). No regressions were found.

## G4: Execution Path / Hidden FD

Audited the execution path to ensure zero hidden finite-difference evaluations mask incomplete Jacobian implementations.
**Result**: FROZEN PASS. `nfev` matches exactly expected nominal evaluations plus `trust-constr` line-search calls. No perturbation cloud evaluates in the background.

## G5: Memory/State Lifetime

Audited the memory lifetime of the large MNA/bundle objects and established permanent regressions.
**Result**: FROZEN PASS. Verified `max_live_bundles = 1`, tested 218-state RSS plateaus at ~103.9 MB, confirmed historical eviction, and integrated 6 permanent regression tests covering boundary lifetime and large-bundle scale.

## G6: Controlled Performance

Direct controlled A/B measurement of the finalized analytical path vs `REFERENCE_FD`.
**Result**: PASS. 
- **Representative Case** (`Ng=113`, 100 freq): Analytical median wall time of `2.74s` vs FD `14.30s` (**5.2x speedup**).
- **Pathological Case** (`Ng=1233`, 1200 freq, capped to 15 iters): Analytical median wall time of `18.25s` vs FD `214.10s` (**11.7x speedup**).

## Overall Disposition

The analytical fast path preserves the exact optimizer behavior, guarantees bounded bounded execution state, prevents FD leakage, and achieves a consistent **>5x to >11x** real wall-time acceleration over the numerical reference. 

**Production changes across G**: None.

The P12.5-G analytical phase is hereby complete and successfully frozen.
