# P12.5-E — Frozen FD vs Analytical Equivalence & Performance Report

Generated: 2026-08-19T23:20:14Z  
Frozen derivative baseline: `4ec27f827a52cf11b534ea85586f3d17e7b6e1af`  
Platform: Windows 11, 4P/8L cores, Python 3.14.2, NumPy 2.4.6, SciPy 1.18.0

Only `local_derivative_mode` differs between the two runs of each pair. Topology/domain, initial point, targets, source/EOM models, objective weights, constraints/bounds, optimizer tolerances/options and iteration limits are held constant by construction (same basin list, same `OptimizationSpec` except that one field).

## Summary

| Case | Np | Ng | FD time | Analytical time | Speedup | FD MNA work | Analytical MNA work | Δ objective | Δ max violation | Verdict |
| ---- | -: | -: | ------: | --------------: | ------: | ----------: | ------------------: | ----------: | --------------: | ------- |
| MULTI_FREQUENCY | 14 | 145 | 139.15 s | 11.90 s | 11.69x | 339966 | 12120+12120=24240 | -0.00393 | 0 | EQUIVALENT |

`Δ objective` and `Δ max violation` are the **worst (most analytical-unfavourable)** signed deltas `analytical  -  FD` across the case's basins: negative means analytical reached a better point. The gate is one-sided — analytical must never be materially worse or infeasible where FD succeeds. `EQUIV_OR_BETTER` means at least one basin converged strictly further under exact gradients; the endpoints then genuinely differ, so Γ/Z_in/V_EOM and coordinates are reported rather than gated for those basins.

`FD MNA work` and `Analytical MNA work` are nominal frequency-point solves. The analytical column is written `evaluator+transaction=total`: the transaction term is the **second** nominal sweep it performs on top of the evaluator's, counted honestly rather than hidden. Back-substitutions are reported separately per case below. Times are the sum of per-basin `minimize` wall times.

## MULTI_FREQUENCY

4 targets (9-12 MHz), default 1-3 cell search, 101-point grid, top-K=2. `local_max_iterations` capped at **40**, identically for both modes. DE (shared, run once): 1.9 s. Basin pairs: 3.

### Work and cost

| Basin | Mode | wall s | nit | nfev | njev | c_nfev | c_njev | evaluator freq solves | txn factorizations | direct backsolves | adjoint backsolves | txn builds | reuse |
| --- | --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| fba586ba9023c237c092c56bb0bd458beb6dd49249dd023910699c0c126d50a0#0 | FD | 43.122 | 40 | 600 | 40 | 600 | 0 | 115140 | 0 | 0 | 0 | 0 | 0 |
| fba586ba9023c237c092c56bb0bd458beb6dd49249dd023910699c0c126d50a0#0 | AN | 3.694 | 40 | 40 | 40 | 40 | 40 | 4040 | 4040 | 2240 | 3880 | 40 | 41 |
| 9cbbe15388c8b6078cae1dcc363d6d145523a4245706d100ac01c590b6cfab30#0 | FD | 47.342 | 40 | 600 | 40 | 600 | 0 | 107666 | 0 | 0 | 0 | 0 | 0 |
| 9cbbe15388c8b6078cae1dcc363d6d145523a4245706d100ac01c590b6cfab30#0 | AN | 4.487 | 40 | 40 | 40 | 40 | 40 | 4040 | 4040 | 2240 | 3880 | 40 | 41 |
| d7260beea27011d3cb56ec01238c3fa4ec34f1500f16c7ff348cff8d687fcd0e#0 | FD | 48.682 | 40 | 600 | 40 | 600 | 0 | 117160 | 0 | 0 | 0 | 0 | 0 |
| d7260beea27011d3cb56ec01238c3fa4ec34f1500f16c7ff348cff8d687fcd0e#0 | AN | 3.718 | 40 | 40 | 40 | 40 | 40 | 4040 | 4040 | 2240 | 3880 | 40 | 41 |

### Scientific equivalence

| Basin | feasible FD/AN | v_max FD/AN | J FD/AN | endpoint | Δ\|Γ\| | ΔZin rel | ΔV_EOM rel | ‖Δu‖∞ | ΔL rel | ΔC rel | Δf_p rel | Verdict |
| --- | --- | --- | --- | --- | --: | --: | --: | --: | --: | --: | --: | --- |
| fba586ba9023#0 | True/True | 0/0 | 19.53263/19.517638 | same J, different u | 0.0317 | 0.0615 | 0.000504 | 0.045 | 0.0897 | 0.0819 | 0.0244 | EQUIVALENT |
| 9cbbe15388c8#0 | True/True | 0/0 | 19.53263/19.519062 | same J, different u | 0.0413 | 0.0793 | 0.000854 | 0.0351 | 0.101 | 0.105 | 0.0221 | EQUIVALENT |
| d7260beea270#0 | True/True | 0/0 | 19.53263/19.528702 | same J, different u | 0.0235 | 0.046 | 0.000277 | 0.0727 | 0.436 | 0.325 | 0.0859 | EQUIVALENT |

`endpoint = same u` -> both runs landed on the same point, so Γ/Z_in/V_EOM are gated. Any other label means the two runs stopped at different points (a flat or multi-optimum set), so those columns are diagnostic there. `same J, different u` means the objective agreed within tolerance while the design coordinates did not. Coordinate columns (‖Δu‖∞, ΔL, ΔC, Δf_p) are always reported, never gated.

### Raw polish endpoints (before Deb pre-polish retention)

Reported separately so an agreement caused by *both* modes being discarded by the frozen pre-polish retention rule is not mistaken for an agreement of the two optimizers.

| Basin | polish kept FD/AN | raw J FD/AN | raw v_max FD/AN | Δ raw J | ‖Δ raw u‖∞ |
| --- | --- | --- | --- | --: | --: |
| fba586ba9023#0 | False/True | 2.7614429/19.517638 | 23.81/0 | 16.8 | 0.467 |
| 9cbbe15388c8#0 | False/True | 1.8170822/19.519062 | 13.89/0 | 17.7 | 0.545 |
| d7260beea270#0 | False/True | 6.1925341/19.528702 | 4.823/0 | 13.3 | 0.115 |

Termination messages (diagnostic only, not gated):

* `fba586ba9023c237c092c56bb0bd458beb6dd49249dd023910699c0c126d50a0#0` FD: The maximum number of function evaluations is exceeded. — AN: The maximum number of function evaluations is exceeded.
* `9cbbe15388c8b6078cae1dcc363d6d145523a4245706d100ac01c590b6cfab30#0` FD: The maximum number of function evaluations is exceeded. — AN: The maximum number of function evaluations is exceeded.
* `d7260beea27011d3cb56ec01238c3fa4ec34f1500f16c7ff348cff8d687fcd0e#0` FD: The maximum number of function evaluations is exceeded. — AN: The maximum number of function evaluations is exceeded.

**Fallbacks observed:** none — every candidate stayed on the analytical path.

Peak RSS during polish: FD 870 MB, analytical 346 MB.

### FD parameter-perturbation multiplier

| Basin | Np | FD nfev/njev | AN nfev/njev | expected FD model Np+1 |
| --- | --: | --: | --: | --: |
| fba586ba9023c237c092c56bb0bd458beb6dd49249dd023910699c0c126d50a0#0 | 14 | 15 | 1 | 15 |
| 9cbbe15388c8b6078cae1dcc363d6d145523a4245706d100ac01c590b6cfab30#0 | 14 | 15 | 1 | 15 |
| d7260beea27011d3cb56ec01238c3fa4ec34f1500f16c7ff348cff8d687fcd0e#0 | 14 | 15 | 1 | 15 |

