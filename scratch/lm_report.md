# P12.5-E — Frozen FD vs Analytical Equivalence & Performance Report

Generated: 2026-08-19T23:12:00Z  
Frozen derivative baseline: `4ec27f827a52cf11b534ea85586f3d17e7b6e1af`  
Platform: Windows 11, 4P/8L cores, Python 3.14.2, NumPy 2.4.6, SciPy 1.18.0

Only `local_derivative_mode` differs between the two runs of each pair. Topology/domain, initial point, targets, source/EOM models, objective weights, constraints/bounds, optimizer tolerances/options and iteration limits are held constant by construction (same basin list, same `OptimizationSpec` except that one field).

## Summary

| Case | Np | Ng | FD time | Analytical time | Speedup | FD MNA work | Analytical MNA work | Δ objective | Δ max violation | Verdict |
| ---- | -: | -: | ------: | --------------: | ------: | ----------: | ------------------: | ----------: | --------------: | ------- |
| LARGE_MULTIPOLE | 14 | 141 | 112.22 s | 9.61 s | 11.68x | 315625 | 11514+12120=23634 | 0 | 0 | EQUIVALENT |
| MULTI_FREQUENCY | - | - | - | - | - | - | - | - | - | NO_BASINS |

`Δ objective` and `Δ max violation` are the **worst (most analytical-unfavourable)** signed deltas `analytical  -  FD` across the case's basins: negative means analytical reached a better point. The gate is one-sided — analytical must never be materially worse or infeasible where FD succeeds. `EQUIV_OR_BETTER` means at least one basin converged strictly further under exact gradients; the endpoints then genuinely differ, so Γ/Z_in/V_EOM and coordinates are reported rather than gated for those basins.

`FD MNA work` and `Analytical MNA work` are nominal frequency-point solves. The analytical column is written `evaluator+transaction=total`: the transaction term is the **second** nominal sweep it performs on top of the evaluator's, counted honestly rather than hidden. Back-substitutions are reported separately per case below. Times are the sum of per-basin `minimize` wall times.

## LARGE_MULTIPOLE

3 targets, 3+3 cells (max Np), 101-point grid, top-K=3. `local_max_iterations` capped at **40**, identically for both modes. DE (shared, run once): 1.8 s. Basin pairs: 3.

### Work and cost

| Basin | Mode | wall s | nit | nfev | njev | c_nfev | c_njev | evaluator freq solves | txn factorizations | direct backsolves | adjoint backsolves | txn builds | reuse |
| --- | --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 6017f6d45e7e25226bddbc5a14e307a9cdc6379dfd1a0c351f059df5f8806fc3#0 | FD | 40.694 | 40 | 585 | 39 | 585 | 0 | 113625 | 0 | 0 | 0 | 0 | 0 |
| 6017f6d45e7e25226bddbc5a14e307a9cdc6379dfd1a0c351f059df5f8806fc3#0 | AN | 3.535 | 40 | 40 | 40 | 40 | 40 | 3838 | 4040 | 1680 | 3920 | 40 | 41 |
| afad4bac964759fb4b603f89e23e093ad8ed817cde7fb35043b8aeb424ca8cfc#0 | FD | 35.741 | 40 | 520 | 40 | 520 | 0 | 101000 | 0 | 0 | 0 | 0 | 0 |
| afad4bac964759fb4b603f89e23e093ad8ed817cde7fb35043b8aeb424ca8cfc#0 | AN | 3.016 | 40 | 40 | 40 | 40 | 40 | 3838 | 4040 | 1440 | 3920 | 40 | 41 |
| 129a8d3272539a9a5be2320824a3f46827b9f048b1a0906d2cf7e6907eb56fca#0 | FD | 35.785 | 40 | 520 | 40 | 520 | 0 | 101000 | 0 | 0 | 0 | 0 | 0 |
| 129a8d3272539a9a5be2320824a3f46827b9f048b1a0906d2cf7e6907eb56fca#0 | AN | 3.056 | 40 | 40 | 40 | 40 | 40 | 3838 | 4040 | 1440 | 3920 | 40 | 41 |

### Scientific equivalence

| Basin | feasible FD/AN | v_max FD/AN | J FD/AN | endpoint | Δ\|Γ\| | ΔZin rel | ΔV_EOM rel | ‖Δu‖∞ | ΔL rel | ΔC rel | Δf_p rel | Verdict |
| --- | --- | --- | --- | --- | --: | --: | --: | --: | --: | --: | --: | --- |
| 6017f6d45e7e#0 | True/True | 0/0 | 13.781587/13.781587 | same | 0 | 0 | 0 | 0 | 0 | 0 | 0 | EQUIVALENT |
| afad4bac9647#0 | True/True | 0/0 | 13.781587/13.781587 | same | 0 | 0 | 0 | 0 | 0 | 0 | 0 | EQUIVALENT |
| 129a8d327253#0 | False/False | 0.09984/0.09984 | 13.781587/13.781587 | same | 0 | 0 | 0 | 0 | 0 | 0 | 0 | EQUIVALENT |

`endpoint = same` -> Γ/Z_in/V_EOM are gated. `AN better` / `differs` -> the two runs stopped at different points, so those columns are diagnostic. Coordinate columns (‖Δu‖∞, ΔL, ΔC, Δf_p) are always reported, never gated.

### Raw polish endpoints (before Deb pre-polish retention)

Reported separately so an agreement caused by *both* modes being discarded by the frozen pre-polish retention rule is not mistaken for an agreement of the two optimizers.

| Basin | polish kept FD/AN | raw J FD/AN | raw v_max FD/AN | Δ raw J | ‖Δ raw u‖∞ |
| --- | --- | --- | --- | --: | --: |
| 6017f6d45e7e#0 | False/False | 2.4144399/2.587428 | 56/29.06 | 0.173 | 0.385 |
| afad4bac9647#0 | False/False | 9.0294241/2.210857 | 2.146/33.97 | -6.82 | 0.172 |
| 129a8d327253#0 | False/False | 12.419098/2.316775 | 0.1757/25.04 | -10.1 | 0.112 |

Termination messages (diagnostic only, not gated):

* `6017f6d45e7e25226bddbc5a14e307a9cdc6379dfd1a0c351f059df5f8806fc3#0` FD: The maximum number of function evaluations is exceeded. — AN: The maximum number of function evaluations is exceeded.
* `afad4bac964759fb4b603f89e23e093ad8ed817cde7fb35043b8aeb424ca8cfc#0` FD: The maximum number of function evaluations is exceeded. — AN: The maximum number of function evaluations is exceeded.
* `129a8d3272539a9a5be2320824a3f46827b9f048b1a0906d2cf7e6907eb56fca#0` FD: The maximum number of function evaluations is exceeded. — AN: The maximum number of function evaluations is exceeded.

**Fallbacks observed:** none — every candidate stayed on the analytical path.

Peak RSS during polish: FD 848 MB, analytical 335 MB.

### FD parameter-perturbation multiplier

| Basin | Np | FD nfev/njev | AN nfev/njev | expected FD model Np+1 |
| --- | --: | --: | --: | --: |
| 6017f6d45e7e25226bddbc5a14e307a9cdc6379dfd1a0c351f059df5f8806fc3#0 | 14 | 15 | 1 | 15 |
| afad4bac964759fb4b603f89e23e093ad8ed817cde7fb35043b8aeb424ca8cfc#0 | 12 | 13 | 1 | 13 |
| 129a8d3272539a9a5be2320824a3f46827b9f048b1a0906d2cf7e6907eb56fca#0 | 12 | 13 | 1 | 13 |

## MULTI_FREQUENCY

5 targets, 2+2 cells, 101-point grid, top-K=2. `local_max_iterations` capped at **40**, identically for both modes. DE (shared, run once): 0.0 s. Basin pairs: 0.

_No basins reached polish for this case._

