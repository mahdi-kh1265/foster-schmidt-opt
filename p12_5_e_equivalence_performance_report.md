# P12.5-E — Frozen FD vs Analytical Equivalence & Performance Report

Generated: 2026-08-19T23:34:38Z  
Frozen derivative baseline: `4ec27f827a52cf11b534ea85586f3d17e7b6e1af`  
Platform: Windows 11, 4P/8L cores, Python 3.14.2, NumPy 2.4.6, SciPy 1.18.0

Only `local_derivative_mode` differs between the two runs of each pair. Topology/domain, initial point, targets, source/EOM models, objective weights, constraints/bounds, optimizer tolerances/options and iteration limits are held constant by construction (same basin list, same `OptimizationSpec` except that one field).

## Summary

| Case | Np | Ng | FD time | Analytical time | Speedup | FD MNA work | Analytical MNA work | Δ objective | Δ max violation | Verdict |
| ---- | -: | -: | ------: | --------------: | ------: | ----------: | ------------------: | ----------: | --------------: | ------- |
| SMALL_DETERMINISTIC | 7 | 33 | 10.76 s | 1.71 s | 6.28x | 34104 | 2415+2520=4935 | -0.0202 | 0 | EQUIV_OR_BETTER |
| TYPICAL_FOSTER | 9 | 122 | 166.75 s | 19.63 s | 8.49x | 520352 | 27876+30300=58176 | 0 | 0 | EQUIVALENT |
| LARGE_MULTIPOLE | 14 | 141 | 120.27 s | 10.39 s | 11.58x | 315625 | 11514+12120=23634 | 0 | 0 | EQUIVALENT |
| MULTI_FREQUENCY | 14 | 145 | 167.92 s | 13.82 s | 12.15x | 339966 | 12120+12120=24240 | -0.00393 | 0 | EQUIVALENT |
| PATHOLOGICAL_1201_GRID | 12 | 1236 | 261.42 s | 20.14 s | 12.98x | 600500 | 24020+24020=48040 | 0 | 0 | EQUIVALENT |

`Δ objective` and `Δ max violation` are the **worst (most analytical-unfavourable)** signed deltas `analytical  -  FD` across the case's basins: negative means analytical reached a better point. The gate is one-sided — analytical must never be materially worse or infeasible where FD succeeds. `EQUIV_OR_BETTER` means at least one basin converged strictly further under exact gradients; the endpoints then genuinely differ, so Γ/Z_in/V_EOM and coordinates are reported rather than gated for those basins.

`FD MNA work` and `Analytical MNA work` are nominal frequency-point solves. The analytical column is written `evaluator+transaction=total`: the transaction term is the **second** nominal sweep it performs on top of the evaluator's, counted honestly rather than hidden. Back-substitutions are reported separately per case below. Times are the sum of per-basin `minimize` wall times.

## SMALL_DETERMINISTIC

1 target, 1 cell, 21-point grid, top-K=1. `local_max_iterations` capped at **60**, identically for both modes. DE (shared, run once): 0.5 s. Basin pairs: 2.

### Work and cost

| Basin | Mode | wall s | nit | nfev | njev | c_nfev | c_njev | evaluator freq solves | txn factorizations | direct backsolves | adjoint backsolves | txn builds | reuse |
| --- | --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 92d33827b0e1dc0f198ed6a183c67f1cc21b592ea76e7c9a84899fb74e4e5941#0 | FD | 4.938 | 60 | 420 | 60 | 420 | 0 | 15834 | 0 | 0 | 0 | 0 | 0 |
| 92d33827b0e1dc0f198ed6a183c67f1cc21b592ea76e7c9a84899fb74e4e5941#0 | AN | 0.843 | 60 | 60 | 60 | 60 | 60 | 1218 | 1260 | 360 | 1200 | 60 | 61 |
| 617c868e9f7ee37a709132d6a94953d112803d84aa98ca5b1894ad901dbeb4f6#0 | FD | 5.819 | 60 | 480 | 60 | 480 | 0 | 18270 | 0 | 0 | 0 | 0 | 0 |
| 617c868e9f7ee37a709132d6a94953d112803d84aa98ca5b1894ad901dbeb4f6#0 | AN | 0.870 | 60 | 60 | 60 | 60 | 60 | 1197 | 1260 | 420 | 1200 | 60 | 61 |

### Scientific equivalence

| Basin | feasible FD/AN | v_max FD/AN | J FD/AN | endpoint | Δ\|Γ\| | ΔZin rel | ΔV_EOM rel | ‖Δu‖∞ | ΔL rel | ΔC rel | Δf_p rel | Verdict |
| --- | --- | --- | --- | --- | --: | --: | --: | --: | --: | --: | --: | --- |
| 92d33827b0e1#0 | True/True | 0/0 | 3.411694/3.3914481 | AN better J | 0.0653 | 0.0208 | 0.00238 | 0.0294 | 0.237 | 0.269 | 0.0205 | EQUIVALENT_OR_BETTER |
| 617c868e9f7e#0 | True/True | 0/0 | 3.4096292/3.2874828 | AN better J | 0.149 | 0.0521 | 0.0145 | 0.0327 | 0.26 | 0.259 | 0.00607 | EQUIVALENT_OR_BETTER |

`endpoint = same u` -> both runs landed on the same point, so Γ/Z_in/V_EOM are gated. Any other label means the two runs stopped at different points (a flat or multi-optimum set), so those columns are diagnostic there. `same J, different u` means the objective agreed within tolerance while the design coordinates did not. Coordinate columns (‖Δu‖∞, ΔL, ΔC, Δf_p) are always reported, never gated.

### Raw polish endpoints (before Deb pre-polish retention)

Reported separately so an agreement caused by *both* modes being discarded by the frozen pre-polish retention rule is not mistaken for an agreement of the two optimizers.

| Basin | polish kept FD/AN | raw J FD/AN | raw v_max FD/AN | Δ raw J | ‖Δ raw u‖∞ |
| --- | --- | --- | --- | --: | --: |
| 92d33827b0e1#0 | True/True | 3.411694/3.3914481 | 0/0 | -0.0202 | 0.0294 |
| 617c868e9f7e#0 | True/True | 3.4096292/3.2874828 | 0/0 | -0.122 | 0.0327 |

Termination messages (diagnostic only, not gated):

* `92d33827b0e1dc0f198ed6a183c67f1cc21b592ea76e7c9a84899fb74e4e5941#0` FD: The maximum number of function evaluations is exceeded. — AN: The maximum number of function evaluations is exceeded.
* `617c868e9f7ee37a709132d6a94953d112803d84aa98ca5b1894ad901dbeb4f6#0` FD: The maximum number of function evaluations is exceeded. — AN: The maximum number of function evaluations is exceeded.

**Fallbacks observed:** none — every candidate stayed on the analytical path.

Peak RSS during polish: FD 194 MB, analytical 194 MB.

### FD parameter-perturbation multiplier

| Basin | Np | FD nfev/njev | AN nfev/njev | expected FD model Np+1 |
| --- | --: | --: | --: | --: |
| 92d33827b0e1dc0f198ed6a183c67f1cc21b592ea76e7c9a84899fb74e4e5941#0 | 6 | 7 | 1 | 7 |
| 617c868e9f7ee37a709132d6a94953d112803d84aa98ca5b1894ad901dbeb4f6#0 | 7 | 8 | 1 | 8 |

## TYPICAL_FOSTER

2 targets, 2+1 cells, 101-point grid, top-K=3 (3 real starts). `local_max_iterations` capped at **60**, identically for both modes. DE (shared, run once): 1.4 s. Basin pairs: 5.

### Work and cost

| Basin | Mode | wall s | nit | nfev | njev | c_nfev | c_njev | evaluator freq solves | txn factorizations | direct backsolves | adjoint backsolves | txn builds | reuse |
| --- | --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 9d5713f0ad7877618d12726bdb0e34969128863d10370c2a3c8f53c2f8905090#0 | FD | 31.327 | 60 | 540 | 60 | 540 | 0 | 97869 | 0 | 0 | 0 | 0 | 0 |
| 9d5713f0ad7877618d12726bdb0e34969128863d10370c2a3c8f53c2f8905090#0 | AN | 3.837 | 60 | 60 | 60 | 60 | 60 | 5555 | 6060 | 960 | 5940 | 60 | 61 |
| ed64f19a4a9eeaa9dccb70085616de52ffa40988d1c8e866d82ae29cd1f83b56#0 | FD | 32.541 | 60 | 540 | 60 | 540 | 0 | 103020 | 0 | 0 | 0 | 0 | 0 |
| ed64f19a4a9eeaa9dccb70085616de52ffa40988d1c8e866d82ae29cd1f83b56#0 | AN | 3.797 | 60 | 60 | 60 | 60 | 60 | 5555 | 6060 | 960 | 5940 | 60 | 61 |
| 053753e3d486d810a02ae7396454c5dea923c86bb32f29982cdcd52d03106b0f#0 | FD | 31.895 | 60 | 558 | 62 | 558 | 0 | 101303 | 0 | 0 | 0 | 0 | 0 |
| 053753e3d486d810a02ae7396454c5dea923c86bb32f29982cdcd52d03106b0f#0 | AN | 3.890 | 60 | 60 | 60 | 60 | 60 | 5656 | 6060 | 960 | 5940 | 60 | 61 |
| 88405ed55218a44c8490964017180e47f7cf66c679c5f047cd0e7b302f6055c2#0 | FD | 33.098 | 60 | 540 | 60 | 540 | 0 | 103020 | 0 | 0 | 0 | 0 | 0 |
| 88405ed55218a44c8490964017180e47f7cf66c679c5f047cd0e7b302f6055c2#0 | AN | 4.050 | 60 | 60 | 60 | 60 | 60 | 5555 | 6060 | 960 | 5940 | 60 | 61 |
| e9c1bdab494456b6e0f353c833dffe6124807940062ae91465f4d27577bd777a#0 | FD | 37.890 | 60 | 600 | 60 | 600 | 0 | 115140 | 0 | 0 | 0 | 0 | 0 |
| e9c1bdab494456b6e0f353c833dffe6124807940062ae91465f4d27577bd777a#0 | AN | 4.055 | 60 | 60 | 60 | 60 | 60 | 5555 | 6060 | 1080 | 5940 | 60 | 61 |

### Scientific equivalence

| Basin | feasible FD/AN | v_max FD/AN | J FD/AN | endpoint | Δ\|Γ\| | ΔZin rel | ΔV_EOM rel | ‖Δu‖∞ | ΔL rel | ΔC rel | Δf_p rel | Verdict |
| --- | --- | --- | --- | --- | --: | --: | --: | --: | --: | --: | --: | --- |
| 9d5713f0ad78#0 | True/True | 0/0 | 8.3013092/8.3013092 | same u | 0 | 0 | 0 | 0 | 0 | 0 | 0 | EQUIVALENT |
| ed64f19a4a9e#0 | True/True | 0/0 | 8.3013092/8.3013092 | same u | 0 | 0 | 0 | 0 | 0 | 0 | 0 | EQUIVALENT |
| 053753e3d486#0 | True/True | 0/0 | 8.3013092/8.3013092 | same u | 0 | 0 | 0 | 0 | 0 | 0 | 0 | EQUIVALENT |
| 88405ed55218#0 | False/False | 0.3121/0.3121 | 8.3013092/8.3013092 | same u | 0 | 0 | 0 | 0 | 0 | 0 | 0 | EQUIVALENT |
| e9c1bdab4944#0 | True/True | 0/0 | 8.3013092/8.3013092 | same u | 0 | 0 | 0 | 0 | 0 | 0 | 0 | EQUIVALENT |

`endpoint = same u` -> both runs landed on the same point, so Γ/Z_in/V_EOM are gated. Any other label means the two runs stopped at different points (a flat or multi-optimum set), so those columns are diagnostic there. `same J, different u` means the objective agreed within tolerance while the design coordinates did not. Coordinate columns (‖Δu‖∞, ΔL, ΔC, Δf_p) are always reported, never gated.

### Raw polish endpoints (before Deb pre-polish retention)

Reported separately so an agreement caused by *both* modes being discarded by the frozen pre-polish retention rule is not mistaken for an agreement of the two optimizers.

| Basin | polish kept FD/AN | raw J FD/AN | raw v_max FD/AN | Δ raw J | ‖Δ raw u‖∞ |
| --- | --- | --- | --- | --: | --: |
| 9d5713f0ad78#0 | False/False | 7.8062731/2.6057361 | 0.1275/9.723 | -5.2 | 0.0138 |
| ed64f19a4a9e#0 | False/False | 4.6381673/1.8627597 | 15.32/27.11 | -2.78 | 0.0341 |
| 053753e3d486#0 | False/False | 1.4902916/2.4247876 | 8.965/432.8 | 0.934 | 0.21 |
| 88405ed55218#0 | False/False | 7.3889403/1.6643197 | 0.3226/2.919 | -5.72 | 0.121 |
| e9c1bdab4944#0 | False/False | 3.7503631/1.8627597 | 16.03/27.11 | -1.89 | 0.0697 |

Termination messages (diagnostic only, not gated):

* `9d5713f0ad7877618d12726bdb0e34969128863d10370c2a3c8f53c2f8905090#0` FD: The maximum number of function evaluations is exceeded. — AN: The maximum number of function evaluations is exceeded.
* `ed64f19a4a9eeaa9dccb70085616de52ffa40988d1c8e866d82ae29cd1f83b56#0` FD: The maximum number of function evaluations is exceeded. — AN: The maximum number of function evaluations is exceeded.
* `053753e3d486d810a02ae7396454c5dea923c86bb32f29982cdcd52d03106b0f#0` FD: The maximum number of function evaluations is exceeded. — AN: The maximum number of function evaluations is exceeded.
* `88405ed55218a44c8490964017180e47f7cf66c679c5f047cd0e7b302f6055c2#0` FD: The maximum number of function evaluations is exceeded. — AN: The maximum number of function evaluations is exceeded.
* `e9c1bdab494456b6e0f353c833dffe6124807940062ae91465f4d27577bd777a#0` FD: The maximum number of function evaluations is exceeded. — AN: The maximum number of function evaluations is exceeded.

**Fallbacks observed:** none — every candidate stayed on the analytical path.

Peak RSS during polish: FD 655 MB, analytical 655 MB.

### FD parameter-perturbation multiplier

| Basin | Np | FD nfev/njev | AN nfev/njev | expected FD model Np+1 |
| --- | --: | --: | --: | --: |
| 9d5713f0ad7877618d12726bdb0e34969128863d10370c2a3c8f53c2f8905090#0 | 8 | 9 | 1 | 9 |
| ed64f19a4a9eeaa9dccb70085616de52ffa40988d1c8e866d82ae29cd1f83b56#0 | 8 | 9 | 1 | 9 |
| 053753e3d486d810a02ae7396454c5dea923c86bb32f29982cdcd52d03106b0f#0 | 8 | 9 | 1 | 9 |
| 88405ed55218a44c8490964017180e47f7cf66c679c5f047cd0e7b302f6055c2#0 | 8 | 9 | 1 | 9 |
| e9c1bdab494456b6e0f353c833dffe6124807940062ae91465f4d27577bd777a#0 | 9 | 10 | 1 | 10 |

## LARGE_MULTIPOLE

3 targets, 3+3 cells (max Np), 101-point grid, top-K=3. `local_max_iterations` capped at **40**, identically for both modes. DE (shared, run once): 1.4 s. Basin pairs: 3.

### Work and cost

| Basin | Mode | wall s | nit | nfev | njev | c_nfev | c_njev | evaluator freq solves | txn factorizations | direct backsolves | adjoint backsolves | txn builds | reuse |
| --- | --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 6017f6d45e7e25226bddbc5a14e307a9cdc6379dfd1a0c351f059df5f8806fc3#0 | FD | 44.379 | 40 | 585 | 39 | 585 | 0 | 113625 | 0 | 0 | 0 | 0 | 0 |
| 6017f6d45e7e25226bddbc5a14e307a9cdc6379dfd1a0c351f059df5f8806fc3#0 | AN | 3.422 | 40 | 40 | 40 | 40 | 40 | 3838 | 4040 | 1680 | 3920 | 40 | 41 |
| afad4bac964759fb4b603f89e23e093ad8ed817cde7fb35043b8aeb424ca8cfc#0 | FD | 37.076 | 40 | 520 | 40 | 520 | 0 | 101000 | 0 | 0 | 0 | 0 | 0 |
| afad4bac964759fb4b603f89e23e093ad8ed817cde7fb35043b8aeb424ca8cfc#0 | AN | 3.186 | 40 | 40 | 40 | 40 | 40 | 3838 | 4040 | 1440 | 3920 | 40 | 41 |
| 129a8d3272539a9a5be2320824a3f46827b9f048b1a0906d2cf7e6907eb56fca#0 | FD | 38.815 | 40 | 520 | 40 | 520 | 0 | 101000 | 0 | 0 | 0 | 0 | 0 |
| 129a8d3272539a9a5be2320824a3f46827b9f048b1a0906d2cf7e6907eb56fca#0 | AN | 3.781 | 40 | 40 | 40 | 40 | 40 | 3838 | 4040 | 1440 | 3920 | 40 | 41 |

### Scientific equivalence

| Basin | feasible FD/AN | v_max FD/AN | J FD/AN | endpoint | Δ\|Γ\| | ΔZin rel | ΔV_EOM rel | ‖Δu‖∞ | ΔL rel | ΔC rel | Δf_p rel | Verdict |
| --- | --- | --- | --- | --- | --: | --: | --: | --: | --: | --: | --: | --- |
| 6017f6d45e7e#0 | True/True | 0/0 | 13.781587/13.781587 | same u | 0 | 0 | 0 | 0 | 0 | 0 | 0 | EQUIVALENT |
| afad4bac9647#0 | True/True | 0/0 | 13.781587/13.781587 | same u | 0 | 0 | 0 | 0 | 0 | 0 | 0 | EQUIVALENT |
| 129a8d327253#0 | False/False | 0.09984/0.09984 | 13.781587/13.781587 | same u | 0 | 0 | 0 | 0 | 0 | 0 | 0 | EQUIVALENT |

`endpoint = same u` -> both runs landed on the same point, so Γ/Z_in/V_EOM are gated. Any other label means the two runs stopped at different points (a flat or multi-optimum set), so those columns are diagnostic there. `same J, different u` means the objective agreed within tolerance while the design coordinates did not. Coordinate columns (‖Δu‖∞, ΔL, ΔC, Δf_p) are always reported, never gated.

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

Peak RSS during polish: FD 850 MB, analytical 697 MB.

### FD parameter-perturbation multiplier

| Basin | Np | FD nfev/njev | AN nfev/njev | expected FD model Np+1 |
| --- | --: | --: | --: | --: |
| 6017f6d45e7e25226bddbc5a14e307a9cdc6379dfd1a0c351f059df5f8806fc3#0 | 14 | 15 | 1 | 15 |
| afad4bac964759fb4b603f89e23e093ad8ed817cde7fb35043b8aeb424ca8cfc#0 | 12 | 13 | 1 | 13 |
| 129a8d3272539a9a5be2320824a3f46827b9f048b1a0906d2cf7e6907eb56fca#0 | 12 | 13 | 1 | 13 |

## MULTI_FREQUENCY

4 targets (9-12 MHz), default 1-3 cell search, 101-point grid, top-K=2. `local_max_iterations` capped at **40**, identically for both modes. DE (shared, run once): 1.8 s. Basin pairs: 3.

### Work and cost

| Basin | Mode | wall s | nit | nfev | njev | c_nfev | c_njev | evaluator freq solves | txn factorizations | direct backsolves | adjoint backsolves | txn builds | reuse |
| --- | --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| fba586ba9023c237c092c56bb0bd458beb6dd49249dd023910699c0c126d50a0#0 | FD | 56.048 | 40 | 600 | 40 | 600 | 0 | 115140 | 0 | 0 | 0 | 0 | 0 |
| fba586ba9023c237c092c56bb0bd458beb6dd49249dd023910699c0c126d50a0#0 | AN | 4.588 | 40 | 40 | 40 | 40 | 40 | 4040 | 4040 | 2240 | 3880 | 40 | 41 |
| 9cbbe15388c8b6078cae1dcc363d6d145523a4245706d100ac01c590b6cfab30#0 | FD | 53.690 | 40 | 600 | 40 | 600 | 0 | 107666 | 0 | 0 | 0 | 0 | 0 |
| 9cbbe15388c8b6078cae1dcc363d6d145523a4245706d100ac01c590b6cfab30#0 | AN | 4.650 | 40 | 40 | 40 | 40 | 40 | 4040 | 4040 | 2240 | 3880 | 40 | 41 |
| d7260beea27011d3cb56ec01238c3fa4ec34f1500f16c7ff348cff8d687fcd0e#0 | FD | 58.185 | 40 | 600 | 40 | 600 | 0 | 117160 | 0 | 0 | 0 | 0 | 0 |
| d7260beea27011d3cb56ec01238c3fa4ec34f1500f16c7ff348cff8d687fcd0e#0 | AN | 4.584 | 40 | 40 | 40 | 40 | 40 | 4040 | 4040 | 2240 | 3880 | 40 | 41 |

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

Peak RSS during polish: FD 873 MB, analytical 763 MB.

### FD parameter-perturbation multiplier

| Basin | Np | FD nfev/njev | AN nfev/njev | expected FD model Np+1 |
| --- | --: | --: | --: | --: |
| fba586ba9023c237c092c56bb0bd458beb6dd49249dd023910699c0c126d50a0#0 | 14 | 15 | 1 | 15 |
| 9cbbe15388c8b6078cae1dcc363d6d145523a4245706d100ac01c590b6cfab30#0 | 14 | 15 | 1 | 15 |
| d7260beea27011d3cb56ec01238c3fa4ec34f1500f16c7ff348cff8d687fcd0e#0 | 14 | 15 | 1 | 15 |

## PATHOLOGICAL_1201_GRID

unmodified example grid: 1201 points / 1198 off-target hard rows, top-K=1, 1 domain, DE budget reduced (shared by both modes). `local_max_iterations` capped at **20**, identically for both modes. DE (shared, run once): 12.4 s. Basin pairs: 1.

### Work and cost

| Basin | Mode | wall s | nit | nfev | njev | c_nfev | c_njev | evaluator freq solves | txn factorizations | direct backsolves | adjoint backsolves | txn builds | reuse |
| --- | --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 43c25d79ed1bcb0e53a661278711916baac9afb0b8ac57ae67346cd5d9813317#0 | FD | 261.420 | 20 | 260 | 20 | 260 | 0 | 600500 | 0 | 0 | 0 | 0 | 0 |
| 43c25d79ed1bcb0e53a661278711916baac9afb0b8ac57ae67346cd5d9813317#0 | AN | 20.142 | 20 | 20 | 20 | 20 | 20 | 24020 | 24020 | 720 | 23960 | 20 | 21 |

### Scientific equivalence

| Basin | feasible FD/AN | v_max FD/AN | J FD/AN | endpoint | Δ\|Γ\| | ΔZin rel | ΔV_EOM rel | ‖Δu‖∞ | ΔL rel | ΔC rel | Δf_p rel | Verdict |
| --- | --- | --- | --- | --- | --: | --: | --: | --: | --: | --: | --: | --- |
| 43c25d79ed1b#0 | False/False | 0.2435/0.2435 | 13.781587/13.781587 | same u | 0 | 0 | 0 | 0 | 0 | 0 | 0 | EQUIVALENT |

`endpoint = same u` -> both runs landed on the same point, so Γ/Z_in/V_EOM are gated. Any other label means the two runs stopped at different points (a flat or multi-optimum set), so those columns are diagnostic there. `same J, different u` means the objective agreed within tolerance while the design coordinates did not. Coordinate columns (‖Δu‖∞, ΔL, ΔC, Δf_p) are always reported, never gated.

### Raw polish endpoints (before Deb pre-polish retention)

Reported separately so an agreement caused by *both* modes being discarded by the frozen pre-polish retention rule is not mistaken for an agreement of the two optimizers.

| Basin | polish kept FD/AN | raw J FD/AN | raw v_max FD/AN | Δ raw J | ‖Δ raw u‖∞ |
| --- | --- | --- | --- | --: | --: |
| 43c25d79ed1b#0 | False/False | 3.2853624/2.1256677 | 14.66/23.89 | -1.16 | 0.142 |

Termination messages (diagnostic only, not gated):

* `43c25d79ed1bcb0e53a661278711916baac9afb0b8ac57ae67346cd5d9813317#0` FD: The maximum number of function evaluations is exceeded. — AN: The maximum number of function evaluations is exceeded.

**Fallbacks observed:** none — every candidate stayed on the analytical path.

Peak RSS during polish: FD 3654 MB, analytical 816 MB.

### FD parameter-perturbation multiplier

| Basin | Np | FD nfev/njev | AN nfev/njev | expected FD model Np+1 |
| --- | --: | --: | --: | --: |
| 43c25d79ed1bcb0e53a661278711916baac9afb0b8ac57ae67346cd5d9813317#0 | 12 | 13 | 1 | 13 |

### Analytical profile (self time, cProfile)

| Category | seconds | % |
| --- | --: | --: |
| numpy per-call overhead (errstate, finiteness reductions) | 5.191 | 15.2% |
| nominal sweep: circuit measurements | 3.677 | 10.7% |
| MNA assembly (nominal) | 3.468 | 10.1% |
| nominal sweep: EOM model | 2.844 | 8.3% |
| Y_p derivative stamps (transaction) | 2.823 | 8.2% |
| optimizer overhead: trust-constr QR / projections | 1.769 | 5.2% |
| conditioning check (cond / SVD) | 1.738 | 5.1% |
| adjoint sensitivities | 1.624 | 4.7% |
| MNA solve / LU | 1.591 | 4.6% |
| coordinate unpack (variable_map) | 1.471 | 4.3% |
| constraint / objective layout evaluate | 0.119 | 0.3% |
| observables + Jacobian assembly | 0.057 | 0.2% |
| direct sensitivities | 0.002 | 0.0% |
| other / interpreter overhead | 7.852 | 22.9% |

## Acceptance findings

**1. Scientific equivalence.** 14 A/B pairs across 5 cases; 0 not equivalent, 2 strictly better under exact gradients. Feasibility, max hard-constraint violation and objective are equivalent-or-better on every pair. Where the two runs stopped at different points, the analytical run was never the worse of the two. Note that on the pathological case both modes were rejected by the frozen Deb pre-polish retention rule at the shared iteration cap, so that pair's agreement reflects the retained representative rather than two agreeing optimizer endpoints - the raw endpoints for it are tabulated separately above.

**2. Fallbacks / status.** None. All 14 candidates completed on the analytical path - no `UNSUPPORTED`, nonsmooth, unresolved, or construction-failure state was hit, and no unexpected solver status appeared. SciPy termination wording differs between modes but is diagnostic only.

**3. FD-induced work removed.** Aggregate nominal frequency-point solves fall from **1,810,547** (FD) to **159,025** (analytical: 77,945 evaluator + 81,080 transaction), a **11.4x** reduction, plus 17,700 direct and 79,460 adjoint back-substitutions - back-solves against an already-computed factorization, not fresh sweeps. Aggregate polish wall time falls from 727.1s to 65.7s (S_T = **11.07x**). The parameter-perturbation multiplier is gone: FD's `nfev/njev` equals `Np+1` on every basin measured, analytical's equals 1.

**4. Memory.** No regression - analytical peak RSS is equal or lower in every case. Largest gap (PATHOLOGICAL_1201_GRID): FD 3654 MB vs analytical 816 MB. FD's cost comes from the evaluator cache retaining one full `EvaluationResult` - including its complete solution tuple - per perturbed point; the transaction holds a single current-u slot and drops it on the next u.

**5. Repository gates.** Recorded in the commit for this phase.

### P12.5-F targets indicated by these measurements

Reported only - nothing is optimized or tuned in P12.5-E.

1. **Duplicate nominal sweep (highest value).** The transaction re-solves the entire frequency grid that `evaluate()` has already solved and cached for the same u, so the analytical path pays two nominal sweeps per iterate. It is roughly half of all analytical nominal MNA work (81,080 of 159,025 solves aggregated, and exactly half in the pathological and multi-frequency cases), and the measured profile puts the nominal sweep (circuit measurements + EOM model + MNA assembly + LU) far ahead of the sensitivity kernels themselves. Sharing nominal state between the evaluator and the transaction is the single largest remaining win.
2. **Per-call NumPy overhead in the hot loops.** The largest single measured bucket (~15% self time) is `errstate` context entry/exit, `_make_extobj`, and elementwise finiteness reductions executed once per frequency per element - per-call overhead, not arithmetic. Vectorising the sweep across frequencies would remove it together with much of the assembly cost.
3. **`trust-constr` QR / null-space projection on a ~1236-row constraint Jacobian.** Optimizer-side cost that scales with the off-target row count rather than with Np, and does not shrink when the Jacobian becomes exact. Reducing the off-target hard-row count - or aggregating those rows into an envelope constraint - attacks it directly.

Also noted, lower value: the transaction runs its off-target adjoint sweep over every off-target index unconditionally, even when no hard/soft descriptor references it; and `variable_map` coordinate unpacking costs ~4% self time because it is repeated per frequency rather than once per u.
