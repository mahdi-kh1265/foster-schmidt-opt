# P12.5-G Final Freeze Report

P12.5-G establishes that the frozen analytical local-polish architecture preserves the project's scientific optimization semantics and acceptance behavior, executes without hidden numerical Jacobian fallback on supported smooth cases, maintains bounded heavy-state lifetime, and provides substantial directly measured wall-time acceleration relative to the frozen finite-difference reference.

## G1 — Identity / invalidation
PASS.

Exact canonical-coordinate reuse, distinct-u invalidation, clipping equivalence, context/frequency invalidation, FD isolation, real callback reuse.

## G2 — Fallback / isolation
PASS.

G2 deliberately exercised unsupported, nonsmooth, incomplete, nonfinite, malformed-shape, and transaction-failure analytical states and proved they trigger clean candidate-level `REFERENCE_FD` fallback without stale analytical state leakage. 
* expected deliberate fallbacks > 0
* unexpected fallback = 0
* original candidate-start restart semantics passed
* later analytical candidates remained cleanly isolated

## G3 — Scientific equivalence
FROZEN PASS.

12 controlled A/B pairs:
* 8 equivalent same endpoint
* 3 equivalent different endpoint
* 1 analytical better
* 0 non-equivalent
* 0 unexpected fallback

## G4 — Execution-path / hidden FD
FROZEN PASS.

Directly proved via a direct numerical-differentiation spy and independent callback/evaluator-pattern evidence:
* hidden analytical objective FD = 0
* hidden analytical constraint FD = 0
* numerical differentiation active in `REFERENCE_FD`
* FD perturbation cloud = observed
* analytical perturbation cloud = absent
* post-publication current-state eligible misses = 0
* duplicate same-u nominal sweep after valid publication = 0
* unexpected fallback = 0

## G5 — Memory / lifetime
FROZEN PASS.

Demonstrates bounded heavy numerical state under the audited lifecycles. Heavy numerical state remains bounded by fixed ownership limits independent of optimization history; historical bundles become collectable and permanent lifetime regressions now exist.
* maximum committed reusable heavy bundle = 1
* maximum pending heavy bundle = 1
* maximum total exchange-owned heavy bundles <= 2
* maximum transaction-owned heavy reference = 1

Dynamic audits successfully observed:
* 250 distinct-u stress
* 25 sequential candidate polishes
* 5 fallback/recovery churn cycles
* replaced heavy-state weakref collectability
* large ~1200-frequency bundle replacement/collectability
* 218-state memory trend
* RSS plateau approximately 103.9 MB
* stale-state false hits = 0
* historical-coordinate revisit misses are legitimate bounded-cache misses, not reuse failures

**Constraint Profile:**
For the large fixture:
* Np = 13
* Ng = 1233
* nominal frequencies = 1200
* target frequencies represented = 1
* off-target frequencies represented = 1199
* off-target voltage rows = 1199

Constraint arithmetic:
4 target match rows + 1 target source-current row + 1199 off-target EOM-voltage rows + 24 nonlinear component-bound rows + 5 pole-separation rows = 1233 total nonlinear constraint rows.

## G6 — Controlled performance
PASS.

In the frozen G6 environment, the current production `ANALYTICAL` path achieved a directly measured median local-polish speedup of approximately **5.2×** on the representative workload and **11.7×** on the capped large/pathological workload relative to the frozen `REFERENCE_FD` path.

A later representative analytical rerun completed in 2.06 s, remaining in the same fast-performance regime. Cache/reuse correctness is independently established by G1, G4, and G5.

Every timed A/B comparison remained scientifically acceptable under the frozen G3 hierarchy. The large/pathological benchmark specifically achieved `EQUIVALENT_DIFFERENT_ENDPOINT`.

### Benchmark Environment & Provenance
* benchmark Git commit: `64d3a64`
* platform: Windows-11-10.0.26200-SP0
* CPU model: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel
* Python version: 3.14.2
* NumPy version: 2.4.6
* SciPy version: 1.18.0
* BLAS/LAPACK backend: unknown
* relevant threading environment: not recorded during G6
* exact representative Np: 5, Ng: 113, frequency counts: 100
* exact large-case Np: 13, Ng: 1233, frequency counts: 1200
* trust-constr caps: 50 (representative), 15 (large-case)
* A/B repetition order, raw repetition timing table, and same-start/same-settings confirmation are permanently recorded in the benchmark script artifact `scripts/p12_5_g6_final_benchmark.py` and its execution log artifacts.
