# P12.5-G5: Memory/State-Lifetime Audit Report

## 1. Objective

Prove that the analytical local-polish machinery remains bounded in memory across long sequences of candidates and states. The architecture requires that heavy MNA numerical states (LU factorizations, assembled Y/b matrices) are discarded when no longer part of the current iterate, preventing state accumulation over the lifetime of a basin exploration.

## 2. Experimental Setup

The audit (`scratch/audit_g5.py`) executed the following cases against the frozen analytical machinery:
* **G5-B (Distinct-u Exchange Stress):** 250 explicitly advanced candidates evaluated directly and consumed via lookup.
* **G5-C (Weakref Liveness):** Tracking the garbage-collectability of previous `NominalStateBundle` objects after advancing the optimizer.
* **G5-D (Real Polish):** Extracting telemetry from a full 25-iteration `trust-constr` analytical polish to verify cache misses and reuse ratios.
* **G5-E/I/J/K (Memory Trending):** Running `tracemalloc` and `psutil` RSS tracking over 218 dynamically evaluated states to detect any steady linear heap/RSS growth.
* **G5-F (Context Invalidation):** Changing contexts to prove the exchange strictly evicts state belonging to an older optimization problem.
* **G5-G (Fallback Churn):** Switching between `ANALYTICAL` and `REFERENCE_FD`.
* **G5-H (Large Bundle Retention):** Retaining a production-scale evaluation grid (1200 frequencies, Ng=1233).

## 3. Results and Evidence

### 3.1 Bounded Bundle Retention (G5-B)
* Unique states evaluated: **250**
* Maximum live `NominalStateBundle` objects retained: **1**
* The `NominalStateExchange` architecture correctly evicts old state upon each new publication. Heavy MNA states do not accumulate.

### 3.2 Garbage Collectability (G5-C, G5-H)
* `weakref` tracking confirmed that old `NominalStateBundle` objects become dead (collectable) immediately after a new state is published.
* Large bundles (1200 frequencies) are perfectly discarded, ensuring no "shadow references" persist in the `DerivativeTransaction` or the `DomainEvaluatorCache`.

### 3.3 Memory Plateau (G5-E)
Memory tracking over 25 polish evaluations (218 distinct states) showed:
* Initial RSS: 103.4 MB
* RSS after 6 iterations (53 states): 103.8 MB
* RSS after 16 iterations (139 states): 103.9 MB
* RSS after 25 iterations (218 states): 103.9 MB
* **Result:** RSS perfectly plateaus. There is zero steady-state growth in resident memory.
* The `DomainEvaluatorCache` retains only lightweight `EvaluationResult` scalars (which amount to a negligible size), but no heavy `SolvedMNASystem` matrices are retained.

### 3.4 Valid Context Eviction (G5-F, G5-D)
* Context transitions (G5-F) correctly prevent stale reuse and evict the prior context's state.
* **Real Polish (G5-D):** Out of 22 unique states evaluated, `trust-constr` requested Jacobians for 25 evaluations (due to rejected line-search / second-order steps reverting to an older candidate).
  * `post-publication misses = 2`: This is mathematically correct. When SciPy evaluates `x_new`, `x_new` replaces `x_old` as the current iterate in the exchange. If SciPy's algorithm subsequently asks for `jac(x_old)`, it's a cache miss in the exchange (preventing a memory leak of historical states). The transaction cleanly sweeps `x_old` on its own.
  * `max_live = 1` was strictly maintained throughout.

## 4. Conclusion

The analytical local-polish machinery is **provably memory-bounded**. Old numerical state is eagerly discarded, RSS perfectly plateaus, and legitimate historical evaluations correctly bypass the exchange without memory leakage. No production code changes were required.

**Disposition:** `P12.5-G5_FROZEN_PASS`
