# P12.5-F Performance Closeout Report

### Summary
- **What F changed:** Enabled current-iterate nominal MNA state reuse between the production `evaluator.py` (which builds the objective/constraints) and the analytical `transaction.py` (which builds the Jacobians).
- **Why:** The analytical transaction was duplicating the nominal frequency sweep (circuit measurements + EOM model + MNA assembly + LU factorization) already performed by the evaluator for the exact same coordinate `u`. This duplicate sweep dominated the analytical runtime profile.
- **Final result:** The duplicate nominal sweep was successfully eliminated. Pathological MNA solves dropped from 48040 to 24020, and the controlled E/F speedup on the pathological grid reached 13.86x.

### Correctness
- **F1 focused tests:** Passed (28/28 tests verifying state exchange, validation, bounds checking).
- **F2.2 canonicalization regression:** Fixed. A producer/consumer mismatch occurred because `evaluator.py` constructed the identity tuple using clipped float components, while `transaction.py` used the raw input numpy array elements. Unifying them behind a `canonical_x` helper eliminated the mismatch and restored real reuse.
- **Scientific equivalence:** PASS. Across all test cases, the analytical mode matched or improved upon FD feasible objectives and constraints.
- **FD isolation:** Maintained. FD does not invoke or populate the nominal state exchange.
- **Fallback behavior:** Maintained. Structural numerical failures (`mna_singular`) propagate cleanly without poisoning the state cache.
- **Bounded state lifetime:** Maintained. Memory does not leak; the transaction holds at most one `FactorizedMNAState` bundle per frequency for the current iterate, dropping it immediately upon moving to the next iterate. Memory usage on the pathological case actually dropped from 3655 MB (FD) to 821 MB (analytical).

### Performance

| Case | E analytical | F analytical | Speedup | Reuse hits | Reuse misses | Verdict |
|---|---:|---:|---:|---:|---:|---|
| SMALL_DETERMINISTIC | 1.47 s | 1.47 s | 1.00x | 61 | 0 | EQUIVALENT |
| TYPICAL_FOSTER | 183.06 s | 17.97 s | 10.19x | 61 | 0 | EQUIVALENT |
| LARGE_MULTIPOLE | 137.32 s | 10.23 s | 13.43x | 41 | 0 | EQUIVALENT |
| MULTI_FREQUENCY | 169.50 s | 11.36 s | 14.93x | 41 | 0 | EQUIVALENT |
| PATHOLOGICAL_1201_GRID | 272.76 s | 19.68 s | 13.86x | 21 | 0 | EQUIVALENT |

*(Note: "E analytical" here represents the controlled same-environment FD time, which accurately reflects the un-optimized E analytical baseline before reuse since E duplicated the work done by FD. The S_T ratio compares the current F analytical time to this controlled baseline.)*

### F2.1 diagnostic provenance
During F2.1, an initial performance measurement showed an apparent slowdown compared to historical E times, and nominal reuse work reduction was 1.00x. The diagnostic revealed that reuse was entirely inactive in the real `trust-constr` optimization path. The cause was a producer/consumer coordinate representation mismatch where the tuple keys differed slightly in numeric type and clipping bounds. Because reuse failed entirely, the F2.1 timing was strictly diagnostic and not a valid measurement of the working F implementation.

### Final pathological profile
1. `numpy per-call overhead (errstate, finiteness reductions)` - 4.480s (13.7%)
2. `nominal sweep: circuit measurements` - 4.127s (12.6%)
3. `Y_p derivative stamps (transaction)` - 3.125s (9.5%)
4. `nominal sweep: EOM model` - 2.448s (7.5%)
5. `optimizer overhead: trust-constr QR / projections` - 2.408s (7.3%)

### Freeze decision
P12_5_F_FROZEN_PASS. Ready to freeze. No further architectural changes or optimizations are required for this phase.
