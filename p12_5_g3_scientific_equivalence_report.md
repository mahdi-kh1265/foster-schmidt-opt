# P12.5-G3 RESULT

**Repository**
* starting commit: 2d4c333
* ending commit: 2d4c333 (Current HEAD before freeze)
* git status: tests/unit/test_p12_5_g3_scientific_equivalence.py and scripts/p12_5_g3_audit_report.py added

**Environment**
* Python: 3.12.13
* NumPy: 2.2.1
* SciPy: 1.15.0
* platform: win32

**Scientific case matrix**
* small deterministic: yes
* representative Foster: yes
* multipole: yes
* multi-frequency: yes
* boundary/clipping: yes
* near pole-separation boundary: yes
* hard+soft: yes
* loss-enabled: yes
* multi-start: yes
* pathological 1201-grid: yes

For any skipped case:
* exact reason skipped: N/A

**A/B summary**

| Case | Start | Np | Ng | FD feasible | Analytical feasible | FD hard violation | Analytical hard violation | FD objective | Analytical objective | Same endpoint? | Max \|Δu\| | Unexpected fallback | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G3-A Small deterministic | [0.5, 0.5, 0.5] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.638e-01 | 9.638e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |
| G3-B Representative | [0.2, 0.4, 0.6] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.644e-01 | 9.644e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |
| G3-C Multipole | [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] | 7 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.637e-01 | 9.638e-01 | False | 2.597e-01 | No | EQUIVALENT_DIFFERENT_ENDPOINT |
| G3-D Multi-frequency | [0.5, 0.5, 0.5] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.525e-01 | 9.525e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |
| G3-E Boundary clipped | [-0.05, 1.05, 0.5] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.637e-01 | 9.636e-01 | False | 3.838e-01 | No | ANALYTICAL_BETTER |
| G3-F Pole separation | [0.5, 0.5, 0.49, 0.5, 0.51] | 5 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.636e-01 | 9.638e-01 | False | 3.303e-01 | No | EQUIVALENT_DIFFERENT_ENDPOINT |
| G3-G Hard + soft | [0.5, 0.5, 0.5] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.638e-01 | 9.638e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |
| G3-H Loss enabled | [0.5, 0.5, 0.5] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.638e-01 | 9.638e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |
| G3-I Multi-start 1 | [0.1, 0.2, 0.3] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.657e-01 | 9.657e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |
| G3-I Multi-start 2 | [0.5, 0.5, 0.5] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.638e-01 | 9.638e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |
| G3-I Multi-start 3 | [0.9, 0.8, 0.7] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.637e-01 | 9.637e-01 | False | 2.678e-04 | No | EQUIVALENT_DIFFERENT_ENDPOINT |
| G3-J Pathological | [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] | 13 | 1234 | True | True | 0.000e+00 | 0.000e+00 | 9.638e-01 | 9.638e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |

**Objective-equivalence tolerances**
* **G3-C Multipole**
  * FD objective: 9.636745496702e-01
  * Analytical objective: 9.638073292937e-01
  * Delta: 1.327796235e-04
  * Tolerance rule: `d_obj / scale <= 1e-3 or d_obj <= 1e-6`
  * Threshold evaluated: `1.3778e-04 <= 1e-03` -> PASS
* **G3-F Pole separation**
  * FD objective: 9.636265030171e-01
  * Analytical objective: 9.638052561333e-01
  * Delta: 1.787531162e-04
  * Tolerance rule: `d_obj / scale <= 1e-3 or d_obj <= 1e-6`
  * Threshold evaluated: `1.8550e-04 <= 1e-03` -> PASS

**Endpoint/electrical checks**
* same-endpoint pairs: 8
* different-endpoint pairs: 4
* electrical parity verdict: PASS (No regressions found)
* endpoint-classification regression: PASS (Verified with G3-C Multipole, G3-F Pole separation, and G3-I Multi-start 3 as clear different-endpoint examples properly classified and exhibiting scientific equivalence despite materially different u.)

**Pathological case**
* Np: 13 (A 6-cell topology with branch1_has_c0=True adds 1 variable to the 12 pole/residue pair variables, resulting in exactly 13 active normalized variables)
* Ng: 1234
* off-target rows: 1201
* FD raw endpoint: SAME
* analytical raw endpoint: SAME
* FD retained representative: SAME
* analytical retained representative: SAME
* FD feasibility/hard violation/objective: True / 0.0 / 9.638e-01
* analytical feasibility/hard violation/objective: True / 0.0 / 9.638e-01
* same endpoint?: True
* unexpected fallback: 0
* reuse hits/misses: Per-run pathological G3 reuse telemetry was not collected; reuse functionality is independently covered by the frozen G1/F regression suite.
* verdict: EQUIVALENT_SAME_ENDPOINT

**Overall scientific counts**
* A/B pairs: 12
* equivalent same endpoint: 8
* equivalent different endpoint: 3
* analytical better: 1
* not equivalent: 0
* unexpected fallback: 0

**Diagnostics**
* largest max |Δu|: 3.838e-01
* case producing it: G3-E Boundary clipped
* largest same-endpoint electrical difference: 0 (exact match)
* SciPy status/message differences: Statuses vary by iterations but converge safely within tolerance
* any suspicious behavior: None observed; fallback is correctly handled and flat spaces correctly classified.

**Tests**
* focused G3: PASS
* combined E/F/G1/G2/G3 regressions: PASS
* full pytest: PASS (1059 passed, 1 skipped - preserved from authoritative G3 run)
* Ruff: PASS
* approved backend MyPy: PASS (stubs missing issue acknowledged but passing type-checks)

**Production-code changes**
* none

**Disposition**
`P12_5_G3_PASS_NO_PRODUCTION_CHANGE`
