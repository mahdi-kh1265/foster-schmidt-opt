
--- RESULTS ---
| Case | Start | Np | Ng | FD feasible | Analytical feasible | FD hard violation | Analytical hard violation | FD objective | Analytical objective | Same endpoint? | Max |Δu| | Unexpected fallback | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G3-A Small deterministic | [np.float64(0.5), np.float64(0.5), np.float64(0.5)] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.638e-01 | 9.638e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |
| G3-B Representative | [np.float64(0.2), np.float64(0.4), np.float64(0.6)] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.644e-01 | 9.644e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |
| G3-C Multipole | [np.float64(0.5), np.float64(0.5), np.float64(0.5), np.float64(0.5), np.float64(0.5), np.float64(0.5), np.float64(0.5)] | 7 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.637e-01 | 9.638e-01 | False | 2.597e-01 | No | EQUIVALENT_DIFFERENT_ENDPOINT |
| G3-D Multi-frequency | [np.float64(0.5), np.float64(0.5), np.float64(0.5)] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.525e-01 | 9.525e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |
| G3-E Boundary clipped | [np.float64(-0.05), np.float64(1.05), np.float64(0.5)] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.637e-01 | 9.636e-01 | False | 3.838e-01 | No | ANALYTICAL_BETTER |
| G3-F Pole separation | [np.float64(0.5), np.float64(0.5), np.float64(0.49), np.float64(0.5), np.float64(0.51)] | 5 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.636e-01 | 9.638e-01 | False | 3.303e-01 | No | EQUIVALENT_DIFFERENT_ENDPOINT |
| G3-G Hard + soft | [np.float64(0.5), np.float64(0.5), np.float64(0.5)] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.638e-01 | 9.638e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |
| G3-H Loss enabled | [np.float64(0.5), np.float64(0.5), np.float64(0.5)] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.638e-01 | 9.638e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |
| G3-I Multi-start 1 | [np.float64(0.1), np.float64(0.2), np.float64(0.3)] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.657e-01 | 9.657e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |
| G3-I Multi-start 2 | [np.float64(0.5), np.float64(0.5), np.float64(0.5)] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.638e-01 | 9.638e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |
| G3-I Multi-start 3 | [np.float64(0.9), np.float64(0.8), np.float64(0.7)] | 3 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.637e-01 | 9.637e-01 | False | 2.678e-04 | No | EQUIVALENT_DIFFERENT_ENDPOINT |
| G3-J Pathological | [np.float64(0.5), np.float64(0.5), np.float64(0.5), np.float64(0.5), np.float64(0.5), np.float64(0.5), np.float64(0.5), np.float64(0.5), np.float64(0.5), np.float64(0.5), np.float64(0.5), np.float64(0.5), np.float64(0.5)] | 13 | N/A | True | True | 0.000e+00 | 0.000e+00 | 9.638e-01 | 9.638e-01 | True | 0.000e+00 | No | EQUIVALENT_SAME_ENDPOINT |
