"""P12.5-G4: Execution-Path Audit."""

import numpy as np
import pytest
from unittest.mock import patch
import scipy.optimize

from foster_eom.domain.objectives import DerivativeMode
from foster_eom.optimize.evaluator import DomainEvaluatorCache
from foster_eom.optimize.local_polish import polish_basin
from tests.unit.test_p12_5_e_analytical_polish import X0, _basin, _spec
from tests.unit.test_p12_5_g3_scientific_equivalence import _build_custom_case, _run_pair

# ---------------------------------------------------------------------------
# G4-A: Callback Wiring Audit
# ---------------------------------------------------------------------------

def test_g4_a_callback_wiring():
    """G4-A: Verify callback wiring to SciPy minimize."""
    ctx = _build_custom_case(n_cells=1)
    
    captured_kwargs_an = {}
    captured_kwargs_fd = {}
    
    original_minimize = scipy.optimize.minimize
    
    def mock_minimize_an(*args, **kwargs):
        captured_kwargs_an.update(kwargs)
        return original_minimize(*args, **kwargs)

    def mock_minimize_fd(*args, **kwargs):
        captured_kwargs_fd.update(kwargs)
        return original_minimize(*args, **kwargs)

    # Run ANALYTICAL
    with patch("scipy.optimize.minimize", side_effect=mock_minimize_an):
        cache_an = DomainEvaluatorCache()
        spec_an = _spec(DerivativeMode.ANALYTICAL, max_iter=2)
        polish_basin(_basin(ctx, cache_an, X0), 0, ctx, cache_an, spec_an)
        
    # Run REFERENCE_FD
    with patch("scipy.optimize.minimize", side_effect=mock_minimize_fd):
        cache_fd = DomainEvaluatorCache()
        spec_fd = _spec(DerivativeMode.REFERENCE_FD, max_iter=2)
        polish_basin(_basin(ctx, cache_fd, X0), 0, ctx, cache_fd, spec_fd)

    # Verify ANALYTICAL callbacks
    assert callable(captured_kwargs_an["jac"]), "ANALYTICAL objective jac must be a callable"
    assert captured_kwargs_an["jac"] not in ["2-point", "3-point", "cs"]
    nlc_an = captured_kwargs_an["constraints"]
    assert callable(nlc_an.jac), "ANALYTICAL constraint jac must be a callable"
    assert nlc_an.jac not in ["2-point", "3-point", "cs"]
    
    # Verify REFERENCE_FD callbacks
    assert captured_kwargs_fd["jac"] == "2-point", "REFERENCE_FD objective jac must be '2-point'"
    nlc_fd = captured_kwargs_fd["constraints"]
    assert nlc_fd.jac == "2-point" or not callable(nlc_fd.jac), "REFERENCE_FD constraint jac must be numerical"

# ---------------------------------------------------------------------------
# G4-B, C, D, E, F, G: Execution Path Invariants
# ---------------------------------------------------------------------------

def test_g4_b_through_g_small_deterministic():
    """G4-B-G: Verify hidden FD, reuse, and duplicate sweeps on a small case."""
    ctx = _build_custom_case(n_cells=1) # Np = 3
    x_start = np.array([0.5, 0.5, 0.5])
    
    pr_fd, pr_an = _run_pair(ctx, x_start, max_iter=5)
    
    t_an = pr_an.telemetry
    t_fd = pr_fd.telemetry
    
    # No fallback masking
    assert t_an.derivative_mode == "analytical"
    assert t_an.fallback_reason is None
    
    Np = 3
    
    # G4-B & G4-C: No hidden FD
    assert t_an.nfev < 20, f"nfev={t_an.nfev} is suspiciously large for 5 iters without FD"
    assert t_fd.nfev >= t_fd.n_iterations * (Np + 1), "REFERENCE_FD should do Np+1 evaluations per iteration"
    
    # G4-D: Perturbation pattern audit
    assert t_an.evaluator_unique_evaluations <= t_an.nfev + 2, "Analytical should not have a perturbation cloud"
    assert t_fd.evaluator_unique_evaluations > t_fd.n_iterations * Np, "FD should have a large perturbation cloud"
    
    # G4-E: Same-u transaction reuse
    assert t_an.transaction_reuse_hits > 0, "Transaction should be reused between obj/constr jac callbacks"
    
    # G4-G: Duplicate sweep audit
    duplicate_sweeps = t_an.transaction_nominal_sweep_solves - (t_an.n_evaluation_frequencies * t_an.nominal_bundle_misses)
    assert duplicate_sweeps == 0, "No duplicate transaction nominal sweeps allowed after publication"
    
def test_g4_i_representative():
    """G4-I: Representative case."""
    ctx = _build_custom_case(base_grid_points=15)
    x_rep = np.array([0.2, 0.4, 0.6])
    pr_fd, pr_an = _run_pair(ctx, x_rep, max_iter=5)
    
    t_an = pr_an.telemetry
    
    assert t_an.derivative_mode == "analytical"
    assert t_an.fallback_reason is None
    assert t_an.transaction_reuse_hits > 0
    duplicate_sweeps = t_an.transaction_nominal_sweep_solves - (t_an.n_evaluation_frequencies * t_an.nominal_bundle_misses)
    assert duplicate_sweeps == 0
    assert t_an.evaluator_unique_evaluations <= t_an.nfev + 2

def test_g4_j_large_constraint_short():
    """G4-J: Large-constraint / pathological short audit."""
    ctx = _build_custom_case(n_cells=6, base_grid_points=1201) # Np = 13, Ng = 1234
    x_start = np.full(13, 0.5)
    
    pr_fd, pr_an = _run_pair(ctx, x_start, max_iter=2)
    
    t_an = pr_an.telemetry
    
    assert t_an.derivative_mode == "analytical"
    assert t_an.fallback_reason is None
    assert t_an.transaction_reuse_hits > 0
    duplicate_sweeps = t_an.transaction_nominal_sweep_solves - (t_an.n_evaluation_frequencies * t_an.nominal_bundle_misses)
    assert duplicate_sweeps == 0
    assert t_an.n_constraint_rows == 1234
    assert t_an.n_params == 13
    assert t_an.evaluator_unique_evaluations <= t_an.nfev + 2

def test_g4_k_hidden_fd_spy():
    """G4-K: Direct hidden-FD spy."""
    ctx = _build_custom_case(n_cells=1)
    x_start = np.array([0.5, 0.5, 0.5])
    
    import sys
    
    def run_with_spy(mode):
        numdiff_calls = []
        def trace_calls(frame, event, arg):
            if event == 'call':
                func_name = frame.f_code.co_name
                if func_name == 'approx_derivative' and 'scipy' in frame.f_globals.get('__name__', ''):
                    try:
                        fun = frame.f_locals.get('fun')
                        x0 = frame.f_locals.get('x0')
                        if fun and x0 is not None:
                            val = fun(x0)
                            shape = np.atleast_2d(val).shape
                            numdiff_calls.append(shape)
                    except Exception:
                        pass
            return trace_calls

        spec = _spec(mode, max_iter=2)
        cache = DomainEvaluatorCache()
        
        sys.settrace(trace_calls)
        try:
            polish_basin(_basin(ctx, cache, x_start), 0, ctx, cache, spec)
        finally:
            sys.settrace(None)
            
        return numdiff_calls

    calls_an = run_with_spy(DerivativeMode.ANALYTICAL)
    calls_fd = run_with_spy(DerivativeMode.REFERENCE_FD)
    
    # Analyze calls
    an_obj_fd = sum(1 for shape in calls_an if shape[1] == 1)
    an_con_fd = sum(1 for shape in calls_an if shape[1] > 1)
    
    fd_obj_fd = sum(1 for shape in calls_fd if shape[1] == 1)
    fd_con_fd = sum(1 for shape in calls_fd if shape[1] > 1)
    
    assert an_obj_fd == 0, "ANALYTICAL should not use numerical differentiation for objective"
    assert an_con_fd == 0, "ANALYTICAL should not use numerical differentiation for constraints"
    
    assert fd_obj_fd > 0, "REFERENCE_FD must use numerical differentiation for objective"
    assert fd_con_fd > 0, "REFERENCE_FD must use numerical differentiation for constraints"
