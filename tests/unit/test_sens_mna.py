import numpy as np
import pytest

from foster_eom.circuit.mna import (
    SolverOptions,
    solve_mna,
    solve_mna_factorized,
)
from foster_eom.errors import CircuitSolveStatus


def test_solve_mna_factorized_ordinary():
    # Ordinary case
    Y = np.array([[2.0 + 1j, -1.0], [-1.0, 2.0 - 1j]], dtype=np.complex128)
    I_vec = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)

    opts = SolverOptions()

    # Run both
    v_nom, stat_nom, diag_nom = solve_mna(Y, I_vec, opts)
    state_fac, stat_fac, diag_fac = solve_mna_factorized(Y, I_vec, opts)

    assert stat_nom == CircuitSolveStatus.OK
    assert stat_fac == CircuitSolveStatus.OK
    assert v_nom is not None
    assert state_fac is not None

    np.testing.assert_allclose(v_nom, state_fac.V_nominal, rtol=1e-12, atol=1e-12)
    assert diag_nom.condition_number == diag_fac.condition_number
    assert diag_nom.residual_norm == pytest.approx(diag_fac.residual_norm)


def test_solve_mna_factorized_singular():
    # Singular case
    Y = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)
    I_vec = np.array([1.0, 2.0], dtype=np.complex128)

    opts = SolverOptions(condition_threshold=1e14)

    v_nom, stat_nom, diag_nom = solve_mna(Y, I_vec, opts)
    state_fac, stat_fac, diag_fac = solve_mna_factorized(Y, I_vec, opts)

    assert stat_nom == CircuitSolveStatus.SINGULAR_OR_ILL_CONDITIONED
    assert stat_fac == CircuitSolveStatus.SINGULAR_OR_ILL_CONDITIONED
    assert v_nom is None
    assert state_fac is None
    assert diag_nom.condition_number == diag_fac.condition_number


def test_solve_mna_factorized_ill_conditioned():
    # Ill-conditioned case (near threshold)
    Y = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-15]], dtype=np.complex128)
    I_vec = np.array([1.0, 2.0], dtype=np.complex128)

    opts = SolverOptions(condition_threshold=1e14)

    v_nom, stat_nom, diag_nom = solve_mna(Y, I_vec, opts)
    state_fac, stat_fac, diag_fac = solve_mna_factorized(Y, I_vec, opts)

    assert stat_nom == CircuitSolveStatus.SINGULAR_OR_ILL_CONDITIONED
    assert stat_fac == CircuitSolveStatus.SINGULAR_OR_ILL_CONDITIONED
    assert v_nom is None
    assert state_fac is None
    assert diag_nom.condition_number == diag_fac.condition_number


def test_solve_mna_factorized_nonfinite():
    # Nonfinite case
    Y = np.array([[1.0, 1.0], [1.0, np.nan]], dtype=np.complex128)
    I_vec = np.array([1.0, 2.0], dtype=np.complex128)

    opts = SolverOptions()

    v_nom, stat_nom, diag_nom = solve_mna(Y, I_vec, opts)
    state_fac, stat_fac, diag_fac = solve_mna_factorized(Y, I_vec, opts)

    assert stat_nom == CircuitSolveStatus.SINGULAR_OR_ILL_CONDITIONED
    assert stat_fac == CircuitSolveStatus.SINGULAR_OR_ILL_CONDITIONED
    assert v_nom is None
    assert state_fac is None
    assert diag_nom.nonfinite_in_matrix
    assert diag_fac.nonfinite_in_matrix


def test_solve_mna_factorized_resonant():
    # Resonant LC circuit exactly at resonance, very low damping.
    # High condition number, but maybe below threshold.
    w = 1e6
    L = 1e-6
    C = 1 / (w**2 * L)
    R_loss = 1e-9

    Y11 = 1 / R_loss + 1j * w * C + 1 / (1j * w * L)
    Y = np.array([[Y11]], dtype=np.complex128)
    I_vec = np.array([1.0], dtype=np.complex128)

    opts = SolverOptions(condition_threshold=1e14)

    v_nom, stat_nom, diag_nom = solve_mna(Y, I_vec, opts)
    state_fac, stat_fac, diag_fac = solve_mna_factorized(Y, I_vec, opts)

    # They should behave identically
    assert stat_nom == stat_fac
    if stat_nom == CircuitSolveStatus.OK:
        np.testing.assert_allclose(v_nom, state_fac.V_nominal, rtol=1e-12, atol=1e-12)
    assert diag_nom.condition_number == diag_fac.condition_number
