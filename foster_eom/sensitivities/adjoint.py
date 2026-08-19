import numpy as np
import scipy.linalg

from foster_eom.circuit.mna import FactorizedMNAState


def compute_adjoint_state(
    state: FactorizedMNAState,
    q: np.ndarray,
) -> np.ndarray:
    """Compute the adjoint state vector lambda from the objective derivative.

    Solves Y^H lambda = 2q efficiently using the cached LU factorization.

    Parameters
    ----------
    state : FactorizedMNAState
        The factorized nominal MNA state.
    q : np.ndarray
        The complex derivative vector q = df / dV^*.
        Shape (N,) or (N, M) for M scalar objectives.

    Returns
    -------
    lam : np.ndarray
        The adjoint state vector lambda, shape (N,) or (N, M).
    """
    RHS = 2.0 * q
    lam = scipy.linalg.lu_solve(state.lu_and_piv, RHS, trans=2)
    return np.asarray(lam)


def compute_adjoint_gradient(
    lam: np.ndarray,
    Y_p_list: list[np.ndarray],
    V_nominal: np.ndarray,
) -> np.ndarray:
    """Compute the real scalar gradient dj/dp from the adjoint state.

    dj/dp = Re(lambda^H (-Y_p V))

    Parameters
    ----------
    lam : np.ndarray
        Adjoint state matrix of shape (N,) or (N, M).
    Y_p_list : list[np.ndarray]
        List of K derivative admittance matrices.
    V_nominal : np.ndarray
        The nominal node voltage vector, shape (N,).

    Returns
    -------
    grad : np.ndarray
        The real gradient matrix of shape (M, K) where entry (m, k) is
        the derivative of objective m with respect to parameter k.
        If lam is 1D (N,), returns shape (K,).
    """
    k = len(Y_p_list)
    is_1d = lam.ndim == 1

    if not Y_p_list:
        if is_1d:
            return np.zeros(0, dtype=np.float64)
        m = lam.shape[1]
        return np.zeros((m, 0), dtype=np.float64)

    m = 1 if is_1d else lam.shape[1]
    grad = np.zeros((m, k), dtype=np.float64)

    lam_H = lam.conj().T if not is_1d else lam.conj().reshape(1, -1)

    for i, Y_p in enumerate(Y_p_list):
        term = -(Y_p @ V_nominal)
        grad[:, i] = np.real(lam_H @ term)

    if is_1d:
        return grad[0, :]
    return grad
