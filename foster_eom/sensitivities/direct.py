import numpy as np
import scipy.linalg

from foster_eom.circuit.mna import FactorizedMNAState

def compute_direct_state_sensitivities(
    state: FactorizedMNAState,
    Y_p_list: list[np.ndarray],
) -> np.ndarray:
    """Compute the direct state sensitivity matrix X_p = Y^{-1} (-Y_p V).

    Uses the cached LU factorization to perform a block back-substitution
    for all parameters simultaneously.

    Parameters
    ----------
    state : FactorizedMNAState
        The factorized nominal MNA state.
    Y_p_list : list[np.ndarray]
        List of length K containing derivative admittance matrices dY/dp_k.

    Returns
    -------
    X_p : np.ndarray
        The state sensitivity matrix of shape (N, K) where column k is dV/dp_k.
    """
    if not Y_p_list:
        n = state.V_nominal.shape[0]
        return np.zeros((n, 0), dtype=np.complex128)
        
    n = state.V_nominal.shape[0]
    k = len(Y_p_list)
    
    RHS = np.zeros((n, k), dtype=np.complex128)
    V = state.V_nominal
    
    for i, Y_p in enumerate(Y_p_list):
        RHS[:, i] = -(Y_p @ V)
        
    X_p = scipy.linalg.lu_solve(state.lu_and_piv, RHS)
    return X_p
