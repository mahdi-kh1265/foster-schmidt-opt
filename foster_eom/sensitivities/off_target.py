import numpy as np

from foster_eom.circuit.graph import CircuitGraph
from foster_eom.circuit.mna import FactorizedMNAState
from foster_eom.sensitivities.adjoint import compute_adjoint_state, compute_adjoint_gradient

def compute_v_eom_adjoint_gradient(
    graph: CircuitGraph,
    node_map: dict[str, int],
    state: FactorizedMNAState,
    Y_p_list: list[np.ndarray],
) -> np.ndarray:
    """Compute the parameter gradient of f = |V_eom| using the adjoint method.

    Parameters
    ----------
    graph : CircuitGraph
        The circuit graph containing the EOM element.
    node_map : dict[str, int]
        Mapping from non-ground node IDs to matrix indices.
    state : FactorizedMNAState
        The factorized nominal MNA state.
    Y_p_list : list[np.ndarray]
        List of K derivative admittance matrices.

    Returns
    -------
    grad : np.ndarray
        Real gradient vector of shape (K,) where entry k is d|V_eom|/dp_k.
    """
    n = state.V_nominal.shape[0]
    k = len(Y_p_list)
    q = np.zeros(n, dtype=np.complex128)
    
    eom_id = graph.eom_element_id
    if eom_id is None or eom_id not in graph.elements:
        return np.zeros(k, dtype=np.float64)
        
    eom_elem = graph.elements[eom_id]
    pos = eom_elem.node_pos
    neg = eom_elem.node_neg
    
    idx_pos = -1 if pos == graph.ground_node_id else node_map[pos]
    idx_neg = -1 if neg == graph.ground_node_id else node_map[neg]
    
    v_pos = 0.0j if idx_pos == -1 else state.V_nominal[idx_pos]
    v_neg = 0.0j if idx_neg == -1 else state.V_nominal[idx_neg]
    
    v_eom = v_pos - v_neg
    abs_v = abs(v_eom)
    
    if abs_v > 0:
        # f = |V_eom|
        # df / dV_eom^* = V_eom / (2 * |V_eom|)
        df_dveom_conj = v_eom / (2.0 * abs_v)
        
        if idx_pos != -1:
            q[idx_pos] += df_dveom_conj
        if idx_neg != -1:
            q[idx_neg] -= df_dveom_conj
            
    lam = compute_adjoint_state(state, q)
    grad = compute_adjoint_gradient(lam, Y_p_list, state.V_nominal)
    
    return grad
