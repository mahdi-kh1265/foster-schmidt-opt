import numpy as np
from dataclasses import dataclass

from foster_eom.circuit.graph import CircuitGraph, GROUND
from foster_eom.domain.source import SourceSpec
from foster_eom.circuit.mna import FactorizedMNAState

@dataclass(frozen=True)
class ObservableDerivatives:
    """Exact target observable derivatives w.r.t continuous parameters.

    All arrays have shape (K,) where K is the number of parameters.
    """
    v_port: np.ndarray       # complex
    i_port: np.ndarray       # complex
    z_in: np.ndarray         # complex
    gamma: np.ndarray        # complex
    p_delivered: np.ndarray  # float (real scalar)


def compute_observable_derivatives(
    graph: CircuitGraph,
    source_spec: SourceSpec,
    node_map: dict[str, int],
    state: FactorizedMNAState,
    X_p: np.ndarray,
    v_port_nom: complex,
    i_port_nom: complex,
    z_in_nom: complex,
) -> ObservableDerivatives:
    """Compute exact derivatives of target observables.

    Parameters
    ----------
    graph : CircuitGraph
        The passive circuit netlist.
    source_spec : SourceSpec
        The source specification.
    node_map : dict[str, int]
        Mapping from non-ground node IDs to matrix indices.
    state : FactorizedMNAState
        The nominal state (used for dimensions, etc if needed).
    X_p : np.ndarray
        Direct state sensitivities of shape (N, K).
    v_port_nom : complex
        Nominal port voltage.
    i_port_nom : complex
        Nominal port current flowing from source into port.
    z_in_nom : complex
        Nominal input impedance Z_in = V_port / I_port.

    Returns
    -------
    ObservableDerivatives
    """
    if X_p.size == 0:
        return ObservableDerivatives(
            v_port=np.zeros(0, dtype=np.complex128),
            i_port=np.zeros(0, dtype=np.complex128),
            z_in=np.zeros(0, dtype=np.complex128),
            gamma=np.zeros(0, dtype=np.complex128),
            p_delivered=np.zeros(0, dtype=np.float64),
        )

    k_params = X_p.shape[1]

    node_pos = graph.input_port.node_pos
    node_neg = graph.input_port.node_neg

    idx_pos = -1 if node_pos == graph.ground_node_id else node_map[node_pos]
    idx_neg = -1 if node_neg == graph.ground_node_id else node_map[node_neg]

    dv_pos = np.zeros(k_params, dtype=np.complex128) if idx_pos == -1 else X_p[idx_pos, :]
    dv_neg = np.zeros(k_params, dtype=np.complex128) if idx_neg == -1 else X_p[idx_neg, :]

    # 1. d V_p / dp
    dv_port = dv_pos - dv_neg

    # 2. d I_s / dp
    # I_s = (V_{th} - V_p) / Z_s
    di_port = - dv_port / source_spec.z_source

    # 3. d Z_in / dp
    # Z_in = V_p / I_p
    # d Z_in / dp = (I_p * d V_p - V_p * d I_p) / I_p^2
    with np.errstate(divide="ignore", invalid="ignore"):
        if abs(i_port_nom) > 0:
            dz_in = (i_port_nom * dv_port - v_port_nom * di_port) / (i_port_nom**2)
        else:
            dz_in = np.full(k_params, complex(np.inf), dtype=np.complex128)

    # 4. d Gamma / dp
    # Gamma = (Z_in - Z0) / (Z_in + Z0)
    # d Gamma / d Z_in = 2 Z0 / (Z_in + Z0)^2
    z0 = source_spec.z_ref_ohm
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = z_in_nom + z0
        if abs(denom) > 0:
            dgamma_dzin = (2.0 * z0) / (denom**2)
            dgamma = dgamma_dzin * dz_in
        else:
            dgamma = np.full(k_params, complex(np.inf), dtype=np.complex128)

    # 5. d P / dp
    # P = Re(V_p * I_p^*)
    # dP/dp = Re( dV_p * I_p^* + V_p * dI_p^* )
    dp_delivered = np.real(dv_port * np.conj(i_port_nom) + v_port_nom * np.conj(di_port))

    return ObservableDerivatives(
        v_port=dv_port,
        i_port=di_port,
        z_in=dz_in,
        gamma=dgamma,
        p_delivered=dp_delivered,
    )
