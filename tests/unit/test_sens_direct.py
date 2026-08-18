import numpy as np
import pytest

from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
from foster_eom.circuit.mna import assemble_mna, solve_mna_factorized
from foster_eom.domain.source import SourceSpec
from foster_eom.sensitivities.stamps import stamp_capacitor_derivative, stamp_inductor_derivative
from foster_eom.sensitivities.direct import compute_direct_state_sensitivities
from foster_eom.errors import CircuitSolveStatus

def test_direct_state_sensitivity():
    """Validates the direct state sensitivity matrices.
    
    1. Direct Residual: Y X_p + Y_p V = 0
    2. Central FD: X_p ~ (V_{p+h} - V_{p-h}) / (2h)
    """
    f_hz = 10e6
    C_val = 100e-12
    L_val = 1e-6
    
    source_spec = SourceSpec(mode="thevenin", thevenin_vrms=1.0, z_source=50.0)
    
    # 1. Compute nominal state
    graph = CircuitGraph("gnd", Port("n1", "gnd"))
    graph.add_node(Node("n1")); graph.add_node(Node("n2")); graph.add_node(Node("n3"))
    graph.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val))
    graph.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val))
    graph.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    
    Y_nom, b_nom, node_map = assemble_mna(graph, source_spec, f_hz)
    
    state, status, _ = solve_mna_factorized(Y_nom, b_nom)
    assert status == CircuitSolveStatus.OK
    assert state is not None
    
    # 2. Compute analytical direct state sensitivities
    Y_p_C = np.zeros_like(Y_nom)
    Y_p_L = np.zeros_like(Y_nom)
    stamp_capacitor_derivative(Y_p_C, graph.elements["C1"], node_map, graph.ground_node_id, f_hz)
    stamp_inductor_derivative(Y_p_L, graph.elements["L1"], node_map, graph.ground_node_id, f_hz)
    
    Y_p_list = [Y_p_C, Y_p_L]
    X_p = compute_direct_state_sensitivities(state, Y_p_list)
    
    assert X_p.shape == (Y_nom.shape[0], 2)
    
    X_p_C = X_p[:, 0]
    X_p_L = X_p[:, 1]
    
    # --- Validation 1: Direct Residual ---
    # Y * X_p_k + Y_p_k * V_nom = 0 (assuming b_nom is invariant wrt p)
    residual_C = Y_nom @ X_p_C + Y_p_C @ state.V_nominal
    residual_L = Y_nom @ X_p_L + Y_p_L @ state.V_nominal
    
    np.testing.assert_allclose(residual_C, 0, atol=1e-5)
    np.testing.assert_allclose(residual_L, 0, atol=1e-5)
    
    # --- Validation 2: Central Finite Differences ---
    h_C = 1e-15
    h_L = 1e-9
    
    # FD for C1
    graph_c_plus = CircuitGraph("gnd", Port("n1", "gnd"))
    graph_c_plus.add_node(Node("n1")); graph_c_plus.add_node(Node("n2")); graph_c_plus.add_node(Node("n3"))
    graph_c_plus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val + h_C))
    graph_c_plus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val))
    graph_c_plus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_c_plus, b_c_plus, _ = assemble_mna(graph_c_plus, source_spec, f_hz)
    state_c_plus, _, _ = solve_mna_factorized(Y_c_plus, b_c_plus)
    
    graph_c_minus = CircuitGraph("gnd", Port("n1", "gnd"))
    graph_c_minus.add_node(Node("n1")); graph_c_minus.add_node(Node("n2")); graph_c_minus.add_node(Node("n3"))
    graph_c_minus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val - h_C))
    graph_c_minus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val))
    graph_c_minus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_c_minus, b_c_minus, _ = assemble_mna(graph_c_minus, source_spec, f_hz)
    state_c_minus, _, _ = solve_mna_factorized(Y_c_minus, b_c_minus)
    
    V_fd_C = (state_c_plus.V_nominal - state_c_minus.V_nominal) / (2 * h_C)
    np.testing.assert_allclose(X_p_C, V_fd_C, rtol=1e-5, atol=1e-12)
    
    # FD for L1
    graph_l_plus = CircuitGraph("gnd", Port("n1", "gnd"))
    graph_l_plus.add_node(Node("n1")); graph_l_plus.add_node(Node("n2")); graph_l_plus.add_node(Node("n3"))
    graph_l_plus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val))
    graph_l_plus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val + h_L))
    graph_l_plus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_l_plus, b_l_plus, _ = assemble_mna(graph_l_plus, source_spec, f_hz)
    state_l_plus, _, _ = solve_mna_factorized(Y_l_plus, b_l_plus)
    
    graph_l_minus = CircuitGraph("gnd", Port("n1", "gnd"))
    graph_l_minus.add_node(Node("n1")); graph_l_minus.add_node(Node("n2")); graph_l_minus.add_node(Node("n3"))
    graph_l_minus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val))
    graph_l_minus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val - h_L))
    graph_l_minus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_l_minus, b_l_minus, _ = assemble_mna(graph_l_minus, source_spec, f_hz)
    state_l_minus, _, _ = solve_mna_factorized(Y_l_minus, b_l_minus)
    
    V_fd_L = (state_l_plus.V_nominal - state_l_minus.V_nominal) / (2 * h_L)
    np.testing.assert_allclose(X_p_L, V_fd_L, rtol=1e-5, atol=1e-12)
