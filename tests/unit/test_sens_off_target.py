import numpy as np
import pytest

from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
from foster_eom.circuit.mna import assemble_mna, solve_mna_factorized
from foster_eom.domain.source import SourceSpec
from foster_eom.sensitivities.stamps import stamp_capacitor_derivative
from foster_eom.sensitivities.off_target import compute_v_eom_adjoint_gradient
from foster_eom.errors import CircuitSolveStatus

def test_v_eom_adjoint_gradient():
    """Validates the off-target |V_eom| adjoint gradient against Central FD."""
    f_hz = 10e6
    C_val = 100e-12
    L_val = 1e-6
    
    source_spec = SourceSpec(mode="thevenin", thevenin_vrms=1.0, z_source=50.0)
    
    graph = CircuitGraph("gnd", Port("n1", "gnd"), eom_element_id="R1")
    graph.add_node(Node("n1")); graph.add_node(Node("n2")); graph.add_node(Node("n3"))
    graph.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val))
    graph.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val))
    graph.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    
    # 1. Nominal
    Y_nom, b_nom, node_map = assemble_mna(graph, source_spec, f_hz)
    state, status, _ = solve_mna_factorized(Y_nom, b_nom)
    assert status == CircuitSolveStatus.OK
    
    # 2. Adjoint gradient
    Y_p_C = np.zeros_like(Y_nom)
    stamp_capacitor_derivative(Y_p_C, graph.elements["C1"], node_map, graph.ground_node_id, f_hz)
    
    grad = compute_v_eom_adjoint_gradient(graph, node_map, state, [Y_p_C])
    
    # 3. FD
    h_C = 1e-15
    
    graph_plus = CircuitGraph("gnd", Port("n1", "gnd"), eom_element_id="R1")
    graph_plus.add_node(Node("n1")); graph_plus.add_node(Node("n2")); graph_plus.add_node(Node("n3"))
    graph_plus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val + h_C))
    graph_plus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val))
    graph_plus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_plus, b_plus, _ = assemble_mna(graph_plus, source_spec, f_hz)
    state_plus, _, _ = solve_mna_factorized(Y_plus, b_plus)
    v_eom_plus = abs(state_plus.V_nominal[node_map["n3"]])
    
    graph_minus = CircuitGraph("gnd", Port("n1", "gnd"), eom_element_id="R1")
    graph_minus.add_node(Node("n1")); graph_minus.add_node(Node("n2")); graph_minus.add_node(Node("n3"))
    graph_minus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val - h_C))
    graph_minus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val))
    graph_minus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_minus, b_minus, _ = assemble_mna(graph_minus, source_spec, f_hz)
    state_minus, _, _ = solve_mna_factorized(Y_minus, b_minus)
    v_eom_minus = abs(state_minus.V_nominal[node_map["n3"]])
    
    fd = (v_eom_plus - v_eom_minus) / (2 * h_C)
    
    np.testing.assert_allclose(grad[0], fd, rtol=1e-5, atol=1e-12)
