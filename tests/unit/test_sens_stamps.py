import numpy as np

from foster_eom.circuit.graph import Element, ElementKind
from foster_eom.circuit.stamps import stamp_element
from foster_eom.sensitivities.stamps import (
    stamp_capacitor_derivative,
    stamp_inductor_derivative,
)


def test_stamp_capacitor_derivative_fd():
    """Verify analytical dY/dC against central finite difference."""
    f_hz = 10e6
    C_val = 100e-12
    h = 1e-15  # Small step for FD

    node_map = {"n1": 0, "n2": 1}
    gnd = "gnd"

    # Analytical
    elem = Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val)
    Y_p = np.zeros((2, 2), dtype=np.complex128)
    stamp_capacitor_derivative(Y_p, elem, node_map, gnd, f_hz)

    # FD
    elem_plus = Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val + h)
    elem_minus = Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val - h)

    Y_plus = np.zeros((2, 2), dtype=np.complex128)
    Y_minus = np.zeros((2, 2), dtype=np.complex128)
    stamp_element(Y_plus, elem_plus, node_map, gnd, f_hz)
    stamp_element(Y_minus, elem_minus, node_map, gnd, f_hz)

    Y_fd = (Y_plus - Y_minus) / (2 * h)

    np.testing.assert_allclose(Y_p, Y_fd, rtol=1e-5, atol=1e-12)


def test_stamp_inductor_derivative_fd():
    """Verify analytical dY/dL against central finite difference."""
    f_hz = 10e6
    L_val = 1e-6
    h = 1e-9  # Small step for FD

    node_map = {"n1": 0, "n2": 1}
    gnd = "gnd"

    # Analytical
    elem = Element("L1", ElementKind.INDUCTOR, "n1", "n2", L_val)
    Y_p = np.zeros((2, 2), dtype=np.complex128)
    stamp_inductor_derivative(Y_p, elem, node_map, gnd, f_hz)

    # FD
    elem_plus = Element("L1", ElementKind.INDUCTOR, "n1", "n2", L_val + h)
    elem_minus = Element("L1", ElementKind.INDUCTOR, "n1", "n2", L_val - h)

    Y_plus = np.zeros((2, 2), dtype=np.complex128)
    Y_minus = np.zeros((2, 2), dtype=np.complex128)
    stamp_element(Y_plus, elem_plus, node_map, gnd, f_hz)
    stamp_element(Y_minus, elem_minus, node_map, gnd, f_hz)

    Y_fd = (Y_plus - Y_minus) / (2 * h)

    np.testing.assert_allclose(Y_p, Y_fd, rtol=1e-5, atol=1e-12)


def test_stamp_capacitor_derivative_gnd():
    """Verify analytical dY/dC against central finite difference with ground."""
    f_hz = 10e6
    C_val = 100e-12
    h = 1e-15

    node_map = {"n1": 0}
    gnd = "gnd"

    elem = Element("C1", ElementKind.CAPACITOR, "n1", gnd, C_val)
    Y_p = np.zeros((1, 1), dtype=np.complex128)
    stamp_capacitor_derivative(Y_p, elem, node_map, gnd, f_hz)

    elem_plus = Element("C1", ElementKind.CAPACITOR, "n1", gnd, C_val + h)
    elem_minus = Element("C1", ElementKind.CAPACITOR, "n1", gnd, C_val - h)

    Y_plus = np.zeros((1, 1), dtype=np.complex128)
    Y_minus = np.zeros((1, 1), dtype=np.complex128)
    stamp_element(Y_plus, elem_plus, node_map, gnd, f_hz)
    stamp_element(Y_minus, elem_minus, node_map, gnd, f_hz)

    Y_fd = (Y_plus - Y_minus) / (2 * h)

    np.testing.assert_allclose(Y_p, Y_fd, rtol=1e-5, atol=1e-12)


def test_full_mna_assembly_fd():
    from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
    from foster_eom.circuit.mna import assemble_mna
    from foster_eom.sensitivities.stamps import (
        stamp_capacitor_derivative,
        stamp_inductor_derivative,
    )

    f_hz = 10e6
    h = 1e-15
    C_val = 100e-12
    L_val = 1e-6

    graph = CircuitGraph("gnd", Port("n1", "gnd"))
    graph.add_node(Node("n1"))
    graph.add_node(Node("n2"))
    graph.add_node(Node("n3"))
    graph.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val))
    graph.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val))
    graph.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))

    from foster_eom.domain.source import SourceSpec

    source_spec = SourceSpec(mode="thevenin", thevenin_vrms=1.0, z_source=50.0)

    Y_nom, b_nom, node_map = assemble_mna(graph, source_spec, f_hz)

    Y_p_C = np.zeros_like(Y_nom)
    Y_p_L = np.zeros_like(Y_nom)

    stamp_capacitor_derivative(Y_p_C, graph.elements["C1"], node_map, graph.ground_node_id, f_hz)
    stamp_inductor_derivative(Y_p_L, graph.elements["L1"], node_map, graph.ground_node_id, f_hz)

    # FD C1
    graph_c_plus = CircuitGraph("gnd", Port("n1", "gnd"))
    graph_c_plus.add_node(Node("n1"))
    graph_c_plus.add_node(Node("n2"))
    graph_c_plus.add_node(Node("n3"))
    graph_c_plus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val + h))
    graph_c_plus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val))
    graph_c_plus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_c_plus, _, _ = assemble_mna(graph_c_plus, source_spec, f_hz)

    graph_c_minus = CircuitGraph("gnd", Port("n1", "gnd"))
    graph_c_minus.add_node(Node("n1"))
    graph_c_minus.add_node(Node("n2"))
    graph_c_minus.add_node(Node("n3"))
    graph_c_minus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val - h))
    graph_c_minus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val))
    graph_c_minus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_c_minus, _, _ = assemble_mna(graph_c_minus, source_spec, f_hz)

    Y_c_fd = (Y_c_plus - Y_c_minus) / (2 * h)
    np.testing.assert_allclose(Y_p_C, Y_c_fd, rtol=1e-5, atol=1e-12)

    # FD L1
    h_l = 1e-9
    graph_l_plus = CircuitGraph("gnd", Port("n1", "gnd"))
    graph_l_plus.add_node(Node("n1"))
    graph_l_plus.add_node(Node("n2"))
    graph_l_plus.add_node(Node("n3"))
    graph_l_plus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val))
    graph_l_plus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val + h_l))
    graph_l_plus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_l_plus, _, _ = assemble_mna(graph_l_plus, source_spec, f_hz)

    graph_l_minus = CircuitGraph("gnd", Port("n1", "gnd"))
    graph_l_minus.add_node(Node("n1"))
    graph_l_minus.add_node(Node("n2"))
    graph_l_minus.add_node(Node("n3"))
    graph_l_minus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val))
    graph_l_minus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val - h_l))
    graph_l_minus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_l_minus, _, _ = assemble_mna(graph_l_minus, source_spec, f_hz)

    Y_l_fd = (Y_l_plus - Y_l_minus) / (2 * h_l)
    np.testing.assert_allclose(Y_p_L, Y_l_fd, rtol=1e-5, atol=1e-12)
