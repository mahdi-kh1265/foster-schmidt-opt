import numpy as np

from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
from foster_eom.circuit.mna import assemble_mna, solve_mna_factorized
from foster_eom.domain.source import SourceSpec
from foster_eom.errors import CircuitSolveStatus
from foster_eom.sensitivities.direct import compute_direct_state_sensitivities
from foster_eom.sensitivities.observables import compute_observable_derivatives
from foster_eom.sensitivities.stamps import stamp_capacitor_derivative


def test_observable_derivatives():
    """Validates observable derivatives against Central FD."""
    f_hz = 10e6
    C_val = 100e-12
    L_val = 1e-6

    source_spec = SourceSpec(mode="thevenin", thevenin_vrms=1.0, z_source=50.0)

    graph = CircuitGraph("gnd", Port("n1", "gnd"))
    graph.add_node(Node("n1"))
    graph.add_node(Node("n2"))
    graph.add_node(Node("n3"))
    graph.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val))
    graph.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val))
    graph.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))

    # 1. Nominal
    Y_nom, b_nom, node_map = assemble_mna(graph, source_spec, f_hz)
    state, status, _ = solve_mna_factorized(Y_nom, b_nom)
    assert status == CircuitSolveStatus.OK

    v_pos_nom = state.V_nominal[node_map["n1"]]
    v_port_nom = v_pos_nom  # since neg is ground
    i_port_nom = (source_spec.vth_phasor - v_port_nom) / source_spec.z_source
    z_in_nom = v_port_nom / i_port_nom

    # 2. Sensitivities
    Y_p_C = np.zeros_like(Y_nom)
    stamp_capacitor_derivative(Y_p_C, graph.elements["C1"], node_map, graph.ground_node_id, f_hz)
    X_p = compute_direct_state_sensitivities(state, [Y_p_C])

    obs = compute_observable_derivatives(
        graph, source_spec, node_map, state, X_p, v_port_nom, i_port_nom, z_in_nom
    )

    # 3. FD
    h_C = 1e-15

    graph_plus = CircuitGraph("gnd", Port("n1", "gnd"))
    graph_plus.add_node(Node("n1"))
    graph_plus.add_node(Node("n2"))
    graph_plus.add_node(Node("n3"))
    graph_plus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val + h_C))
    graph_plus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val))
    graph_plus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_plus, b_plus, _ = assemble_mna(graph_plus, source_spec, f_hz)
    state_plus, _, _ = solve_mna_factorized(Y_plus, b_plus)
    v_port_plus = state_plus.V_nominal[node_map["n1"]]
    i_port_plus = (source_spec.vth_phasor - v_port_plus) / source_spec.z_source
    z_in_plus = v_port_plus / i_port_plus
    gamma_plus = (z_in_plus - source_spec.z_ref_ohm) / (z_in_plus + source_spec.z_ref_ohm)
    p_plus = np.real(v_port_plus * np.conj(i_port_plus))

    graph_minus = CircuitGraph("gnd", Port("n1", "gnd"))
    graph_minus.add_node(Node("n1"))
    graph_minus.add_node(Node("n2"))
    graph_minus.add_node(Node("n3"))
    graph_minus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val - h_C))
    graph_minus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val))
    graph_minus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_minus, b_minus, _ = assemble_mna(graph_minus, source_spec, f_hz)
    state_minus, _, _ = solve_mna_factorized(Y_minus, b_minus)
    v_port_minus = state_minus.V_nominal[node_map["n1"]]
    i_port_minus = (source_spec.vth_phasor - v_port_minus) / source_spec.z_source
    z_in_minus = v_port_minus / i_port_minus
    gamma_minus = (z_in_minus - source_spec.z_ref_ohm) / (z_in_minus + source_spec.z_ref_ohm)
    p_minus = np.real(v_port_minus * np.conj(i_port_minus))

    fd_v = (v_port_plus - v_port_minus) / (2 * h_C)
    fd_i = (i_port_plus - i_port_minus) / (2 * h_C)
    fd_z = (z_in_plus - z_in_minus) / (2 * h_C)
    fd_gamma = (gamma_plus - gamma_minus) / (2 * h_C)
    fd_p = (p_plus - p_minus) / (2 * h_C)

    np.testing.assert_allclose(obs.v_port[0], fd_v, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(obs.i_port[0], fd_i, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(obs.z_in[0], fd_z, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(obs.gamma[0], fd_gamma, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(obs.p_delivered[0], fd_p, rtol=1e-5, atol=1e-12)
