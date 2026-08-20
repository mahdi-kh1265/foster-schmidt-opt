import numpy as np

from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
from foster_eom.circuit.mna import assemble_mna, solve_mna_factorized
from foster_eom.domain.source import SourceSpec
from foster_eom.errors import CircuitSolveStatus
from foster_eom.sensitivities.adjoint import compute_adjoint_gradient, compute_adjoint_state
from foster_eom.sensitivities.direct import compute_direct_state_sensitivities
from foster_eom.sensitivities.stamps import stamp_capacitor_derivative, stamp_inductor_derivative


def test_adjoint_identity():
    """Validates the complex adjoint identity: w^H Y v = (Y^H w)^H v."""
    N = 5
    np.random.seed(42)
    Y = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    v = np.random.randn(N) + 1j * np.random.randn(N)
    w = np.random.randn(N) + 1j * np.random.randn(N)

    left = w.conj().T @ Y @ v
    right = (Y.conj().T @ w).conj().T @ v

    np.testing.assert_allclose(left, right, atol=1e-12)


def test_adjoint_gradient_validation():
    """Validates adjoint gradient vs Direct and Central FD.

    Uses objective J = 0.5 * sum(W_i * |V_i|^2)
    q = dJ/dV* = 0.5 * W * V
    """
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

    Y_nom, b_nom, node_map = assemble_mna(graph, source_spec, f_hz)
    state, status, _ = solve_mna_factorized(Y_nom, b_nom)
    assert status == CircuitSolveStatus.OK

    # 1. Objective J = 0.5 * sum(|V_i|^2) -> W = I
    V = state.V_nominal
    np.eye(len(V))
    0.5 * np.sum(np.abs(V) ** 2)
    q = 0.5 * V  # Since J = 0.5 V^H V, dJ/dV* = 0.5 V

    # 2. Adjoint
    lam = compute_adjoint_state(state, q)

    # Adjoint residual validation: Y^H lam = 2q
    residual_adj = Y_nom.conj().T @ lam - 2 * q
    np.testing.assert_allclose(residual_adj, 0, atol=1e-12)

    Y_p_C = np.zeros_like(Y_nom)
    Y_p_L = np.zeros_like(Y_nom)
    stamp_capacitor_derivative(Y_p_C, graph.elements["C1"], node_map, graph.ground_node_id, f_hz)
    stamp_inductor_derivative(Y_p_L, graph.elements["L1"], node_map, graph.ground_node_id, f_hz)

    grad_adj = compute_adjoint_gradient(lam, [Y_p_C, Y_p_L], V)

    # 3. Direct
    X_p = compute_direct_state_sensitivities(state, [Y_p_C, Y_p_L])
    # J = 0.5 V^H V => dJ/dp = Re(V^H X_p)
    grad_dir = np.zeros(2)
    grad_dir[0] = np.real(V.conj().T @ X_p[:, 0])
    grad_dir[1] = np.real(V.conj().T @ X_p[:, 1])

    np.testing.assert_allclose(grad_adj, grad_dir, rtol=1e-10, atol=1e-12)

    # 4. Central FD
    h_C = 1e-15
    h_L = 1e-9

    # FD C1
    graph_c_plus = CircuitGraph("gnd", Port("n1", "gnd"))
    graph_c_plus.add_node(Node("n1"))
    graph_c_plus.add_node(Node("n2"))
    graph_c_plus.add_node(Node("n3"))
    graph_c_plus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val + h_C))
    graph_c_plus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val))
    graph_c_plus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_c_plus, b_c_plus, _ = assemble_mna(graph_c_plus, source_spec, f_hz)
    state_c_plus, _, _ = solve_mna_factorized(Y_c_plus, b_c_plus)
    J_c_plus = 0.5 * np.sum(np.abs(state_c_plus.V_nominal) ** 2)

    graph_c_minus = CircuitGraph("gnd", Port("n1", "gnd"))
    graph_c_minus.add_node(Node("n1"))
    graph_c_minus.add_node(Node("n2"))
    graph_c_minus.add_node(Node("n3"))
    graph_c_minus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val - h_C))
    graph_c_minus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val))
    graph_c_minus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_c_minus, b_c_minus, _ = assemble_mna(graph_c_minus, source_spec, f_hz)
    state_c_minus, _, _ = solve_mna_factorized(Y_c_minus, b_c_minus)
    J_c_minus = 0.5 * np.sum(np.abs(state_c_minus.V_nominal) ** 2)

    grad_fd_C = (J_c_plus - J_c_minus) / (2 * h_C)

    # FD L1
    graph_l_plus = CircuitGraph("gnd", Port("n1", "gnd"))
    graph_l_plus.add_node(Node("n1"))
    graph_l_plus.add_node(Node("n2"))
    graph_l_plus.add_node(Node("n3"))
    graph_l_plus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val))
    graph_l_plus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val + h_L))
    graph_l_plus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_l_plus, b_l_plus, _ = assemble_mna(graph_l_plus, source_spec, f_hz)
    state_l_plus, _, _ = solve_mna_factorized(Y_l_plus, b_l_plus)
    J_l_plus = 0.5 * np.sum(np.abs(state_l_plus.V_nominal) ** 2)

    graph_l_minus = CircuitGraph("gnd", Port("n1", "gnd"))
    graph_l_minus.add_node(Node("n1"))
    graph_l_minus.add_node(Node("n2"))
    graph_l_minus.add_node(Node("n3"))
    graph_l_minus.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val))
    graph_l_minus.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val - h_L))
    graph_l_minus.add_element(Element("R1", ElementKind.RESISTOR, "n3", graph.ground_node_id, 50.0))
    Y_l_minus, b_l_minus, _ = assemble_mna(graph_l_minus, source_spec, f_hz)
    state_l_minus, _, _ = solve_mna_factorized(Y_l_minus, b_l_minus)
    J_l_minus = 0.5 * np.sum(np.abs(state_l_minus.V_nominal) ** 2)

    grad_fd_L = (J_l_plus - J_l_minus) / (2 * h_L)

    grad_fd = np.array([grad_fd_C, grad_fd_L])

    np.testing.assert_allclose(grad_adj, grad_fd, rtol=1e-5, atol=1e-12)
