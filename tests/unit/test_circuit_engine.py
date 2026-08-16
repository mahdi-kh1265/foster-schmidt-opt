"""Analytical acceptance tests for the circuit engine (spec §32.1).

Every test uses a known analytical result for a simple circuit.
All quantities use RMS phasor convention: S = V · conj(I), no 1/2.
"""

import numpy as np
import pytest

from foster_eom.circuit import (
    CircuitGraph,
    Element,
    ElementKind,
    Node,
    Port,
    solve_circuit_single,
)
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.errors import CircuitSolveStatus
from foster_eom.models import create_synthetic_mbvd
from foster_eom.units import z_to_gamma

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SOURCE_50 = SourceSpec(
    mode=SourceMode.THEVENIN,
    thevenin_vrms=1.0,
    z_source_real_ohm=50.0,
    z_ref_ohm=50.0,
)

_F = 10e6  # 10 MHz default test frequency


def _build_single_load(
    element: Element,
    ground_id: str = "gnd",
    source_node: str = "in",
) -> CircuitGraph:
    """Helper: source_node → element → ground."""
    g = CircuitGraph(
        ground_node_id=ground_id,
        input_port=Port(node_pos=source_node, node_neg=ground_id),
        eom_element_id=element.id,
    )
    g.add_node(Node(id=ground_id, is_ground=True))
    g.add_node(Node(id=source_node))
    g.add_element(element)
    return g


def _assert_power_balance(sol, atol=1e-10, rtol=1e-6):
    """Assert complex-power balance within tolerance."""
    assert sol.status == CircuitSolveStatus.OK
    assert sol.power_balance_ok, (
        f"Power balance failed: residual={sol.power_balance_residual}, "
        f"S_delivered={sol.s_source_delivered}"
    )
    # Extra strictness
    assert abs(sol.power_balance_residual) < atol + rtol * abs(sol.s_source_delivered)


def _assert_port_current_agreement(sol, rtol=1e-10):
    """Assert I_port (from passive branches) agrees with I_droop."""
    assert sol.i_port is not None
    assert sol.i_source_droop is not None
    if abs(sol.i_port) > 1e-20:
        rel = abs(sol.i_port - sol.i_source_droop) / abs(sol.i_port)
        assert rel < rtol, (
            f"I_port vs I_droop mismatch: I_port={sol.i_port}, "
            f"I_droop={sol.i_source_droop}, rel={rel}"
        )


# ---------------------------------------------------------------------------
# §1  Simple loads
# ---------------------------------------------------------------------------


class TestBareR:
    def test_zin_and_gamma(self):
        g = _build_single_load(
            Element(id="R1", kind=ElementKind.RESISTOR, node_pos="in", node_neg="gnd", value=50.0)
        )
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        assert sol.status == CircuitSolveStatus.OK
        assert np.isclose(sol.z_in, 50.0)
        assert np.isclose(abs(sol.gamma), 0.0, atol=1e-12)

    def test_voltage_divider(self):
        g = _build_single_load(
            Element(id="R1", kind=ElementKind.RESISTOR, node_pos="in", node_neg="gnd", value=50.0)
        )
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        # V_load = V_th * R / (Rs + R) = 1.0 * 50/(50+50) = 0.5
        assert np.isclose(abs(sol.v_eom), 0.5)

    def test_power(self):
        g = _build_single_load(
            Element(id="R1", kind=ElementKind.RESISTOR, node_pos="in", node_neg="gnd", value=50.0)
        )
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        # P_load = V^2 / R = 0.5^2 / 50 = 0.005 W
        assert np.isclose(sol.element_measurements["R1"].real_power_w, 0.005)
        _assert_power_balance(sol)
        _assert_port_current_agreement(sol)

    def test_mismatched_load(self):
        g = _build_single_load(
            Element(id="R1", kind=ElementKind.RESISTOR, node_pos="in", node_neg="gnd", value=100.0)
        )
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        assert np.isclose(sol.z_in, 100.0)
        expected_gamma = z_to_gamma(100.0, 50.0)
        assert np.isclose(sol.gamma, expected_gamma)
        _assert_power_balance(sol)


class TestBareC:
    def test_zin(self):
        C = 100e-12
        g = _build_single_load(
            Element(id="C1", kind=ElementKind.CAPACITOR, node_pos="in", node_neg="gnd", value=C)
        )
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        omega = 2 * np.pi * _F
        z_c = 1.0 / (1j * omega * C)
        assert np.isclose(sol.z_in, z_c)
        _assert_power_balance(sol)
        _assert_port_current_agreement(sol)

    def test_lossless_power(self):
        """Pure C: real dissipation should be ~0."""
        C = 100e-12
        g = _build_single_load(
            Element(id="C1", kind=ElementKind.CAPACITOR, node_pos="in", node_neg="gnd", value=C)
        )
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        assert abs(sol.element_measurements["C1"].real_power_w) < 1e-15


class TestBareL:
    def test_zin(self):
        L = 1e-6
        g = _build_single_load(
            Element(id="L1", kind=ElementKind.INDUCTOR, node_pos="in", node_neg="gnd", value=L)
        )
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        omega = 2 * np.pi * _F
        z_l = 1j * omega * L
        assert np.isclose(sol.z_in, z_l)
        _assert_power_balance(sol)
        _assert_port_current_agreement(sol)


# ---------------------------------------------------------------------------
# §2  Series / Parallel combinations
# ---------------------------------------------------------------------------


class TestSeriesRC:
    def test_zin(self):
        R, C = 100.0, 100e-12
        g = CircuitGraph(
            ground_node_id="gnd",
            input_port=Port("in", "gnd"),
        )
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        g.add_node(Node("mid"))
        g.add_element(Element("R1", ElementKind.RESISTOR, "in", "mid", value=R))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "mid", "gnd", value=C))
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        omega = 2 * np.pi * _F
        z_expected = R + 1.0 / (1j * omega * C)
        assert np.isclose(sol.z_in, z_expected)
        _assert_power_balance(sol)
        _assert_port_current_agreement(sol)


class TestParallelRC:
    def test_zin(self):
        R, C = 100.0, 100e-12
        g = CircuitGraph(
            ground_node_id="gnd",
            input_port=Port("in", "gnd"),
        )
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("R1", ElementKind.RESISTOR, "in", "gnd", value=R))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "in", "gnd", value=C))
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        omega = 2 * np.pi * _F
        y_expected = 1.0 / R + 1j * omega * C
        z_expected = 1.0 / y_expected
        assert np.isclose(sol.z_in, z_expected)
        _assert_power_balance(sol)
        _assert_port_current_agreement(sol)


class TestSeriesRLC:
    def test_zin(self):
        R, L, C = 100.0, 1e-6, 100e-12
        g = CircuitGraph(ground_node_id="gnd", input_port=Port("in", "gnd"))
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        g.add_node(Node("n1"))
        g.add_node(Node("n2"))
        g.add_element(Element("R1", ElementKind.RESISTOR, "in", "n1", value=R))
        g.add_element(Element("L1", ElementKind.INDUCTOR, "n1", "n2", value=L))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "n2", "gnd", value=C))
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        omega = 2 * np.pi * _F
        z_expected = R + 1j * omega * L + 1.0 / (1j * omega * C)
        assert np.isclose(sol.z_in, z_expected)
        _assert_power_balance(sol)
        _assert_port_current_agreement(sol)


class TestParallelRLC:
    def test_zin(self):
        R, L, C = 1000.0, 1e-6, 100e-12
        g = CircuitGraph(ground_node_id="gnd", input_port=Port("in", "gnd"))
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("R1", ElementKind.RESISTOR, "in", "gnd", value=R))
        g.add_element(Element("L1", ElementKind.INDUCTOR, "in", "gnd", value=L))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "in", "gnd", value=C))
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        omega = 2 * np.pi * _F
        y_expected = 1.0 / R + 1j * omega * C + 1.0 / (1j * omega * L)
        z_expected = 1.0 / y_expected
        assert np.isclose(sol.z_in, z_expected)
        _assert_power_balance(sol)
        _assert_port_current_agreement(sol)


# ---------------------------------------------------------------------------
# §3  Composite topologies
# ---------------------------------------------------------------------------


class TestParallelLCInSeries:
    """Parallel LC cell in series with a resistive load."""

    def test_zin(self):
        L, C, R_load = 1e-6, 100e-12, 50.0
        g = CircuitGraph(ground_node_id="gnd", input_port=Port("in", "gnd"))
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        g.add_node(Node("mid"))
        # Parallel LC between "in" and "mid"
        g.add_element(Element("L1", ElementKind.INDUCTOR, "in", "mid", value=L))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "in", "mid", value=C))
        # Load
        g.add_element(Element("R1", ElementKind.RESISTOR, "mid", "gnd", value=R_load))
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        omega = 2 * np.pi * _F
        y_lc = 1.0 / (1j * omega * L) + 1j * omega * C
        z_lc = 1.0 / y_lc
        z_expected = z_lc + R_load
        assert np.isclose(sol.z_in, z_expected)
        _assert_power_balance(sol)
        _assert_port_current_agreement(sol)


class TestLNetwork:
    """Simple L-network: series L, shunt C, load R."""

    def test_zin(self):
        L_s, C_sh, R_load = 1e-6, 100e-12, 200.0
        g = CircuitGraph(ground_node_id="gnd", input_port=Port("in", "gnd"))
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        g.add_node(Node("mid"))
        g.add_element(Element("Ls", ElementKind.INDUCTOR, "in", "mid", value=L_s))
        g.add_element(Element("Csh", ElementKind.CAPACITOR, "mid", "gnd", value=C_sh))
        g.add_element(Element("Rload", ElementKind.RESISTOR, "mid", "gnd", value=R_load))
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        omega = 2 * np.pi * _F
        # Shunt: C || R_load
        y_shunt = 1.0 / R_load + 1j * omega * C_sh
        z_shunt = 1.0 / y_shunt
        z_expected = 1j * omega * L_s + z_shunt
        assert np.isclose(sol.z_in, z_expected)
        _assert_power_balance(sol)
        _assert_port_current_agreement(sol)


# ---------------------------------------------------------------------------
# §4  Node renumbering invariance
# ---------------------------------------------------------------------------


class TestNodeInvariance:
    """Same circuit with different node ID strings and different insertion
    order must produce identical results."""

    def _build_circuit(self, ids: dict[str, str], reverse_order: bool = False) -> CircuitGraph:
        """Build a series RC circuit with parametrized node IDs."""
        g = CircuitGraph(
            ground_node_id=ids["gnd"],
            input_port=Port(ids["in"], ids["gnd"]),
        )
        nodes = [
            Node(ids["gnd"], is_ground=True),
            Node(ids["in"]),
            Node(ids["mid"]),
        ]
        elems = [
            Element("R1", ElementKind.RESISTOR, ids["in"], ids["mid"], value=100.0),
            Element("C1", ElementKind.CAPACITOR, ids["mid"], ids["gnd"], value=100e-12),
        ]
        if reverse_order:
            nodes = list(reversed(nodes))
            elems = list(reversed(elems))
        for n in nodes:
            g.add_node(n)
        for e in elems:
            g.add_element(e)
        return g

    def test_renamed_ids(self):
        ids_a = {"gnd": "gnd", "in": "in", "mid": "mid"}
        ids_b = {"gnd": "ground_ref", "in": "port_input", "mid": "internal_47"}
        sol_a = solve_circuit_single(self._build_circuit(ids_a), _SOURCE_50, _F)
        sol_b = solve_circuit_single(self._build_circuit(ids_b), _SOURCE_50, _F)
        assert np.isclose(sol_a.z_in, sol_b.z_in)
        assert np.isclose(sol_a.v_port, sol_b.v_port)
        assert np.isclose(sol_a.gamma, sol_b.gamma)

    def test_insertion_order(self):
        ids = {"gnd": "gnd", "in": "in", "mid": "mid"}
        sol_fwd = solve_circuit_single(self._build_circuit(ids, False), _SOURCE_50, _F)
        sol_rev = solve_circuit_single(self._build_circuit(ids, True), _SOURCE_50, _F)
        assert np.isclose(sol_fwd.z_in, sol_rev.z_in)
        assert np.isclose(sol_fwd.v_port, sol_rev.v_port)
        assert np.isclose(sol_fwd.gamma, sol_rev.gamma)


# ---------------------------------------------------------------------------
# §5  Complex-power balance
# ---------------------------------------------------------------------------


class TestPowerBalance:
    def test_lossy_rc(self):
        """Series RC: real dissipation in R, reactive in C."""
        g = CircuitGraph(ground_node_id="gnd", input_port=Port("in", "gnd"))
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        g.add_node(Node("mid"))
        g.add_element(Element("R1", ElementKind.RESISTOR, "in", "mid", value=100.0))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "mid", "gnd", value=100e-12))
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        _assert_power_balance(sol)
        # Resistor has positive real power, capacitor has zero real power
        assert sol.element_measurements["R1"].real_power_w > 0
        assert abs(sol.element_measurements["C1"].real_power_w) < 1e-15

    def test_lossless_lc(self):
        """Pure LC: P ≈ 0, but Im(S) balance must hold."""
        g = CircuitGraph(ground_node_id="gnd", input_port=Port("in", "gnd"))
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("L1", ElementKind.INDUCTOR, "in", "gnd", value=1e-6))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "in", "gnd", value=100e-12))
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        _assert_power_balance(sol)
        # Both elements: real power ≈ 0
        assert abs(sol.element_measurements["L1"].real_power_w) < 1e-15
        assert abs(sol.element_measurements["C1"].real_power_w) < 1e-15


# ---------------------------------------------------------------------------
# §6  Reflection coefficient
# ---------------------------------------------------------------------------


class TestReflection:
    def test_known_load_100ohm(self):
        g = _build_single_load(Element("R1", ElementKind.RESISTOR, "in", "gnd", value=100.0))
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        expected_gamma = (100.0 - 50.0) / (100.0 + 50.0)
        assert np.isclose(sol.gamma, expected_gamma)

    def test_zref_differs_from_zsource(self):
        """Z_source=75Ω, Z_ref=50Ω, Z_load=50Ω.
        Gamma uses z_ref, not z_source."""
        source_75 = SourceSpec(
            mode=SourceMode.THEVENIN,
            thevenin_vrms=1.0,
            z_source_real_ohm=75.0,
            z_ref_ohm=50.0,
        )
        g = _build_single_load(Element("R1", ElementKind.RESISTOR, "in", "gnd", value=50.0))
        sol = solve_circuit_single(g, source_75, _F)
        # Z_in = 50Ω, Z_ref = 50Ω → Γ = 0
        assert np.isclose(sol.z_in, 50.0)
        assert np.isclose(abs(sol.gamma), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# §7  Source voltage convention
# ---------------------------------------------------------------------------


class TestSourceConvention:
    def test_resistive_divider(self):
        """V_port = V_th * Z_load / (Z_s + Z_load) for resistive divider."""
        R_load = 200.0
        g = _build_single_load(Element("R1", ElementKind.RESISTOR, "in", "gnd", value=R_load))
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        v_expected = _SOURCE_50.vth_phasor * R_load / (_SOURCE_50.z_source + R_load)
        assert np.isclose(sol.v_port, v_expected)


# ---------------------------------------------------------------------------
# §8  I_port vs I_droop agreement
# ---------------------------------------------------------------------------


class TestPortCurrentAgreement:
    """Every simple circuit: I_port from passive branches must agree with
    I_droop = (V_th - V_port) / Z_source."""

    @pytest.mark.parametrize("R_load", [10.0, 50.0, 100.0, 1000.0])
    def test_various_loads(self, R_load):
        g = _build_single_load(Element("R1", ElementKind.RESISTOR, "in", "gnd", value=R_load))
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        _assert_port_current_agreement(sol)


# ---------------------------------------------------------------------------
# §9  Singular / ill-conditioned handling
# ---------------------------------------------------------------------------


class TestSingularHandling:
    def test_floating_node(self):
        """Circuit with an unconnected internal node → singular."""
        g = CircuitGraph(ground_node_id="gnd", input_port=Port("in", "gnd"))
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        g.add_node(Node("floating"))  # no element connects to this
        g.add_element(Element("R1", ElementKind.RESISTOR, "in", "gnd", value=50.0))
        sol = solve_circuit_single(g, _SOURCE_50, _F)
        assert sol.status == CircuitSolveStatus.SINGULAR_OR_ILL_CONDITIONED

    def test_nonfinite_stamp_inductor_at_dc(self):
        """Ideal inductor at f=0 produces y=Inf → nonfinite pre-screen.

        We test the MNA assembly/solve path directly at f=0 to produce a
        genuinely nonfinite admittance stamp.
        """
        from foster_eom.circuit.mna import assemble_mna, solve_mna

        g = _build_single_load(Element("L1", ElementKind.INDUCTOR, "in", "gnd", value=1e-6))
        Y, I_vec, _node_map = assemble_mna(g, _SOURCE_50, 0.0)
        # Y should contain Inf from the inductor stamp
        assert not np.all(np.isfinite(Y)), "Y should contain Inf at f=0"
        V, status, diag = solve_mna(Y, I_vec)
        assert status == CircuitSolveStatus.SINGULAR_OR_ILL_CONDITIONED
        assert diag.nonfinite_in_matrix
        assert V is None

    def test_exact_cancellation(self):
        """Two parallel elements whose admittances exactly cancel at a
        specific frequency → singular Y matrix.

        At f_0 where ωL = 1/(ωC), the parallel LC admittance is zero.
        If the only path to ground is through this LC pair, the network
        node becomes floating.
        """
        # Choose L, C so resonance is at f_0 = 10 MHz
        f_0 = 10e6
        omega_0 = 2 * np.pi * f_0
        L = 1e-6
        C = 1.0 / (omega_0**2 * L)

        # in → [L||C] → mid → nothing (mid is only connected via LC pair)
        # At resonance, Y_L + Y_C = 0, so mid is effectively floating.
        g = CircuitGraph(ground_node_id="gnd", input_port=Port("in", "gnd"))
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        g.add_node(Node("mid"))
        g.add_element(Element("L1", ElementKind.INDUCTOR, "in", "mid", value=L))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "in", "mid", value=C))
        # mid only connects to in via L||C; at resonance Y_LC = 0 → mid is floating
        g.add_element(Element("Rload", ElementKind.RESISTOR, "mid", "gnd", value=50.0))

        # Off resonance: should solve fine
        sol_off = solve_circuit_single(g, _SOURCE_50, f_0 * 1.1)
        assert sol_off.status == CircuitSolveStatus.OK

        # At exact resonance: the series LC between in and mid has Z=∞,
        # but mid still connects to ground via Rload, so it's not floating.
        # Instead, test a truly singular case: no Rload, just LC to mid.
        g2 = CircuitGraph(ground_node_id="gnd", input_port=Port("in", "gnd"))
        g2.add_node(Node("gnd", is_ground=True))
        g2.add_node(Node("in"))
        g2.add_node(Node("mid"))
        g2.add_element(Element("L1", ElementKind.INDUCTOR, "in", "mid", value=L))
        g2.add_element(Element("C1", ElementKind.CAPACITOR, "in", "mid", value=C))
        # At resonance Y_L + Y_C = 0 between in and mid → mid is floating
        sol_res = solve_circuit_single(g2, _SOURCE_50, f_0)
        assert sol_res.status == CircuitSolveStatus.SINGULAR_OR_ILL_CONDITIONED


# ---------------------------------------------------------------------------
# §10  OnePortModel element
# ---------------------------------------------------------------------------


class TestOnePortModelElement:
    def test_mbvd_eom_as_load(self):
        """Synthetic mBVD EOM connected directly to 50Ω source."""
        eom = create_synthetic_mbvd()
        g = _build_single_load(Element("EOM", ElementKind.ONE_PORT_MODEL, "in", "gnd", model=eom))
        f = 10e6
        sol = solve_circuit_single(g, _SOURCE_50, f)
        assert sol.status == CircuitSolveStatus.OK

        # Z_in should equal Z_EOM
        z_eom = eom.z(f)
        assert np.isclose(sol.z_in, z_eom, rtol=1e-10)

        # Gamma
        expected_gamma = z_to_gamma(z_eom, 50.0)
        assert np.isclose(sol.gamma, expected_gamma)

        _assert_power_balance(sol)
        _assert_port_current_agreement(sol)


# ---------------------------------------------------------------------------
# §11  EOM V/I with non-ground negative terminal
# ---------------------------------------------------------------------------


class TestEOMNonGroundNeg:
    """EOM element between two non-ground nodes: V_EOM = V_pos - V_neg."""

    def test_veom_between_nodes(self):
        # Source → R_series → node_a → EOM → node_b → R_load → GND
        R_s_val, R_load = 100.0, 100.0
        eom = create_synthetic_mbvd()
        g = CircuitGraph(
            ground_node_id="gnd",
            input_port=Port("in", "gnd"),
            eom_element_id="EOM",
        )
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        g.add_node(Node("a"))
        g.add_node(Node("b"))
        g.add_element(Element("Rs", ElementKind.RESISTOR, "in", "a", value=R_s_val))
        g.add_element(Element("EOM", ElementKind.ONE_PORT_MODEL, "a", "b", model=eom))
        g.add_element(Element("Rload", ElementKind.RESISTOR, "b", "gnd", value=R_load))

        sol = solve_circuit_single(g, _SOURCE_50, _F)
        assert sol.status == CircuitSolveStatus.OK

        # V_EOM should be V[a] - V[b], not just V[a]
        v_a = sol.node_voltages["a"]
        v_b = sol.node_voltages["b"]
        assert np.isclose(sol.v_eom, v_a - v_b)
        assert not np.isclose(sol.v_eom, v_a)  # Must not be just V[a]

        _assert_power_balance(sol)
        _assert_port_current_agreement(sol)


# ---------------------------------------------------------------------------
# §12  Graph validation
# ---------------------------------------------------------------------------


class TestGraphValidation:
    def test_self_loop_rejected(self):
        with pytest.raises(ValueError, match="self-loop"):
            Element("R1", ElementKind.RESISTOR, "a", "a", value=50.0)

    def test_duplicate_node_id(self):
        g = CircuitGraph(ground_node_id="gnd", input_port=Port("in", "gnd"))
        g.add_node(Node("gnd", is_ground=True))
        with pytest.raises(ValueError, match="Duplicate node"):
            g.add_node(Node("gnd", is_ground=True))

    def test_duplicate_element_id(self):
        g = CircuitGraph(ground_node_id="gnd", input_port=Port("in", "gnd"))
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("R1", ElementKind.RESISTOR, "in", "gnd", value=50.0))
        with pytest.raises(ValueError, match="Duplicate element"):
            g.add_element(Element("R1", ElementKind.RESISTOR, "in", "gnd", value=100.0))

    def test_dangling_node_ref(self):
        g = CircuitGraph(ground_node_id="gnd", input_port=Port("in", "gnd"))
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("R1", ElementKind.RESISTOR, "in", "missing", value=50.0))
        with pytest.raises(ValueError, match="unknown node_neg"):
            g.validate()

    def test_missing_ground(self):
        g = CircuitGraph(ground_node_id="gnd", input_port=Port("in", "gnd"))
        g.add_node(Node("gnd"))  # is_ground=False!
        g.add_node(Node("in"))
        with pytest.raises(ValueError, match="is_ground"):
            g.validate()

    def test_missing_eom_element(self):
        g = CircuitGraph(
            ground_node_id="gnd",
            input_port=Port("in", "gnd"),
            eom_element_id="nonexistent",
        )
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        with pytest.raises(ValueError, match="EOM element"):
            g.validate()

    def test_primitive_missing_value(self):
        with pytest.raises(ValueError, match="requires a value"):
            Element("R1", ElementKind.RESISTOR, "a", "b", value=None)

    def test_primitive_negative_value(self):
        with pytest.raises(ValueError, match="strictly positive"):
            Element("R1", ElementKind.RESISTOR, "a", "b", value=-10.0)

    def test_model_element_missing_model(self):
        with pytest.raises(ValueError, match="model is None"):
            Element("E1", ElementKind.ONE_PORT_MODEL, "a", "b", model=None)
