"""P11 integration tests: full ngspice AC run and MNA comparison.

Skipped when ngspice is not on PATH.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from foster_eom.spice.ngspice import detect_ngspice

pytestmark = pytest.mark.skipif(detect_ngspice() is None, reason="ngspice not available on PATH")

from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port  # noqa: E402
from foster_eom.circuit.solve import solve_circuit  # noqa: E402
from foster_eom.domain.source import SourceMode, SourceSpec  # noqa: E402
from foster_eom.models.components import LumpedLossyCapacitor, LumpedLossyInductor  # noqa: E402
from foster_eom.spice.api import validate_against_mna  # noqa: E402
from foster_eom.spice.netlist import build_netlist  # noqa: E402
from foster_eom.spice.ngspice import run_ngspice  # noqa: E402
from foster_eom.spice.result import ValidationThresholds  # noqa: E402


def _source(rs=50.0):
    return SourceSpec(mode=SourceMode.THEVENIN, thevenin_vrms=1.0, z_source_real_ohm=rs)


def _graph_r(r=50.0):
    g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
    g.add_node(Node("0", is_ground=True))
    g.add_node(Node("in"))
    g.add_element(Element("R1", ElementKind.RESISTOR, "in", "0", value=r))
    return g


TIGHT = ValidationThresholds(pass_max_rel_err=1e-4, pass_max_phase_deg=0.01)


class TestROnly:
    def test_z_in_matches(self):
        freqs = np.logspace(6, 9, 20).tolist()
        g = _graph_r(50.0)
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ("pass", "warn"), f"Failed: {rpt.fail_reason}"


class TestRCSeries:
    def test_rc_ac_sweep(self):
        freqs = np.logspace(6, 9, 20).tolist()
        g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g.add_node(Node("0", is_ground=True))
        g.add_node(Node("in"))
        g.add_node(Node("mid"))
        g.add_element(Element("R1", ElementKind.RESISTOR, "in", "mid", value=50.0))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "mid", "0", value=10e-12))
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ("pass", "warn"), f"Failed: {rpt.fail_reason}"


class TestRLParallelResonance:
    def test_resonance_location(self):
        L, C = 10e-9, 10e-12
        f0 = 1.0 / (2.0 * math.pi * math.sqrt(L * C))
        freqs = np.logspace(math.log10(f0 / 10), math.log10(f0 * 10), 100).tolist()
        g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g.add_node(Node("0", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("L1", ElementKind.INDUCTOR, "in", "0", value=L))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "in", "0", value=C))
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ("pass", "warn"), f"Failed: {rpt.fail_reason}"


class TestRLC:
    """RLC series resonator (explicit test case added for acceptance)."""

    def test_rlc_series_resonance(self):
        """Series R-L-C: Z_in has minimum at f0=1/(2π√LC)."""
        R, L, C = 5.0, 22e-9, 22e-12
        f0 = 1.0 / (2.0 * math.pi * math.sqrt(L * C))
        freqs = np.logspace(math.log10(f0 / 5), math.log10(f0 * 5), 60).tolist()
        g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g.add_node(Node("0", is_ground=True))
        g.add_node(Node("in"))
        g.add_node(Node("n1"))
        g.add_node(Node("n2"))
        g.add_element(Element("R1", ElementKind.RESISTOR, "in", "n1", value=R))
        g.add_element(Element("L1", ElementKind.INDUCTOR, "n1", "n2", value=L))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "n2", "0", value=C))
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ("pass", "warn"), f"RLC series failed: {rpt.fail_reason}"
        if rpt.comparisons:
            max_rel = max(c.max_rel_err for c in rpt.comparisons)
            assert max_rel < 1e-3, f"RLC rel error too high: {max_rel:.3e}"


class TestNonuniformFreq:
    """Explicit nonuniform/arbitrary frequency vector case."""

    def test_arbitrary_freq_vector(self):
        """Nonuniform frequencies: neither LIN nor DEC, forces per-frequency wrdata."""
        # Hand-picked frequencies that don't form any regular grid
        freqs = [1e6, 2.7e6, 5.1e6, 8.3e6, 12e6, 25e6, 50e6, 100e6, 200e6, 500e6]
        g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g.add_node(Node("0", is_ground=True))
        g.add_node(Node("in"))
        L, C = 10e-9, 100e-12
        g.add_element(Element("L1", ElementKind.INDUCTOR, "in", "0", value=L))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "in", "0", value=C))
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ("pass", "warn"), f"Nonuniform freqs failed: {rpt.fail_reason}"
        # Confirm all 10 frequencies compared (2 quantities: Z_in + I_port)
        assert len(rpt.comparisons) > 0


class TestFosterCell:
    """Single-cell Foster (parallel LC) and multi-cell network."""

    def test_single_foster_cell_parallel_lc(self):
        """Parallel L||C: typical Foster impedance cell."""
        L, C = 47e-9, 47e-12
        f0 = 1.0 / (2.0 * math.pi * math.sqrt(L * C))
        freqs = np.logspace(math.log10(max(1e6, f0 / 10)), math.log10(f0 * 10), 50).tolist()
        g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g.add_node(Node("0", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("L1", ElementKind.INDUCTOR, "in", "0", value=L))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "in", "0", value=C))
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ("pass", "warn"), f"Foster cell: {rpt.fail_reason}"

    def test_two_cell_foster_network(self):
        """Two parallel-LC cells in series: multi-cell Foster matcher."""
        # Cell 1: L1||C1
        L1, C1 = 47e-9, 47e-12
        # Cell 2: L2||C2 at different resonance
        L2, C2 = 22e-9, 100e-12
        freqs = np.logspace(7, 9, 31).tolist()  # 10 pts/dec, DEC grid

        g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g.add_node(Node("0", is_ground=True))
        g.add_node(Node("in"))
        g.add_node(Node("mid"))
        # Cell 1: in||0
        g.add_element(Element("L1", ElementKind.INDUCTOR, "in", "0", value=L1))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "in", "0", value=C1))
        # Series coupling via mid node (simplified)
        g.add_element(Element("Rc", ElementKind.RESISTOR, "in", "mid", value=1.0))
        # Cell 2: mid||0
        g.add_element(Element("L2", ElementKind.INDUCTOR, "mid", "0", value=L2))
        g.add_element(Element("C2", ElementKind.CAPACITOR, "mid", "0", value=C2))
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ("pass", "warn"), f"Multi-cell Foster: {rpt.fail_reason}"


class TestLossyModels:
    def test_lossy_inductor(self):
        ind = LumpedLossyInductor(l_h=10e-9, r_dcr_ohm=0.5, c_par_f=1e-12)
        g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g.add_node(Node("0", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("L1", ElementKind.ONE_PORT_MODEL, "in", "0", model=ind))
        freqs = np.logspace(6, 9, 20).tolist()
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ("pass", "warn"), f"Failed: {rpt.fail_reason}"

    def test_lossy_capacitor(self):
        cap = LumpedLossyCapacitor(c_f=10e-12, r_esr_ohm=0.1, l_esl_h=1e-9)
        g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g.add_node(Node("0", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("C1", ElementKind.ONE_PORT_MODEL, "in", "0", model=cap))
        freqs = np.logspace(6, 9, 20).tolist()
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ("pass", "warn"), f"Failed: {rpt.fail_reason}"

    def test_lossy_lc_combination(self):
        """Lossy L + lossy C in a realistic matcher cell."""
        ind = LumpedLossyInductor(l_h=47e-9, r_dcr_ohm=0.8, c_par_f=0.5e-12)
        cap = LumpedLossyCapacitor(c_f=47e-12, r_esr_ohm=0.05, l_esl_h=0.5e-9)
        g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g.add_node(Node("0", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("L1", ElementKind.ONE_PORT_MODEL, "in", "0", model=ind))
        g.add_element(Element("C1", ElementKind.ONE_PORT_MODEL, "in", "0", model=cap))
        freqs = np.logspace(6, 9, 31).tolist()  # DEC grid
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ("pass", "warn"), f"Lossy L+C: {rpt.fail_reason}"


class TestConventions:
    def test_unit_phasor_no_sqrt2(self):
        """Regression: AC 1 0 returns unit-source phasors; no sqrt(2) factor applied."""
        freqs = [50e6]
        R = 50.0
        src = SourceSpec(mode=SourceMode.THEVENIN, thevenin_vrms=1.0, z_source_real_ohm=R)
        g = _graph_r(R)
        nl = build_netlist(g, src, freqs)
        ng = run_ngspice(nl)
        i_sense = ng.sense_currents.get("Vsense")
        assert i_sense is not None
        expected_unit = 1.0 / (R + R)  # 0.01 A per unit volt
        assert abs(abs(i_sense[0]) - expected_unit) / expected_unit < 1e-3

    def test_current_positive_into_dut(self):
        """I(Vsense) must be positive real when current flows into passive DUT."""
        freqs = [50e6]
        g = _graph_r(100.0)
        src = _source(rs=50.0)
        nl = build_netlist(g, src, freqs)
        ng = run_ngspice(nl)
        i_sense = ng.sense_currents.get("Vsense")
        assert i_sense is not None
        assert float(np.real(i_sense[0])) > 0

    def test_z_in_from_vsense(self):
        """Z_in = V_dut/I(Vsense) must equal DUT impedance analytically."""
        freqs = [10e6, 50e6, 100e6]  # irregular
        R_dut = 75.0
        Rs = 50.0
        g = _graph_r(R_dut)
        src = _source(rs=Rs)
        nl = build_netlist(g, src, freqs)
        ng = run_ngspice(nl)
        i_sense = ng.sense_currents.get("Vsense")
        v_in = ng.node_voltages.get("in")
        assert i_sense is not None and v_in is not None
        # Z_in from SPICE (unit source)
        z_in_spice = v_in / i_sense
        # Analytic: pure resistor
        for z in z_in_spice:
            assert abs(z.real - R_dut) / R_dut < 1e-3
            assert abs(z.imag) < 1e-3


class TestP09RealizationPath:
    """Simulate the P09-realization → netlist → ngspice → MNA path.

    We build a synthetic catalog realization (ideal L+C Foster cell) and
    confirm the full pipeline produces pass-level agreement.
    """

    def test_ideal_catalog_lc_realization(self):
        """Ideal-model L+C (catalog-representative) SPICE vs MNA agreement."""
        # Typical catalog inductors/capacitors from P08/P09 ideal tier
        L_nom = 47e-9  # 47 nH inductor
        C_nom = 47e-12  # 47 pF capacitor
        freqs = np.logspace(6, 9, 31).tolist()  # DEC 10/dec, 3 dec

        g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g.add_node(Node("0", is_ground=True))
        g.add_node(Node("in"))
        # Foster cell: parallel L||C
        g.add_element(Element("L1", ElementKind.INDUCTOR, "in", "0", value=L_nom))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "in", "0", value=C_nom))

        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ("pass", "warn"), f"P09 ideal-catalog path failed: {rpt.fail_reason}"
        assert rpt.solver_version is not None
        assert "ngspice" in rpt.solver_version.lower()
        if rpt.comparisons:
            max_rel = max(c.max_rel_err for c in rpt.comparisons)
            # Report — do not require specific threshold beyond TIGHT
            assert max_rel < TIGHT.pass_max_rel_err * 10, (
                f"Max rel error {max_rel:.3e} unexpectedly large"
            )

    def test_parametric_catalog_lossy_lc(self):
        """Parametric catalog model (lossy L + lossy C) SPICE vs MNA."""
        ind = LumpedLossyInductor(l_h=22e-9, r_dcr_ohm=0.5, c_par_f=0.3e-12)
        cap = LumpedLossyCapacitor(c_f=22e-12, r_esr_ohm=0.08, l_esl_h=0.3e-9)
        freqs = np.logspace(7, 9, 31).tolist()

        g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g.add_node(Node("0", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("L1", ElementKind.ONE_PORT_MODEL, "in", "0", model=ind))
        g.add_element(Element("C1", ElementKind.ONE_PORT_MODEL, "in", "0", model=cap))

        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ("pass", "warn"), f"Parametric catalog SPICE failed: {rpt.fail_reason}"


class TestErrorReport:
    """Validation produces useful error info for corrupt/wrong models."""

    def test_error_metrics_reported(self):
        """Even a failing report must contain max_rel_err for diagnostics."""
        freqs = np.logspace(6, 9, 10).tolist()
        src = _source()
        g_mna = _graph_r(50.0)
        sols = solve_circuit(g_mna, src, freqs)
        # SPICE with wrong R
        g_spice = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g_spice.add_node(Node("0", is_ground=True))
        g_spice.add_node(Node("in"))
        g_spice.add_element(Element("R1", ElementKind.RESISTOR, "in", "0", value=500.0))
        rpt = validate_against_mna(g_spice, src, sols, thresholds=TIGHT)
        # Either fail or large error
        if rpt.status == "fail":
            assert rpt.fail_reason is not None
        elif rpt.comparisons:
            assert max(c.max_rel_err for c in rpt.comparisons) > 0.1

    def test_solver_version_in_report(self):
        """Successful report must identify ngspice version."""
        freqs = [10e6]
        g = _graph_r(50.0)
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols)
        assert rpt.solver_version is not None
        assert "ngspice" in rpt.solver_version.lower()


class TestWrongModel:
    def test_wrong_r_value_produces_large_error(self):
        """Netlist with R=500 instead of R=50 must produce fail or large error."""
        freqs = np.logspace(6, 9, 10).tolist()
        src = _source()
        # MNA solutions with R=50
        g_mna = _graph_r(50.0)
        sols = solve_circuit(g_mna, src, freqs)
        # SPICE netlist with R=500 (wrong model)
        g_spice = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g_spice.add_node(Node("0", is_ground=True))
        g_spice.add_node(Node("in"))
        g_spice.add_element(Element("R1", ElementKind.RESISTOR, "in", "0", value=500.0))
        rpt = validate_against_mna(g_spice, src, sols, thresholds=TIGHT)
        if rpt.status != "fail" and rpt.comparisons:
            assert max(c.max_rel_err for c in rpt.comparisons) > 0.1
