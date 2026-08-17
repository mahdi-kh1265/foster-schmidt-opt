"""P11 integration tests: full ngspice AC run and MNA comparison.

Skipped when ngspice is not on PATH.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from foster_eom.spice.ngspice import detect_ngspice

pytestmark = pytest.mark.skipif(
    detect_ngspice() is None,
    reason='ngspice not available on PATH'
)

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
    g = CircuitGraph(ground_node_id='0', input_port=Port('in', '0'))
    g.add_node(Node('0', is_ground=True))
    g.add_node(Node('in'))
    g.add_element(Element('R1', ElementKind.RESISTOR, 'in', '0', value=r))
    return g


TIGHT = ValidationThresholds(pass_max_rel_err=1e-4, pass_max_phase_deg=0.01)


class TestROnly:
    def test_z_in_matches(self):
        freqs = np.logspace(6, 9, 20).tolist()
        g = _graph_r(50.0)
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ('pass', 'warn'), f'Failed: {rpt.fail_reason}'


class TestRCSeries:
    def test_rc_ac_sweep(self):
        freqs = np.logspace(6, 9, 20).tolist()
        g = CircuitGraph(ground_node_id='0', input_port=Port('in', '0'))
        g.add_node(Node('0', is_ground=True))
        g.add_node(Node('in'))
        g.add_node(Node('mid'))
        g.add_element(Element('R1', ElementKind.RESISTOR, 'in', 'mid', value=50.0))
        g.add_element(Element('C1', ElementKind.CAPACITOR, 'mid', '0', value=10e-12))
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ('pass', 'warn'), f'Failed: {rpt.fail_reason}'


class TestRLParallelResonance:
    def test_resonance_location(self):
        L, C = 10e-9, 10e-12
        f0 = 1.0 / (2.0 * math.pi * math.sqrt(L * C))
        freqs = np.logspace(math.log10(f0/10), math.log10(f0*10), 100).tolist()
        g = CircuitGraph(ground_node_id='0', input_port=Port('in', '0'))
        g.add_node(Node('0', is_ground=True))
        g.add_node(Node('in'))
        g.add_element(Element('L1', ElementKind.INDUCTOR, 'in', '0', value=L))
        g.add_element(Element('C1', ElementKind.CAPACITOR, 'in', '0', value=C))
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ('pass', 'warn'), f'Failed: {rpt.fail_reason}'


class TestLossyModels:
    def test_lossy_inductor(self):
        ind = LumpedLossyInductor(l_h=10e-9, r_dcr_ohm=0.5, c_par_f=1e-12)
        g = CircuitGraph(ground_node_id='0', input_port=Port('in', '0'))
        g.add_node(Node('0', is_ground=True))
        g.add_node(Node('in'))
        g.add_element(Element('L1', ElementKind.ONE_PORT_MODEL, 'in', '0', model=ind))
        freqs = np.logspace(6, 9, 20).tolist()
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ('pass', 'warn'), f'Failed: {rpt.fail_reason}'

    def test_lossy_capacitor(self):
        cap = LumpedLossyCapacitor(c_f=10e-12, r_esr_ohm=0.1, l_esl_h=1e-9)
        g = CircuitGraph(ground_node_id='0', input_port=Port('in', '0'))
        g.add_node(Node('0', is_ground=True))
        g.add_node(Node('in'))
        g.add_element(Element('C1', ElementKind.ONE_PORT_MODEL, 'in', '0', model=cap))
        freqs = np.logspace(6, 9, 20).tolist()
        src = _source()
        sols = solve_circuit(g, src, freqs)
        rpt = validate_against_mna(g, src, sols, thresholds=TIGHT)
        assert rpt.status in ('pass', 'warn'), f'Failed: {rpt.fail_reason}'


class TestConventions:
    def test_unit_phasor_no_sqrt2(self):
        """Regression: AC 1 0 returns unit-source phasors; no sqrt(2) factor applied."""
        freqs = [50e6]
        R = 50.0
        src = SourceSpec(mode=SourceMode.THEVENIN, thevenin_vrms=1.0, z_source_real_ohm=R)
        g = _graph_r(R)
        nl = build_netlist(g, src, freqs)
        ng = run_ngspice(nl)
        i_sense = ng.sense_currents.get('Vsense')
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
        i_sense = ng.sense_currents.get('Vsense')
        assert i_sense is not None
        assert float(np.real(i_sense[0])) > 0


class TestWrongModel:
    def test_wrong_r_value_produces_large_error(self):
        """Netlist with R=500 instead of R=50 must produce fail or large error."""
        freqs = np.logspace(6, 9, 10).tolist()
        src = _source()
        # MNA solutions with R=50
        g_mna = _graph_r(50.0)
        sols = solve_circuit(g_mna, src, freqs)
        # SPICE netlist with R=500 (wrong model)
        g_spice = CircuitGraph(ground_node_id='0', input_port=Port('in', '0'))
        g_spice.add_node(Node('0', is_ground=True))
        g_spice.add_node(Node('in'))
        g_spice.add_element(Element('R1', ElementKind.RESISTOR, 'in', '0', value=500.0))
        rpt = validate_against_mna(g_spice, src, sols, thresholds=TIGHT)
        if rpt.status != 'fail' and rpt.comparisons:
            assert max(c.max_rel_err for c in rpt.comparisons) > 0.1
