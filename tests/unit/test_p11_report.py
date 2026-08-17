"""P11 report field tests.  No ngspice required."""

from __future__ import annotations

import numpy as np

from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.models.components import TabularImpedanceComponent
from foster_eom.spice.api import validate_against_mna
from foster_eom.spice.result import SpiceValidationReport, ValidationThresholds


def _source():
    return SourceSpec(mode=SourceMode.THEVENIN, thevenin_vrms=1.0)


def _graph_with_tabular():
    fhz = np.array([1e6, 2e6, 3e6])
    z = np.array([50 + 0j, 50 + 10j, 50 + 20j])
    tab = TabularImpedanceComponent(fhz, z)
    g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
    g.add_node(Node("0", is_ground=True))
    g.add_node(Node("in"))
    g.add_element(Element("T1", ElementKind.ONE_PORT_MODEL, "in", "0", model=tab))
    return g


class TestConventionFields:
    def _make_report(self, status="solver_unavailable") -> SpiceValidationReport:
        # Use a trivial graph to trigger solver_unavailable (no ngspice in test env)
        g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g.add_node(Node("0", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("R1", ElementKind.RESISTOR, "in", "0", value=50.0))
        from foster_eom.circuit.measurements import CircuitSolution, SolveDiagnostics
        from foster_eom.errors import CircuitSolveStatus

        sol = CircuitSolution(
            f_hz=1e6, status=CircuitSolveStatus.OK, diagnostics=SolveDiagnostics()
        )
        return validate_against_mna(g, _source(), [sol])

    def test_source_convention_fixed_string(self):
        r = self._make_report()
        assert r.source_convention == "spice=AC_1_0_unit_phasor,scale=vth_phasor_in_python"

    def test_current_direction_fixed_string(self):
        r = self._make_report()
        assert r.current_direction_convention == "Vsense_oriented:I(Vsense)>0_into_DUT"

    def test_phase_convention_fixed_string(self):
        r = self._make_report()
        assert r.phase_convention == "angle(spice_conj_mna),masked_below_mag_floor"

    def test_solver_unavailable_no_ngspice(self):
        from unittest.mock import patch

        g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g.add_node(Node("0", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("R1", ElementKind.RESISTOR, "in", "0", value=50.0))
        from foster_eom.circuit.measurements import CircuitSolution, SolveDiagnostics
        from foster_eom.errors import CircuitSolveStatus

        sol = CircuitSolution(
            f_hz=1e6, status=CircuitSolveStatus.OK, diagnostics=SolveDiagnostics()
        )
        with patch("foster_eom.spice.api.detect_ngspice", return_value=None):
            r = validate_against_mna(g, _source(), [sol])
        assert r.status == "solver_unavailable"
        assert r.solver_version is None
        assert r.fail_reason is None


class TestUnsupportedStatus:
    def test_tabular_returns_unsupported(self):
        from foster_eom.circuit.measurements import CircuitSolution, SolveDiagnostics
        from foster_eom.errors import CircuitSolveStatus

        g = _graph_with_tabular()
        sol = CircuitSolution(
            f_hz=1e6, status=CircuitSolveStatus.OK, diagnostics=SolveDiagnostics()
        )
        r = validate_against_mna(g, _source(), [sol])
        assert r.status == "unsupported"
        assert "T1" in r.unsupported_elements
        assert r.comparisons == []
        # Solver version not set (SPICE not run)
        assert r.solver_version is None

    def test_fail_reason_mentions_elements(self):
        from foster_eom.circuit.measurements import CircuitSolution, SolveDiagnostics
        from foster_eom.errors import CircuitSolveStatus

        g = _graph_with_tabular()
        sol = CircuitSolution(
            f_hz=1e6, status=CircuitSolveStatus.OK, diagnostics=SolveDiagnostics()
        )
        r = validate_against_mna(g, _source(), [sol])
        assert r.fail_reason is not None
        assert "T1" in r.fail_reason


class TestReportFields:
    def test_netlist_sha_present(self):
        from unittest.mock import patch

        g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g.add_node(Node("0", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("R1", ElementKind.RESISTOR, "in", "0", value=50.0))
        from foster_eom.circuit.measurements import CircuitSolution, SolveDiagnostics
        from foster_eom.errors import CircuitSolveStatus

        sol = CircuitSolution(
            f_hz=1e6, status=CircuitSolveStatus.OK, diagnostics=SolveDiagnostics()
        )
        with patch("foster_eom.spice.api.detect_ngspice", return_value=None):
            r = validate_against_mna(g, _source(), [sol])
        assert r.netlist_sha256 is not None and len(r.netlist_sha256) == 64

    def test_thresholds_stored(self):
        from unittest.mock import patch

        g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
        g.add_node(Node("0", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("R1", ElementKind.RESISTOR, "in", "0", value=50.0))
        from foster_eom.circuit.measurements import CircuitSolution, SolveDiagnostics
        from foster_eom.errors import CircuitSolveStatus

        sol = CircuitSolution(
            f_hz=1e6, status=CircuitSolveStatus.OK, diagnostics=SolveDiagnostics()
        )
        thr = ValidationThresholds(pass_max_rel_err=1e-5)
        with patch("foster_eom.spice.api.detect_ngspice", return_value=None):
            r = validate_against_mna(g, _source(), [sol], thresholds=thr)
        assert r.thresholds.pass_max_rel_err == 1e-5
