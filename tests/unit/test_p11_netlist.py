"""P11 unit tests: SPICE netlist builder.  No ngspice required."""

from __future__ import annotations

import hashlib

import numpy as np

from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.models.components import (
    LumpedLossyCapacitor,
    LumpedLossyInductor,
    TabularImpedanceComponent,
)
from foster_eom.spice.netlist import _detect_grid, _sanitize, build_netlist
from foster_eom.spice.result import MeasurementPlan


def _source(rs=50.0):
    return SourceSpec(mode=SourceMode.THEVENIN, thevenin_vrms=1.0, z_source_real_ohm=rs)


def _simple_graph(elements):
    g = CircuitGraph(ground_node_id="0", input_port=Port("in", "0"))
    g.add_node(Node("0", is_ground=True))
    g.add_node(Node("in"))
    for e in elements:
        g.add_element(e)
    return g


def _freqs():
    return [1e6, 2e6, 3e6, 4e6, 5e6]


class TestDetectGrid:
    def test_lin(self):
        assert _detect_grid([1e6, 2e6, 3e6]) is not None
        assert "LIN" in _detect_grid([1e6, 2e6, 3e6])

    def test_irregular_none(self):
        assert _detect_grid([1e6, 1.5e6, 3.7e6]) is None

    def test_single_none(self):
        assert _detect_grid([1e6]) is None


class TestSanitize:
    def test_special_chars(self):
        used = set()
        s = _sanitize("b1/L[0]", "n", used)
        assert "/" not in s and "[" not in s

    def test_digit_prefix(self):
        used = set()
        s = _sanitize("1bad", "n", used)
        assert not s[0].isdigit()

    def test_collision_safe(self):
        used = {"n_abc"}
        s = _sanitize("abc", "n", used)
        assert s != "n_abc"

    def test_deterministic(self):
        s1 = _sanitize("x/y", "n", set())
        s2 = _sanitize("x/y", "n", set())
        assert s1 == s2


class TestBuildNetlist:
    def test_r_line(self):
        g = _simple_graph([Element("r1", ElementKind.RESISTOR, "in", "0", value=50.0)])
        nl = build_netlist(g, _source(), _freqs())
        assert "50" in nl.netlist_text

    def test_ground_zero(self):
        g = _simple_graph([Element("r1", ElementKind.RESISTOR, "in", "0", value=50.0)])
        assert build_netlist(g, _source(), _freqs()).node_map["0"] == "0"

    def test_sha_deterministic(self):
        g = _simple_graph([Element("r1", ElementKind.RESISTOR, "in", "0", value=50.0)])
        assert (
            build_netlist(g, _source(), _freqs()).sha256
            == build_netlist(g, _source(), _freqs()).sha256
        )

    def test_sha_matches_text(self):
        g = _simple_graph([Element("r1", ElementKind.RESISTOR, "in", "0", value=50.0)])
        nl = build_netlist(g, _source(), _freqs())
        assert nl.sha256 == hashlib.sha256(nl.netlist_text.encode()).hexdigest()

    def test_vsrc_ac_1_0(self):
        g = _simple_graph([Element("r1", ElementKind.RESISTOR, "in", "0", value=50.0)])
        assert "AC 1 0" in build_netlist(g, _source(), _freqs()).netlist_text

    def test_no_sqrt2_in_vsrc(self):
        src = SourceSpec(mode=SourceMode.THEVENIN, thevenin_vrms=0.5)
        g = _simple_graph([Element("r1", ElementKind.RESISTOR, "in", "0", value=50.0)])
        nl = build_netlist(g, src, _freqs())
        for line in nl.netlist_text.split("\n"):
            if line.strip().startswith("Vsrc"):
                assert "AC 1 0" in line

    def test_vsense_present(self):
        g = _simple_graph([Element("r1", ElementKind.RESISTOR, "in", "0", value=50.0)])
        nl = build_netlist(g, _source(), _freqs())
        assert "Vsense" in nl.netlist_text
        assert "__input__" in nl.sense_source_map

    def test_lossy_cap_subckt(self):
        cap = LumpedLossyCapacitor(c_f=10e-12, r_esr_ohm=0.1, l_esl_h=1e-9)
        g = _simple_graph([Element("C1", ElementKind.ONE_PORT_MODEL, "in", "0", model=cap)])
        nl = build_netlist(g, _source(), _freqs())
        assert not nl.unsupported_elements
        assert ".SUBCKT" in nl.netlist_text and "Resr" in nl.netlist_text

    def test_lossy_ind_subckt(self):
        ind = LumpedLossyInductor(l_h=10e-9, r_dcr_ohm=0.2, c_par_f=1e-12)
        g = _simple_graph([Element("L1", ElementKind.ONE_PORT_MODEL, "in", "0", model=ind)])
        nl = build_netlist(g, _source(), _freqs())
        assert not nl.unsupported_elements
        assert ".SUBCKT" in nl.netlist_text and "Rdcr" in nl.netlist_text

    def test_tabular_unsupported(self):
        fhz = np.array([1e6, 2e6, 3e6])
        z = np.array([50 + 0j, 50 + 10j, 50 + 20j])
        tab = TabularImpedanceComponent(fhz, z)
        g = _simple_graph([Element("T1", ElementKind.ONE_PORT_MODEL, "in", "0", model=tab)])
        nl = build_netlist(g, _source(), _freqs())
        assert "T1" in nl.unsupported_elements
        assert "tabular_component" in nl.unsupported_model_reasons["T1"]

    def test_branch_sense_source(self):
        g = _simple_graph([Element("R1", ElementKind.RESISTOR, "in", "0", value=50.0)])
        plan = MeasurementPlan(branch_element_ids=("R1",))
        nl = build_netlist(g, _source(), _freqs(), measurement_plan=plan)
        assert "R1" in nl.sense_source_map
        assert nl.sense_source_map["R1"] in nl.netlist_text

    def test_irregular_freqs_control_block(self):
        freqs = [1e6, 1.5e6, 3.7e6, 5e6]
        g = _simple_graph([Element("R1", ElementKind.RESISTOR, "in", "0", value=50.0)])
        nl = build_netlist(g, _source(), freqs)
        assert ".control" in nl.ac_command.lower() or "lin 1" in nl.ac_command.lower()

    def test_source_vth_recorded(self):
        src = SourceSpec(mode=SourceMode.THEVENIN, thevenin_vrms=2.0, phase_deg=30.0)
        g = _simple_graph([Element("R1", ElementKind.RESISTOR, "in", "0", value=50.0)])
        nl = build_netlist(g, src, _freqs())
        assert abs(abs(nl.source_vth_phasor) - 2.0) < 1e-10
