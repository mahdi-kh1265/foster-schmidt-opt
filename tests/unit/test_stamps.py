"""Tests for low-level MNA stamps."""

import numpy as np

from foster_eom.circuit.graph import GROUND, Element, ElementKind
from foster_eom.circuit.stamps import stamp_admittance, stamp_element
from foster_eom.models import IdealCapacitorEOM


class TestStampAdmittance:
    """Canonical 2x2 admittance stamp."""

    def test_basic_pattern(self):
        Y = np.zeros((3, 3), dtype=np.complex128)
        stamp_admittance(Y, 0, 1, 2.0 + 1j)
        assert Y[0, 0] == 2.0 + 1j
        assert Y[1, 1] == 2.0 + 1j
        assert Y[0, 1] == -(2.0 + 1j)
        assert Y[1, 0] == -(2.0 + 1j)
        # Other entries untouched
        assert Y[2, 2] == 0.0
        assert Y[0, 2] == 0.0

    def test_ground_node_a(self):
        Y = np.zeros((2, 2), dtype=np.complex128)
        stamp_admittance(Y, GROUND, 0, 3.0)
        # Only Y[b,b] gets the stamp
        assert Y[0, 0] == 3.0
        # No Y[a,...] writes
        assert Y[1, 1] == 0.0

    def test_ground_node_b(self):
        Y = np.zeros((2, 2), dtype=np.complex128)
        stamp_admittance(Y, 1, GROUND, 5.0)
        assert Y[1, 1] == 5.0
        assert Y[0, 0] == 0.0

    def test_both_ground(self):
        Y = np.zeros((2, 2), dtype=np.complex128)
        stamp_admittance(Y, GROUND, GROUND, 10.0)
        assert np.all(Y == 0.0)

    def test_accumulation(self):
        """Multiple stamps at the same positions accumulate."""
        Y = np.zeros((2, 2), dtype=np.complex128)
        stamp_admittance(Y, 0, 1, 1.0)
        stamp_admittance(Y, 0, 1, 2.0)
        assert Y[0, 0] == 3.0
        assert Y[0, 1] == -3.0


class TestStampElement:
    """Element-type dispatch."""

    def _make_node_map(self) -> dict[str, int]:
        return {"a": 0, "b": 1}

    def test_resistor(self):
        Y = np.zeros((2, 2), dtype=np.complex128)
        elem = Element(id="R1", kind=ElementKind.RESISTOR, node_pos="a", node_neg="b", value=100.0)
        stamp_element(Y, elem, self._make_node_map(), "gnd", 1e6)
        expected_y = 1.0 / 100.0
        assert np.isclose(Y[0, 0], expected_y)
        assert np.isclose(Y[1, 1], expected_y)
        assert np.isclose(Y[0, 1], -expected_y)
        assert np.isclose(Y[1, 0], -expected_y)

    def test_inductor(self):
        Y = np.zeros((2, 2), dtype=np.complex128)
        f = 10e6
        L = 1e-6
        elem = Element(id="L1", kind=ElementKind.INDUCTOR, node_pos="a", node_neg="b", value=L)
        stamp_element(Y, elem, self._make_node_map(), "gnd", f)
        omega = 2 * np.pi * f
        expected_y = 1.0 / (1j * omega * L)
        assert np.isclose(Y[0, 0], expected_y)

    def test_capacitor(self):
        Y = np.zeros((2, 2), dtype=np.complex128)
        f = 10e6
        C = 10e-12
        elem = Element(id="C1", kind=ElementKind.CAPACITOR, node_pos="a", node_neg="b", value=C)
        stamp_element(Y, elem, self._make_node_map(), "gnd", f)
        omega = 2 * np.pi * f
        expected_y = 1j * omega * C
        assert np.isclose(Y[0, 0], expected_y)

    def test_one_port_model(self):
        Y = np.zeros((2, 2), dtype=np.complex128)
        f = 10e6
        model = IdealCapacitorEOM(c0_f=10e-12)
        elem = Element(
            id="EOM",
            kind=ElementKind.ONE_PORT_MODEL,
            node_pos="a",
            node_neg="b",
            model=model,
        )
        stamp_element(Y, elem, self._make_node_map(), "gnd", f)
        expected_y = model.y(f)
        assert np.isclose(Y[0, 0], expected_y)

    def test_element_to_ground(self):
        """Element with one terminal at ground."""
        Y = np.zeros((1, 1), dtype=np.complex128)
        elem = Element(id="R1", kind=ElementKind.RESISTOR, node_pos="a", node_neg="gnd", value=50.0)
        stamp_element(Y, elem, {"a": 0}, "gnd", 1e6)
        assert np.isclose(Y[0, 0], 1.0 / 50.0)
