"""Tests for network circuit construction (Prompt 04B)."""

from __future__ import annotations

import pytest

from foster_eom.circuit.graph import ElementKind
from foster_eom.domain.topology import LOrientation
from foster_eom.foster.foster_form import FosterCell, FosterComponents
from foster_eom.foster.network_builder import build_foster_circuit
from foster_eom.foster.schmidt import BranchRealization
from foster_eom.foster.sign_search import SignPattern
from foster_eom.foster.topology_enum import TopologyCandidate
from foster_eom.models.base import OnePortModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyModel(OnePortModel):
    """Trivial model returning constant impedance."""

    def _z_impl(self, f_hz):
        return 50.0 + 0j

    def _y_impl(self, f_hz):
        return 1.0 / (50.0 + 0j)

    def metadata(self):
        return {"type": "dummy"}


def _make_sign_pattern(
    orientation: LOrientation = LOrientation.SCHMIDT_SHUNT_THEN_SERIES,
    b1_real: BranchRealization = BranchRealization.FINITE_FOSTER,
    b2_real: BranchRealization = BranchRealization.FINITE_FOSTER,
) -> SignPattern:
    return SignPattern(
        orientation=orientation,
        signs=(1,),
        series_targets=(),
        shunt_targets=(),
        branch1_required_intervals=(),
        branch2_required_intervals=(),
        branch1_realization=b1_real,
        branch2_realization=b2_real,
    )


def _make_topology(
    orientation: LOrientation = LOrientation.SCHMIDT_SHUNT_THEN_SERIES,
    b1_cells: int = 1,
    b2_cells: int = 1,
    b1_c0: bool = False,
    b1_linf: bool = False,
    b2_c0: bool = False,
    b2_linf: bool = False,
) -> TopologyCandidate:
    p1 = b1_cells + (1 if b1_c0 else 0) + (1 if b1_linf else 0)
    p2 = b2_cells + (1 if b2_c0 else 0) + (1 if b2_linf else 0)
    n1 = 2 * b1_cells + (1 if b1_c0 else 0) + (1 if b1_linf else 0)
    n2 = 2 * b2_cells + (1 if b2_c0 else 0) + (1 if b2_linf else 0)
    return TopologyCandidate(
        orientation=orientation,
        branch1_cells=b1_cells,
        branch2_cells=b2_cells,
        branch1_has_c0=b1_c0,
        branch1_has_linf=b1_linf,
        branch2_has_c0=b2_c0,
        branch2_has_linf=b2_linf,
        branch1_n_coefficients=p1,
        branch2_n_coefficients=p2,
        n_reactive=n1 + n2,
        structurally_valid=True,
        prune_reason=None,
    )


def _make_components(
    n_cells: int = 1, c0: float | None = None, linf: float | None = None
) -> FosterComponents:
    cells = tuple(FosterCell(l_h=10e-6, c_f=100e-12, f_pole_hz=5e6) for _ in range(n_cells))
    return FosterComponents(c0_f=c0, l_inf_h=linf, cells=cells)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildFosterCircuit:
    """Circuit construction tests."""

    def test_orientation_mismatch_raises(self) -> None:
        """Topology/sign_pattern orientation mismatch → ValueError."""
        topo = _make_topology(orientation=LOrientation.SCHMIDT_SHUNT_THEN_SERIES)
        sign = _make_sign_pattern(orientation=LOrientation.ALTERNATE_L_ORIENTATION)
        with pytest.raises(ValueError, match="Orientation mismatch"):
            build_foster_circuit(
                topo,
                sign,
                _make_components(),
                _make_components(),
                _DummyModel(),
            )

    def test_basic_standard_circuit_has_eom(self) -> None:
        """Standard orientation circuit contains EOM element."""
        topo = _make_topology()
        sign = _make_sign_pattern()
        built = build_foster_circuit(
            topo,
            sign,
            _make_components(),
            _make_components(),
            _DummyModel(),
        )
        assert built.eom_element_id in built.graph.elements
        eom_elem = built.graph.elements[built.eom_element_id]
        assert eom_elem.kind == ElementKind.ONE_PORT_MODEL

    def test_standard_circuit_has_branch_elements(self) -> None:
        """Standard circuit creates branch elements."""
        topo = _make_topology()
        sign = _make_sign_pattern()
        built = build_foster_circuit(
            topo,
            sign,
            _make_components(),
            _make_components(),
            _DummyModel(),
        )
        assert len(built.branch1_element_ids) > 0
        assert len(built.branch2_element_ids) > 0

    def test_standard_circuit_validates(self) -> None:
        """Standard circuit passes graph validation."""
        topo = _make_topology()
        sign = _make_sign_pattern()
        built = build_foster_circuit(
            topo,
            sign,
            _make_components(),
            _make_components(),
            _DummyModel(),
        )
        built.graph.validate()  # Should not raise

    def test_zero_impedance_series_aliases_nodes(self) -> None:
        """ZERO_IMPEDANCE series branch → node aliasing (no elements)."""
        topo = _make_topology(b2_cells=0)
        sign = _make_sign_pattern(b2_real=BranchRealization.ZERO_IMPEDANCE)
        built = build_foster_circuit(
            topo,
            sign,
            _make_components(),
            None,
            _DummyModel(),
        )
        # No branch2 elements
        assert len(built.branch2_element_ids) == 0
        built.graph.validate()

    def test_open_shunt_branch_no_elements(self) -> None:
        """OPEN_OMITTED shunt branch → no shunt elements."""
        topo = _make_topology(b1_cells=0)
        sign = _make_sign_pattern(b1_real=BranchRealization.OPEN_OMITTED)
        built = build_foster_circuit(
            topo,
            sign,
            None,
            _make_components(),
            _DummyModel(),
        )
        assert len(built.branch1_element_ids) == 0
        built.graph.validate()

    def test_trivial_branch_with_components_raises(self) -> None:
        """Non-FINITE_FOSTER branch with components → ValueError."""
        topo = _make_topology(b1_cells=0)
        sign = _make_sign_pattern(b1_real=BranchRealization.OPEN_OMITTED)
        with pytest.raises(ValueError, match="non-None components"):
            build_foster_circuit(
                topo,
                sign,
                _make_components(),
                _make_components(),
                _DummyModel(),
            )

    def test_foster_branch_missing_components_raises(self) -> None:
        """FINITE_FOSTER branch with None components → ValueError."""
        topo = _make_topology()
        sign = _make_sign_pattern()
        with pytest.raises(ValueError, match="components are None"):
            build_foster_circuit(
                topo,
                sign,
                None,
                _make_components(),
                _DummyModel(),
            )

    def test_cell_count_mismatch_raises(self) -> None:
        """Component cell count mismatch → ValueError."""
        topo = _make_topology(b1_cells=2)  # expects 2 cells
        sign = _make_sign_pattern()
        with pytest.raises(ValueError, match="cell count mismatch"):
            build_foster_circuit(
                topo,
                sign,
                _make_components(n_cells=1),
                _make_components(),
                _DummyModel(),
            )

    def test_dual_orientation_circuit(self) -> None:
        """Dual orientation circuit builds and validates."""
        topo = _make_topology(orientation=LOrientation.ALTERNATE_L_ORIENTATION)
        sign = _make_sign_pattern(orientation=LOrientation.ALTERNATE_L_ORIENTATION)
        built = build_foster_circuit(
            topo,
            sign,
            _make_components(),
            _make_components(),
            _DummyModel(),
        )
        built.graph.validate()
        assert built.eom_element_id in built.graph.elements

    def test_endpoint_capacitor(self) -> None:
        """Topology with C0 endpoint creates capacitor element."""
        topo = _make_topology(b1_c0=True)
        sign = _make_sign_pattern()
        comp1 = _make_components(c0=200e-12)
        built = build_foster_circuit(
            topo,
            sign,
            comp1,
            _make_components(),
            _DummyModel(),
        )
        # Find the C0 element
        c0_elements = [
            e for e in built.graph.elements.values() if e.symbolic_role and "C0" in e.symbolic_role
        ]
        assert len(c0_elements) == 1
        assert c0_elements[0].kind == ElementKind.CAPACITOR
        built.graph.validate()

    def test_isolated_foster_branch_mna_numeric(self):
        # 11. Isolated Foster branch MNA numerically equals jX_Foster
        import numpy as np

        from foster_eom.circuit.solve import solve_circuit_single
        from foster_eom.domain.source import SourceMode, SourceSpec

        # We test this by making a SERIES branch and a SHORT shunt branch in DUAL orientation.
        # DUAL: series then shunt.
        # Wait, if shunt is OPEN, then Z_in = Z_series + Z_L
        topo = _make_topology(
            orientation=LOrientation.ALTERNATE_L_ORIENTATION, b1_cells=0, b2_cells=1
        )
        sign = _make_sign_pattern(
            orientation=LOrientation.ALTERNATE_L_ORIENTATION,
            b1_real=BranchRealization.OPEN_OMITTED,
            b2_real=BranchRealization.FINITE_FOSTER,
        )
        b1_comp = None
        b2_comp = _make_components(n_cells=1)  # L=10u, C=100p, f_pole=5M

        model = _DummyModel()
        built = build_foster_circuit(topo, sign, b1_comp, b2_comp, model)

        source = SourceSpec(mode=SourceMode.THEVENIN, thevenin_vrms=1.0, z_ref_ohm=50.0)
        f_hz = 1e6
        sol = solve_circuit_single(built.graph, source, f_hz)

        # Z_in should be Z_series + Z_L. But Z_L is OPEN! Wait.
        # If shunt is OPEN, load is connected to series. Z_in = Z_series + 50
        assert sol.z_in is not None

        # Analytical Z_series:
        # cell 1: L=10u, C=100p.
        w = 2 * np.pi * f_hz
        z_l = 1j * w * 10e-6
        z_c = 1 / (1j * w * 100e-12)
        z_cell = 1 / (1 / z_l + 1 / z_c)
        z_expected = z_cell + 50.0

        np.testing.assert_allclose(sol.z_in, z_expected, rtol=1e-5)

    def test_standard_whole_network_mna(self):
        # 12. STANDARD whole-network MNA numerically equals Z_shunt || (Z_series + Z_L)
        import numpy as np

        from foster_eom.circuit.solve import solve_circuit_single
        from foster_eom.domain.source import SourceMode, SourceSpec

        topo = _make_topology(
            orientation=LOrientation.SCHMIDT_SHUNT_THEN_SERIES, b1_cells=1, b2_cells=1
        )
        sign = _make_sign_pattern(
            orientation=LOrientation.SCHMIDT_SHUNT_THEN_SERIES,
            b1_real=BranchRealization.FINITE_FOSTER,
            b2_real=BranchRealization.FINITE_FOSTER,
        )

        b1_comp = _make_components(n_cells=1)  # cell 1
        b2_comp = _make_components(n_cells=1)  # cell 2

        model = _DummyModel()
        built = build_foster_circuit(topo, sign, b1_comp, b2_comp, model)

        source = SourceSpec(mode=SourceMode.THEVENIN, thevenin_vrms=1.0, z_ref_ohm=50.0)
        f_hz = 1e6
        sol = solve_circuit_single(built.graph, source, f_hz)

        w = 2 * np.pi * f_hz
        z_l = 1j * w * 10e-6
        z_c = 1 / (1j * w * 100e-12)
        z_cell = 1 / (1 / z_l + 1 / z_c)

        z_shunt = z_cell
        z_series = z_cell
        z_l_load = 50.0

        y_expected = (1 / z_shunt) + 1 / (z_series + z_l_load)
        z_expected = 1 / y_expected

        np.testing.assert_allclose(sol.z_in, z_expected, rtol=1e-5)

    def test_dual_whole_network_mna(self):
        # 13. DUAL whole-network MNA numerically equals Z_series + (Z_L || Z_shunt)
        import numpy as np

        from foster_eom.circuit.solve import solve_circuit_single
        from foster_eom.domain.source import SourceMode, SourceSpec

        topo = _make_topology(
            orientation=LOrientation.ALTERNATE_L_ORIENTATION, b1_cells=1, b2_cells=1
        )
        sign = _make_sign_pattern(
            orientation=LOrientation.ALTERNATE_L_ORIENTATION,
            b1_real=BranchRealization.FINITE_FOSTER,
            b2_real=BranchRealization.FINITE_FOSTER,
        )

        b1_comp = _make_components(n_cells=1)
        b2_comp = _make_components(n_cells=1)

        model = _DummyModel()
        built = build_foster_circuit(topo, sign, b1_comp, b2_comp, model)

        source = SourceSpec(mode=SourceMode.THEVENIN, thevenin_vrms=1.0, z_ref_ohm=50.0)
        f_hz = 1e6
        sol = solve_circuit_single(built.graph, source, f_hz)

        w = 2 * np.pi * f_hz
        z_l = 1j * w * 10e-6
        z_c = 1 / (1j * w * 100e-12)
        z_cell = 1 / (1 / z_l + 1 / z_c)

        z_series = z_cell
        z_shunt = z_cell
        z_l_load = 50.0

        z_expected = z_series + 1 / ((1 / z_shunt) + (1 / z_l_load))

        np.testing.assert_allclose(sol.z_in, z_expected, rtol=1e-5)
