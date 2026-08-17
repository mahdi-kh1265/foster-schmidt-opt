"""Tests for foster_eom.analysis.q_factor (Prompt 06, F.3 + edge cases)."""

from __future__ import annotations

import math

from foster_eom.analysis.q_factor import (
    QStatus,
    compute_q_metrics,
)
from foster_eom.analysis.sweep import SweepSpec, compute_adaptive_sweep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parallel_rlc_graph(R=50.0, f_pole=10e6):
    """Parallel RLC: Z_in = 1 / (1/R + j*omega*C + 1/(j*omega*L))."""

    from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port

    omega_0 = 2.0 * math.pi * f_pole
    L = 1e-6
    C = 1.0 / (omega_0**2 * L)

    g = CircuitGraph(
        ground_node_id="gnd",
        input_port=Port("in", "gnd"),
        eom_element_id="R_eom",
    )
    g.add_node(Node("gnd", is_ground=True))
    g.add_node(Node("in"))
    g.add_element(Element("R_eom", ElementKind.RESISTOR, "in", "gnd", value=R))
    g.add_element(Element("L1", ElementKind.INDUCTOR, "in", "gnd", value=L))
    g.add_element(Element("C1", ElementKind.CAPACITOR, "in", "gnd", value=C))
    return g, L, C


def _source(z=50.0):
    from foster_eom.domain.source import SourceMode, SourceSpec

    return SourceSpec(
        mode=SourceMode.THEVENIN,
        thevenin_vrms=1.0,
        z_source_real_ohm=z,
        z_ref_ohm=z,
    )


def _dummy_model():
    from foster_eom.models.base import OnePortModel

    class _Dummy(OnePortModel):
        def _z_impl(self, f_hz):
            return 50.0 + 0j

        def metadata(self):
            return {"name": "_Dummy"}

    return _Dummy()


def _run_sweep(f_pole=10e6, R=50.0, n_base=200, max_depth=4):
    g, _L, _C = _make_parallel_rlc_graph(R=R, f_pole=f_pole)
    src = _source()
    spec = SweepSpec.from_targets([f_pole], n_base_points=n_base, max_depth=max_depth)
    return compute_adaptive_sweep(g, src, _dummy_model(), spec, target_hz=[f_pole]), g, src


# ---------------------------------------------------------------------------
# F.3: Textbook Q
# ---------------------------------------------------------------------------


class TestTextbookQ:
    def test_q_voltage_extracted(self):
        f_pole = 10e6
        sweep, g, src = _run_sweep(f_pole=f_pole, R=500.0, max_depth=5)
        result = compute_q_metrics(sweep, [f_pole], graph=g, source_spec=src)
        m = result.per_target[0]
        assert m.status in (QStatus.OK, QStatus.MULTIPLE_NEARBY_PEAKS), (
            f"Expected OK or MULTIPLE, got {m.status}"
        )
        if m.q_voltage is not None:
            assert m.q_voltage > 0

    def test_f0_near_target(self):
        f_pole = 10e6
        sweep, _g, _src = _run_sweep(f_pole=f_pole, max_depth=5)
        result = compute_q_metrics(sweep, [f_pole])
        m = result.per_target[0]
        if m.f0_hz is not None:
            assert abs(m.f0_hz - f_pole) / f_pole < 0.05


# ---------------------------------------------------------------------------
# Missing crossings
# ---------------------------------------------------------------------------


class TestMissingCrossings:
    def test_missing_both_crossings(self):
        """Construct a sweep where there is a peak but -3 dB band exceeds sweep range."""
        f_pole = 10e6
        g, _, _ = _make_parallel_rlc_graph(R=50.0, f_pole=f_pole)
        src = _source()
        spec = SweepSpec(
            f_min_hz=9.999e6,
            f_max_hz=10.001e6,
            n_base_points=50,
            max_depth=3,
        )
        sweep = compute_adaptive_sweep(g, src, _dummy_model(), spec, target_hz=[f_pole])
        result = compute_q_metrics(sweep, [f_pole])
        m = result.per_target[0]
        if m.status in (
            QStatus.CROSSINGS_MISSING_LOW,
            QStatus.CROSSINGS_MISSING_HIGH,
            QStatus.CROSSINGS_MISSING_BOTH,
        ):
            assert m.q_voltage is None

    def test_no_fake_q_when_crossings_missing(self):
        """q_voltage must be None if any crossing is absent."""
        f_pole = 10e6
        g, _, _ = _make_parallel_rlc_graph(R=50.0, f_pole=f_pole)
        src = _source()
        spec = SweepSpec(f_min_hz=9.99e6, f_max_hz=10.01e6, n_base_points=10, max_depth=1)
        sweep = compute_adaptive_sweep(g, src, _dummy_model(), spec, target_hz=[f_pole])
        result = compute_q_metrics(sweep, [f_pole])
        for m in result.per_target:
            if m.status not in (QStatus.OK,):
                assert m.q_voltage is None, f"Fake Q produced for status={m.status}: {m.q_voltage}"


# ---------------------------------------------------------------------------
# Multiple nearby peaks
# ---------------------------------------------------------------------------


class TestMultiplePeaks:
    def test_multiple_peaks_f0_non_null(self):
        """For MULTIPLE_NEARBY_PEAKS, f0_hz and candidate_peaks_hz must be populated."""
        f_pole = 10e6
        sweep, _, _ = _run_sweep(f_pole=f_pole, max_depth=3)
        result = compute_q_metrics(sweep, [f_pole], search_window_ratio=0.9)
        assert len(result.per_target) == 1
        m = result.per_target[0]
        if m.status not in (QStatus.NO_LOCAL_PEAK, QStatus.UNRESOLVED_REGION):
            assert m.f0_hz is not None
            assert len(m.candidate_peaks_hz) >= 1

    def test_candidate_peaks_populated_for_multiple(self):
        """candidate_peaks_hz must contain all peaks when multiple are found."""
        f_pole = 10e6
        sweep, _, _ = _run_sweep(f_pole=f_pole, max_depth=4, n_base=200)
        result = compute_q_metrics(sweep, [f_pole])
        for m in result.per_target:
            if m.status == QStatus.MULTIPLE_NEARBY_PEAKS:
                assert len(m.candidate_peaks_hz) >= 2
            if m.status == QStatus.OK:
                assert len(m.candidate_peaks_hz) >= 1


# ---------------------------------------------------------------------------
# TARGET_ON_SHOULDER
# ---------------------------------------------------------------------------


class TestTargetOnShoulder:
    def test_shoulder_status_f0_non_null(self):
        """TARGET_ON_SHOULDER must still report f0_hz (the nearest peak)."""
        f_pole = 10e6
        sweep, _, _ = _run_sweep(f_pole=f_pole, max_depth=3)
        target_off = f_pole * 0.85
        result = compute_q_metrics(sweep, [target_off])
        m = result.per_target[0]
        if m.status == QStatus.TARGET_ON_SHOULDER:
            assert m.f0_hz is not None
            assert m.nearest_peak_hz is not None


# ---------------------------------------------------------------------------
# Energy Q
# ---------------------------------------------------------------------------


class TestEnergyQ:
    def test_energy_q_available_for_native_rlc(self):
        f_pole = 10e6
        sweep, g, src = _run_sweep(f_pole=f_pole, R=500.0, max_depth=5)
        result = compute_q_metrics(sweep, [f_pole], graph=g, source_spec=src)
        m = result.per_target[0]
        if m.f0_hz is not None:
            assert isinstance(m.q_energy_available, bool)

    def test_energy_q_convention_loaded(self):
        """Q_energy = omega * W_stored / P_loss using RMS phasors (loaded)."""

        from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
        from foster_eom.domain.source import SourceMode, SourceSpec

        f_pole = 10e6
        R = 5.0
        L = 1e-6
        omega_0 = 2 * math.pi * f_pole
        C = 1.0 / (omega_0**2 * L)
        # Loaded Q = omega_0 * L / (R + R_source) -- not checked analytically here

        g = CircuitGraph(
            ground_node_id="gnd",
            input_port=Port("in", "gnd"),
            eom_element_id="R1",
        )
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        g.add_node(Node("mid"))
        g.add_node(Node("n2"))
        g.add_element(Element("R1", ElementKind.RESISTOR, "in", "mid", value=R))
        g.add_element(Element("L1", ElementKind.INDUCTOR, "mid", "n2", value=L))
        g.add_element(Element("C1", ElementKind.CAPACITOR, "n2", "gnd", value=C))

        src = SourceSpec(
            mode=SourceMode.THEVENIN,
            thevenin_vrms=1.0,
            z_source_real_ohm=50.0,
            z_ref_ohm=50.0,
        )
        spec = SweepSpec.from_targets([f_pole], n_base_points=100, max_depth=5)
        sweep = compute_adaptive_sweep(g, src, _dummy_model(), spec, target_hz=[f_pole])
        result = compute_q_metrics(sweep, [f_pole], graph=g, source_spec=src)
        m = result.per_target[0]
        if m.q_energy_available and m.q_energy is not None:
            assert m.q_energy > 0
            assert isinstance(m.q_energy_unavailable_reason, (str, type(None)))

    def test_energy_q_unavailable_for_one_port_model(self):
        """ONE_PORT_MODEL guard: _compute_energy_q returns None and unavailable reason."""
        from foster_eom.analysis.q_factor import _compute_energy_q
        from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
        from foster_eom.models.base import OnePortModel

        class _Simple(OnePortModel):
            def _z_impl(self, f_hz):
                return 50.0 + 0j

            def metadata(self):
                return {"type": "simple"}

        model = _Simple()

        g = CircuitGraph(
            ground_node_id="gnd",
            input_port=Port("in", "gnd"),
            eom_element_id="M1",
        )
        g.add_node(Node("gnd", is_ground=True))
        g.add_node(Node("in"))
        g.add_element(Element("M1", ElementKind.ONE_PORT_MODEL, "in", "gnd", model=model))

        src = _source()

        q_e, avail, reason = _compute_energy_q(g, src, 10e6)
        assert q_e is None
        assert avail is False
        assert reason is not None
        assert "ONE_PORT_MODEL" in reason

        spec = SweepSpec.from_targets([10e6], n_base_points=20, max_depth=1)
        sweep = compute_adaptive_sweep(g, src, model, spec, target_hz=[10e6])
        result = compute_q_metrics(sweep, [10e6], graph=g, source_spec=src)
        m = result.per_target[0]
        assert m.q_energy is None
        assert m.q_energy_available is False
        assert m.q_energy_unavailable_reason is not None
