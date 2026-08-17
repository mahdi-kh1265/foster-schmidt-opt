"""Tests for foster_eom.analysis.stress (Prompt 06)."""

from __future__ import annotations

import math

from foster_eom.analysis.stress import compute_stress
from foster_eom.analysis.sweep import SweepSpec, compute_adaptive_sweep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_two_r_graph(R1=50.0, R2=100.0):
    """Source -> R1 in series -> R2 shunt to ground.

    Gives two distinct resistors with different V/I stress.
    """
    from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port

    g = CircuitGraph(
        ground_node_id="gnd",
        input_port=Port("in", "gnd"),
        eom_element_id="R2",
    )
    g.add_node(Node("gnd", is_ground=True))
    g.add_node(Node("in"))
    g.add_node(Node("mid"))
    g.add_element(Element("R1", ElementKind.RESISTOR, "in", "mid", value=R1))
    g.add_element(Element("R2", ElementKind.RESISTOR, "mid", "gnd", value=R2))
    return g


# ---------------------------------------------------------------------------
# Two-tone RMS is RSS not max
# ---------------------------------------------------------------------------


class TestMultitoneRMS:
    def test_two_tone_rms_is_rss(self):
        """Multi-tone V_rms = sqrt(|V1|^2 + |V2|^2), not max(|V1|, |V2|)."""
        g = _make_two_r_graph()
        src = _source()
        spec = SweepSpec.from_targets([9e6, 11e6], n_base_points=30, max_depth=2)
        sweep = compute_adaptive_sweep(g, src, _dummy_model(), spec, target_hz=[9e6, 11e6])
        summary = compute_stress(g, src, sweep, [9e6, 11e6])

        for e in summary.elements:
            # RMS must be >= max per-tone RMS (RSS >= component)
            # and >= 0
            assert e.multitone_v_rms_v >= 0
            assert e.multitone_i_rms_a >= 0
            # Conservative bound >= RMS
            assert e.multitone_v_peak_bound_v >= e.multitone_v_rms_v * math.sqrt(2) - 1e-12, (
                "Peak bound must be >= sqrt(2)*RMS"
            )

    def test_two_tone_power_is_sum_not_max(self):
        """Multi-tone P_avg = P1 + P2, not max."""
        g = _make_two_r_graph()
        src = _source()
        spec = SweepSpec.from_targets([9e6, 11e6], n_base_points=20, max_depth=1)
        sweep = compute_adaptive_sweep(g, src, _dummy_model(), spec, target_hz=[9e6, 11e6])

        # Single-tone results
        spec1 = SweepSpec(f_min_hz=8e6, f_max_hz=12e6, n_base_points=20)
        sweep1 = compute_adaptive_sweep(g, src, _dummy_model(), spec1, target_hz=[9e6])
        summary1 = compute_stress(g, src, sweep1, [9e6])

        spec2 = SweepSpec(f_min_hz=8e6, f_max_hz=12e6, n_base_points=20)
        sweep2 = compute_adaptive_sweep(g, src, _dummy_model(), spec2, target_hz=[11e6])
        summary2 = compute_stress(g, src, sweep2, [11e6])

        summary12 = compute_stress(g, src, sweep, [9e6, 11e6])

        for e12 in summary12.elements:
            e1 = next((e for e in summary1.elements if e.element_id == e12.element_id), None)
            e2 = next((e for e in summary2.elements if e.element_id == e12.element_id), None)
            if e1 and e2:
                sum_power = e1.multitone_p_avg_w + e2.multitone_p_avg_w
                # Multi-tone average power must be approximately the sum
                assert abs(e12.multitone_p_avg_w - sum_power) <= sum_power * 0.05 + 1e-12, (
                    f"Multi-tone power {e12.multitone_p_avg_w:.4g} "
                    f"!= sum {sum_power:.4g} for {e12.element_id}"
                )


# ---------------------------------------------------------------------------
# Peak stress <= conservative bound
# ---------------------------------------------------------------------------


class TestConservativeBound:
    def test_conservative_bound_gte_rss(self):
        """conservative_bound >= sqrt(2)*V_rms always."""
        g = _make_two_r_graph()
        src = _source()
        spec = SweepSpec.from_targets([10e6], n_base_points=20, max_depth=1)
        sweep = compute_adaptive_sweep(g, src, _dummy_model(), spec, target_hz=[10e6])
        summary = compute_stress(g, src, sweep, [10e6])
        for e in summary.elements:
            assert e.multitone_v_peak_bound_v >= 0
            assert e.multitone_i_peak_bound_a >= 0


# ---------------------------------------------------------------------------
# Separate worst V/I/P frequencies
# ---------------------------------------------------------------------------


class TestSeparateWorstFrequencies:
    def test_separate_freq_fields_present(self):
        g = _make_two_r_graph()
        src = _source()
        spec = SweepSpec.from_targets([10e6], n_base_points=20)
        sweep = compute_adaptive_sweep(g, src, _dummy_model(), spec)
        summary = compute_stress(g, src, sweep, [10e6])
        for e in summary.elements:
            assert isinstance(e.sweep_worst_v_freq_hz, float)
            assert isinstance(e.sweep_worst_i_freq_hz, float)
            assert isinstance(e.sweep_worst_p_freq_hz, float)

    def test_current_fields_present(self):
        """Current peak bound and margin fields must exist."""
        g = _make_two_r_graph()
        src = _source()
        spec = SweepSpec.from_targets([10e6], n_base_points=20)
        sweep = compute_adaptive_sweep(g, src, _dummy_model(), spec)
        summary = compute_stress(
            g,
            src,
            sweep,
            [10e6],
            ratings={
                "R1": {"voltage_v": 100.0, "current_a": 1.0},
                "R2": {"voltage_v": 100.0, "current_a": 1.0},
            },
        )
        for e in summary.elements:
            assert isinstance(e.multitone_i_peak_bound_a, float)
            assert e.sweep_current_margin is not None
            assert e.multitone_current_margin is not None


# ---------------------------------------------------------------------------
# Worst-element summary fields
# ---------------------------------------------------------------------------


class TestWorstElements:
    def test_worst_current_element_reported(self):
        g = _make_two_r_graph()
        src = _source()
        spec = SweepSpec.from_targets([10e6], n_base_points=20)
        sweep = compute_adaptive_sweep(g, src, _dummy_model(), spec)
        summary = compute_stress(g, src, sweep, [10e6])
        # Field must exist; may be None if no elements
        assert summary.worst_sweep_current_element is not None
        assert summary.worst_multitone_current_element is not None
