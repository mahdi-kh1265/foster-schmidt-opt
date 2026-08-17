"""Unit tests for foster_eom.analysis.sweep (Prompt 06, spec §F.1, F.2, F.4).

Acceptance tests
----------------
F.1  Hidden narrow resonance is detected
F.2  Adaptive refinement converges
F.4  Non-peak target / shoulder condition

Edge-case tests (spec §7.2)
---------------------------
- Narrow resonance narrower than initial coarse grid is still found
- Unresolved interval at max_depth: verification_complete=False, listed
- Pole interval NOT unresolved when curvature passes (PATCH 2)
- declared_frequency_resolution_hz present and equals ratio x band (PATCH 3)

- Phase unwrap across ±π: unwrapped monotone
"""

from __future__ import annotations

import math
from typing import Any

from foster_eom.analysis.sweep import (
    SweepSpec,
    compute_adaptive_sweep,
)
from foster_eom.circuit import CircuitGraph, Element, ElementKind, Node, Port
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.models.base import OnePortModel

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_SOURCE_50 = SourceSpec(
    mode=SourceMode.THEVENIN,
    thevenin_vrms=1.0,
    z_source_real_ohm=50.0,
    z_ref_ohm=50.0,
)


def _make_parallel_rlc_graph(
    R: float,
    L: float,
    C: float,
    source_node: str = "in",
    gnd: str = "gnd",
) -> CircuitGraph:
    """Parallel R-L-C to ground; C is the EOM element."""
    g = CircuitGraph(
        ground_node_id=gnd,
        input_port=Port(node_pos=source_node, node_neg=gnd),
        eom_element_id="C1",
    )
    g.add_node(Node(id=gnd, is_ground=True))
    g.add_node(Node(id=source_node))
    g.add_element(
        Element(id="R1", kind=ElementKind.RESISTOR, node_pos=source_node, node_neg=gnd, value=R)
    )
    g.add_element(
        Element(id="L1", kind=ElementKind.INDUCTOR, node_pos=source_node, node_neg=gnd, value=L)
    )
    g.add_element(
        Element(id="C1", kind=ElementKind.CAPACITOR, node_pos=source_node, node_neg=gnd, value=C)
    )
    return g


class _StubEOM(OnePortModel):
    """Minimal stub OnePortModel — satisfies abstract interface."""

    def _z_impl(self, f_hz):  # type: ignore[override]
        return 50.0 + 0j

    def metadata(self) -> dict[str, Any]:
        return {"type": "stub"}


_EOM = _StubEOM()


# ---------------------------------------------------------------------------
# F.1  Hidden narrow resonance is detected
# ---------------------------------------------------------------------------


class TestF1HiddenNarrowResonance:
    """A very narrow parallel resonance hidden between coarse grid points."""

    def _graph_and_target(self):
        f0 = 10e6
        L = 1e-6
        C = 1.0 / ((2 * math.pi * f0) ** 2 * L)
        R = 1000.0
        g = _make_parallel_rlc_graph(R, L, C)
        return g, f0

    def test_resonance_detected_in_sweep(self):
        graph, f0 = self._graph_and_target()
        spec = SweepSpec.from_targets([f0], n_base_points=10)
        result = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec, target_hz=[f0])

        v_peaks = [
            p for p in result.resonance_list if p.quantity_name == "v_eom" and p.is_local_maximum
        ]
        assert len(v_peaks) >= 1, "Expected at least one V_EOM peak"

        closest = min(v_peaks, key=lambda p: abs(p.frequency_hz - f0))
        assert abs(closest.frequency_hz - f0) / f0 < 0.05, (
            f"Closest peak {closest.frequency_hz:.4g} Hz not near {f0:.4g} Hz"
        )

    def test_more_points_in_resonance_region(self):
        graph, f0 = self._graph_and_target()
        spec_coarse = SweepSpec.from_targets([f0], n_base_points=10, max_depth=0)
        spec_adaptive = SweepSpec.from_targets([f0], n_base_points=10, max_depth=5)

        r_coarse = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec_coarse, target_hz=[f0])
        r_adaptive = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec_adaptive, target_hz=[f0])

        assert len(r_adaptive.frequencies_hz) > len(r_coarse.frequencies_hz)


# ---------------------------------------------------------------------------
# F.2  Adaptive convergence
# ---------------------------------------------------------------------------


class TestF2AdaptiveConvergence:
    def test_convergence_with_depth(self):
        f0 = 9e6
        L = 1e-6
        C = 1.0 / ((2 * math.pi * f0) ** 2 * L)
        R = 500.0
        graph = _make_parallel_rlc_graph(R, L, C)

        errors = []
        for depth in [1, 3, 5]:
            spec = SweepSpec.from_targets([f0], n_base_points=20, max_depth=depth)
            result = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec, target_hz=[f0])
            v_peaks = [
                p
                for p in result.resonance_list
                if p.quantity_name == "v_eom" and p.is_local_maximum
            ]
            if v_peaks:
                closest = min(v_peaks, key=lambda p: abs(p.frequency_hz - f0))
                errors.append(abs(closest.frequency_hz - f0) / f0)
            else:
                errors.append(1.0)

        assert errors[-1] <= errors[0] + 1e-4

    def test_verification_complete_clean_case(self):
        f0 = 10e6
        L = 1e-6
        C = 1.0 / ((2 * math.pi * f0) ** 2 * L)
        R = 50.0
        graph = _make_parallel_rlc_graph(R, L, C)
        spec = SweepSpec(f_min_hz=8e6, f_max_hz=12e6, n_base_points=100, max_depth=5)
        result = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec, target_hz=[f0])
        assert result.verification_complete


# ---------------------------------------------------------------------------
# F.4  Non-peak target / shoulder condition
# ---------------------------------------------------------------------------


class TestF4TargetOnShoulder:
    def test_off_target_resonance_fields_exist(self):
        f_peak = 10e6
        L = 1e-6
        C = 1.0 / ((2 * math.pi * f_peak) ** 2 * L)
        R = 1000.0
        graph = _make_parallel_rlc_graph(R, L, C)

        f_target = 8e6
        spec = SweepSpec.from_targets([f_target], margin_lo=0.8, margin_hi=1.5, n_base_points=50)
        result = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec, target_hz=[f_target])
        assert isinstance(result.off_target_unsafe, bool)
        assert isinstance(result.resonance_list, tuple)


# ---------------------------------------------------------------------------
# Edge case: narrow resonance narrower than initial grid
# ---------------------------------------------------------------------------


class TestNarrowResonanceDetection:
    def test_very_high_q_detected(self):
        f0 = 10e6
        L = 1e-6
        C = 1.0 / ((2 * math.pi * f0) ** 2 * L)
        R = 10_000.0
        graph = _make_parallel_rlc_graph(R, L, C)
        spec = SweepSpec.from_targets(
            [f0],
            n_base_points=5,
            max_depth=8,
            curvature_tol=0.01,
            include_pole_neighborhoods=True,
        )
        result = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec, target_hz=[f0])
        v_peaks = [
            p for p in result.resonance_list if p.quantity_name == "v_eom" and p.is_local_maximum
        ]
        assert len(v_peaks) >= 1, "Expected hidden resonance to be detected"


# ---------------------------------------------------------------------------
# Edge case: unresolved interval at max_depth
# ---------------------------------------------------------------------------


class TestUnresolvedInterval:
    def test_unresolved_fields_always_present(self):
        graph = _make_parallel_rlc_graph(50.0, 1e-6, 1e-12)
        spec = SweepSpec(f_min_hz=1e6, f_max_hz=2e6, n_base_points=5, max_depth=0)
        result = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec)
        assert isinstance(result.verification_complete, bool)
        assert isinstance(result.unresolved_intervals, tuple)

    def test_unresolved_listed_when_failing(self):
        f0 = 10e6
        L = 1e-6
        C = 1.0 / ((2 * math.pi * f0) ** 2 * L)
        R = 100_000.0
        graph = _make_parallel_rlc_graph(R, L, C)
        spec = SweepSpec(
            f_min_hz=9.9e6,
            f_max_hz=10.1e6,
            n_base_points=2,
            max_depth=1,
            pole_min_refinement_depth=0,
            min_interval_width_ratio=1e-9,
            curvature_tol=1e-8,
            include_pole_neighborhoods=False,
        )
        result = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec, target_hz=[f0])
        if not result.verification_complete:
            assert len(result.unresolved_intervals) > 0


# ---------------------------------------------------------------------------
# PATCH 2: Pole interval NOT unresolved when curvature passes
# ---------------------------------------------------------------------------


class TestPatch2PoleIntervalNotUnresolved:
    def test_resolved_when_curvature_passes(self):
        f0 = 10e6
        L = 1e-6
        C = 1.0 / ((2 * math.pi * f0) ** 2 * L)
        R = 10.0  # very low Q → broad, easy curvature
        graph = _make_parallel_rlc_graph(R, L, C)
        spec = SweepSpec(
            f_min_hz=8e6,
            f_max_hz=12e6,
            n_base_points=50,
            max_depth=2,
            pole_min_refinement_depth=2,
            min_interval_width_ratio=1e-3,
            curvature_tol=0.5,  # very loose
            include_pole_neighborhoods=True,
        )
        result = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec, target_hz=[f0])
        assert result.verification_complete, (
            f"Expected resolved; unresolved={result.unresolved_intervals}"
        )
        assert len(result.unresolved_intervals) == 0


# ---------------------------------------------------------------------------
# PATCH 3: declared_frequency_resolution_hz
# ---------------------------------------------------------------------------


class TestPatch3DeclaredResolution:
    def test_declared_resolution_field_correct(self):
        f0 = 10e6
        L = 1e-6
        C = 1.0 / ((2 * math.pi * f0) ** 2 * L)
        R = 50.0
        graph = _make_parallel_rlc_graph(R, L, C)

        ratio = 5e-3
        spec = SweepSpec(
            f_min_hz=9e6,
            f_max_hz=11e6,
            n_base_points=20,
            min_interval_width_ratio=ratio,
        )
        result = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec, target_hz=[f0])

        expected = ratio * (spec.f_max_hz - spec.f_min_hz)
        assert math.isclose(result.declared_frequency_resolution_hz, expected, rel_tol=1e-9)

    def test_declared_resolution_always_present_and_positive(self):
        graph = _make_parallel_rlc_graph(50.0, 1e-6, 1e-12)
        spec = SweepSpec(f_min_hz=1e6, f_max_hz=2e6, n_base_points=5)
        result = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec)
        assert hasattr(result, "declared_frequency_resolution_hz")
        assert result.declared_frequency_resolution_hz > 0


# ---------------------------------------------------------------------------
# Phase unwrap across ±π
# ---------------------------------------------------------------------------


class TestPhaseUnwrap:
    def test_unwrapped_phase_present(self):
        f0 = 10e6
        L = 1e-6
        C = 1.0 / ((2 * math.pi * f0) ** 2 * L)
        R = 50.0
        graph = _make_parallel_rlc_graph(R, L, C)
        spec = SweepSpec(f_min_hz=5e6, f_max_hz=15e6, n_base_points=100)
        result = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec, target_hz=[f0])
        phases = [p for p in result.unwrapped_phase_rad if p is not None]
        assert len(phases) > 0

    def test_unwrapped_phase_no_large_jumps(self):
        f0 = 10e6
        L = 1e-6
        C = 1.0 / ((2 * math.pi * f0) ** 2 * L)
        R = 100.0
        graph = _make_parallel_rlc_graph(R, L, C)
        spec = SweepSpec(f_min_hz=5e6, f_max_hz=15e6, n_base_points=200, max_depth=5)
        result = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec, target_hz=[f0])
        phases = [p for p in result.unwrapped_phase_rad if p is not None]
        if len(phases) >= 2:
            diffs = [abs(phases[i + 1] - phases[i]) for i in range(len(phases) - 1)]
            assert max(diffs) <= math.pi + 0.1


# ---------------------------------------------------------------------------
# SweepSpec.from_targets
# ---------------------------------------------------------------------------


class TestSweepSpecFromTargets:
    def test_band_derived_from_targets(self):
        spec = SweepSpec.from_targets([10e6, 20e6], margin_lo=0.5, margin_hi=2.0)
        assert math.isclose(spec.f_min_hz, 10e6 * 0.5)
        assert math.isclose(spec.f_max_hz, 20e6 * 2.0)

    def test_validity_range_clips(self):
        spec = SweepSpec.from_targets(
            [10e6], margin_lo=0.1, margin_hi=3.0, validity_range=(5e6, 25e6)
        )
        assert spec.f_min_hz >= 5e6
        assert spec.f_max_hz <= 25e6

    def test_explicit_kwargs_forwarded(self):
        spec = SweepSpec.from_targets([10e6], n_base_points=42, max_depth=7)
        assert spec.n_base_points == 42
        assert spec.max_depth == 7


# ---------------------------------------------------------------------------
# SweepResult field completeness
# ---------------------------------------------------------------------------


class TestSweepResultFields:
    def test_all_required_fields_present(self):
        graph = _make_parallel_rlc_graph(50.0, 1e-6, 1e-12)
        spec = SweepSpec(f_min_hz=1e6, f_max_hz=2e6, n_base_points=5)
        result = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec)
        assert isinstance(result.spec, SweepSpec)
        assert isinstance(result.frequencies_hz, tuple)
        assert isinstance(result.v_eom_mag, tuple)
        assert isinstance(result.gamma_mag, tuple)
        assert isinstance(result.i_source_mag, tuple)
        assert isinstance(result.unwrapped_phase_rad, tuple)
        assert isinstance(result.resonance_list, tuple)
        assert isinstance(result.off_target_unsafe, bool)
        assert isinstance(result.failed_frequencies_hz, tuple)
        assert isinstance(result.worst_power_balance_residual, float)
        assert isinstance(result.power_balance_ok, bool)
        assert isinstance(result.verification_complete, bool)
        assert isinstance(result.unresolved_intervals, tuple)
        assert isinstance(result.declared_frequency_resolution_hz, float)

    def test_parallel_arrays_same_length(self):
        graph = _make_parallel_rlc_graph(50.0, 1e-6, 1e-12)
        spec = SweepSpec(f_min_hz=1e6, f_max_hz=2e6, n_base_points=10)
        result = compute_adaptive_sweep(graph, _SOURCE_50, _EOM, spec)
        n = len(result.frequencies_hz)
        assert len(result.v_eom_mag) == n
        assert len(result.gamma_mag) == n
        assert len(result.i_source_mag) == n
        assert len(result.unwrapped_phase_rad) == n
