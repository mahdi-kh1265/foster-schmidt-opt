"""End-to-end Prompt-06 smoke test (P06 §smoke).

Uses a simple synthetic circuit and runs the complete
sweep -> Q -> stress -> time-reconstruction pipeline.
"""

from __future__ import annotations

import math

from foster_eom.analysis.q_factor import QStatus, compute_q_metrics
from foster_eom.analysis.stress import compute_stress
from foster_eom.analysis.sweep import SweepSpec, compute_adaptive_sweep
from foster_eom.analysis.time_reconstruction import compute_time_domain
from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
from foster_eom.domain.objectives import TimeDomainPhaseMode
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.models.base import OnePortModel


class _Dummy50(OnePortModel):
    def _z_impl(self, f_hz: float) -> complex:
        return 50.0 + 0j

    def metadata(self) -> dict:
        return {"name": "_Dummy50"}


def _make_circuit(f_pole=10e6):
    """Parallel RLC with resonance near 10 MHz."""
    omega_0 = 2.0 * math.pi * f_pole
    L = 1e-6
    C = 1.0 / (omega_0**2 * L)
    R = 200.0  # moderate Q

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
    return g


class TestP06Smoke:
    """Full Prompt-06 pipeline smoke test: no mocks, real execution path."""

    def test_p06_full_pipeline(self):
        targets = [9e6, 10e6, 11e6]
        g = _make_circuit(f_pole=10e6)
        src = SourceSpec(
            mode=SourceMode.THEVENIN,
            thevenin_vrms=1.0,
            z_source_real_ohm=50.0,
            z_ref_ohm=50.0,
        )
        model = _Dummy50()

        # 1. Adaptive sweep
        spec = SweepSpec.from_targets(targets, n_base_points=80, max_depth=4)
        sweep = compute_adaptive_sweep(g, src, model, spec, target_hz=targets)

        assert len(sweep.frequencies_hz) > 80, "Adaptive refinement should add points"
        assert isinstance(sweep.verification_complete, bool)
        assert isinstance(sweep.declared_frequency_resolution_hz, float)
        assert sweep.declared_frequency_resolution_hz > 0
        assert isinstance(sweep.worst_power_balance_residual, float)

        # 2. Q metrics
        q_result = compute_q_metrics(sweep, targets, graph=g, source_spec=src)
        assert len(q_result.per_target) == 3
        for m in q_result.per_target:
            assert isinstance(m.status, QStatus)
            assert isinstance(m.candidate_peaks_hz, tuple)
            if m.f0_hz is not None:
                assert m.f0_hz > 0

        # 3. Stress
        stress = compute_stress(g, src, sweep, targets)
        assert len(stress.elements) > 0
        for e in stress.elements:
            assert e.multitone_v_peak_bound_v >= e.multitone_v_rms_v * math.sqrt(2) - 1e-12
            assert e.multitone_i_peak_bound_a >= 0

        # 4. Time-domain reconstruction
        # Use v_eom phasors from sweep at target frequencies
        v_phasors_eom = []
        for ft in targets:
            idx = min(
                range(len(sweep.frequencies_hz)), key=lambda i: abs(sweep.frequencies_hz[i] - ft)
            )
            v = sweep.v_eom_mag[idx]
            v_phasors_eom.append(float(v) if v is not None else 0.0)

        tones = list(zip(targets, [complex(v) for v in v_phasors_eom], strict=True))

        td = compute_time_domain(tones, TimeDomainPhaseMode.ALL_ZERO)

        assert td.eom_signal.conservative_bound >= td.eom_signal.peak_val - 1e-12
        assert td.eom_signal.conservative_bound >= 0
        assert td.n_points > 0
        assert not td.point_count_capped or td.n_points == 50_000

        # 5. CandidateResult integration
        from foster_eom.domain.results import CandidateResult

        cr = CandidateResult(
            candidate_id="p06_smoke",
            sweep_result=sweep,
            q_result=q_result,
            stress_summary=stress,
            time_domain_result=td,
            verification_complete=sweep.verification_complete,
        )
        assert cr.sweep_result is sweep
        assert cr.verification_complete == sweep.verification_complete
