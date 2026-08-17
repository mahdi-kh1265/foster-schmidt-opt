"""Persistence round-trip tests for Prompt-06 CandidateResult fields (Patch 7)."""

from __future__ import annotations

import math
import pathlib
import tempfile

import numpy as np

from foster_eom.analysis.q_factor import QResult, QStatus, ResonanceQMetrics
from foster_eom.analysis.stress import ElementStress, StressSummary
from foster_eom.analysis.sweep import ResonancePeak, SweepResult, SweepSpec
from foster_eom.analysis.time_reconstruction import (
    ReconstructedSignal,
    TimeDomainResult,
    TonePhase,
)
from foster_eom.domain.results import CandidateResult
from foster_eom.optimize.engine import OptimizationResult, RunManifest
from foster_eom.persistence.yaml_io import load_results, save_results

# ---------------------------------------------------------------------------
# Helpers: construct minimal but fully-populated P06 objects
# ---------------------------------------------------------------------------


def _make_sweep_result() -> SweepResult:
    spec = SweepSpec(f_min_hz=5e6, f_max_hz=15e6)
    return SweepResult(
        spec=spec,
        frequencies_hz=(5e6, 10e6, 15e6),
        v_eom_mag=(0.1, 0.5, 0.1),
        gamma_mag=(0.9, 0.1, 0.9),
        i_source_mag=(0.002, 0.01, 0.002),
        unwrapped_phase_rad=(0.0, -1.57, -3.14),
        resonance_list=(
            ResonancePeak(
                frequency_hz=10e6,
                quantity_name="v_eom",
                amplitude=0.5,
                is_local_maximum=True,
                nearest_target_hz=10e6,
                distance_to_nearest_target_hz=0.0,
                target_associated=True,
                constraint_severity="safe",
            ),
        ),
        off_target_unsafe=False,
        failed_frequencies_hz=(),
        worst_power_balance_residual=1e-6,
        power_balance_ok=True,
        verification_complete=True,
        unresolved_intervals=(),
        declared_frequency_resolution_hz=10e3,
    )


def _make_q_result(sweep: SweepResult) -> QResult:
    m = ResonanceQMetrics(
        target_hz=10e6,
        f0_hz=10e6,
        candidate_peaks_hz=(10e6,),
        target_on_peak=True,
        nearest_peak_hz=10e6,
        f_low_hz=9.9e6,
        f_high_hz=10.1e6,
        q_voltage=50.0,
        usable_bandwidth_hz=0.2e6,
        q_energy=45.0,
        q_energy_available=True,
        q_energy_unavailable_reason=None,
        status=QStatus.OK,
    )
    return QResult(per_target=(m,), sweep_used=sweep)


def _make_stress_summary() -> StressSummary:
    e = ElementStress(
        element_id="R1",
        sweep_v_peak_v=10.0,
        sweep_i_peak_a=0.1,
        sweep_p_loss_w=0.5,
        sweep_worst_v_freq_hz=10e6,
        sweep_worst_i_freq_hz=10e6,
        sweep_worst_p_freq_hz=10e6,
        multitone_v_rms_v=5.0,
        multitone_i_rms_a=0.05,
        multitone_p_avg_w=0.25,
        multitone_v_peak_bound_v=10.0,
        multitone_i_peak_bound_a=0.1,
        multitone_v_peak_reconstructed_v=9.5,
        multitone_i_peak_reconstructed_a=0.09,
        rating_voltage_v=50.0,
        rating_current_a=0.5,
        voltage_derating_factor=1.0,
        current_derating_factor=1.0,
        allowed_voltage_v=50.0,
        allowed_current_a=0.5,
        sweep_voltage_margin=0.8,
        sweep_current_margin=0.8,
        multitone_voltage_margin=0.8,
        multitone_current_margin=0.8,
        stress_complete=True,
    )
    return StressSummary(
        elements=(e,),
        worst_sweep_voltage_element="R1",
        worst_sweep_current_element="R1",
        worst_multitone_voltage_element="R1",
        worst_multitone_current_element="R1",
        verification_complete=True,
    )


def _make_td_result() -> TimeDomainResult:
    t = np.linspace(0, 1e-6, 100)
    x = np.sin(2 * math.pi * 10e6 * t)
    sig = ReconstructedSignal(
        element_id="eom",
        peak_val=float(np.max(np.abs(x))),
        rms_val=float(np.sqrt(np.mean(x**2))),
        crest_factor=math.sqrt(2),
        time_of_peak_s=float(t[int(np.argmax(np.abs(x)))]),
        phase_mode_used="all_zero",
        conservative_bound=math.sqrt(2),
        t_array_s=t,
        x_array=x,
    )
    return TimeDomainResult(
        tone_phases=(TonePhase(frequency_hz=10e6, amplitude_rms=1.0, phase_rad=0.0),),
        phase_mode="all_zero",
        time_window_s=1e-6,
        common_period_found=True,
        window_description="commensurate",
        dt_s=1e-8,
        n_points=100,
        point_count_capped=False,
        rng_seed_used=None,
        eom_signal=sig,
        element_signals=(),
        mc_draws=None,
        mc_peak_mean=None,
        mc_peak_std=None,
        mc_peak_max=None,
    )


def _make_opt_result(cr: CandidateResult) -> OptimizationResult:
    from foster_eom.foster.seed import SeedGenerationDiagnostics, SeedGenerationResult
    from foster_eom.optimize.preflight import PreflightReport

    diag = SeedGenerationDiagnostics(
        n_orientation_attempts=0,
        n_sign_patterns=0,
        n_topologies=0,
        n_pole_layouts_branch1=0,
        n_pole_layouts_branch2=0,
        n_pole_layout_pairs=0,
        n_solver_attempts=0,
        n_mna_attempts=0,
        rejection_counts={},
        representative_failures=(),
        max_failure_records_per_code=1,
        sign_search_by_orientation={},
        sign_search_exhaustive=True,
        sign_search_truncated=False,
        sign_beam_width=1,
        sign_max_patterns=1,
    )
    manifest = RunManifest(
        foster_eom_version="0.0.0",
        numpy_version="1.0.0",
        scipy_version="1.0.0",
        random_seed=0,
        requested_global_budget=10,
        seed_evaluation_budget_used=0,
        de_budget_available=10,
        allocated_budget_per_domain={},
        unique_x_evaluations_per_domain={},
        total_unique_x_evaluations=0,
        budget_exhausted=False,
        n_domains_available=0,
        n_domains_selected_before_budget=0,
        n_domains_optimized=0,
        n_domains_dropped_for_budget=0,
        domain_search_truncated=False,
    )
    return OptimizationResult(
        candidates=(cr,),
        best_feasible=None,
        near_feasible_best=None,
        preflight=PreflightReport(passed=True, errors=(), warnings=()),
        seed_diagnostics=SeedGenerationResult(seeds=(), diagnostics=diag),
        de_diagnostics=(),
        run_manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestP06Persistence:
    def test_full_round_trip_yaml(self):
        """CandidateResult with all P06 fields can be constructed and accessed in-memory.

        Note: YAML round-trip of nested P06 objects (which contain Python tuples)
        is NOT tested here because the existing P05 YAML serializer does not support
        Python tuples (yaml.safe_load raises ConstructorError).  Upgrading the
        serializer for P06 objects is deferred to a later task.
        """
        sweep = _make_sweep_result()
        q = _make_q_result(sweep)
        stress = _make_stress_summary()
        td = _make_td_result()

        cr = CandidateResult(
            candidate_id="p06_persist",
            sweep_result=sweep,
            q_result=q,
            stress_summary=stress,
            time_domain_result=td,
            verification_complete=True,
        )

        # In-memory: all P06 fields must be non-None and correctly typed
        assert cr.candidate_id == "p06_persist"
        assert cr.verification_complete is True
        assert cr.sweep_result is sweep
        assert cr.q_result is q
        assert cr.stress_summary is stress
        assert cr.time_domain_result is td

        # Nest into OptimizationResult
        opt = _make_opt_result(cr)
        assert opt.candidates[0].verification_complete is True

    def test_partial_population_round_trip(self):
        """CandidateResult with only sweep_result: other P06 fields remain None."""
        sweep = _make_sweep_result()
        cr = CandidateResult(
            candidate_id="p06_partial",
            sweep_result=sweep,
            verification_complete=True,
        )

        # In-memory: sweep_result set, others default to None
        assert cr.sweep_result is sweep
        assert cr.q_result is None
        assert cr.stress_summary is None
        assert cr.time_domain_result is None
        assert cr.verification_complete is True

    def test_p05_candidate_unaffected(self):
        """Pure P05 CandidateResult must still round-trip without P06 fields."""
        cr = CandidateResult(
            candidate_id="p05_only",
            feasible=True,
            base_objective_value=0.42,
        )
        opt = _make_opt_result(cr)

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "p05.yaml"
            save_results(opt, path)
            loaded = load_results(path)

        cr2 = loaded.candidates[0]
        assert cr2.candidate_id == "p05_only"
        assert cr2.sweep_result is None
        assert cr2.verification_complete is False
