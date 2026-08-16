"""Tests for domain schemas and validation.

Covers valid construction, invalid-case rejection, and serialization
of all domain Pydantic models.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from foster_eom.domain.component import ComponentPolicy, ContinuousLimits
from foster_eom.domain.constraints import (
    ConstraintRecord,
    ConstraintSeverity,
    MatchConstraints,
    StressConstraints,
)
from foster_eom.domain.eom import EOMModelSpec, EOMModelType, ExtrapolationPolicy, MotionalBranch
from foster_eom.domain.frequency_plan import ExclusionBand, FrequencyPlan, FrequencyTarget
from foster_eom.domain.project import ProjectSpec
from foster_eom.domain.results import CandidateResult
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.domain.topology import (
    PoleInterval,
    PoleMode,
    PoleSpec,
    TopologySearchSpec,
)

# ---------------------------------------------------------------------------
# SourceSpec
# ---------------------------------------------------------------------------


class TestSourceSpec:
    def test_available_power_valid(self) -> None:
        s = SourceSpec(mode=SourceMode.AVAILABLE_POWER, available_power_dbm=20.0)
        assert s.vth_rms > 0

    def test_thevenin_valid(self) -> None:
        s = SourceSpec(mode=SourceMode.THEVENIN, thevenin_vrms=5.0)
        assert s.vth_rms == pytest.approx(5.0)

    def test_generator_valid(self) -> None:
        s = SourceSpec(
            mode=SourceMode.GENERATOR_INTO_Z0,
            generator_display_v=1.0,
            generator_display_convention="rms_into_z0",
        )
        assert s.vth_rms == pytest.approx(2.0)

    def test_available_power_missing_field(self) -> None:
        with pytest.raises(ValueError, match="available_power_dbm or available_power_w"):
            SourceSpec(mode=SourceMode.AVAILABLE_POWER)

    def test_thevenin_missing_field(self) -> None:
        with pytest.raises(ValueError, match="thevenin_vrms or thevenin_vpp"):
            SourceSpec(mode=SourceMode.THEVENIN)

    def test_generator_missing_convention(self) -> None:
        with pytest.raises(ValueError, match="generator_display_convention"):
            SourceSpec(
                mode=SourceMode.GENERATOR_INTO_Z0,
                generator_display_v=1.0,
            )

    def test_negative_z_source_real(self) -> None:
        with pytest.raises(ValueError):
            SourceSpec(
                mode=SourceMode.AVAILABLE_POWER,
                available_power_dbm=20.0,
                z_source_real_ohm=-10.0,
            )

    def test_z_source_complex(self) -> None:
        s = SourceSpec(
            mode=SourceMode.AVAILABLE_POWER,
            available_power_dbm=20.0,
            z_source_real_ohm=50.0,
            z_source_imag_ohm=10.0,
        )
        assert s.z_source == complex(50.0, 10.0)

    def test_frozen(self) -> None:
        s = SourceSpec(mode=SourceMode.AVAILABLE_POWER, available_power_dbm=20.0)
        with pytest.raises((ValidationError, AttributeError)):
            s.z_source_real_ohm = 75.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FrequencyPlan
# ---------------------------------------------------------------------------


class TestFrequencyPlan:
    def _make_targets(self, freqs: list[float]) -> list[FrequencyTarget]:
        return [
            FrequencyTarget(
                label=f"f{i}",
                frequency_hz=f,
                voltage_target_rms_v=20.0,
            )
            for i, f in enumerate(freqs)
        ]

    def test_valid_plan(self) -> None:
        plan = FrequencyPlan(
            targets=self._make_targets([9e6, 10e6, 11e6]),
            sweep_f_min_hz=5e6,
            sweep_f_max_hz=15e6,
        )
        assert len(plan.enabled_targets) == 3

    def test_duplicate_frequencies_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            FrequencyPlan(
                targets=self._make_targets([10e6, 10e6]),
                sweep_f_min_hz=5e6,
                sweep_f_max_hz=15e6,
            )

    def test_nonpositive_frequency_rejected(self) -> None:
        with pytest.raises(ValueError):
            FrequencyTarget(frequency_hz=0.0)

    def test_negative_frequency_rejected(self) -> None:
        with pytest.raises(ValueError):
            FrequencyTarget(frequency_hz=-1e6)

    def test_target_outside_sweep_band(self) -> None:
        with pytest.raises(ValueError, match="outside verification band"):
            FrequencyPlan(
                targets=self._make_targets([100e6]),
                sweep_f_min_hz=5e6,
                sweep_f_max_hz=15e6,
            )

    def test_invalid_sweep_band(self) -> None:
        with pytest.raises(ValueError):
            FrequencyPlan(
                targets=self._make_targets([10e6]),
                sweep_f_min_hz=15e6,
                sweep_f_max_hz=5e6,
            )

    def test_voltage_range_inverted(self) -> None:
        with pytest.raises(ValueError, match="voltage_min_rms_v"):
            FrequencyTarget(
                frequency_hz=10e6,
                voltage_min_rms_v=30.0,
                voltage_max_rms_v=10.0,
            )

    def test_voltage_target_below_min(self) -> None:
        with pytest.raises(ValueError, match="voltage_target_rms_v"):
            FrequencyTarget(
                frequency_hz=10e6,
                voltage_target_rms_v=5.0,
                voltage_min_rms_v=10.0,
            )

    def test_negative_voltage_target(self) -> None:
        with pytest.raises(ValueError):
            FrequencyTarget(frequency_hz=10e6, voltage_target_rms_v=-1.0)

    def test_target_frequencies_sorted(self) -> None:
        plan = FrequencyPlan(
            targets=self._make_targets([11e6, 9e6, 10e6]),
            sweep_f_min_hz=5e6,
            sweep_f_max_hz=15e6,
        )
        freqs = plan.target_frequencies_hz
        assert freqs == sorted(freqs)

    def test_exclusion_band_invalid(self) -> None:
        with pytest.raises(ValueError, match="Exclusion band"):
            ExclusionBand(f_min_hz=10e6, f_max_hz=5e6)

    def test_empty_targets_rejected(self) -> None:
        with pytest.raises(ValueError):
            FrequencyPlan(
                targets=[],
                sweep_f_min_hz=5e6,
                sweep_f_max_hz=15e6,
            )


# ---------------------------------------------------------------------------
# Topology and Pole specs
# ---------------------------------------------------------------------------


class TestTopologySpec:
    def test_valid_default(self) -> None:
        t = TopologySearchSpec()
        assert len(t.orientations) >= 1

    def test_inverted_cell_range(self) -> None:
        with pytest.raises(ValueError, match="branch1_cells_min"):
            TopologySearchSpec(branch1_cells_min=5, branch1_cells_max=2)

    def test_empty_orientations(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            TopologySearchSpec(orientations=[])

    def test_pole_interval_inverted(self) -> None:
        with pytest.raises(ValueError, match="min_hz"):
            PoleInterval(min_hz=15e6, max_hz=5e6)

    def test_pole_interval_initial_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="initial_hz"):
            PoleInterval(min_hz=5e6, max_hz=15e6, initial_hz=20e6)

    def test_pole_locked_without_initial(self) -> None:
        with pytest.raises(ValueError, match="locked"):
            PoleInterval(min_hz=5e6, max_hz=15e6, locked=True)

    def test_fixed_mode_no_poles(self) -> None:
        with pytest.raises(ValueError, match="fixed mode"):
            PoleSpec(mode=PoleMode.FIXED)

    def test_intervals_mode_no_intervals(self) -> None:
        with pytest.raises(ValueError, match="intervals mode"):
            PoleSpec(mode=PoleMode.INTERVALS)

    def test_allowed_band_inverted(self) -> None:
        with pytest.raises(ValueError, match="allowed_band_hz"):
            PoleSpec(allowed_band_hz=(15e6, 5e6))


# ---------------------------------------------------------------------------
# Component policy
# ---------------------------------------------------------------------------


class TestComponentPolicy:
    def test_valid_default(self) -> None:
        cp = ComponentPolicy()
        assert cp.continuous_limits.l_min_h < cp.continuous_limits.l_max_h

    def test_inverted_l_range(self) -> None:
        with pytest.raises(ValueError, match="l_min_h"):
            ContinuousLimits(l_min_h=100e-6, l_max_h=10e-9)

    def test_inverted_c_range(self) -> None:
        with pytest.raises(ValueError, match="c_min_f"):
            ContinuousLimits(c_min_f=20e-9, c_max_f=0.2e-12)

    def test_invalid_derating(self) -> None:
        with pytest.raises(ValueError):
            ComponentPolicy(voltage_derating_fraction=1.5)

    def test_zero_derating(self) -> None:
        with pytest.raises(ValueError):
            ComponentPolicy(voltage_derating_fraction=0.0)


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class TestConstraints:
    def test_match_constraints_valid(self) -> None:
        mc = MatchConstraints()
        assert mc.gamma_max > 0

    def test_match_constraints_inverted_resistance(self) -> None:
        with pytest.raises(ValueError, match="resistance_min_ohm"):
            MatchConstraints(resistance_min_ohm=70.0, resistance_max_ohm=35.0)

    def test_stress_valid(self) -> None:
        sc = StressConstraints()
        assert sc.source_current_rms_max_a > 0

    def test_constraint_record(self) -> None:
        cr = ConstraintRecord(
            name="S11 limit",
            severity=ConstraintSeverity.HARD,
            limit=-15.0,
            unit="dB",
        )
        assert cr.name == "S11 limit"


# ---------------------------------------------------------------------------
# EOM model spec
# ---------------------------------------------------------------------------


class TestEOMModelSpec:
    def test_ideal_capacitor_valid(self) -> None:
        eom = EOMModelSpec(
            model_type=EOMModelType.IDEAL_CAPACITOR,
            c0_f=12e-12,
        )
        assert eom.c0_f == pytest.approx(12e-12)
        assert eom.extrapolation_policy == ExtrapolationPolicy.ERROR

    def test_extrapolation_policy_custom(self) -> None:
        eom = EOMModelSpec(
            model_type=EOMModelType.IDEAL_CAPACITOR,
            c0_f=12e-12,
            extrapolation_policy=ExtrapolationPolicy.CLAMP,
        )
        assert eom.extrapolation_policy == ExtrapolationPolicy.CLAMP

    def test_ideal_capacitor_missing_c0(self) -> None:
        with pytest.raises(ValueError, match="c0_f"):
            EOMModelSpec(model_type=EOMModelType.IDEAL_CAPACITOR)

    def test_mbvd_valid(self) -> None:
        eom = EOMModelSpec(
            model_type=EOMModelType.MBVD,
            name="SYNTHETIC_TEST_ONLY",
            c0_f=12e-12,
            g0_s=2e-5,
            rs_ohm=0.5,
            ls_h=15e-9,
            motional_branches=[
                MotionalBranch(rm_ohm=8.0, lm_h=50e-6, cm_f=9e-12),
            ],
            validity_hz=(1e6, 30e6),
        )
        assert len(eom.motional_branches) == 1

    def test_mbvd_missing_c0(self) -> None:
        with pytest.raises(ValueError, match="c0_f"):
            EOMModelSpec(model_type=EOMModelType.MBVD)

    def test_tabular_missing_data_file(self) -> None:
        with pytest.raises(ValueError, match="data_file"):
            EOMModelSpec(model_type=EOMModelType.TABULAR)

    def test_validity_hz_inverted(self) -> None:
        with pytest.raises(ValueError, match="validity_hz"):
            EOMModelSpec(
                model_type=EOMModelType.IDEAL_CAPACITOR,
                c0_f=12e-12,
                validity_hz=(30e6, 1e6),
            )

    def test_motional_branch_negative_lm(self) -> None:
        with pytest.raises(ValueError):
            MotionalBranch(rm_ohm=8.0, lm_h=-50e-6, cm_f=9e-12)


# ---------------------------------------------------------------------------
# ProjectSpec (minimal aggregate test)
# ---------------------------------------------------------------------------


class TestProjectSpec:
    def _make_project(self) -> ProjectSpec:
        return ProjectSpec(
            source=SourceSpec(
                mode=SourceMode.AVAILABLE_POWER,
                available_power_dbm=20.0,
            ),
            eom=EOMModelSpec(
                model_type=EOMModelType.MBVD,
                name="TEST",
                c0_f=12e-12,
                g0_s=2e-5,
                rs_ohm=0.5,
                ls_h=15e-9,
                motional_branches=[
                    MotionalBranch(rm_ohm=8.0, lm_h=50e-6, cm_f=9e-12),
                ],
                validity_hz=(1e6, 30e6),
            ),
            frequencies=FrequencyPlan(
                targets=[
                    FrequencyTarget(label="f1", frequency_hz=9e6, voltage_target_rms_v=20.0),
                    FrequencyTarget(label="f2", frequency_hz=10e6, voltage_target_rms_v=20.0),
                    FrequencyTarget(label="f3", frequency_hz=11e6, voltage_target_rms_v=20.0),
                ],
                sweep_f_min_hz=5e6,
                sweep_f_max_hz=15e6,
            ),
        )

    def test_valid_project(self) -> None:
        p = self._make_project()
        assert p.schema_version == "0.2"
        assert len(p.frequencies.targets) == 3

    def test_frozen(self) -> None:
        p = self._make_project()
        with pytest.raises((ValidationError, AttributeError)):
            p.schema_version = "0.3"  # type: ignore[misc]

    def test_candidate_result_skeleton(self) -> None:
        r = CandidateResult(candidate_id="c001")
        assert r.candidate_id == "c001"
        assert not r.feasible
