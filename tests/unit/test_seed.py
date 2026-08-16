"""Tests for seed synthesis pipeline (Prompt 04B)."""

from __future__ import annotations

import numpy as np
import pytest

from foster_eom.domain.component import ContinuousLimits
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.domain.topology import TopologySearchSpec
from foster_eom.foster.seed import (
    SeedCandidate,
    SeedGenerationResult,
    SignSearchOptions,
    generate_seeds,
)
from foster_eom.models.base import OnePortModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ResistiveLoad(OnePortModel):
    """Purely resistive EOM load for testing."""

    def __init__(self, r_ohm: float = 50.0) -> None:
        super().__init__()
        self._r = r_ohm

    def _z_impl(self, f_hz):
        return np.full_like(f_hz, complex(self._r, 0.0), dtype=complex)

    def _y_impl(self, f_hz):
        return np.full_like(f_hz, complex(1.0 / self._r, 0.0), dtype=complex)

    def metadata(self):
        return {"type": "resistive", "r_ohm": self._r}


class _ReactiveLoad(OnePortModel):
    """EOM load with real + imaginary part for testing."""

    def __init__(self, z: complex) -> None:
        super().__init__()
        self._z = z

    def _z_impl(self, f_hz):
        return self._z

    def _y_impl(self, f_hz):
        return 1.0 / self._z

    def metadata(self):
        return {"type": "reactive", "z": str(self._z)}


def _default_source() -> SourceSpec:
    return SourceSpec(
        mode=SourceMode.THEVENIN,
        thevenin_vrms=1.0,
        z_source_real_ohm=50.0,
    )


def _default_component_limits() -> ContinuousLimits:
    return ContinuousLimits(
        c_min_f=1e-12,
        c_max_f=1e-9,
        l_min_h=10e-9,
        l_max_h=100e-6,
    )


def _default_topo_spec(**kw) -> TopologySearchSpec:
    defaults = dict(
        branch1_cells_min=0,
        branch1_cells_max=2,
        branch2_cells_min=0,
        branch2_cells_max=2,
        max_total_reactive_components=14,
    )
    defaults.update(kw)
    return TopologySearchSpec(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateSeeds:
    """Seed generation pipeline tests."""

    def test_invalid_r_match_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            generate_seeds(
                r_match_ohm=0.0,
                source_spec=_default_source(),
                eom_model=_ResistiveLoad(),
                f_targets_hz=np.array([10e6]),
                topo_spec=_default_topo_spec(),
                component_limits=_default_component_limits(),
            )

    def test_invalid_frequencies_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            generate_seeds(
                r_match_ohm=50.0,
                source_spec=_default_source(),
                eom_model=_ResistiveLoad(),
                f_targets_hz=np.array([-1.0]),
                topo_spec=_default_topo_spec(),
                component_limits=_default_component_limits(),
            )

    def test_non_increasing_frequencies_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            generate_seeds(
                r_match_ohm=50.0,
                source_spec=_default_source(),
                eom_model=_ResistiveLoad(),
                f_targets_hz=np.array([10e6, 9e6]),
                topo_spec=_default_topo_spec(),
                component_limits=_default_component_limits(),
            )

    def test_result_type_is_seed_generation_result(self) -> None:
        """Pipeline always returns SeedGenerationResult."""
        result = generate_seeds(
            r_match_ohm=50.0,
            source_spec=_default_source(),
            eom_model=_ResistiveLoad(),
            f_targets_hz=np.array([10e6]),
            topo_spec=_default_topo_spec(),
            component_limits=_default_component_limits(),
        )
        assert isinstance(result, SeedGenerationResult)

    def test_diagnostics_always_present(self) -> None:
        """Diagnostics are always populated, even with no seeds."""
        result = generate_seeds(
            r_match_ohm=50.0,
            source_spec=_default_source(),
            eom_model=_ResistiveLoad(),
            f_targets_hz=np.array([10e6]),
            topo_spec=_default_topo_spec(),
            component_limits=_default_component_limits(),
        )
        d = result.diagnostics
        assert d.n_orientation_attempts >= 1
        assert isinstance(d.rejection_counts, dict)
        assert isinstance(d.sign_search_by_orientation, dict)

    def test_accepted_seeds_are_seed_candidates(self) -> None:
        """All entries in result.seeds are SeedCandidate instances."""
        result = generate_seeds(
            r_match_ohm=50.0,
            source_spec=_default_source(),
            eom_model=_ReactiveLoad(25.0 + 10j),
            f_targets_hz=np.array([10e6]),
            topo_spec=_default_topo_spec(),
            component_limits=_default_component_limits(),
            match_tolerance=1.0,  # Very loose for testing
        )
        for seed in result.seeds:
            assert isinstance(seed, SeedCandidate)

    def test_sign_search_budget_propagates(self) -> None:
        """SignSearchOptions budget is reported in diagnostics."""
        sso = SignSearchOptions(beam_width=500, max_patterns=100)
        result = generate_seeds(
            r_match_ohm=50.0,
            source_spec=_default_source(),
            eom_model=_ResistiveLoad(),
            f_targets_hz=np.array([10e6]),
            topo_spec=_default_topo_spec(),
            component_limits=_default_component_limits(),
            sign_search_options=sso,
        )
        assert result.diagnostics.sign_beam_width == 500
        assert result.diagnostics.sign_max_patterns == 100

    def test_max_seeds_caps_output(self) -> None:
        """max_seeds limits the number of returned seeds."""
        result = generate_seeds(
            r_match_ohm=50.0,
            source_spec=_default_source(),
            eom_model=_ReactiveLoad(25.0 + 10j),
            f_targets_hz=np.array([10e6]),
            topo_spec=_default_topo_spec(),
            component_limits=_default_component_limits(),
            match_tolerance=1.0,
            max_seeds=1,
        )
        assert len(result.seeds) <= 1


class TestSeedValidation:
    """Seed validation contract tests."""

    def test_accepted_seed_has_validation(self) -> None:
        """Accepted seeds have non-None validation."""
        result = generate_seeds(
            r_match_ohm=50.0,
            source_spec=_default_source(),
            eom_model=_ReactiveLoad(25.0 + 10j),
            f_targets_hz=np.array([10e6]),
            topo_spec=_default_topo_spec(),
            component_limits=_default_component_limits(),
            match_tolerance=1.0,
        )
        for seed in result.seeds:
            v = seed.validation
            assert v is not None
            assert v.all_rmatch_satisfied
            assert v.all_power_balance_ok
            assert len(v.match_error_at_targets) == 1

    def test_seed_sort_order_deterministic(self) -> None:
        """Seeds are sorted by match error (deterministic)."""
        result = generate_seeds(
            r_match_ohm=50.0,
            source_spec=_default_source(),
            eom_model=_ReactiveLoad(25.0 + 10j),
            f_targets_hz=np.array([10e6]),
            topo_spec=_default_topo_spec(),
            component_limits=_default_component_limits(),
            match_tolerance=1.0,
        )
        if len(result.seeds) >= 2:
            errors = [s.validation.max_match_error for s in result.seeds]
            assert errors == sorted(errors)


class TestSignSearchOptionsValidation:
    """SignSearchOptions validation."""

    def test_invalid_beam_width_raises(self) -> None:
        with pytest.raises(ValueError, match="beam_width"):
            SignSearchOptions(beam_width=0)

    def test_invalid_max_patterns_raises(self) -> None:
        with pytest.raises(ValueError, match="max_patterns"):
            SignSearchOptions(max_patterns=0)
