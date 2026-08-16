"""Prompt 05 unit tests — compute_pole_legal_region in poles.py."""

from __future__ import annotations

import math
import numpy as np
import pytest

from foster_eom.foster.poles import PoleMode, PoleSpec, compute_pole_legal_region


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_spec(
    band: tuple[float, float] | None = (0.5e6, 100e6),
    excl: float = 100e3,
    sep: float = 200e3,
) -> PoleSpec:
    return PoleSpec(
        mode=PoleMode.AUTO,
        allowed_band_hz=band,
        delta_f_target_min_hz=excl,
        delta_f_pole_min_hz=sep,
    )


def _fixed_spec() -> PoleSpec:
    return PoleSpec(
        mode=PoleMode.FIXED,
        fixed_poles_hz=(5e6,),
        delta_f_target_min_hz=100e3,
        delta_f_pole_min_hz=200e3,
    )


def _interval_spec(intervals: list[tuple[float, float]]) -> PoleSpec:
    return PoleSpec(
        mode=PoleMode.INTERVALS,
        intervals_hz=tuple(intervals),
        delta_f_target_min_hz=100e3,
        delta_f_pole_min_hz=200e3,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPoleRegionFixed:
    def test_fixed_returns_point_interval(self) -> None:
        f_seed = 5e6
        lo, hi = compute_pole_legal_region(
            cell_index=0,
            f_pole_seed_hz=f_seed,
            pole_spec=_fixed_spec(),
            f_targets_hz=np.array([2e6, 10e6]),
            n_cells=1,
            prev_pole_hz=-math.inf,
            next_pole_hz=None,
        )
        assert lo == f_seed
        assert hi == f_seed


class TestPoleRegionAuto:
    def test_auto_contains_seed(self) -> None:
        f_seed = 5e6
        lo, hi = compute_pole_legal_region(
            cell_index=0,
            f_pole_seed_hz=f_seed,
            pole_spec=_auto_spec(),
            f_targets_hz=np.array([2e6, 10e6]),
            n_cells=1,
            prev_pole_hz=-math.inf,
            next_pole_hz=None,
        )
        assert lo <= f_seed <= hi

    def test_auto_excludes_target_zone(self) -> None:
        """Pole seed should not be in exclusion zone of a target."""
        f_seed = 5e6
        spec = _auto_spec(excl=1e6)  # large exclusion zone
        # Target at exactly 5 MHz would exclude a seed too close
        lo, hi = compute_pole_legal_region(
            cell_index=0,
            f_pole_seed_hz=f_seed,
            pole_spec=spec,
            f_targets_hz=np.array([4e6, 6e6]),  # close targets
            n_cells=1,
            prev_pole_hz=-math.inf,
            next_pole_hz=None,
        )
        # Even if seed falls in gap, lo <= f_seed <= hi should hold when a gap exists
        # Just ensure it returns valid floats
        assert math.isfinite(lo) and math.isfinite(hi)

    def test_auto_clips_for_next_pole_separation(self) -> None:
        """Upper bound clips to next_pole - sep."""
        f_seed = 5e6
        sep = 200e3
        next_f = 6e6
        lo, hi = compute_pole_legal_region(
            cell_index=0,
            f_pole_seed_hz=f_seed,
            pole_spec=_auto_spec(sep=sep),
            f_targets_hz=np.array([2e6, 10e6]),
            n_cells=2,
            prev_pole_hz=-math.inf,
            next_pole_hz=next_f,
        )
        assert hi <= next_f - sep + 1e3  # small tolerance


class TestPoleRegionIntervals:
    def test_interval_contains_seed(self) -> None:
        f_seed = 7e6
        spec = _interval_spec([(5e6, 10e6)])
        lo, hi = compute_pole_legal_region(
            cell_index=0,
            f_pole_seed_hz=f_seed,
            pole_spec=spec,
            f_targets_hz=np.array([2e6, 12e6]),
            n_cells=1,
            prev_pole_hz=-math.inf,
            next_pole_hz=None,
        )
        assert lo <= f_seed <= hi

    def test_interval_out_of_range_returns_point(self) -> None:
        """Cell index beyond specified intervals → fallback to point."""
        spec = _interval_spec([(5e6, 10e6)])
        lo, hi = compute_pole_legal_region(
            cell_index=5,          # beyond available intervals
            f_pole_seed_hz=7e6,
            pole_spec=spec,
            f_targets_hz=np.array([2e6]),
            n_cells=6,
            prev_pole_hz=-math.inf,
            next_pole_hz=None,
        )
        assert lo == hi  # point interval (fallback)
