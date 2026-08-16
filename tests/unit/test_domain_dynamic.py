"""Prompt 05 Tests for Dynamic L/C and Pole Spacing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from foster_eom.foster.foster_form import coefficients_to_components
from foster_eom.optimize.domain import _check_fixed_fixed_separation


def test_dynamic_lc_relationships():
    """Verify C_m = 1/k_m and L_m = k_m / (2pi f_p)^2."""
    km = np.array([1e10, 2e10])
    fp = np.array([1e6, 2e6])

    comp = coefficients_to_components(None, None, km, fp)

    c_m1, c_m2 = comp.cells[0].c_f, comp.cells[1].c_f
    l_m1, l_m2 = comp.cells[0].l_h, comp.cells[1].l_h

    assert c_m1 == pytest.approx(1.0 / km[0])
    assert c_m2 == pytest.approx(1.0 / km[1])

    q_m1 = (2 * math.pi * fp[0])**2
    q_m2 = (2 * math.pi * fp[1])**2

    assert l_m1 == pytest.approx(km[0] / q_m1)
    assert l_m2 == pytest.approx(km[1] / q_m2)

def test_moving_fp_changes_l_m():
    """Verify that moving f_p changes L_m but not C_m."""
    km = np.array([1e10])
    fp1 = np.array([1e6])
    fp2 = np.array([2e6])

    c1 = coefficients_to_components(None, None, km, fp1)
    c2 = coefficients_to_components(None, None, km, fp2)

    assert c1.cells[0].c_f == c2.cells[0].c_f
    assert c1.cells[0].l_h != c2.cells[0].l_h
    assert c1.cells[0].l_h > c2.cells[0].l_h  # Higher f_p means lower L

def test_complete_pole_spacing_coverage():
    """Test all cases of adjacent pole spacing."""
    # The actual constraint evaluator uses exactly `fp[m+1] - fp[m]`.
    # We test the constraint behavior directly or the pre-flight check.

    # FIXED -> FIXED (fails before DE if invalid)
    regions = ((1e6, 1e6), (1.1e6, 1.1e6))
    f_poles = (1e6, 1.1e6)
    ok, _ = _check_fixed_fixed_separation(f_poles, regions, 200e3)
    assert not ok

    ok, _ = _check_fixed_fixed_separation(f_poles, regions, 50e3)
    assert ok

    # The following cases don't fail pre-flight; they are dynamically constrained by DE.
    # FIXED -> MOVABLE
    regions_fm = ((1e6, 1e6), (1.1e6, 2e6))
    ok, _ = _check_fixed_fixed_separation((1e6, 1.1e6), regions_fm, 200e3)
    assert ok  # Should pass pre-flight, DE handles it

    # MOVABLE -> FIXED
    regions_mf = ((1e6, 2e6), (2.1e6, 2.1e6))
    ok, _ = _check_fixed_fixed_separation((1e6, 2.1e6), regions_mf, 200e3)
    assert ok

    # MOVABLE -> MOVABLE
    regions_mm = ((1e6, 2e6), (1.1e6, 2.5e6))
    ok, _ = _check_fixed_fixed_separation((1e6, 1.1e6), regions_mm, 200e3)
    assert ok

