"""Unit tests for P10 sampler."""

from __future__ import annotations

import numpy as np
import pytest

from foster_eom.robustness.sampler import (
    RobustnessSpec,
    ci_method_for_spec,
    draw_samples,
    inverse_transform_draw,
)
from foster_eom.robustness.uncertainty import (
    PerturbMethod,
    SlotUncertainty,
    UncertaintySource,
    UncertaintyTerm,
)


def _make_su(
    eid: str, tol: float | None, perturb: PerturbMethod = PerturbMethod.IDEAL_LC
) -> SlotUncertainty:
    terms: tuple[UncertaintyTerm, ...] = ()
    if tol is not None and tol > 0:
        terms = (UncertaintyTerm(source=UncertaintySource.MANUFACTURING_TOLERANCE, tol_frac=tol),)
    return SlotUncertainty(
        element_id=eid,
        terms=terms,
        has_tol_frac=tol is not None,
        catalog_tol_frac=tol,
        perturb_method=perturb if terms else PerturbMethod.NONE,
    )


class TestDrawSamples:
    def test_random_shape(self) -> None:
        sus = [_make_su("b1_L1", 0.05), _make_su("b1_C1", 0.10)]
        spec = RobustnessSpec(n_samples=100, seed=1, method="random")
        dm = draw_samples(sus, spec)
        assert dm.u.shape == (100, 2)
        assert dm.slot_order == ["b1_L1", "b1_C1"]

    def test_random_reproducible(self) -> None:
        sus = [_make_su("b1_L1", 0.05)]
        spec = RobustnessSpec(n_samples=50, seed=42, method="random")
        dm1 = draw_samples(sus, spec)
        dm2 = draw_samples(sus, spec)
        np.testing.assert_array_equal(dm1.u, dm2.u)

    def test_random_uniform_marginals(self) -> None:
        """KS test: random draws from uniform should pass at p>0.05."""
        from scipy.stats import kstest

        sus = [_make_su("b1_L1", 0.05)]
        spec = RobustnessSpec(n_samples=500, seed=7, method="random")
        dm = draw_samples(sus, spec)
        _stat, p = kstest(dm.u[:, 0], "uniform")
        assert p > 0.05, f"KS p={p:.4f} too small; draws not uniform"

    def test_lhs_shape(self) -> None:
        sus = [_make_su("b1_L1", 0.05), _make_su("b1_C1", 0.02)]
        spec = RobustnessSpec(n_samples=64, seed=0, method="lhs")
        dm = draw_samples(sus, spec)
        assert dm.u.shape == (64, 2)
        # LHS: each column should have one sample per stratum
        col = np.sort(dm.u[:, 0])
        # Each stratum [i/n, (i+1)/n] should contain exactly one sample
        n = 64
        for i in range(n):
            assert np.any((col >= i / n) & (col < (i + 1) / n))

    def test_non_stochastic_slot_excluded_from_matrix(self) -> None:
        sus = [_make_su("b1_L1", 0.05), _make_su("b1_C1", None)]  # C1 deterministic
        spec = RobustnessSpec(n_samples=10, seed=0)
        dm = draw_samples(sus, spec)
        assert dm.u.shape == (10, 1)
        assert dm.slot_order == ["b1_L1"]

    def test_all_non_stochastic_empty_matrix(self) -> None:
        sus = [_make_su("b1_L1", None), _make_su("b1_C1", None)]
        spec = RobustnessSpec(n_samples=10, seed=0)
        dm = draw_samples(sus, spec)
        assert dm.u.shape == (10, 0)
        assert dm.slot_order == []


class TestInverseTransform:
    def test_uniform_within_bounds(self) -> None:
        sus = [_make_su("b1_L1", 0.10)]
        nom = {"b1_L1": 10e-9}
        spec = RobustnessSpec(n_samples=1000, seed=0)
        dm = draw_samples(sus, spec)
        for i in range(min(100, spec.n_samples)):
            draw = inverse_transform_draw(dm.u[i], sus, nom)
            v = draw["b1_L1"]
            assert 9e-9 <= v <= 11e-9, f"draw={v} outside ±10% of 10nH"

    def test_non_stochastic_slot_returns_nominal(self) -> None:
        sus = [_make_su("b1_C1", None)]
        nom = {"b1_C1": 10e-12}
        spec = RobustnessSpec(n_samples=10, seed=0)
        dm = draw_samples(sus, spec)
        for i in range(10):
            draw = inverse_transform_draw(dm.u[i], sus, nom)
            assert draw["b1_C1"] == pytest.approx(10e-12)

    def test_normal_3sigma_mostly_within_bounds(self) -> None:
        """99.7% of normal_3sigma draws should be within +/- 3 sigma = +/- tol."""
        from foster_eom.robustness.uncertainty import UncertaintySource, UncertaintyTerm

        t = UncertaintyTerm(
            source=UncertaintySource.MODEL_UNCERTAINTY,
            tol_frac=0.09,
            distribution="normal_3sigma",
        )
        su = SlotUncertainty(
            element_id="b1_L1",
            terms=(t,),
            has_tol_frac=False,
            catalog_tol_frac=None,
            perturb_method=PerturbMethod.IDEAL_LC,
        )
        nom = {"b1_L1": 10e-9}
        spec = RobustnessSpec(n_samples=1000, seed=0)
        dm = draw_samples([su], spec)
        outside = 0
        for i in range(1000):
            draw = inverse_transform_draw(dm.u[i], [su], nom)
            v = draw["b1_L1"]
            delta = abs(v / 10e-9 - 1.0)
            # Soft clip at u in [1e-4, 1-1e-4] means draws are within ~3.72 sigma max
            # Use generous boundary (3.72 sigma ~= 1.12 * 3 sigma) to account for clip point
            if delta > 0.09 * 1.25:  # ~3.7 sigma for sigma=tol/3=0.03
                outside += 1
        # Effectively no draws should be far outside (clipping is tight)
        assert outside == 0, f"{outside} draws outside soft-clip boundary"


class TestCIMethod:
    def test_random_gives_wilson(self) -> None:
        spec = RobustnessSpec(method="random")
        assert ci_method_for_spec(spec) == "wilson"

    def test_lhs_gives_none(self) -> None:
        spec = RobustnessSpec(method="lhs")
        assert ci_method_for_spec(spec) is None

    def test_sobol_gives_none(self) -> None:
        spec = RobustnessSpec(method="sobol")
        assert ci_method_for_spec(spec) is None
