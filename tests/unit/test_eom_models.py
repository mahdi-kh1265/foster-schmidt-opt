"""Tests for EOM models."""

import warnings

import numpy as np
import pytest

from foster_eom.domain.eom import (
    EOMModelSpec,
    EOMModelType,
    ExtrapolationPolicy,
    MotionalBranch,
)
from foster_eom.errors import ModelValidityError
from foster_eom.models import (
    IdealCapacitorEOM,
    LossyCapacitorEOM,
    MBVDModel,
    TabularEOM,
    build_eom_model,
    create_synthetic_mbvd,
)
from foster_eom.models.base import EOMModel, OnePortModel


# ---------------------------------------------------------------------------
# §1  Ideal capacitor EOM
# ---------------------------------------------------------------------------


class TestIdealCapacitorEOM:
    def test_known_value_scalar(self):
        model = IdealCapacitorEOM(c0_f=1e-12)
        f = 1e6
        omega = 2 * np.pi * f
        z_expected = 1 / (1j * omega * 1e-12)
        assert np.isclose(model.z(f), z_expected)
        assert np.isclose(model.y(f), 1 / z_expected)

    def test_vectorized(self):
        model = IdealCapacitorEOM(c0_f=1e-12)
        f_arr = np.array([1e6, 2e6, 5e6])
        z_arr = model.z(f_arr)
        assert z_arr.shape == (3,)
        for i, f in enumerate(f_arr):
            assert np.isclose(z_arr[i], 1 / (1j * 2 * np.pi * f * 1e-12))

    def test_no_nans_wide_sweep(self):
        model = IdealCapacitorEOM(c0_f=10e-12)
        f_sweep = np.logspace(3, 11, 500)  # 1 kHz to 100 GHz
        z = model.z(f_sweep)
        assert np.all(np.isfinite(z))
        y = model.y(f_sweep)
        assert np.all(np.isfinite(y))

    def test_validity_is_none(self):
        assert IdealCapacitorEOM(c0_f=1e-12).validity_range() is None

    def test_metadata_provenance(self):
        m = IdealCapacitorEOM(c0_f=5e-12).metadata()
        assert m["c0_f"] == 5e-12
        assert m["mathematical_only"] is True

    def test_negative_c0_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            IdealCapacitorEOM(-1e-12)

    def test_zero_c0_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            IdealCapacitorEOM(0.0)

    def test_negative_frequency_rejected(self):
        with pytest.raises(ValueError):
            IdealCapacitorEOM(c0_f=1e-12).z(-1.0)


# ---------------------------------------------------------------------------
# §2  Lossy capacitor EOM
# ---------------------------------------------------------------------------


class TestLossyCapacitorEOM:
    def test_known_value(self):
        model = LossyCapacitorEOM(c0_f=10e-12, rs_ohm=1.0, ls_h=5e-9, g0_s=1e-6)
        f = 10e6
        omega = 2 * np.pi * f
        z_expected = 1.0 + 1j * omega * 5e-9 + 1 / (1e-6 + 1j * omega * 10e-12)
        assert np.isclose(model.z(f), z_expected)

    def test_limit_recovers_ideal(self):
        """Rs=Ls=G0=0 must recover ideal capacitor."""
        model = LossyCapacitorEOM(c0_f=10e-12)
        f = 10e6
        omega = 2 * np.pi * f
        assert np.isclose(model.z(f), 1 / (1j * omega * 10e-12))

    def test_no_nans_wide_sweep(self):
        model = LossyCapacitorEOM(c0_f=10e-12, rs_ohm=0.5, ls_h=10e-9, g0_s=1e-5)
        z = model.z(np.logspace(3, 10, 500))
        assert np.all(np.isfinite(z))


# ---------------------------------------------------------------------------
# §3  mBVD model — lossless single-mode analytic tests
# ---------------------------------------------------------------------------


class TestMBVDModel:
    """Analytic resonance/antiresonance tests use a simplified lossless
    single-mode mBVD (Rm=0, Rs=Ls=G0=0) so the formulas are exact."""

    @staticmethod
    def _make_lossless(Lm: float, Cm: float, C0: float) -> MBVDModel:
        return MBVDModel(
            c0_f=C0,
            motional_branches=[MotionalBranch(rm_ohm=0.0, lm_h=Lm, cm_f=Cm)],
        )

    def test_series_resonance(self):
        Lm, Cm, C0 = 50e-6, 10e-12, 20e-12
        model = self._make_lossless(Lm, Cm, C0)
        f_r = 1.0 / (2 * np.pi * np.sqrt(Lm * Cm))
        # Slightly off exact resonance to avoid singularity
        z_near = model.z(f_r * (1 + 1e-9))
        assert np.abs(z_near) < 1e-3, f"|Z| at near-resonance should be ~0, got {np.abs(z_near)}"

    def test_antiresonance(self):
        Lm, Cm, C0 = 50e-6, 10e-12, 20e-12
        model = self._make_lossless(Lm, Cm, C0)
        f_r = 1.0 / (2 * np.pi * np.sqrt(Lm * Cm))
        f_a = f_r * np.sqrt(1 + Cm / C0)
        y_near = model.y(f_a * (1 + 1e-9))
        assert np.abs(y_near) < 1e-3, (
            f"|Y| at near-antiresonance should be ~0, got {np.abs(y_near)}"
        )

    def test_exact_resonance_no_nan(self):
        """Evaluating exactly at f_r must not produce NaN."""
        Lm, Cm, C0 = 50e-6, 10e-12, 20e-12
        model = self._make_lossless(Lm, Cm, C0)
        f_r = 1.0 / (2 * np.pi * np.sqrt(Lm * Cm))
        z_exact = model.z(f_r)
        # May be 0 or very small, but must not be NaN
        assert not np.isnan(z_exact), "Z at exact resonance must not be NaN"

    def test_exact_antiresonance_no_nan(self):
        """Evaluating exactly at f_a must not produce NaN."""
        Lm, Cm, C0 = 50e-6, 10e-12, 20e-12
        model = self._make_lossless(Lm, Cm, C0)
        f_r = 1.0 / (2 * np.pi * np.sqrt(Lm * Cm))
        f_a = f_r * np.sqrt(1 + Cm / C0)
        z_exact = model.z(f_a)
        # May be inf, but must not be NaN
        assert not np.isnan(z_exact), "Z at exact antiresonance must not be NaN"

    def test_no_branches_recovers_ideal(self):
        """mBVD with no motional branches and Rs=Ls=G0=0 is an ideal cap."""
        model = MBVDModel(c0_f=10e-12)
        f = 10e6
        omega = 2 * np.pi * f
        assert np.isclose(model.z(f), 1 / (1j * omega * 10e-12))

    def test_passivity_positive_real(self):
        """Passivity: Re(Z) >= 0 across sweep for positive parameters."""
        synthetic = create_synthetic_mbvd()
        f_sweep = np.linspace(1e6, 20e6, 500)
        z_sweep = synthetic.z(f_sweep)
        # Numerical tolerance: Re(Z) must not be significantly negative
        assert np.all(np.real(z_sweep) >= -1e-12)

    def test_no_nans_wide_sweep(self):
        model = create_synthetic_mbvd()
        z = model.z(np.logspace(5, 8, 500))
        assert np.all(np.isfinite(z))


# ---------------------------------------------------------------------------
# §4  Tabular EOM — all extrapolation policies
# ---------------------------------------------------------------------------


class TestTabularEOM:
    @staticmethod
    def _tabular_data():
        f = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
        z = np.array([10 - 100j, 15 - 50j, 20 - 30j, 25 - 15j, 30 - 5j])
        return f, z

    def test_exact_data_points(self):
        f, z = self._tabular_data()
        model = TabularEOM(f, z)
        for fi, zi in zip(f, z):
            assert np.isclose(model.z(fi), zi)

    def test_linear_interpolation_midpoint(self):
        f, z = self._tabular_data()
        model = TabularEOM(f, z)
        z_mid = model.z(1.5e6)
        assert np.isclose(z_mid, 12.5 - 75j)

    def test_error_policy_raises(self):
        f, z = self._tabular_data()
        model = TabularEOM(f, z, extrapolation_policy=ExtrapolationPolicy.ERROR)
        with pytest.raises(ModelValidityError):
            model.z(6e6)
        with pytest.raises(ModelValidityError):
            model.z(0.5e6)

    def test_warn_policy_emits_then_silences(self):
        f, z = self._tabular_data()
        model = TabularEOM(f, z, extrapolation_policy=ExtrapolationPolicy.WARN)

        # First call: warning emitted
        with pytest.warns(UserWarning, match="MODEL_EXTRAPOLATION"):
            result = model.z(6e6)
        assert np.isfinite(result)

        # Second call: no warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result2 = model.z(6e6)
        assert np.isfinite(result2)

    def test_allow_policy_silent(self):
        f, z = self._tabular_data()
        model = TabularEOM(f, z, extrapolation_policy=ExtrapolationPolicy.ALLOW)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = model.z(6e6)
        assert np.isfinite(result)

    def test_clamp_policy_scalar(self):
        f, z = self._tabular_data()
        model = TabularEOM(f, z, extrapolation_policy=ExtrapolationPolicy.CLAMP)
        # Above range → clamps to f_max
        assert np.isclose(model.z(10e6), z[-1])
        # Below range → clamps to f_min
        assert np.isclose(model.z(0.1e6), z[0])

    def test_clamp_policy_mixed_vector(self):
        """Mixed vector with in-range and out-of-range frequencies."""
        f, z = self._tabular_data()
        model = TabularEOM(f, z, extrapolation_policy=ExtrapolationPolicy.CLAMP)

        query = np.array([0.5e6, 2e6, 3e6, 10e6])
        result = model.z(query)
        assert result.shape == (4,)
        # out-of-range low → clamp to f_min value
        assert np.isclose(result[0], z[0])
        # in-range points → normal interpolation
        assert np.isclose(result[1], z[1])
        assert np.isclose(result[2], z[2])
        # out-of-range high → clamp to f_max value
        assert np.isclose(result[3], z[-1])


# ---------------------------------------------------------------------------
# §5  Tabular strict validation
# ---------------------------------------------------------------------------


class TestTabularValidation:
    def test_not_1d(self):
        with pytest.raises(ValueError):
            TabularEOM(np.array([[1, 2], [3, 4]]), np.array([1, 2, 3, 4]))

    def test_mismatched_length(self):
        with pytest.raises(ValueError):
            TabularEOM(np.array([1, 2]), np.array([1, 2, 3]))

    def test_fewer_than_2_points(self):
        with pytest.raises(ValueError):
            TabularEOM(np.array([1]), np.array([1]))

    def test_non_finite_freq(self):
        with pytest.raises(ValueError):
            TabularEOM(np.array([1, np.nan]), np.array([1, 2]))

    def test_non_finite_z(self):
        with pytest.raises(ValueError):
            TabularEOM(np.array([1.0, 2.0]), np.array([1.0, np.inf]))

    def test_zero_frequency(self):
        with pytest.raises(ValueError):
            TabularEOM(np.array([0, 1]), np.array([1, 2]))

    def test_not_strictly_increasing(self):
        with pytest.raises(ValueError):
            TabularEOM(np.array([2, 1]), np.array([1, 2]))

    def test_duplicates(self):
        with pytest.raises(ValueError):
            TabularEOM(np.array([1, 1, 2]), np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# §6  OnePortModel reciprocal behavior
# ---------------------------------------------------------------------------


class TestReciprocal:
    """Verify y(f) == 1/z(f) wherever finite, and that subclasses
    overriding only one of _z_impl/_y_impl behave consistently."""

    def test_y_equals_reciprocal_z_ideal_cap(self):
        model = IdealCapacitorEOM(c0_f=5e-12)
        f = np.array([1e6, 5e6, 10e6])
        z = model.z(f)
        y = model.y(f)
        assert np.allclose(y, 1.0 / z)

    def test_y_equals_reciprocal_z_lossy(self):
        model = LossyCapacitorEOM(c0_f=10e-12, rs_ohm=1.0, ls_h=5e-9, g0_s=1e-6)
        f = np.array([1e6, 5e6, 10e6])
        z = model.z(f)
        y = model.y(f)
        assert np.allclose(y, 1.0 / z)

    def test_y_equals_reciprocal_z_mbvd(self):
        model = create_synthetic_mbvd()
        f = np.linspace(1e6, 20e6, 50)
        z = model.z(f)
        y = model.y(f)
        # Avoid infinities from exact resonance
        finite = np.isfinite(z) & np.isfinite(y)
        assert np.allclose(y[finite], 1.0 / z[finite])

    def test_recursion_guard(self):
        """A subclass that overrides neither _z_impl nor _y_impl must raise."""

        class BareModel(OnePortModel):
            def metadata(self):
                return {}

        with pytest.raises(NotImplementedError, match="must override"):
            BareModel().z(1e6)

        with pytest.raises(NotImplementedError, match="must override"):
            BareModel().y(1e6)


# ---------------------------------------------------------------------------
# §7  _warned_extrapolation flag audit
# ---------------------------------------------------------------------------


class TestWarnedExtrapolationInvariants:
    """The _warned_extrapolation flag must be diagnostic-only and never
    affect numerical results."""

    def test_flag_does_not_affect_numerical_output(self):
        f, z = np.array([1e6, 2e6]), np.array([10 - 50j, 20 - 30j])
        m1 = TabularEOM(f, z, extrapolation_policy=ExtrapolationPolicy.WARN)
        m2 = TabularEOM(f, z, extrapolation_policy=ExtrapolationPolicy.WARN)

        # Evaluate m1 once (sets _warned_extrapolation = True)
        with pytest.warns(UserWarning):
            r1 = m1.z(3e6)

        # m2 is still False
        assert not m2._warned_extrapolation

        # Both must return the same numerical value
        with pytest.warns(UserWarning):
            r2 = m2.z(3e6)
        assert np.isclose(r1, r2)

    def test_reset_warnings(self):
        f, z = np.array([1e6, 2e6]), np.array([10 - 50j, 20 - 30j])
        model = TabularEOM(f, z, extrapolation_policy=ExtrapolationPolicy.WARN)

        with pytest.warns(UserWarning):
            model.z(3e6)
        assert model._warned_extrapolation

        model.reset_warnings()
        assert not model._warned_extrapolation

        # Should warn again after reset
        with pytest.warns(UserWarning):
            model.z(3e6)


# ---------------------------------------------------------------------------
# §8  Factory completeness
# ---------------------------------------------------------------------------


class TestFactory:
    """Verify every supported EOMModelSpec type maps correctly and all
    parameters survive schema → runtime construction."""

    def test_ideal_capacitor(self):
        spec = EOMModelSpec(
            model_type=EOMModelType.IDEAL_CAPACITOR,
            c0_f=5e-12,
            name="test_ideal",
            extrapolation_policy=ExtrapolationPolicy.WARN,
        )
        model = build_eom_model(spec)
        assert isinstance(model, IdealCapacitorEOM)
        assert model.c0_f == 5e-12
        assert model.extrapolation_policy == ExtrapolationPolicy.WARN

    def test_lossy_capacitor_all_params(self):
        spec = EOMModelSpec(
            model_type=EOMModelType.LOSSY_CAPACITOR,
            c0_f=12e-12,
            rs_ohm=0.5,
            ls_h=10e-9,
            g0_s=1e-5,
            validity_hz=(1e6, 50e6),
            extrapolation_policy=ExtrapolationPolicy.CLAMP,
            name="test_lossy",
        )
        model = build_eom_model(spec)
        assert isinstance(model, LossyCapacitorEOM)
        assert model.c0_f == 12e-12
        assert model.rs_ohm == 0.5
        assert model.ls_h == 10e-9
        assert model.g0_s == 1e-5
        assert model.validity_range() == (1e6, 50e6)
        assert model.extrapolation_policy == ExtrapolationPolicy.CLAMP

    def test_mbvd_all_params(self):
        branches = [
            MotionalBranch(rm_ohm=8.0, lm_h=50e-6, cm_f=9e-12),
            MotionalBranch(rm_ohm=12.0, lm_h=30e-6, cm_f=5e-12),
        ]
        spec = EOMModelSpec(
            model_type=EOMModelType.MBVD,
            c0_f=12e-12,
            g0_s=2e-5,
            rs_ohm=0.5,
            ls_h=15e-9,
            motional_branches=branches,
            validity_hz=(1e6, 30e6),
            extrapolation_policy=ExtrapolationPolicy.ALLOW,
            name="test_mbvd",
        )
        model = build_eom_model(spec)
        assert isinstance(model, MBVDModel)
        assert model.c0_f == 12e-12
        assert model.g0_s == 2e-5
        assert model.rs_ohm == 0.5
        assert model.ls_h == 15e-9
        assert len(model.motional_branches) == 2
        assert model.motional_branches[0].rm_ohm == 8.0
        assert model.motional_branches[1].cm_f == 5e-12
        assert model.validity_range() == (1e6, 30e6)
        assert model.extrapolation_policy == ExtrapolationPolicy.ALLOW

    def test_tabular_deferred(self):
        spec = EOMModelSpec(model_type=EOMModelType.TABULAR, data_file="dummy.csv")
        with pytest.raises(NotImplementedError):
            build_eom_model(spec)

    def test_rational_deferred(self):
        spec = EOMModelSpec(model_type=EOMModelType.RATIONAL, data_file="dummy.csv")
        with pytest.raises(NotImplementedError):
            build_eom_model(spec)


# ---------------------------------------------------------------------------
# §9  Synthetic fixture
# ---------------------------------------------------------------------------


class TestSyntheticFixture:
    def test_label(self):
        model = create_synthetic_mbvd()
        assert model.metadata()["label"] == "SYNTHETIC_TEST_ONLY"

    def test_parameters(self):
        model = create_synthetic_mbvd()
        assert model.c0_f == 12e-12
        assert model.rs_ohm == 0.5
        assert model.ls_h == 15e-9
        assert model.g0_s == 2e-5
        assert len(model.motional_branches) == 1
        b = model.motional_branches[0]
        assert b.rm_ohm == 8.0
        assert b.lm_h == 50e-6
        assert b.cm_f == 9e-12

    def test_is_eom_model(self):
        assert isinstance(create_synthetic_mbvd(), EOMModel)

    def test_beta_per_v_default_none(self):
        assert create_synthetic_mbvd().beta_per_v(1e6) is None
