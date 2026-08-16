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


def test_ideal_capacitor_eom():
    model = IdealCapacitorEOM(c0_f=1e-12)
    f = 1e6
    omega = 2 * np.pi * f
    z_expected = 1 / (1j * omega * 1e-12)

    # Scalar eval
    assert np.isclose(model.z(f), z_expected)
    assert np.isclose(model.y(f), 1 / z_expected)

    # Vectorized
    f_arr = np.array([f, 2 * f])
    z_arr = model.z(f_arr)
    assert z_arr.shape == (2,)
    assert np.isclose(z_arr[0], z_expected)
    assert np.isclose(z_arr[1], 1 / (1j * 2 * omega * 1e-12))

    # Base validations
    assert model.validity_range() is None
    assert model.metadata()["c0_f"] == 1e-12

    with pytest.raises(ValueError):
        model.z(-1.0)

    with pytest.raises(ValueError, match="strictly positive"):
        IdealCapacitorEOM(-1e-12)


def test_lossy_capacitor_eom():
    model = LossyCapacitorEOM(c0_f=10e-12, rs_ohm=1.0, ls_h=5e-9, g0_s=1e-6)
    f = 10e6
    omega = 2 * np.pi * f

    z_expected = 1.0 + 1j * omega * 5e-9 + 1 / (1e-6 + 1j * omega * 10e-12)
    assert np.isclose(model.z(f), z_expected)

    # Limit test: Rs=0, Ls=0, G0=0 recovers ideal
    ideal = LossyCapacitorEOM(c0_f=10e-12, rs_ohm=0.0, ls_h=0.0, g0_s=0.0)
    assert np.isclose(ideal.z(f), 1 / (1j * omega * 10e-12))


def test_mbvd_model():
    # Lossless single mode for analytic check
    # Rm=0, Lm=50uH, Cm=10pF
    # C0=20pF
    Lm = 50e-6
    Cm = 10e-12
    C0 = 20e-12
    model = MBVDModel(c0_f=C0, motional_branches=[MotionalBranch(rm_ohm=0.0, lm_h=Lm, cm_f=Cm)])

    # Resonance: f_r = 1 / (2pi sqrt(Lm * Cm))
    f_r = 1.0 / (2 * np.pi * np.sqrt(Lm * Cm))
    # Anti-resonance: f_a = f_r * sqrt(1 + Cm/C0)
    f_a = f_r * np.sqrt(1 + Cm / C0)

    # At resonance, Y_core should go to infinity (Z_m -> 0), so Z_EOM -> 0
    # Wait, exact float evaluation at f_r might hit divide by zero or be very small.
    # Let's test a very close frequency.
    z_res = model.z(f_r * 1.000000001)
    assert np.abs(z_res) < 1e-4

    # At antiresonance, Y_core should go to 0, so Z_EOM -> infinity
    # Thus Y_EOM -> 0
    y_anti = model.y(f_a * 1.000000001)
    assert np.abs(y_anti) < 1e-4
    
    # PR check for positive parameters on synthetic
    synthetic = create_synthetic_mbvd()
    f_sweep = np.linspace(1e6, 20e6, 100)
    z_sweep = synthetic.z(f_sweep)
    # Passivity: Re(Z) >= 0 (allow small numerical tolerance)
    assert np.all(np.real(z_sweep) >= -1e-12)


def test_tabular_eom():
    f_hz = np.array([1e6, 2e6, 3e6])
    z_data = np.array([10 - 100j, 15 - 50j, 20 - 30j])
    
    model = TabularEOM(f_hz, z_data, interpolation="linear")
    
    # Exact data points
    assert np.allclose(model.z(1e6), 10 - 100j)
    
    # Linear interpolation at midpoint
    assert np.allclose(model.z(1.5e6), 12.5 - 75j)
    
    # Metadata
    assert model.validity_range() == (1e6, 3e6)
    
    # Extrapolation policy default ERROR
    with pytest.raises(ModelValidityError):
        model.z(5e6)
        
    # Extrapolation CLAMP
    model_clamp = TabularEOM(
        f_hz, z_data, extrapolation_policy=ExtrapolationPolicy.CLAMP
    )
    assert np.allclose(model_clamp.z(5e6), 20 - 30j)
    
    # Extrapolation WARN (should warn once, then silence)
    model_warn = TabularEOM(
        f_hz, z_data, extrapolation_policy=ExtrapolationPolicy.WARN
    )
    with pytest.warns(UserWarning, match="MODEL_EXTRAPOLATION"):
        assert np.allclose(model_warn.z(5e6), 30 + 10j) # Extrapolated linearly by scipy

    # Second call should not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model_warn.z(5e6)


def test_tabular_strict_validation():
    # Not 1-D
    with pytest.raises(ValueError):
        TabularEOM(np.array([[1, 2], [3, 4]]), np.array([1, 2, 3, 4]))

    # Mismatched length
    with pytest.raises(ValueError):
        TabularEOM(np.array([1, 2]), np.array([1, 2, 3]))

    # Not enough points
    with pytest.raises(ValueError):
        TabularEOM(np.array([1]), np.array([1]))

    # Non-finite
    with pytest.raises(ValueError):
        TabularEOM(np.array([1, np.nan]), np.array([1, 2]))

    # Negative/zero frequency
    with pytest.raises(ValueError):
        TabularEOM(np.array([0, 1]), np.array([1, 2]))

    # Not strictly increasing
    with pytest.raises(ValueError):
        TabularEOM(np.array([2, 1]), np.array([1, 2]))

    # Duplicates
    with pytest.raises(ValueError):
        TabularEOM(np.array([1, 1, 2]), np.array([1, 2, 3]))


def test_factory():
    spec_ideal = EOMModelSpec(model_type=EOMModelType.IDEAL_CAPACITOR, c0_f=5e-12)
    model_ideal = build_eom_model(spec_ideal)
    assert isinstance(model_ideal, IdealCapacitorEOM)
    assert model_ideal.c0_f == 5e-12

    spec_lossy = EOMModelSpec(model_type=EOMModelType.LOSSY_CAPACITOR, c0_f=5e-12, rs_ohm=1.0)
    model_lossy = build_eom_model(spec_lossy)
    assert isinstance(model_lossy, LossyCapacitorEOM)
    assert model_lossy.rs_ohm == 1.0

    spec_mbvd = EOMModelSpec(
        model_type=EOMModelType.MBVD,
        c0_f=5e-12,
        motional_branches=[MotionalBranch(rm_ohm=1, lm_h=1e-3, cm_f=1e-12)],
    )
    model_mbvd = build_eom_model(spec_mbvd)
    assert isinstance(model_mbvd, MBVDModel)
    assert len(model_mbvd.motional_branches) == 1

    # Tabular is deferred
    spec_tab = EOMModelSpec(model_type=EOMModelType.TABULAR, data_file="dummy.csv")
    with pytest.raises(NotImplementedError):
        build_eom_model(spec_tab)
