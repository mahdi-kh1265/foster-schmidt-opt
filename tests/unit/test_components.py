"""Tests for primitive component models."""

import numpy as np
import pytest

from foster_eom.models import (
    IdealCapacitor,
    IdealInductor,
    IdealResistor,
    LumpedLossyCapacitor,
    LumpedLossyInductor,
    TabularImpedanceComponent,
)


def test_ideal_resistor():
    model = IdealResistor(r_ohm=50.0)
    assert np.isclose(model.z(1e6), 50.0 + 0j)

    with pytest.raises(ValueError):
        IdealResistor(-1.0)


def test_ideal_inductor():
    model = IdealInductor(l_h=1e-6)
    f = 10e6
    omega = 2 * np.pi * f
    assert np.isclose(model.z(f), 1j * omega * 1e-6)

    with pytest.raises(ValueError):
        IdealInductor(0.0)


def test_ideal_capacitor():
    model = IdealCapacitor(c_f=1e-12)
    f = 10e6
    omega = 2 * np.pi * f
    assert np.isclose(model.z(f), 1 / (1j * omega * 1e-12))

    with pytest.raises(ValueError):
        IdealCapacitor(0.0)


def test_lossy_inductor():
    # Topology: (R_dcr + jwL) || 1/(jwC_par)
    L = 1e-6
    R = 0.5
    C = 1e-12
    model = LumpedLossyInductor(l_h=L, r_dcr_ohm=R, c_par_f=C)

    f = 10e6
    omega = 2 * np.pi * f

    z_series = R + 1j * omega * L
    y_series = 1.0 / z_series
    y_par = 1j * omega * C
    z_expected = 1.0 / (y_series + y_par)

    assert np.isclose(model.z(f), z_expected)
    assert model.metadata()["r_dcr_ohm"] == 0.5

    with pytest.raises(ValueError):
        LumpedLossyInductor(-1e-6)
    with pytest.raises(ValueError):
        LumpedLossyInductor(1e-6, r_dcr_ohm=-1.0)
    with pytest.raises(ValueError):
        LumpedLossyInductor(1e-6, c_par_f=-1.0)


def test_lossy_capacitor():
    # Topology: R_esr + jwL_esl + 1/(jwC)
    C = 1e-12
    R = 0.1
    L = 1e-9
    model = LumpedLossyCapacitor(c_f=C, r_esr_ohm=R, l_esl_h=L)

    f = 10e6
    omega = 2 * np.pi * f

    z_expected = R + 1j * omega * L + 1.0 / (1j * omega * C)
    assert np.isclose(model.z(f), z_expected)

    with pytest.raises(ValueError):
        LumpedLossyCapacitor(-1e-12)
    with pytest.raises(ValueError):
        LumpedLossyCapacitor(1e-12, r_esr_ohm=-1.0)
    with pytest.raises(ValueError):
        LumpedLossyCapacitor(1e-12, l_esl_h=-1.0)


def test_tabular_component():
    f_hz = np.array([1e6, 2e6, 3e6])
    z_data = np.array([10 - 10j, 15 - 5j, 20 + 0j])

    model = TabularImpedanceComponent(f_hz, z_data, interpolation="linear")

    assert np.allclose(model.z(1.5e6), 12.5 - 7.5j)
    assert model.validity_range() == (1e6, 3e6)
    assert model.metadata()["interpolation"] == "linear"
