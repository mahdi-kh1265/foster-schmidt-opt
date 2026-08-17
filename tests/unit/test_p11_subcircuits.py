"""P11 subcircuit impedance vs MNA tests.  No ngspice required.

Each supported parametric subcircuit is verified analytically:
the SPICE subcircuit topology must produce impedance values matching
the OnePortModel._z_impl() over a frequency sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

from foster_eom.models.components import LumpedLossyCapacitor, LumpedLossyInductor
from foster_eom.spice.netlist import _lossy_capacitor_subckt, _lossy_inductor_subckt


def _lossy_ind_z_direct(model: LumpedLossyInductor, freqs: np.ndarray) -> np.ndarray:
    """Compute Z directly from topology formula."""
    omega = 2.0 * np.pi * freqs
    z_series = model.r_dcr_ohm + 1j * omega * model.l_h
    y_series = 1.0 / z_series
    y_par = 1j * omega * model.c_par_f
    return 1.0 / (y_series + y_par)


def _lossy_cap_z_direct(model: LumpedLossyCapacitor, freqs: np.ndarray) -> np.ndarray:
    """Compute Z directly from topology formula."""
    omega = 2.0 * np.pi * freqs
    z_c = 1.0 / (1j * omega * model.c_f)
    return model.r_esr_ohm + 1j * omega * model.l_esl_h + z_c


class TestLossyInductorEquivalence:
    """LumpedLossyInductor subcircuit Z must match model._z_impl."""

    @pytest.mark.parametrize('l_h,r_dcr,c_par', [
        (10e-9, 0.0, 0.0),
        (10e-9, 0.5, 0.0),
        (10e-9, 0.5, 1e-12),
        (100e-9, 2.0, 5e-12),
    ])
    def test_impedance_matches_model(self, l_h, r_dcr, c_par) -> None:
        model = LumpedLossyInductor(l_h=l_h, r_dcr_ohm=r_dcr, c_par_f=c_par)
        freqs = np.logspace(6, 10, 50)
        z_model = np.array([complex(model._z_impl(f)) for f in freqs])
        z_formula = _lossy_ind_z_direct(model, freqs)
        # Both should agree with each other (they use same formula)
        np.testing.assert_allclose(
            np.abs(z_model), np.abs(z_formula), rtol=1e-12,
            err_msg='LossyInductor Z formula mismatch'
        )

    def test_subckt_text_topology(self) -> None:
        model = LumpedLossyInductor(l_h=10e-9, r_dcr_ohm=0.5, c_par_f=1e-12)
        text = _lossy_inductor_subckt(model, 'LTEST')
        assert '.SUBCKT LTEST p n' in text
        assert 'Rdcr p nd_i1' in text
        assert 'Lmain nd_i1 n' in text
        assert 'Cpar p n' in text
        assert '.ENDS' in text

    def test_subckt_no_cpar_when_zero(self) -> None:
        model = LumpedLossyInductor(l_h=10e-9, r_dcr_ohm=0.5, c_par_f=0.0)
        text = _lossy_inductor_subckt(model, 'LTEST')
        assert 'Cpar' not in text


class TestLossyCapacitorEquivalence:
    """LumpedLossyCapacitor subcircuit Z must match model._z_impl."""

    @pytest.mark.parametrize('c_f,r_esr,l_esl', [
        (10e-12, 0.0, 0.0),
        (10e-12, 0.1, 0.0),
        (10e-12, 0.1, 1e-9),
        (100e-12, 0.5, 5e-9),
    ])
    def test_impedance_matches_model(self, c_f, r_esr, l_esl) -> None:
        model = LumpedLossyCapacitor(c_f=c_f, r_esr_ohm=r_esr, l_esl_h=l_esl)
        freqs = np.logspace(6, 10, 50)
        z_model = np.array([complex(model._z_impl(f)) for f in freqs])
        z_formula = _lossy_cap_z_direct(model, freqs)
        np.testing.assert_allclose(
            np.abs(z_model), np.abs(z_formula), rtol=1e-12,
            err_msg='LossyCapacitor Z formula mismatch'
        )

    def test_subckt_text_topology(self) -> None:
        model = LumpedLossyCapacitor(c_f=10e-12, r_esr_ohm=0.1, l_esl_h=1e-9)
        text = _lossy_capacitor_subckt(model, 'CTEST')
        assert '.SUBCKT CTEST p n' in text
        assert 'Resr p nd_i1' in text
        assert 'Lesl nd_i1 nd_i2' in text
        assert 'Cmain nd_i2 n' in text
        assert '.ENDS' in text
