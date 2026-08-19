from foster_eom.sensitivities.coverage import (
    CoordinateConstraintKind,
    DerivativeRoute,
    MNAStampKind,
    get_derivative_coverage,
)


def test_logk0_coverage():
    cov = get_derivative_coverage("logk0")
    assert cov.route == DerivativeRoute.MNA_DERIVED
    assert cov.mna_stamps == (MNAStampKind.CAPACITOR,)
    assert not cov.coordinate_constraints


def test_logkinf_coverage():
    cov = get_derivative_coverage("logkinf")
    assert cov.route == DerivativeRoute.MNA_DERIVED
    assert cov.mna_stamps == (MNAStampKind.INDUCTOR,)
    assert not cov.coordinate_constraints


def test_logkm_coverage():
    cov = get_derivative_coverage("logkm")
    assert cov.route == DerivativeRoute.MNA_DERIVED
    assert MNAStampKind.CAPACITOR in cov.mna_stamps
    assert MNAStampKind.INDUCTOR in cov.mna_stamps
    assert not cov.coordinate_constraints


def test_fp_coverage():
    cov = get_derivative_coverage("fp")
    assert cov.route == DerivativeRoute.COMBINED
    assert cov.mna_stamps == (MNAStampKind.INDUCTOR,)
    assert CoordinateConstraintKind.POLE_SEPARATION in cov.coordinate_constraints


def test_unknown_variable_is_unsupported():
    cov = get_derivative_coverage("unknown_future_var")
    assert cov.route == DerivativeRoute.UNSUPPORTED
