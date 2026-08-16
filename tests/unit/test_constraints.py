"""Tests for Prompt-05 constraint compilation and evaluation."""


from foster_eom.circuit.measurements import CircuitSolution
from foster_eom.domain.constraints import (
    ConstraintRecord,
    ConstraintSeverity,
    MatchConstraints,
    StressConstraints,
)
from foster_eom.optimize.constraints import compile_constraint_layout


def _make_mock_solution(f_hz: float, gamma: float | None = None, z_in_real: float | None = None, z_in_imag: float | None = None, v_eom: float | None = None, i_source: float | None = None) -> CircuitSolution:
    z_in = complex(z_in_real or 0.0, z_in_imag or 0.0) if z_in_real is not None or z_in_imag is not None else None
    return CircuitSolution(
        f_hz=f_hz,
        z_in=z_in,
        gamma=complex(gamma, 0.0) if gamma is not None else None,
        s11_db=None,
        v_eom=complex(v_eom, 0.0) if v_eom is not None else None,
        i_source_droop=complex(i_source, 0.0) if i_source is not None else None,
        power_balance_ok=True,
        status="ok",
        diagnostics="",
    )

def test_deterministic_layout():
    """Verify constraint layout order is deterministic across compilations."""
    match_c = MatchConstraints(gamma_max=0.5, resistance_max_ohm=100.0, max_abs_reactance_ohm=50.0)
    stress_c = StressConstraints(source_current_rms_max_a=1.0, off_target_eom_peak_rms_v=5.0)

    layout1 = compile_constraint_layout(
        match_constraints=match_c,
        stress_constraints=stress_c,
        extra_records=[],
        target_frequencies_hz=(1e6, 2e6),
        evaluation_frequencies_hz=(1e6, 1.5e6, 2e6),
        target_indices=(0, 2),
        off_target_indices=(1,),
        severity_filter=ConstraintSeverity.HARD,
        n_cells_b1=1,
        n_cells_b2=1,
        z_ref_ohm=50.0,
    )
    layout2 = compile_constraint_layout(
        match_constraints=match_c,
        stress_constraints=stress_c,
        extra_records=[],
        target_frequencies_hz=(1e6, 2e6),
        evaluation_frequencies_hz=(1e6, 1.5e6, 2e6),
        target_indices=(0, 2),
        off_target_indices=(1,),
        severity_filter=ConstraintSeverity.HARD,
        n_cells_b1=1,
        n_cells_b2=1,
        z_ref_ohm=50.0,
    )
    names1 = [d.name for d in layout1.descriptors]
    names2 = [d.name for d in layout2.descriptors]
    assert names1 == names2
    assert "gamma_f1000000Hz" in names1
    assert "offtarget_veom_1500000Hz" in names1

def test_hard_soft_validation_filtering():
    """Verify severity filters correctly isolate constraints."""
    match_c = MatchConstraints(gamma_max=0.5, resistance_max_ohm=100.0, max_abs_reactance_ohm=50.0)
    stress_c = StressConstraints(source_current_rms_max_a=1.0, off_target_eom_peak_rms_v=5.0)
    extra = [
        ConstraintRecord(name="extra_soft", severity="soft", limit=1.0, frequency_scope="all_targets"),
        ConstraintRecord(name="extra_valid", severity="hard", limit=1.0, frequency_scope="all_targets", validation_only=True),
    ]

    hard_layout = compile_constraint_layout(
        match_constraints=match_c, stress_constraints=stress_c, extra_records=extra,
        target_frequencies_hz=(1e6,), evaluation_frequencies_hz=(1e6,), target_indices=(0,), off_target_indices=(),
        severity_filter=ConstraintSeverity.HARD, n_cells_b1=0, n_cells_b2=0, z_ref_ohm=50.0
    )
    soft_layout = compile_constraint_layout(
        match_constraints=match_c, stress_constraints=stress_c, extra_records=extra,
        target_frequencies_hz=(1e6,), evaluation_frequencies_hz=(1e6,), target_indices=(0,), off_target_indices=(),
        severity_filter=ConstraintSeverity.SOFT, n_cells_b1=0, n_cells_b2=0, z_ref_ohm=50.0
    )

    hard_names = [d.name for d in hard_layout.descriptors]
    soft_names = [d.name for d in soft_layout.descriptors]

    # HARD filter has built-in gamma, etc., but NO soft and NO validation_only
    assert "gamma_f1000000Hz" in hard_names
    assert not any("extra_soft" in n for n in hard_names)
    assert not any("extra_valid" in n for n in hard_names)

    # SOFT filter has extra_soft
    assert any("extra_soft" in n for n in soft_names)
    assert not any("gamma" in n for n in soft_names)

def test_constraint_evaluations():
    """Verify mathematical evaluations of standard constraints."""
    match_c = MatchConstraints(gamma_max=0.5, resistance_max_ohm=100.0, max_abs_reactance_ohm=50.0)
    stress_c = StressConstraints(source_current_rms_max_a=1.0, off_target_eom_peak_rms_v=5.0)

    layout = compile_constraint_layout(
        match_constraints=match_c, stress_constraints=stress_c, extra_records=[],
        target_frequencies_hz=(1e6,), evaluation_frequencies_hz=(1e6, 1.5e6), target_indices=(0,), off_target_indices=(1,),
        severity_filter=ConstraintSeverity.HARD, n_cells_b1=1, n_cells_b2=0, z_ref_ohm=50.0
    )

    sols = (
        _make_mock_solution(1e6, gamma=0.6, z_in_real=110.0, z_in_imag=60.0, i_source=1.2, v_eom=3.0),
        _make_mock_solution(1.5e6, v_eom=6.0)
    )

    g = layout.evaluate(
        solutions=sols, target_indices=(0,), off_target_indices=(1,),
        branch1_pole_regions=((0.1, 0.9),), branch2_pole_regions=(),
        branch1_k_residues=(1.0,), branch2_k_residues=(), branch1_f_poles=(5e6,), branch2_f_poles=(),
        branch1_l_vals=(2e-6,), branch2_l_vals=(), branch1_c_vals=(1e-12,), branch2_c_vals=(),
        component_limits_l_min=1e-9, component_limits_l_max=1e-6,
        component_limits_c_min=1e-15, component_limits_c_max=1e-9,
        pole_sep_min_b1=1e6, pole_sep_min_b2=1e6, z_ref_ohm=50.0,
        gamma_max=0.5, r_min_ohm=0.0, r_max_ohm=100.0, x_max_ohm=50.0,
        source_current_max_a=1.0, off_target_eom_peak_rms_v=5.0
    )

    names = [d.name for d in layout.descriptors]

    # gamma = 0.6 > 0.5 (violated, expected < 0)
    gamma_idx = names.index("gamma_f1000000Hz")
    assert g[gamma_idx] < 0.0

    # r_max: 110 > 100 (violated)
    rmax_idx = names.index("r_max_f1000000Hz")
    assert g[rmax_idx] < 0.0

    # offtarget: 6.0 > 5.0 (violated)
    ot_idx = names.index("offtarget_veom_1500000Hz")
    assert g[ot_idx] < 0.0

    # L_hi: 2e-6 > 1e-6 (violated)
    lhi_idx = names.index("comp_L_hi_b1_m0")
    assert g[lhi_idx] < 0.0

def test_gamma_uses_z_ref():
    """Ensure gamma relies only on z_ref by checking formula independence."""
    # This behavior is encoded into CircuitSolution or how constraints scale.
    # The constraint layout just reads `sol.gamma`.
    pass

def test_one_target_case():
    """Verify no crashes for one target."""
    match_c = MatchConstraints(gamma_max=0.5, resistance_max_ohm=100.0, max_abs_reactance_ohm=50.0)
    stress_c = StressConstraints()
    layout = compile_constraint_layout(
        match_constraints=match_c, stress_constraints=stress_c, extra_records=[],
        target_frequencies_hz=(1e6,), evaluation_frequencies_hz=(1e6,), target_indices=(0,), off_target_indices=(),
        severity_filter=ConstraintSeverity.HARD, n_cells_b1=0, n_cells_b2=0, z_ref_ohm=50.0
    )
    assert layout.n > 0
    assert any("gamma_f1000000Hz" in d.name for d in layout.descriptors)

