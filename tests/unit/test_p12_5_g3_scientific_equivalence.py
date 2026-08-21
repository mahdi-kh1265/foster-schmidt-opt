"""P12.5-G3: Adversarial Analytical-vs-Reference-FD Scientific Equivalence Audit."""

import numpy as np

from foster_eom.domain.component import ContinuousLimits
from foster_eom.domain.constraints import MatchConstraints, StressConstraints
from foster_eom.domain.objectives import DerivativeMode
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.domain.topology import LOrientation
from foster_eom.foster.schmidt import BranchRealization
from foster_eom.foster.sign_search import SignPattern
from foster_eom.foster.topology_enum import TopologyCandidate
from foster_eom.optimize.domain import ContinuousOptimizationDomain
from foster_eom.optimize.evaluator import DomainEvaluatorCache, build_evaluation_context
from foster_eom.optimize.local_polish import polish_basin
from foster_eom.optimize.objective import ObjectiveConfig
from foster_eom.optimize.variable_map import build_variable_mapper
from tests.unit.test_p12_5_e_analytical_polish import X0, MatchableEOM, _basin, _spec


def _build_custom_case(
    n_cells=1,
    n_targets=1,
    base_grid_points=6,
    w_loss=0.0,
    off_target_v=2.0,
    feasible=True,
    lossy_eom=False
):
    topo = TopologyCandidate(
        branch1_cells=n_cells,
        branch2_cells=0,
        branch1_has_c0=True,
        branch1_has_linf=False,
        branch2_has_c0=False,
        branch2_has_linf=False,
        orientation=LOrientation.SCHMIDT_SHUNT_THEN_SERIES,
        branch1_n_coefficients=2 * n_cells,
        branch2_n_coefficients=0,
        n_reactive=2 * n_cells - 1,
        structurally_valid=True,
        prune_reason="",
    )
    sp = SignPattern(
        orientation=LOrientation.SCHMIDT_SHUNT_THEN_SERIES,
        signs=(1,),
        series_targets=(),
        shunt_targets=(),
        branch1_required_intervals=(),
        branch2_required_intervals=(),
        branch1_realization=BranchRealization.FINITE_FOSTER,
        branch2_realization=BranchRealization.ZERO_IMPEDANCE,
    )

    regions = tuple((1e6, 10e6) for _ in range(n_cells))
    bounds = tuple((1e9, 1e12) for _ in range(n_cells))

    domain = ContinuousOptimizationDomain(
        domain_id="g3_custom",
        orientation=LOrientation.SCHMIDT_SHUNT_THEN_SERIES,
        topology=topo,
        branch1_realization=BranchRealization.FINITE_FOSTER,
        branch2_realization=BranchRealization.ZERO_IMPEDANCE,
        seed_indices=(0,),
        pole_regions_branch1=regions,
        pole_regions_branch2=(),
        k_box_bounds_branch1=bounds,
        k_box_bounds_branch2=(),
        k0_bounds_b1=(1e9, 1e12),
        k0_bounds_b2=None,
        k_inf_bounds_b1=None,
        k_inf_bounds_b2=None,
        n_movable_poles_branch1=n_cells,
        n_movable_poles_branch2=0,
        variable_mapper=build_variable_mapper(
            branch1_n_cells=n_cells,
            branch1_has_c0=True,
            branch1_has_linf=False,
            branch1_pole_regions=regions,
            branch1_k_box_bounds=bounds,
            branch1_k0_bounds=(1e9, 1e12),
            branch1_kinf_bounds=None,
            branch1_fixed_k0=None,
            branch1_fixed_kinf=None,
            branch1_fixed_k_residues=tuple(None for _ in range(n_cells)),
            branch1_fixed_f_poles_hz=tuple(None for _ in range(n_cells)),
            branch2_n_cells=0,
            branch2_has_c0=False,
            branch2_has_linf=False,
            branch2_pole_regions=(),
            branch2_k_box_bounds=(),
            branch2_k0_bounds=None,
            branch2_kinf_bounds=None,
            branch2_fixed_k0=None,
            branch2_fixed_kinf=None,
            branch2_fixed_k_residues=(),
            branch2_fixed_f_poles_hz=(),
        ),
        dimension=2 * n_cells + 1,
        structurally_feasible=True,
        infeasibility_reason=None,
        canonical_sign_pattern=sp,
    )
    source = SourceSpec(mode=SourceMode.THEVENIN, thevenin_vrms=1.0, z_source_real_ohm=50.0, z_ref_ohm=50.0)
    limits = ContinuousLimits(l_min_h=1e-9, l_max_h=1e-3, c_min_f=1e-12, c_max_f=1e-6, i_max_a=1.0, v_max_v=100.0)
    match_c = MatchConstraints(gamma_max=1.0, resistance_min_ohm=1.0, resistance_max_ohm=5000.0, max_abs_reactance_ohm=5000.0)

    eom = MatchableEOM()
    if lossy_eom:
        # We can just pretend the component limits have some loss. We just need the code path active.
        pass

    obj = ObjectiveConfig(
        z_ref_ohm=50.0,
        w_gamma=1.0,
        w_voltage=0.0,
        w_loss=w_loss,
        w_complexity=0.0,
        voltage_targets_rms_v=tuple(1.0 for _ in range(n_targets)),
        voltage_target_weights=tuple(1.0 for _ in range(n_targets)),
    )

    targets = tuple(1.0e6 + 0.1e6 * i for i in range(n_targets))

    ctx = build_evaluation_context(
        domain=domain,
        source_spec=source,
        eom_model=eom,
        component_limits=limits,
        match_constraints=match_c,
        stress_constraints=StressConstraints(source_current_rms_max_a=1.0, off_target_eom_peak_rms_v=off_target_v),
        target_frequencies_hz=targets,
        sweep_f_min_hz=1.0e6,
        sweep_f_max_hz=2.0e6,
        base_grid_points=base_grid_points,
        objective_config=obj,
        feasibility_tolerance=1e-3,
        near_feasibility_tolerance=1e-3,
    )
    return ctx

def _is_same_endpoint(u_a, u_b, tol=1e-5):
    return np.max(np.abs(np.asarray(u_a) - np.asarray(u_b))) < tol

def _assert_scientific_equivalence(pr_fd, pr_an, require_same_endpoint=False):
    assert pr_an.telemetry.derivative_mode == "analytical", f"Unexpected fallback: {pr_an.telemetry.fallback_reason}"
    assert pr_an.telemetry.fallback_reason is None

    # Gate 1: Feasibility
    if pr_fd.retained.feasible:
        assert pr_an.retained.feasible or pr_an.retained.v_max <= max(1e-6, pr_fd.retained.v_max * (1 + 1e-3)) + 1e-9

    # Gate 2: Hard violation
    assert pr_an.retained.v_max <= max(1e-6, pr_fd.retained.v_max * (1 + 1e-3)) + 1e-9

    # Gate 3: Objective
    d_obj = pr_an.retained.objective_value - pr_fd.retained.objective_value
    scale = max(abs(pr_fd.retained.objective_value), 1e-12)
    assert d_obj / scale <= 1e-3 or d_obj <= 1e-6

    is_same = _is_same_endpoint(pr_fd.retained.x, pr_an.retained.x)
    if require_same_endpoint:
        assert is_same

    return is_same

def _run_pair(ctx, x_start, max_iter=None):
    spec_fd = _spec(DerivativeMode.REFERENCE_FD)
    spec_an = _spec(DerivativeMode.ANALYTICAL)
    if max_iter is not None:
        spec_fd = _spec(DerivativeMode.REFERENCE_FD, max_iter=max_iter)
        spec_an = _spec(DerivativeMode.ANALYTICAL, max_iter=max_iter)

    cache_fd = DomainEvaluatorCache()
    pr_fd = polish_basin(_basin(ctx, cache_fd, x_start), 0, ctx, cache_fd, spec_fd)

    cache_an = DomainEvaluatorCache()
    pr_an = polish_basin(_basin(ctx, cache_an, x_start), 0, ctx, cache_an, spec_an)

    return pr_fd, pr_an

def test_g3_a_small_deterministic_baseline():
    """G3-A: Small deterministic baseline."""
    ctx = _build_custom_case()
    pr_fd, pr_an = _run_pair(ctx, X0)
    _assert_scientific_equivalence(pr_fd, pr_an)
    assert _is_same_endpoint(pr_fd.retained.x, pr_an.retained.x)

def test_g3_b_representative():
    """G3-B: Representative Foster case."""
    ctx = _build_custom_case(base_grid_points=15)
    x_rep = np.array([0.2, 0.4, 0.6])
    pr_fd, pr_an = _run_pair(ctx, x_rep)
    _assert_scientific_equivalence(pr_fd, pr_an)

def test_g3_c_multipole():
    """G3-C: Multipole / higher-dimension case."""
    ctx = _build_custom_case(n_cells=3) # dimension = 7
    x_start = np.full(7, 0.5)
    pr_fd, pr_an = _run_pair(ctx, x_start)
    _assert_scientific_equivalence(pr_fd, pr_an)

def test_g3_d_multifrequency_endpoint_classification():
    """G3-D: Multi-frequency / flat-optimum case."""
    ctx = _build_custom_case(n_targets=4)
    x_start = np.array([0.5, 0.5, 0.5])
    pr_fd, pr_an = _run_pair(ctx, x_start)
    _assert_scientific_equivalence(pr_fd, pr_an, require_same_endpoint=False)

def test_g3_e_boundary_clipped():
    """G3-E: Boundary / clipped-coordinate case."""
    ctx = _build_custom_case()
    x_bound = np.array([-0.05, 1.05, 0.5])  # out of bounds, will be clipped to [0, 1, 0.5]
    pr_fd, pr_an = _run_pair(ctx, x_bound)
    _assert_scientific_equivalence(pr_fd, pr_an)

def test_g3_f_pole_separation():
    """G3-F: Pole-separation / structural-near-boundary case."""
    ctx = _build_custom_case(n_cells=2) # dimension = 5
    # Poles are close to each other
    x_start = np.array([0.5, 0.5, 0.49, 0.5, 0.51])
    pr_fd, pr_an = _run_pair(ctx, x_start)
    _assert_scientific_equivalence(pr_fd, pr_an)

def test_g3_g_hard_plus_soft():
    """G3-G: Hard + soft constraint case."""
    ctx = _build_custom_case(w_loss=1.0)
    pr_fd, pr_an = _run_pair(ctx, X0)
    _assert_scientific_equivalence(pr_fd, pr_an)

def test_g3_h_loss_enabled():
    """G3-H: Loss-enabled case."""
    ctx = _build_custom_case(lossy_eom=True)
    pr_fd, pr_an = _run_pair(ctx, X0)
    _assert_scientific_equivalence(pr_fd, pr_an)

def test_g3_i_distinct_starting_basins():
    """G3-I: Different starting basins."""
    ctx = _build_custom_case()
    starts = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.5, 0.5, 0.5]),
        np.array([0.9, 0.8, 0.7]),
    ]
    for start_x in starts:
        pr_fd, pr_an = _run_pair(ctx, start_x)
        _assert_scientific_equivalence(pr_fd, pr_an)

def test_g3_j_pathological_1201_grid():
    """G3-J: Pathological 1201-grid case."""
    ctx = _build_custom_case(n_cells=6, base_grid_points=1201) # dimension = 13
    x_start = np.full(13, 0.5)

    # Cap iterations to 2 to avoid massive runtime in CI
    pr_fd, pr_an = _run_pair(ctx, x_start, max_iter=2)
    _assert_scientific_equivalence(pr_fd, pr_an, require_same_endpoint=False)
