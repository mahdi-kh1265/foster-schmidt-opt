import gc

import numpy as np

from foster_eom.domain.component import ContinuousLimits
from foster_eom.domain.constraints import MatchConstraints, StressConstraints
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.domain.topology import LOrientation
from foster_eom.foster.schmidt import BranchRealization
from foster_eom.foster.topology_enum import TopologyCandidate
from foster_eom.optimize.domain import ContinuousOptimizationDomain
from foster_eom.optimize.objective import ObjectiveConfig
from foster_eom.sensitivities.transaction import DerivativeTransaction
from tests.unit.test_sens_e2e import DummyEOM, build_evaluation_context, build_variable_mapper


def test_transaction_memory_lifecycle():
    topo = TopologyCandidate(
        branch1_cells=1,
        branch2_cells=0,
        branch1_has_c0=True,
        branch1_has_linf=False,
        branch2_has_c0=False,
        branch2_has_linf=False,
        orientation=LOrientation.SCHMIDT_SHUNT_THEN_SERIES,
        branch1_n_coefficients=2,
        branch2_n_coefficients=0,
        n_reactive=1,
        structurally_valid=True,
        prune_reason="",
    )
    from foster_eom.foster.sign_search import SignPattern

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
    domain = ContinuousOptimizationDomain(
        domain_id="test_dom",
        orientation=LOrientation.SCHMIDT_SHUNT_THEN_SERIES,
        topology=topo,
        branch1_realization=BranchRealization.FINITE_FOSTER,
        branch2_realization=BranchRealization.ZERO_IMPEDANCE,
        seed_indices=(0,),
        pole_regions_branch1=((1e6, 10e6),),
        pole_regions_branch2=(),
        k_box_bounds_branch1=((1e9, 1e12),),
        k_box_bounds_branch2=(),
        k0_bounds_b1=(1e9, 1e12),
        k0_bounds_b2=None,
        k_inf_bounds_b1=None,
        k_inf_bounds_b2=None,
        n_movable_poles_branch1=1,
        n_movable_poles_branch2=0,
        variable_mapper=build_variable_mapper(
            branch1_n_cells=1,
            branch1_has_c0=True,
            branch1_has_linf=False,
            branch1_pole_regions=((1e6, 10e6),),
            branch1_k_box_bounds=((1e9, 1e12),),
            branch1_k0_bounds=(1e9, 1e12),
            branch1_kinf_bounds=None,
            branch1_fixed_k0=None,
            branch1_fixed_kinf=None,
            branch1_fixed_k_residues=(None,),
            branch1_fixed_f_poles_hz=(None,),
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
        dimension=3,
        structurally_feasible=True,
        infeasibility_reason=None,
        canonical_sign_pattern=sp,
    )

    ctx = build_evaluation_context(
        domain=domain,
        source_spec=SourceSpec(
            mode=SourceMode.THEVENIN, thevenin_vrms=1.0, z_source_real_ohm=50.0, z_ref_ohm=50.0
        ),
        eom_model=DummyEOM(),
        component_limits=ContinuousLimits(
            l_min_h=1e-9, l_max_h=1e-6, c_min_f=1e-12, c_max_f=1e-9, i_max_a=1.0, v_max_v=100.0
        ),
        match_constraints=MatchConstraints(gamma_max=0.5, resistance_max_ohm=50.0),
        stress_constraints=StressConstraints(
            source_current_rms_max_a=1.0, off_target_eom_peak_rms_v=2.0
        ),
        target_frequencies_hz=(1.0e6,),
        sweep_f_min_hz=1.0e6,
        sweep_f_max_hz=2.0e6,
        base_grid_points=5,
        objective_config=ObjectiveConfig(
            z_ref_ohm=50.0,
            w_gamma=1.0,
            w_voltage=0.0,
            w_loss=0.0,
            w_complexity=0.0,
            voltage_targets_rms_v=(),
            voltage_target_weights=(),
        ),
        feasibility_tolerance=1e-3,
        near_feasibility_tolerance=1e-3,
    )

    txn = DerivativeTransaction(ctx)
    x0 = np.array([0.5, 0.5, 0.5])

    # Run multiple evaluations and ensure memory isn't leaking drastically
    for _ in range(50):
        _, _ = txn.evaluate_jacobians(x0 + np.random.uniform(-0.01, 0.01, 3))

    # Assert telemetry
    assert txn.metrics["jacobian_evals"] == 50
    assert txn.metrics["factorizations"] > 0
    assert txn.metrics["direct_substitutions"] > 0
    assert txn.metrics["adjoint_substitutions"] > 0

    # Force gc to verify we have no uncollectible objects
    gc.collect()
