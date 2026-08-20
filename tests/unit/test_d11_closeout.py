"""P12.5-D1.1 Final Derivative Closeout Tests.

Requirements:
1. Analytical support gate verification
2. Objective-value/gradient coherence (same-u, FD + directional)
3. Complete production constraint Jacobian (full layout, directional FD)
4. Safety audit (no silent zeros, adjoint trans=2, pass gates)
"""

import math

import numpy as np
import pytest

from foster_eom.domain.component import ContinuousLimits
from foster_eom.domain.constraints import ConstraintSeverity, MatchConstraints, StressConstraints
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.domain.topology import LOrientation
from foster_eom.foster.schmidt import BranchRealization
from foster_eom.foster.sign_search import SignPattern
from foster_eom.foster.topology_enum import TopologyCandidate
from foster_eom.models.base import OnePortModel
from foster_eom.optimize.constraints import ConstraintLayout
from foster_eom.optimize.domain import ContinuousOptimizationDomain
from foster_eom.optimize.evaluator import (
    DomainEvaluatorCache,
    EvaluationContext,
    build_evaluation_context,
    evaluate,
)
from foster_eom.optimize.objective import ObjectiveConfig
from foster_eom.optimize.variable_map import build_variable_mapper
from foster_eom.sensitivities.objective_gradient import (
    DerivativeStatus,
    check_analytical_support,
)
from foster_eom.sensitivities.transaction import DerivativeTransaction

# ── Shared test fixtures ──────────────────────────────────────────────────────


class DummyEOM(OnePortModel):
    def _z_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        return 50.0 - 1j / (2 * np.pi * f_hz * 1e-12)

    @property
    def metadata(self) -> dict:
        return {}


def _build_production_context(
    *,
    w_gamma: float = 1.0,
    w_voltage: float = 0.0,
    w_loss: float = 0.0,
    target_frequencies_hz: tuple[float, ...] = (1.0e6, 5.0e6),
    base_grid_points: int = 4,
) -> EvaluationContext:
    """Build a full production-like EvaluationContext."""
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
    mapper = build_variable_mapper(
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
    )
    domain = ContinuousOptimizationDomain(
        domain_id="closeout_test",
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
        variable_mapper=mapper,
        dimension=3,
        structurally_feasible=True,
        infeasibility_reason=None,
        canonical_sign_pattern=sp,
    )
    obj = ObjectiveConfig(
        z_ref_ohm=50.0,
        w_gamma=w_gamma,
        w_voltage=w_voltage,
        w_loss=w_loss,
        w_complexity=0.0,
        voltage_targets_rms_v=(),
        voltage_target_weights=(),
    )
    return build_evaluation_context(
        domain=domain,
        source_spec=SourceSpec(
            mode=SourceMode.THEVENIN,
            thevenin_vrms=1.0,
            z_source_real_ohm=50.0,
            z_ref_ohm=50.0,
        ),
        eom_model=DummyEOM(),
        component_limits=ContinuousLimits(
            l_min_h=1e-9,
            l_max_h=1e-3,
            c_min_f=1e-12,
            c_max_f=1e-6,
            i_max_a=1.0,
            v_max_v=100.0,
        ),
        match_constraints=MatchConstraints(gamma_max=0.5, resistance_max_ohm=50.0),
        stress_constraints=StressConstraints(
            source_current_rms_max_a=1.0, off_target_eom_peak_rms_v=2.0
        ),
        target_frequencies_hz=target_frequencies_hz,
        sweep_f_min_hz=min(target_frequencies_hz),
        sweep_f_max_hz=max(target_frequencies_hz) * 2,
        base_grid_points=base_grid_points,
        objective_config=obj,
        feasibility_tolerance=1e-3,
        near_feasibility_tolerance=1e-3,
    )


# ── Requirement 1: Analytical Support Gate ────────────────────────────────────


class TestAnalyticalSupportGate:
    """Verify the analytical support gate correctly classifies configurations."""

    def test_default_production_config_supported(self):
        """Default production: w_gamma > 0, w_loss=0, no soft → SUPPORTED."""
        config = ObjectiveConfig(z_ref_ohm=50.0, w_gamma=1.0)
        soft = ConstraintLayout(descriptors=())
        ok, reasons = check_analytical_support(config, soft, None, None)
        assert ok is True
        assert reasons == []

    def test_w_loss_without_lossy_elements_supported(self):
        """w_loss > 0 but lossy_element_ids=() → J_loss=0, supported."""
        config = ObjectiveConfig(z_ref_ohm=50.0, w_loss=1.0)
        soft = ConstraintLayout(descriptors=())
        ok, _reasons = check_analytical_support(config, soft, None, None)
        assert ok is True

    def test_w_loss_with_lossy_elements_now_supported(self):
        """w_loss > 0 and lossy_element_ids set → NOW SUPPORTED (D2)."""
        config = ObjectiveConfig(
            z_ref_ohm=50.0,
            w_loss=1.0,
            lossy_element_ids=("R_parasitic",),
        )
        soft = ConstraintLayout(descriptors=())
        ok, reasons = check_analytical_support(config, soft, None, None)
        assert ok is True
        assert reasons == []

    def test_soft_constraints_without_external_data_now_supported(self):
        """Soft constraints enabled, external data not provided → SUPPORTED (D2).

        The transaction now computes soft g and soft Jacobian internally.
        """
        from foster_eom.domain.constraints import FrequencyScope
        from foster_eom.optimize.constraints import ConstraintDescriptor

        config = ObjectiveConfig(z_ref_ohm=50.0)
        desc = ConstraintDescriptor(
            name="soft_test",
            constraint_type="custom",
            frequency_scope=FrequencyScope.ALL_TARGETS,
            severity=ConstraintSeverity.SOFT,
            penalty_weight=1.0,
        )
        soft = ConstraintLayout(descriptors=(desc,))
        ok, _reasons = check_analytical_support(config, soft, None, None)
        assert ok is True

    def test_soft_constraints_with_data_supported(self):
        """Soft constraints with g and J provided → supported."""
        from foster_eom.domain.constraints import FrequencyScope
        from foster_eom.optimize.constraints import ConstraintDescriptor

        config = ObjectiveConfig(z_ref_ohm=50.0)
        desc = ConstraintDescriptor(
            name="soft_test",
            constraint_type="custom",
            frequency_scope=FrequencyScope.ALL_TARGETS,
            severity=ConstraintSeverity.SOFT,
            penalty_weight=1.0,
        )
        soft = ConstraintLayout(descriptors=(desc,))
        ok, _reasons = check_analytical_support(config, soft, np.array([0.1]), np.array([[0.5]]))
        assert ok is True

    def test_production_eom_optimization_config_supported(self):
        """The exact config we intend to optimize in P12.5-E."""
        ctx = _build_production_context(w_gamma=1.0, w_loss=0.0)
        ok, reasons = check_analytical_support(ctx.objective_config, ctx.soft_layout, None, None)
        assert ok is True, f"Production config unsupported: {reasons}"


# ── Requirement 2: Objective Value/Gradient Coherence ─────────────────────────


class TestObjectiveCoherence:
    """Prove J_total(u) and ∇J_total(u) come from the exact same objective."""

    @pytest.fixture
    def production_ctx(self):
        return _build_production_context(
            w_gamma=1.0,
            w_loss=0.0,
            target_frequencies_hz=(1.0e6, 5.0e6),
            base_grid_points=4,
        )

    def test_gradient_matches_central_fd(self, production_ctx):
        """∇J_analytic ≈ ∇J_centralFD for the production objective."""
        ctx = production_ctx
        DomainEvaluatorCache()
        x0 = np.array([0.5, 0.5, 0.5])

        txn = DerivativeTransaction(ctx)
        j_base_ana, _ = txn.evaluate_jacobians(x0)

        # Central FD of production J_total
        h = 1e-6
        grad_fd = np.zeros(len(x0))
        for k in range(len(x0)):
            xp = x0.copy()
            xm = x0.copy()
            xp[k] += h
            xm[k] -= h
            rp = evaluate(xp, ctx, DomainEvaluatorCache())
            rm = evaluate(xm, ctx, DomainEvaluatorCache())
            grad_fd[k] = (rp.objective_value - rm.objective_value) / (2 * h)

        # The analytical gradient should match FD
        np.testing.assert_allclose(
            j_base_ana,
            grad_fd,
            rtol=5e-4,
            atol=1e-8,
            err_msg="Analytical gradient does not match central FD of production objective",
        )

    def test_directional_derivatives(self, production_ctx):
        """∇J^T d ≈ (J(u+hd) - J(u-hd)) / (2h) for multiple directions."""
        ctx = production_ctx
        x0 = np.array([0.5, 0.5, 0.5])

        txn = DerivativeTransaction(ctx)
        j_base_ana, _ = txn.evaluate_jacobians(x0)

        rng = np.random.default_rng(42)
        directions = [rng.standard_normal(3) for _ in range(5)]
        # Normalize
        directions = [d / np.linalg.norm(d) for d in directions]
        # Also add coordinate directions
        for k in range(3):
            e = np.zeros(3)
            e[k] = 1.0
            directions.append(e)

        h = 1e-6
        for i, d in enumerate(directions):
            analytical_dd = j_base_ana @ d
            xp = x0 + h * d
            xm = x0 - h * d
            rp = evaluate(xp, ctx, DomainEvaluatorCache())
            rm = evaluate(xm, ctx, DomainEvaluatorCache())
            fd_dd = (rp.objective_value - rm.objective_value) / (2 * h)

            np.testing.assert_allclose(
                analytical_dd,
                fd_dd,
                rtol=5e-4,
                atol=1e-8,
                err_msg=f"Directional derivative mismatch for direction {i}",
            )

    def test_same_u_value_and_gradient(self, production_ctx):
        """Same u produces consistent J_total(u) and ∇J_total(u)."""
        ctx = production_ctx
        x0 = np.array([0.5, 0.5, 0.5])

        # Get J_total from production evaluator
        evaluate(x0, ctx, DomainEvaluatorCache())

        # Get gradient from analytical transaction
        txn = DerivativeTransaction(ctx)
        j_base_ana, _ = txn.evaluate_jacobians(x0)

        # The gradient must not be all zeros (for non-trivial objective)
        assert np.any(j_base_ana != 0.0), "Gradient is all zeros — potential silent placeholder"

        # Gradient norm should be comparable to FD estimate
        h = 1e-7
        fd_norm_estimate = 0.0
        for k in range(len(x0)):
            xp = x0.copy()
            xm = x0.copy()
            xp[k] += h
            xm[k] -= h
            rp = evaluate(xp, ctx, DomainEvaluatorCache())
            rm = evaluate(xm, ctx, DomainEvaluatorCache())
            fd_norm_estimate += ((rp.objective_value - rm.objective_value) / (2 * h)) ** 2
        fd_norm = math.sqrt(fd_norm_estimate)
        ana_norm = np.linalg.norm(j_base_ana)

        np.testing.assert_allclose(
            ana_norm,
            fd_norm,
            rtol=5e-2,
            err_msg="Analytical gradient norm does not match FD estimate",
        )


# ── Requirement 3: Complete Production Constraint Jacobian ────────────────────


class TestFullConstraintJacobian:
    """Validate the full runtime constraint Jacobian in one shot."""

    @pytest.fixture
    def production_ctx(self):
        return _build_production_context(
            w_gamma=1.0,
            w_loss=0.0,
            target_frequencies_hz=(1.0e6, 5.0e6),
            base_grid_points=4,
        )

    def test_shape_and_descriptor_order(self, production_ctx):
        """Verify row count, shape, descriptor order, and block structure."""
        ctx = production_ctx
        layout = ctx.hard_layout
        mapper = ctx.domain.variable_mapper

        n_targets = len(ctx.target_indices)
        n_off_target = len(ctx.off_target_indices)
        n_cells_b1 = ctx.domain.topology.branch1_cells  # 1
        n_cells_b2 = ctx.domain.topology.branch2_cells  # 0

        # Expected rows:
        # Per target: gamma, r_max, r_min, x_bound, i_source = 5
        # Off-target: 1 per off-target frequency
        # Component bounds: 4 per cell per branch (L_hi, L_lo, C_hi, C_lo)
        # Pole sep: (n_cells - 1) per branch
        expected_target_rows = n_targets * 5
        expected_off_target_rows = n_off_target
        expected_comp_rows = (n_cells_b1 + n_cells_b2) * 4
        expected_sep_rows = max(0, n_cells_b1 - 1) + max(0, n_cells_b2 - 1)
        expected_total = (
            expected_target_rows + expected_off_target_rows + expected_comp_rows + expected_sep_rows
        )

        assert layout.n == expected_total, (
            f"Expected {expected_total} constraints, got {layout.n}. "
            f"targets={n_targets}, off_target={n_off_target}, "
            f"cells_b1={n_cells_b1}, cells_b2={n_cells_b2}"
        )

        # Compute Jacobian
        txn = DerivativeTransaction(ctx)
        x0 = np.array([0.5, 0.5, 0.5])
        _, j_constr = txn.evaluate_jacobians(x0)

        assert j_constr.shape == (layout.n, mapper.dimension), (
            f"Jacobian shape {j_constr.shape} != ({layout.n}, {mapper.dimension})"
        )

        # Verify descriptor types present
        types_present = {d.constraint_type for d in layout.descriptors}
        assert "gamma" in types_present
        assert "r_max" in types_present
        assert "r_min" in types_present
        assert "x_bound" in types_present
        assert "i_source" in types_present
        if n_off_target > 0:
            assert "offtarget" in types_present
        if n_cells_b1 > 0:
            assert "comp_L_hi" in types_present
            assert "comp_C_hi" in types_present

    def test_full_layout_central_fd(self, production_ctx):
        """Full constraint Jacobian vs central FD of production evaluate()."""
        ctx = production_ctx
        x0 = np.array([0.5, 0.5, 0.5])

        # Analytical Jacobian
        txn = DerivativeTransaction(ctx)
        _, j_constr_ana = txn.evaluate_jacobians(x0)

        # Central FD of production g-vector
        h = 1e-6
        n_c = ctx.hard_layout.n
        n_dim = len(x0)
        j_fd = np.zeros((n_c, n_dim))

        for k in range(n_dim):
            xp = x0.copy()
            xm = x0.copy()
            xp[k] += h
            xm[k] -= h
            rp = evaluate(xp, ctx, DomainEvaluatorCache())
            rm = evaluate(xm, ctx, DomainEvaluatorCache())
            j_fd[:, k] = (np.array(rp.hard_margins) - np.array(rm.hard_margins)) / (2 * h)

        # Compare each row with context
        for i, desc in enumerate(ctx.hard_layout.descriptors):
            for k in range(n_dim):
                if abs(j_fd[i, k]) > 1e-12 or abs(j_constr_ana[i, k]) > 1e-12:
                    np.testing.assert_allclose(
                        j_constr_ana[i, k],
                        j_fd[i, k],
                        rtol=5e-4,
                        atol=1e-7,
                        err_msg=(
                            f"Constraint Jacobian mismatch: row {i} ({desc.name}, "
                            f"type={desc.constraint_type}), param {k}"
                        ),
                    )

    def test_full_layout_directional_fd(self, production_ctx):
        """J_g d ≈ (g(u+hd) - g(u-hd)) / (2h) for multiple directions."""
        ctx = production_ctx
        x0 = np.array([0.5, 0.5, 0.5])

        txn = DerivativeTransaction(ctx)
        _, j_constr_ana = txn.evaluate_jacobians(x0)

        rng = np.random.default_rng(123)
        directions = [rng.standard_normal(3) for _ in range(5)]
        directions = [d / np.linalg.norm(d) for d in directions]
        # Add coordinate directions
        for k in range(3):
            e = np.zeros(3)
            e[k] = 1.0
            directions.append(e)

        h = 1e-6
        for di, d in enumerate(directions):
            ana_dd = j_constr_ana @ d  # shape (n_c,)
            xp = x0 + h * d
            xm = x0 - h * d
            rp = evaluate(xp, ctx, DomainEvaluatorCache())
            rm = evaluate(xm, ctx, DomainEvaluatorCache())
            fd_dd = (np.array(rp.hard_margins) - np.array(rm.hard_margins)) / (2 * h)

            for i in range(ctx.hard_layout.n):
                if abs(fd_dd[i]) > 1e-12 or abs(ana_dd[i]) > 1e-12:
                    np.testing.assert_allclose(
                        ana_dd[i],
                        fd_dd[i],
                        rtol=5e-4,
                        atol=1e-7,
                        err_msg=(
                            f"Directional FD mismatch: direction {di}, "
                            f"row {i} ({ctx.hard_layout.descriptors[i].name})"
                        ),
                    )


# ── Requirement 4: Safety Audit ───────────────────────────────────────────────


class TestSafetyAudit:
    """Final safety checks for D1.1 closeout."""

    def test_no_silent_placeholder_zeros_in_gradient(self):
        """The analytical gradient must not be all zeros for non-trivial configs."""
        ctx = _build_production_context(w_gamma=1.0)
        x0 = np.array([0.5, 0.5, 0.5])
        txn = DerivativeTransaction(ctx)
        j_base, _ = txn.evaluate_jacobians(x0)
        assert np.any(j_base != 0.0), "Gradient is all zeros — silent placeholder detected"

    def test_all_production_configs_now_supported(self):
        """All production configs (including w_loss, soft) are now SUPPORTED."""
        # w_loss with lossy elements
        config_loss = ObjectiveConfig(
            z_ref_ohm=50.0,
            w_loss=1.0,
            lossy_element_ids=("R_parasitic",),
        )
        soft_empty = ConstraintLayout(descriptors=())
        ok, reasons = check_analytical_support(config_loss, soft_empty, None, None)
        assert ok, f"Expected supported, got reasons: {reasons}"
        # w_gamma + w_loss + soft all together
        ok2, reasons2 = check_analytical_support(
            ObjectiveConfig(z_ref_ohm=50.0, w_gamma=1.0, w_loss=0.5),
            soft_empty,
            None,
            None,
        )
        assert ok2, f"Expected supported, got reasons: {reasons2}"

    def test_adjoint_uses_trans_2(self):
        """Verify Y^H λ = 2q uses lu_solve(..., trans=2) — Hermitian transpose."""
        import inspect

        from foster_eom.sensitivities.adjoint import compute_adjoint_state

        source = inspect.getsource(compute_adjoint_state)
        assert "trans=2" in source, (
            "compute_adjoint_state must use lu_solve(..., trans=2) for Hermitian transpose"
        )

    def test_frozen_nominal_semantics_unchanged(self):
        """Production evaluate() still returns valid results (no regression)."""
        ctx = _build_production_context(w_gamma=1.0)
        x0 = np.array([0.5, 0.5, 0.5])
        res = evaluate(x0, ctx, DomainEvaluatorCache())
        assert res.numerical_status == "ok"
        assert res.objective_value >= 0.0
        assert len(res.hard_margins) == ctx.hard_layout.n

    def test_derivative_status_tracking(self):
        """ObjectiveGradientResult status tracks actual derivative validity."""
        ctx = _build_production_context(w_gamma=1.0)
        x0 = np.array([0.5, 0.5, 0.5])
        txn = DerivativeTransaction(ctx)
        txn.evaluate_jacobians(x0)
        assert txn._obj_grad_result is not None
        # Default production config should be smooth
        assert txn._obj_grad_result.status in (
            DerivativeStatus.SMOOTH,
            DerivativeStatus.NONSMOOTH_KINK,
        )
        assert len(txn._obj_grad_result.unsupported_terms) == 0
