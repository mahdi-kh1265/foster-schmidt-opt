"""P12.5-D2.2 — Evidence-Only Final Closeout Tests.

Tests with ACTUAL violated/inactive soft constraints using production layouts.
Reports max relative/absolute errors for every gradient comparison.

Key insight: The production `_eval_one("gamma", ...)` evaluator uses the
`gamma_max` parameter passed from `MatchConstraints.gamma_max`, not the
descriptor's `normalization_scale`. So to create violated soft gamma
constraints, we set `gamma_max` in MatchConstraints to a very tight value.
"""

import dataclasses

import numpy as np

from foster_eom.domain.component import ContinuousLimits
from foster_eom.domain.constraints import (
    ConstraintSeverity,
    FrequencyScope,
    MatchConstraints,
    StressConstraints,
)
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.domain.topology import LOrientation
from foster_eom.foster.schmidt import BranchRealization
from foster_eom.foster.sign_search import SignPattern
from foster_eom.foster.topology_enum import TopologyCandidate
from foster_eom.models.base import OnePortModel
from foster_eom.optimize.constraints import (
    ConstraintDescriptor,
    ConstraintLayout,
)
from foster_eom.optimize.domain import ContinuousOptimizationDomain
from foster_eom.optimize.evaluator import (
    DomainEvaluatorCache,
    EvaluationContext,
    build_evaluation_context,
    evaluate,
)
from foster_eom.optimize.objective import ObjectiveConfig
from foster_eom.optimize.variable_map import build_variable_mapper
from foster_eom.sensitivities.transaction import DerivativeTransaction

# ── Shared fixtures ───────────────────────────────────────────────────────────


class DummyEOM(OnePortModel):
    def _z_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        return 50.0 - 1j / (2 * np.pi * f_hz * 1e-12)

    @property
    def metadata(self) -> dict:
        return {}


def _build_ctx(
    *,
    w_gamma: float = 1.0,
    w_loss: float = 0.0,
    gamma_max: float = 0.5,
    lossy_element_ids: tuple[str, ...] = (),
) -> EvaluationContext:
    """Build a production-like EvaluationContext with NO soft constraints."""
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
        domain_id="d22_test",
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
    target_freqs = (1.0e6, 5.0e6)
    obj = ObjectiveConfig(
        z_ref_ohm=50.0,
        w_gamma=w_gamma,
        w_voltage=0.0,
        w_loss=w_loss,
        w_complexity=0.0,
        lossy_element_ids=lossy_element_ids,
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
        match_constraints=MatchConstraints(
            gamma_max=gamma_max,
            resistance_max_ohm=50.0,
        ),
        stress_constraints=StressConstraints(
            source_current_rms_max_a=1.0, off_target_eom_peak_rms_v=2.0
        ),
        target_frequencies_hz=target_freqs,
        sweep_f_min_hz=min(target_freqs),
        sweep_f_max_hz=max(target_freqs) * 2,
        base_grid_points=4,
        objective_config=obj,
        feasibility_tolerance=1e-3,
        near_feasibility_tolerance=1e-3,
    )


def _make_soft_gamma_layout(
    ctx: EvaluationContext,
    penalty_weight: float = 10.0,
) -> ConstraintLayout:
    """Build soft gamma descriptors matching the context's target indices.

    The actual gamma limit comes from ctx.match_constraints.gamma_max,
    which is what _eval_one uses for the "gamma" constraint type.
    """
    descs = []
    for ti, fi in enumerate(ctx.target_indices):
        descs.append(
            ConstraintDescriptor(
                name=f"soft_gamma_f{fi}",
                constraint_type="gamma",
                frequency_scope=FrequencyScope.ALL_TARGETS,
                severity=ConstraintSeverity.SOFT,
                target_index=ti,
                freq_index=fi,
                normalization_scale=max(ctx.match_constraints.gamma_max, 1e-6),
                penalty_weight=penalty_weight,
            )
        )
    return ConstraintLayout(descriptors=tuple(descs))


def _inject_soft(ctx: EvaluationContext, soft: ConstraintLayout) -> EvaluationContext:
    """Replace soft_layout on a frozen EvaluationContext."""
    return dataclasses.replace(ctx, soft_layout=soft)


def _eval_j(ctx: EvaluationContext, x: np.ndarray) -> float:
    """Production J_total."""
    return evaluate(x, ctx, DomainEvaluatorCache()).objective_value


def _central_fd(ctx: EvaluationContext, x0: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Componentwise central-FD gradient."""
    grad = np.zeros(len(x0))
    for k in range(len(x0)):
        xp, xm = x0.copy(), x0.copy()
        xp[k] += h
        xm[k] -= h
        grad[k] = (_eval_j(ctx, xp) - _eval_j(ctx, xm)) / (2 * h)
    return grad


def _dir_fd(ctx: EvaluationContext, x0: np.ndarray, d: np.ndarray, h: float = 1e-5) -> float:
    """Central-FD directional derivative."""
    return (_eval_j(ctx, x0 + h * d) - _eval_j(ctx, x0 - h * d)) / (2 * h)


def _report(label: str, ana: np.ndarray, fd: np.ndarray) -> dict:
    """Compute and print error metrics."""
    abs_diff = np.abs(ana - fd)
    denom = np.where(np.abs(fd) > 1e-15, np.abs(fd), 1.0)
    rel_diff = abs_diff / denom
    m = {
        "max_abs": float(np.max(abs_diff)),
        "max_rel": float(np.max(rel_diff)),
    }
    print(f"\n  {label}:")
    print(f"    max|abs err| = {m['max_abs']:.3e}")
    print(f"    max|rel err| = {m['max_rel']:.3e}")
    print(f"    analytical = {ana}")
    print(f"    FD         = {fd}")
    return m


# ── Requirement 1: Violated soft (g < 0) ─────────────────────────────────────


class TestViolatedSoft:
    """gamma_max=0.01 → |Γ| >> 0.01 → g_soft < 0 → violated."""

    X0 = np.array([0.5, 0.5, 0.5])

    def _ctx(self, *, w_gamma: float = 1.0, w_loss: float = 0.0):
        ctx_base = _build_ctx(w_gamma=w_gamma, w_loss=w_loss, gamma_max=0.01)
        soft = _make_soft_gamma_layout(ctx_base, penalty_weight=10.0)
        return _inject_soft(ctx_base, soft)

    def test_actually_violated(self):
        """At least one soft margin is < 0."""
        ctx = self._ctx()
        res = evaluate(self.X0, ctx, DomainEvaluatorCache())
        assert res.soft_penalty_total > 0, (
            f"Expected violated, got penalty={res.soft_penalty_total}"
        )
        print(f"\n  soft_penalty_total = {res.soft_penalty_total:.6e}")

    def test_componentwise(self):
        """Componentwise ∇J matches FD."""
        ctx = self._ctx()
        txn = DerivativeTransaction(ctx)
        grad_ana, _ = txn.evaluate_jacobians(self.X0)
        grad_fd = _central_fd(ctx, self.X0)
        m = _report("violated_soft_comp", grad_ana, grad_fd)
        np.testing.assert_allclose(grad_ana, grad_fd, rtol=2e-2, atol=1e-8)

    def test_directional(self):
        """Directional ∇J^T d matches FD (5 random directions)."""
        ctx = self._ctx()
        txn = DerivativeTransaction(ctx)
        grad_ana, _ = txn.evaluate_jacobians(self.X0)
        rng = np.random.default_rng(100)
        max_rel = 0.0
        for _ in range(5):
            d = rng.standard_normal(3)
            d /= np.linalg.norm(d)
            ana_dd = float(grad_ana @ d)
            fd_dd = _dir_fd(ctx, self.X0, d)
            if abs(fd_dd) > 1e-15:
                max_rel = max(max_rel, abs(ana_dd - fd_dd) / abs(fd_dd))
            np.testing.assert_allclose(ana_dd, fd_dd, rtol=2e-2, atol=1e-8)
        print(f"\n  violated_soft_dir max_rel = {max_rel:.3e}")


# ── Requirement 2: Inactive soft (g > 0) ─────────────────────────────────────


class TestInactiveSoft:
    """gamma_max=1.0 → |Γ| < 1.0 → g_soft > 0 → inactive."""

    X0 = np.array([0.5, 0.5, 0.5])

    def _ctx(self):
        ctx_base = _build_ctx(w_gamma=1.0, gamma_max=1.0)
        soft = _make_soft_gamma_layout(ctx_base, penalty_weight=10.0)
        return _inject_soft(ctx_base, soft)

    def test_actually_inactive(self):
        """All soft margins > 0."""
        ctx = self._ctx()
        res = evaluate(self.X0, ctx, DomainEvaluatorCache())
        assert res.soft_penalty_total == 0.0, f"Expected inactive, got {res.soft_penalty_total}"

    def test_componentwise(self):
        """Gradient matches FD (soft adds zero penalty)."""
        ctx = self._ctx()
        txn = DerivativeTransaction(ctx)
        grad_ana, _ = txn.evaluate_jacobians(self.X0)
        grad_fd = _central_fd(ctx, self.X0)
        m = _report("inactive_soft_comp", grad_ana, grad_fd)
        np.testing.assert_allclose(grad_ana, grad_fd, rtol=2e-2, atol=1e-8)

    def test_equals_no_soft_baseline(self):
        """Inactive soft gradient == baseline (no soft) gradient."""
        ctx_base = _build_ctx(w_gamma=1.0, gamma_max=1.0)
        ctx_soft = self._ctx()
        txn_base = DerivativeTransaction(ctx_base)
        grad_base, _ = txn_base.evaluate_jacobians(self.X0)
        txn_soft = DerivativeTransaction(ctx_soft)
        grad_soft, _ = txn_soft.evaluate_jacobians(self.X0)
        np.testing.assert_allclose(grad_soft, grad_base, rtol=1e-12, atol=1e-15)


# ── Requirement 3: Soft-only objective ────────────────────────────────────────


class TestSoftOnly:
    """w_gamma=0, w_loss=0, gamma_max=0.01 → J = J_soft (violated)."""

    X0 = np.array([0.5, 0.5, 0.5])

    def _ctx(self):
        ctx_base = _build_ctx(w_gamma=0.0, w_loss=0.0, gamma_max=0.01)
        soft = _make_soft_gamma_layout(ctx_base, penalty_weight=5.0)
        return _inject_soft(ctx_base, soft)

    def test_j_parity(self):
        """J_production > 0 for soft-only violated config."""
        ctx = self._ctx()
        j = _eval_j(ctx, self.X0)
        assert j > 0, f"Expected J > 0, got {j}"
        print(f"\n  soft_only J = {j:.6e}")

    def test_componentwise(self):
        """Componentwise ∇J matches FD."""
        ctx = self._ctx()
        txn = DerivativeTransaction(ctx)
        grad_ana, _ = txn.evaluate_jacobians(self.X0)
        grad_fd = _central_fd(ctx, self.X0)
        m = _report("soft_only_comp", grad_ana, grad_fd)
        np.testing.assert_allclose(grad_ana, grad_fd, rtol=2e-2, atol=1e-8)

    def test_directional(self):
        """Directional ∇J^T d matches FD (5 random)."""
        ctx = self._ctx()
        txn = DerivativeTransaction(ctx)
        grad_ana, _ = txn.evaluate_jacobians(self.X0)
        rng = np.random.default_rng(200)
        max_rel = 0.0
        for _ in range(5):
            d = rng.standard_normal(3)
            d /= np.linalg.norm(d)
            ana_dd = float(grad_ana @ d)
            fd_dd = _dir_fd(ctx, self.X0, d)
            if abs(fd_dd) > 1e-15:
                max_rel = max(max_rel, abs(ana_dd - fd_dd) / abs(fd_dd))
            np.testing.assert_allclose(ana_dd, fd_dd, rtol=2e-2, atol=1e-8)
        print(f"\n  soft_only_dir max_rel = {max_rel:.3e}")

    def test_gradient_nonzero(self):
        """Soft-only gradient must be materially nonzero when violated."""
        ctx = self._ctx()
        txn = DerivativeTransaction(ctx)
        grad_ana, _ = txn.evaluate_jacobians(self.X0)
        norm = float(np.linalg.norm(grad_ana))
        print(f"\n  soft_only grad norm = {norm:.6e}")
        assert norm > 1e-6, f"Gradient too small: {norm}"


# ── Requirement 4: w_loss > 0 + violated soft ────────────────────────────────


class TestLossPlusSoft:
    """w_gamma=1, w_loss=1, gamma_max=0.01, soft violated simultaneously."""

    X0 = np.array([0.5, 0.5, 0.5])

    def _ctx(self):
        ctx_base = _build_ctx(w_gamma=1.0, w_loss=1.0, gamma_max=0.01, lossy_element_ids=())
        soft = _make_soft_gamma_layout(ctx_base, penalty_weight=10.0)
        return _inject_soft(ctx_base, soft)

    def test_j_parity(self):
        """J_production has both base and soft penalty > 0."""
        ctx = self._ctx()
        res = evaluate(self.X0, ctx, DomainEvaluatorCache())
        assert res.soft_penalty_total > 0, f"Soft penalty = {res.soft_penalty_total}"
        assert res.base_objective_value > 0, f"Base = {res.base_objective_value}"
        print(
            f"\n  loss+soft: J_total={res.objective_value:.6e}, "
            f"J_base={res.base_objective_value:.6e}, "
            f"J_soft={res.soft_penalty_total:.6e}"
        )

    def test_componentwise(self):
        """Componentwise ∇J matches FD."""
        ctx = self._ctx()
        txn = DerivativeTransaction(ctx)
        grad_ana, _ = txn.evaluate_jacobians(self.X0)
        grad_fd = _central_fd(ctx, self.X0)
        m = _report("loss_plus_soft_comp", grad_ana, grad_fd)
        np.testing.assert_allclose(grad_ana, grad_fd, rtol=2e-2, atol=1e-8)

    def test_directional(self):
        """Directional ∇J^T d matches FD (5 random)."""
        ctx = self._ctx()
        txn = DerivativeTransaction(ctx)
        grad_ana, _ = txn.evaluate_jacobians(self.X0)
        rng = np.random.default_rng(300)
        max_rel = 0.0
        for _ in range(5):
            d = rng.standard_normal(3)
            d /= np.linalg.norm(d)
            ana_dd = float(grad_ana @ d)
            fd_dd = _dir_fd(ctx, self.X0, d)
            if abs(fd_dd) > 1e-15:
                max_rel = max(max_rel, abs(ana_dd - fd_dd) / abs(fd_dd))
            np.testing.assert_allclose(ana_dd, fd_dd, rtol=2e-2, atol=1e-8)
        print(f"\n  loss_plus_soft_dir max_rel = {max_rel:.3e}")
