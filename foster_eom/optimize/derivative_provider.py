"""Analytical derivative provider for local polish (P12.5-E).

Adapts the validated P12.5-D :class:`DerivativeTransaction` to the callback
signature ``trust-constr`` expects, and refuses to hand SciPy anything partial.

Design contract
---------------
* Objective and constraint *values* are **not** produced here.  They remain the
  frozen production ``evaluate()`` callbacks in
  :mod:`foster_eom.optimize.local_polish`.  This module supplies Jacobians only.
* Both Jacobians for one parameter vector come from a single current-``u``
  transaction.  A new ``u`` invalidates it; no historical heavy state is kept.
* Any unsupported / nonsmooth / unresolved derivative state, incomplete nominal
  coverage, non-finite entry, or shape mismatch raises
  :class:`DerivativeUnavailable` so the caller can fall back to the frozen FD
  path for that candidate.  A partially-populated Jacobian is never returned.
* P12.5-F: the transaction reuses the nominal per-frequency MNA state the
  production ``evaluate()`` already computed for the *current* iterate, via the
  :class:`~foster_eom.optimize.nominal_state.NominalStateExchange` attached to
  the shared :class:`DomainEvaluatorCache`.  Reuse requires exact identity of
  ``u``, context, domain and frequency grid; anything else re-solves.  Only
  ``ANALYTICAL`` attaches an exchange, so ``REFERENCE_FD`` is untouched.
"""

from __future__ import annotations

import numpy as np

from foster_eom.optimize.evaluator import DomainEvaluatorCache, EvaluationContext
from foster_eom.optimize.nominal_state import NominalStateExchange
from foster_eom.sensitivities.objective_gradient import (
    DerivativeStatus,
    check_analytical_support,
)
from foster_eom.sensitivities.transaction import DerivativeTransaction

#: Constraint types whose Jacobian rows come from the off-target adjoint sweep.
_OFF_TARGET_CONSTRAINT_TYPES = frozenset({"offtarget", "v_min", "v_max"})


class DerivativeUnavailable(Exception):
    """Raised when no trustworthy analytical Jacobian can be produced.

    Carries a compact machine-readable ``reason`` for telemetry.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _required_off_target_indices(context: EvaluationContext) -> frozenset[int]:
    """Off-target frequency indices that any hard/soft constraint row depends on."""
    needed: set[int] = set()
    for layout in (context.hard_layout, context.soft_layout):
        for desc in layout.descriptors:
            if desc.constraint_type in _OFF_TARGET_CONSTRAINT_TYPES and desc.freq_index is not None:
                needed.add(int(desc.freq_index))
    return frozenset(needed)


class AnalyticalDerivativeProvider:
    """Supplies ``grad J(u)`` and ``J_g(u)`` from one shared transaction.

    Parameters
    ----------
    context : EvaluationContext
        The same frozen context the production evaluator uses.
    cache : DomainEvaluatorCache | None
        The same cache the production objective/constraint callbacks use.  When
        given, a :class:`NominalStateExchange` is attached to it so the current
        iterate's nominal MNA state is shared instead of swept twice (P12.5-F).
        When ``None``, the transaction sweeps the grid itself exactly as in
        P12.5-E.
    """

    def __init__(
        self,
        context: EvaluationContext,
        cache: DomainEvaluatorCache | None = None,
    ) -> None:
        self.context = context
        self.cache = cache

        exchange: NominalStateExchange | None = None
        if cache is not None:
            exchange = cache.nominal_exchange
            if exchange is None:
                exchange = NominalStateExchange()
                cache.nominal_exchange = exchange
            exchange.enable()
        self.exchange = exchange
        self.transaction = DerivativeTransaction(context, exchange=exchange)

        self.n_params: int = context.domain.variable_mapper.dimension
        self.n_hard: int = context.hard_layout.n
        #: Row count SciPy sees.  ``_g_vec`` substitutes a single constant row
        #: when the hard layout is empty, so the Jacobian must match that shape.
        self.n_constraint_rows: int = self.n_hard if self.n_hard > 0 else 1

        self._required_off_target = _required_off_target_indices(context)

        # Telemetry: distinct-u transaction builds vs same-u reuse hits.
        self.n_transaction_evaluations: int = 0
        self.n_reuse_hits: int = 0
        self.n_objective_jac_calls: int = 0
        self.n_constraint_jac_calls: int = 0

    # -- pre-flight ---------------------------------------------------------

    def preflight(self, x0: np.ndarray) -> None:
        """Validate the configuration and one full build at ``x0``.

        Raises
        ------
        DerivativeUnavailable
            If the configuration is unsupported or the build at ``x0`` is not
            trustworthy.  The caller must then use the frozen FD path.
        """
        supported, reasons = check_analytical_support(
            config=self.context.objective_config,
            soft_layout=self.context.soft_layout,
            soft_g_vector=None,
            soft_jacobian=None,
        )
        if not supported:
            raise DerivativeUnavailable("unsupported_config:" + ",".join(reasons))
        self._ensure(x0)

    # -- SciPy callbacks ----------------------------------------------------

    def objective_jac(self, x: np.ndarray) -> np.ndarray:
        """Return ``grad J(u)`` with shape ``(n_params,)``."""
        self.n_objective_jac_calls += 1
        j_base, _ = self._ensure(x)
        return j_base

    def constraint_jac(self, x: np.ndarray) -> np.ndarray:
        """Return ``J_g(u)`` with shape ``(n_constraint_rows, n_params)``."""
        self.n_constraint_jac_calls += 1
        _, j_constr = self._ensure(x)
        if self.n_hard == 0:
            # Parity with the constant placeholder row ``_g_vec`` returns.
            return np.zeros((1, self.n_params), dtype=np.float64)
        return j_constr

    # -- internals ----------------------------------------------------------

    def _ensure(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build (or reuse) the current-``u`` transaction and validate it."""
        from foster_eom.optimize.evaluator import canonical_x, evaluate

        x_eval, x_key = canonical_x(x)

        # SciPy's trust-constr often evaluates Jacobians before the objective.
        # Force the production evaluator to run first, publishing its nominal state,
        # so the transaction can reuse it.  SciPy's subsequent objective evaluation
        # will hit the cache.
        if self.cache is not None:
            evaluate(x_eval, self.context, self.cache)

        prev_x = self.transaction.current_x
        reused = prev_x is not None and np.array_equal(prev_x, x_eval)

        try:
            j_base, j_constr = self.transaction.evaluate_jacobians(x_eval, x_key=x_key)
        except Exception as exc:  # any build failure means fall back to FD
            raise DerivativeUnavailable(f"construction_failed:{type(exc).__name__}:{exc}") from exc

        if reused:
            self.n_reuse_hits += 1
        else:
            self.n_transaction_evaluations += 1

        self._validate(j_base, j_constr)
        return j_base, j_constr

    def _validate(self, j_base: np.ndarray, j_constr: np.ndarray) -> None:
        """Reject anything that is not a complete, finite, correctly-shaped Jacobian."""
        status = self.transaction.last_status
        if status is None:
            raise DerivativeUnavailable("no_status")
        if status is not DerivativeStatus.SMOOTH:
            result = self.transaction.last_result
            detail = ""
            if result is not None:
                terms = result.unsupported_terms or result.nonsmooth_terms
                if terms:
                    detail = ":" + ",".join(terms[:3])
            raise DerivativeUnavailable(f"status_{status.value}{detail}")

        missing_targets = set(self.context.target_indices) - self.transaction.solved_target_indices
        if missing_targets:
            raise DerivativeUnavailable(
                f"nominal_target_solve_failed:{sorted(missing_targets)[:3]}"
            )

        missing_off = self._required_off_target - self.transaction.solved_off_target_indices
        if missing_off:
            raise DerivativeUnavailable(f"nominal_off_target_solve_failed:n={len(missing_off)}")

        if j_base.shape != (self.n_params,):
            raise DerivativeUnavailable(f"objective_jac_shape:{j_base.shape}")
        if self.n_hard > 0 and j_constr.shape != (self.n_hard, self.n_params):
            raise DerivativeUnavailable(f"constraint_jac_shape:{j_constr.shape}")

        if not np.all(np.isfinite(j_base)):
            raise DerivativeUnavailable("objective_jac_nonfinite")
        if self.n_hard > 0 and not np.all(np.isfinite(j_constr)):
            raise DerivativeUnavailable("constraint_jac_nonfinite")

    # -- telemetry ----------------------------------------------------------

    def metrics_snapshot(self) -> dict[str, int]:
        """Flat counter snapshot for polish telemetry."""
        m = self.transaction.metrics
        snapshot = {
            "transaction_evaluations": self.n_transaction_evaluations,
            "transaction_reuse_hits": self.n_reuse_hits,
            "objective_jac_calls": self.n_objective_jac_calls,
            "constraint_jac_calls": self.n_constraint_jac_calls,
            "jacobian_evals": int(m["jacobian_evals"]),
            "factorizations": int(m["factorizations"]),
            "direct_substitutions": int(m["direct_substitutions"]),
            "adjoint_substitutions": int(m["adjoint_substitutions"]),
            # P12.5-F nominal-work accounting
            "transaction_nominal_sweep_solves": int(m["nominal_sweep_solves"]),
            "transaction_nominal_states_reused": int(m["nominal_states_reused"]),
            "nominal_bundle_hits": int(m["bundle_hits"]),
            "nominal_bundle_misses": int(m["bundle_misses"]),
        }
        if self.exchange is not None:
            snapshot.update(self.exchange.counters())
        return snapshot

    def release(self) -> None:
        """Stop capturing and free every retained nominal state bundle."""
        if self.exchange is not None:
            self.exchange.disable()
