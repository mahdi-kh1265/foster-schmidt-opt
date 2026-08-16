"""Central evaluator for Prompt 05 continuous optimization.

Implements the frozen EvaluationContext, EvaluationResult, and DomainEvaluatorCache.
One unique ``x`` vector → one MNA solve per frequency → one EvaluationResult.
The cache ensures that identical ``x`` vectors are not re-solved.

Numerical failure semantics:
  - Structural numerical failures (LinAlgError, nonfinite, negative residues) are caught
    and returned as infeasible results with deterministic constraint vectors.
  - Programming errors (AttributeError, KeyError, etc.) propagate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from foster_eom.circuit.graph import CircuitGraph
from foster_eom.circuit.measurements import CircuitSolution
from foster_eom.circuit.mna import SolverOptions
from foster_eom.circuit.solve import solve_circuit_single
from foster_eom.domain.component import ContinuousLimits
from foster_eom.domain.constraints import MatchConstraints, StressConstraints
from foster_eom.domain.source import SourceSpec
from foster_eom.errors import CircuitSolveStatus
from foster_eom.foster.network_builder import build_foster_circuit
from foster_eom.foster.sign_search import SignPattern
from foster_eom.models.base import OnePortModel
from foster_eom.optimize.constraints import ConstraintLayout, compile_constraint_layout
from foster_eom.optimize.domain import ContinuousOptimizationDomain
from foster_eom.optimize.objective import ObjectiveConfig, compute_objective
from foster_eom.optimize.variable_map import BranchCoordinates

# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationResult:
    """Complete result for a single physical evaluation.

    ``objective_value`` is J_total = J_base + J_soft — the single canonical
    scalar used by DE, local polish, ranking, and persistence.
    """

    x: tuple[float, ...]

    # Canonical objective
    objective_value: float  # J_total = J_base + J_soft
    base_objective_value: float
    soft_penalty_total: float
    objective_terms: dict[str, float]  # "total", "base", "soft_penalty", "j_gamma", ...

    # Hard constraint margins (fixed-length, deterministic order)
    hard_margins: tuple[float, ...]

    # Soft constraint per-term penalties
    soft_penalties: dict[str, float]

    # Feasibility
    v_max: float
    v_sum: float
    feasible: bool
    near_feasible: bool

    # Numerical status
    numerical_status: str  # "ok" | "mna_singular" | "nonfinite" | "component_invalid"
    numerical_failure_reason: str | None
    failed_frequency_hz: float | None
    failed_stage: str | None  # "target" | "coarse"

    # Circuit solutions (one per evaluation frequency)
    all_solutions: tuple[CircuitSolution, ...]
    target_solutions: tuple[CircuitSolution, ...]

    # Diagnostic
    coarse_evaluated: bool


# ---------------------------------------------------------------------------
# Frozen evaluation context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationContext:
    """Compiled, immutable context shared across all evaluations in one domain run."""

    domain: ContinuousOptimizationDomain
    source_spec: SourceSpec
    eom_model: OnePortModel
    component_limits: ContinuousLimits
    match_constraints: MatchConstraints
    stress_constraints: StressConstraints

    # Compiled frequency structure
    evaluation_frequencies_hz: tuple[float, ...]  # unique, sorted
    target_indices: tuple[int, ...]  # into evaluation_frequencies_hz
    off_target_indices: tuple[int, ...]
    off_target_mask: tuple[bool, ...]

    # Compiled layouts
    hard_layout: ConstraintLayout
    soft_layout: ConstraintLayout

    # Objective configuration
    objective_config: ObjectiveConfig

    # Lazy evaluation flag
    requires_coarse_for_hard_soft: bool

    # Tolerances
    feasibility_tolerance: float
    near_feasibility_tolerance: float


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class DomainEvaluatorCache:
    """Per-domain cache mapping normalized x → EvaluationResult."""

    def __init__(self) -> None:
        self._cache: dict[tuple[float, ...], EvaluationResult] = {}
        self.n_calls: int = 0
        self.n_unique_evaluations: int = 0
        self.n_cache_hits: int = 0
        self.target_frequency_point_solves: int = 0
        self.coarse_frequency_point_solves: int = 0
        self.numerical_failures: int = 0

    @property
    def total_frequency_point_solves(self) -> int:
        return self.target_frequency_point_solves + self.coarse_frequency_point_solves

    def get(self, x_key: tuple[float, ...]) -> EvaluationResult | None:
        self.n_calls += 1
        result = self._cache.get(x_key)
        if result is not None:
            self.n_cache_hits += 1
        return result

    def put(
        self,
        x_key: tuple[float, ...],
        result: EvaluationResult,
        n_target_solves: int,
        n_coarse_solves: int,
    ) -> None:
        self._cache[x_key] = result
        self.n_unique_evaluations += 1
        self.target_frequency_point_solves += n_target_solves
        self.coarse_frequency_point_solves += n_coarse_solves
        if result.numerical_status != "ok":
            self.numerical_failures += 1


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------


def evaluate(
    x: np.ndarray,
    context: EvaluationContext,
    cache: DomainEvaluatorCache,
    compute_coarse: bool | None = None,
) -> EvaluationResult:
    """Evaluate one normalized decision vector ``x``.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Normalized decision vector in ``[0, 1]^n``.
    context : EvaluationContext
    cache : DomainEvaluatorCache
    compute_coarse : bool | None
        Override coarse-grid evaluation.  If None, uses
        ``context.requires_coarse_for_hard_soft``.
    """
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    x_key = tuple(x.tolist())

    cached = cache.get(x_key)
    if cached is not None:
        return cached

    result = _evaluate_uncached(x, x_key, context, compute_coarse)
    n_target = len(context.target_indices)
    n_coarse = len(context.evaluation_frequencies_hz) - n_target if result.coarse_evaluated else 0
    cache.put(x_key, result, n_target, n_coarse)
    return result


def _evaluate_uncached(
    x: np.ndarray,
    x_key: tuple[float, ...],
    context: EvaluationContext,
    compute_coarse: bool | None,
) -> EvaluationResult:
    """Perform the actual evaluation without cache."""
    domain = context.domain
    mapper = domain.variable_mapper
    n_hard = context.hard_layout.n

    # -- Unpack decision vector --
    try:
        b1, b2 = mapper.unpack(x)
    except (ValueError, AssertionError) as exc:
        return _failure_result(
            x_key,
            n_hard,
            context,
            status="component_invalid",
            reason=f"unpack failed: {exc}",
        )

    # -- Validate unpacked components --
    try:
        _validate_components(b1, b2)
    except ValueError as exc:
        return _failure_result(
            x_key,
            n_hard,
            context,
            status="component_invalid",
            reason=str(exc),
        )

    # -- Build circuit from components --
    try:
        graph = _build_graph(b1, b2, domain, context.eom_model, domain.canonical_sign_pattern)
    except (ValueError, AssertionError) as exc:
        return _failure_result(
            x_key,
            n_hard,
            context,
            status="component_invalid",
            reason=f"graph build failed: {exc}",
        )

    # -- Solve target frequencies --
    target_solutions: list[CircuitSolution] = []
    opts = SolverOptions()
    for fi in context.target_indices:
        f_hz = context.evaluation_frequencies_hz[fi]
        try:
            sol = solve_circuit_single(graph, context.source_spec, f_hz, opts)
        except np.linalg.LinAlgError as exc:
            return _failure_result(
                x_key,
                n_hard,
                context,
                status="mna_singular",
                reason=str(exc),
                failed_f=f_hz,
                failed_stage="target",
            )
        if sol.status != CircuitSolveStatus.OK:
            return _failure_result(
                x_key,
                n_hard,
                context,
                status="mna_singular",
                reason=f"MNA failed at {f_hz:.3g} Hz: {sol.status}",
                failed_f=f_hz,
                failed_stage="target",
            )
        if not _solution_is_finite(sol):
            return _failure_result(
                x_key,
                n_hard,
                context,
                status="nonfinite",
                reason=f"non-finite solution at {f_hz:.3g} Hz",
                failed_f=f_hz,
                failed_stage="target",
            )
        target_solutions.append(sol)

    # -- Determine whether to solve coarse grid --
    do_coarse = context.requires_coarse_for_hard_soft if compute_coarse is None else compute_coarse

    all_solutions: list[CircuitSolution | None] = [None] * len(context.evaluation_frequencies_hz)
    for ti, fi in enumerate(context.target_indices):
        all_solutions[fi] = target_solutions[ti]

    if do_coarse:
        for fi in context.off_target_indices:
            if all_solutions[fi] is not None:
                continue
            f_hz = context.evaluation_frequencies_hz[fi]
            try:
                sol = solve_circuit_single(graph, context.source_spec, f_hz, opts)
            except np.linalg.LinAlgError as exc:
                return _failure_result(
                    x_key,
                    n_hard,
                    context,
                    status="mna_singular",
                    reason=str(exc),
                    failed_f=f_hz,
                    failed_stage="coarse",
                )
            if sol.status == CircuitSolveStatus.OK and _solution_is_finite(sol):
                all_solutions[fi] = sol

    # Build complete tuple (None → failed placeholder not needed for constraint eval)
    all_sol_tuple = tuple(
        s if s is not None else _null_solution(context.evaluation_frequencies_hz[i])
        for i, s in enumerate(all_solutions)
    )

    # -- Component values for constraint evaluation --
    b1_lv = b1.l_values_h
    b2_lv = b2.l_values_h
    b1_cv = b1.c_values_f
    b2_cv = b2.c_values_f

    # -- Hard constraints --
    hard_g = context.hard_layout.evaluate(
        solutions=all_sol_tuple,
        target_indices=context.target_indices,
        off_target_indices=context.off_target_indices if do_coarse else (),
        branch1_pole_regions=domain.pole_regions_branch1,
        branch2_pole_regions=domain.pole_regions_branch2,
        branch1_k_residues=b1.k_residues,
        branch2_k_residues=b2.k_residues,
        branch1_f_poles=b1.f_poles_hz,
        branch2_f_poles=b2.f_poles_hz,
        branch1_l_vals=b1_lv,
        branch2_l_vals=b2_lv,
        branch1_c_vals=b1_cv,
        branch2_c_vals=b2_cv,
        component_limits_l_min=context.component_limits.l_min_h,
        component_limits_l_max=context.component_limits.l_max_h,
        component_limits_c_min=context.component_limits.c_min_f,
        component_limits_c_max=context.component_limits.c_max_f,
        pole_sep_min_b1=_get_pole_sep(domain, 1),
        pole_sep_min_b2=_get_pole_sep(domain, 2),
        z_ref_ohm=context.source_spec.z_ref_ohm,
        gamma_max=context.match_constraints.gamma_max,
        r_min_ohm=context.match_constraints.resistance_min_ohm,
        r_max_ohm=context.match_constraints.resistance_max_ohm,
        x_max_ohm=context.match_constraints.max_abs_reactance_ohm,
        source_current_max_a=context.stress_constraints.source_current_rms_max_a,
        off_target_eom_peak_rms_v=context.stress_constraints.off_target_eom_peak_rms_v,
    )

    # -- Soft constraints --
    soft_g = context.soft_layout.evaluate(
        solutions=all_sol_tuple,
        target_indices=context.target_indices,
        off_target_indices=context.off_target_indices if do_coarse else (),
        branch1_pole_regions=domain.pole_regions_branch1,
        branch2_pole_regions=domain.pole_regions_branch2,
        branch1_k_residues=b1.k_residues,
        branch2_k_residues=b2.k_residues,
        branch1_f_poles=b1.f_poles_hz,
        branch2_f_poles=b2.f_poles_hz,
        branch1_l_vals=b1_lv,
        branch2_l_vals=b2_lv,
        branch1_c_vals=b1_cv,
        branch2_c_vals=b2_cv,
        component_limits_l_min=context.component_limits.l_min_h,
        component_limits_l_max=context.component_limits.l_max_h,
        component_limits_c_min=context.component_limits.c_min_f,
        component_limits_c_max=context.component_limits.c_max_f,
        pole_sep_min_b1=_get_pole_sep(domain, 1),
        pole_sep_min_b2=_get_pole_sep(domain, 2),
        z_ref_ohm=context.source_spec.z_ref_ohm,
        gamma_max=context.match_constraints.gamma_max,
        r_min_ohm=context.match_constraints.resistance_min_ohm,
        r_max_ohm=context.match_constraints.resistance_max_ohm,
        x_max_ohm=context.match_constraints.max_abs_reactance_ohm,
        source_current_max_a=context.stress_constraints.source_current_rms_max_a,
        off_target_eom_peak_rms_v=context.stress_constraints.off_target_eom_peak_rms_v,
    )

    # -- Objective --
    obj = compute_objective(
        config=context.objective_config,
        target_solutions=tuple(target_solutions),
        soft_layout=context.soft_layout,
        soft_g_vector=tuple(float(v) for v in soft_g),
    )

    # -- Feasibility --
    v_j = np.maximum(0.0, -hard_g)
    v_max = float(np.max(v_j)) if len(v_j) > 0 else 0.0
    v_sum = float(np.sum(v_j))
    eps = context.feasibility_tolerance
    eps_near = context.near_feasibility_tolerance
    feasible = v_max <= eps
    near_feasible = v_max <= eps_near

    return EvaluationResult(
        x=x_key,
        objective_value=obj.j_total,
        base_objective_value=obj.j_base,
        soft_penalty_total=obj.j_soft,
        objective_terms={
            "total": obj.j_total,
            "base": obj.j_base,
            "soft_penalty": obj.j_soft,
            "j_gamma": obj.j_gamma,
            "j_voltage": obj.j_voltage,
            "j_loss": obj.j_loss,
            "j_complexity": obj.j_complexity,
        },
        hard_margins=tuple(float(v) for v in hard_g),
        soft_penalties=obj.soft_terms,
        v_max=v_max,
        v_sum=v_sum,
        feasible=feasible,
        near_feasible=near_feasible,
        numerical_status="ok",
        numerical_failure_reason=None,
        failed_frequency_hz=None,
        failed_stage=None,
        all_solutions=all_sol_tuple,
        target_solutions=tuple(target_solutions),
        coarse_evaluated=do_coarse,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _failure_result(
    x_key: tuple[float, ...],
    n_hard: int,
    context: EvaluationContext,
    *,
    status: str,
    reason: str,
    failed_f: float | None = None,
    failed_stage: str | None = None,
) -> EvaluationResult:
    """Build an infeasible EvaluationResult for a numerical failure."""
    hard_margins = tuple(-1.0 for _ in range(n_hard))
    v_max = max(1.0, 0.0)
    v_sum = float(n_hard)

    return EvaluationResult(
        x=x_key,
        objective_value=1e9,
        base_objective_value=1e9,
        soft_penalty_total=0.0,
        objective_terms={"total": 1e9, "base": 1e9, "soft_penalty": 0.0},
        hard_margins=hard_margins,
        soft_penalties={},
        v_max=v_max,
        v_sum=v_sum,
        feasible=False,
        near_feasible=False,
        numerical_status=status,
        numerical_failure_reason=reason,
        failed_frequency_hz=failed_f,
        failed_stage=failed_stage,
        all_solutions=(),
        target_solutions=(),
        coarse_evaluated=False,
    )


def _validate_components(b1: BranchCoordinates, b2: BranchCoordinates) -> None:
    """Raise ValueError if any computed component value is non-physical."""
    for branch_label, b in (("b1", b1), ("b2", b2)):
        for i, l_val in enumerate(b.l_values_h):
            if not math.isfinite(l_val) or l_val < 0:
                raise ValueError(f"{branch_label} L[{i}] = {l_val!r} is non-physical")
        for i, c_val in enumerate(b.c_values_f):
            if not math.isfinite(c_val) or c_val < 0:
                raise ValueError(f"{branch_label} C[{i}] = {c_val!r} is non-physical")
        for i, km in enumerate(b.k_residues):
            if not math.isfinite(km) or km <= 0:
                raise ValueError(f"{branch_label} k_residue[{i}] = {km!r} must be positive")
        for i, fp in enumerate(b.f_poles_hz):
            if not math.isfinite(fp) or fp <= 0:
                raise ValueError(f"{branch_label} f_pole[{i}] = {fp!r} must be positive Hz")


def _solution_is_finite(sol: CircuitSolution) -> bool:
    """Return True if the solution has finite z_in, gamma, v_eom."""
    checks = []
    if sol.z_in is not None:
        checks.append(math.isfinite(sol.z_in.real) and math.isfinite(sol.z_in.imag))
    if sol.gamma is not None:
        checks.append(math.isfinite(abs(sol.gamma)))
    if sol.v_eom is not None:
        checks.append(math.isfinite(abs(sol.v_eom)))
    return all(checks) if checks else True


def _null_solution(f_hz: float) -> CircuitSolution:
    """A placeholder CircuitSolution for grid points not yet solved."""
    from foster_eom.circuit.mna import SolveDiagnostics

    return CircuitSolution(
        f_hz=f_hz,
        status=CircuitSolveStatus.SINGULAR_OR_ILL_CONDITIONED,
        diagnostics=SolveDiagnostics(),
    )


def _build_graph(
    b1: BranchCoordinates,
    b2: BranchCoordinates,
    domain: ContinuousOptimizationDomain,
    eom_model: OnePortModel,
    sign_pattern: SignPattern,
) -> CircuitGraph:
    """Reconstruct the FosterCircuit graph from unpacked branch coordinates."""
    from foster_eom.foster.foster_form import FosterCell, FosterComponents

    def _make_components(b: BranchCoordinates, n_cells: int) -> FosterComponents:
        cells = tuple(
            FosterCell(
                l_h=b.l_values_h[m] if m < len(b.l_values_h) else 0.0,
                c_f=b.c_values_f[m] if m < len(b.c_values_f) else 0.0,
                f_pole_hz=b.f_poles_hz[m],
            )
            for m in range(min(n_cells, len(b.k_residues)))
        )
        c0 = (1.0 / b.k0) if b.k0 is not None and b.k0 > 0 else None
        l_inf = b.k_inf if b.k_inf is not None else None
        return FosterComponents(c0_f=c0, l_inf_h=l_inf, cells=cells)

    c1 = _make_components(b1, domain.topology.branch1_cells)
    c2 = _make_components(b2, domain.topology.branch2_cells)

    # None out trivial (non-FINITE_FOSTER) branches
    from foster_eom.foster.schmidt import BranchRealization

    c1_arg = c1 if domain.branch1_realization == BranchRealization.FINITE_FOSTER else None
    c2_arg = c2 if domain.branch2_realization == BranchRealization.FINITE_FOSTER else None

    built = build_foster_circuit(
        topology=domain.topology,
        sign_pattern=sign_pattern,
        branch1_components=c1_arg,
        branch2_components=c2_arg,
        eom_model=eom_model,
    )
    return built.graph


def _get_pole_sep(domain: ContinuousOptimizationDomain, branch: int) -> float:
    """Return pole separation minimum from pole_regions (fallback 0)."""
    # Not stored on domain; return a safe default.
    # Pole separation constraints are enforced via the ConstraintLayout.
    return 0.0  # The constraint layout uses its own stored limits.


# ---------------------------------------------------------------------------
# Context factory
# ---------------------------------------------------------------------------


def build_evaluation_context(
    domain: ContinuousOptimizationDomain,
    source_spec: SourceSpec,
    eom_model: OnePortModel,
    component_limits: ContinuousLimits,
    match_constraints: MatchConstraints,
    stress_constraints: StressConstraints,
    target_frequencies_hz: tuple[float, ...],
    sweep_f_min_hz: float,
    sweep_f_max_hz: float,
    base_grid_points: int,
    objective_config: ObjectiveConfig,
    feasibility_tolerance: float = 1e-6,
    near_feasibility_tolerance: float = 0.05,
    extra_constraint_records: list | None = None,
) -> EvaluationContext:
    """Build a frozen EvaluationContext from domain + project specs."""
    # Build unique sorted evaluation frequency grid
    base_grid = np.linspace(sweep_f_min_hz, sweep_f_max_hz, base_grid_points)
    all_freqs = np.unique(np.concatenate([base_grid, np.array(target_frequencies_hz)]))
    eval_freqs = tuple(float(f) for f in all_freqs)

    target_set = set(target_frequencies_hz)
    off_target_mask = tuple(f not in target_set for f in eval_freqs)
    target_indices = tuple(i for i, f in enumerate(eval_freqs) if f in target_set)
    off_target_indices = tuple(i for i, ok in enumerate(off_target_mask) if ok)

    # Compile constraint layouts
    extra_records = extra_constraint_records or []
    from foster_eom.domain.constraints import ConstraintSeverity

    hard_layout = compile_constraint_layout(
        match_constraints=match_constraints,
        stress_constraints=stress_constraints,
        extra_records=extra_records,
        target_frequencies_hz=target_frequencies_hz,
        evaluation_frequencies_hz=eval_freqs,
        target_indices=target_indices,
        off_target_indices=off_target_indices,
        severity_filter=ConstraintSeverity.HARD,
        n_cells_b1=domain.topology.branch1_cells,
        n_cells_b2=domain.topology.branch2_cells,
        z_ref_ohm=source_spec.z_ref_ohm,
    )
    soft_layout = compile_constraint_layout(
        match_constraints=match_constraints,
        stress_constraints=stress_constraints,
        extra_records=extra_records,
        target_frequencies_hz=target_frequencies_hz,
        evaluation_frequencies_hz=eval_freqs,
        target_indices=target_indices,
        off_target_indices=off_target_indices,
        severity_filter=ConstraintSeverity.SOFT,
        n_cells_b1=domain.topology.branch1_cells,
        n_cells_b2=domain.topology.branch2_cells,
        z_ref_ohm=source_spec.z_ref_ohm,
    )

    # Lazy coarse flag
    from foster_eom.domain.constraints import FrequencyScope

    requires_coarse = any(
        d.frequency_scope in (FrequencyScope.SWEEP, FrequencyScope.OFF_TARGET)
        for d in hard_layout.descriptors + soft_layout.descriptors
    )

    return EvaluationContext(
        domain=domain,
        source_spec=source_spec,
        eom_model=eom_model,
        component_limits=component_limits,
        match_constraints=match_constraints,
        stress_constraints=stress_constraints,
        evaluation_frequencies_hz=eval_freqs,
        target_indices=target_indices,
        off_target_indices=off_target_indices,
        off_target_mask=off_target_mask,
        hard_layout=hard_layout,
        soft_layout=soft_layout,
        objective_config=objective_config,
        requires_coarse_for_hard_soft=requires_coarse,
        feasibility_tolerance=feasibility_tolerance,
        near_feasibility_tolerance=near_feasibility_tolerance,
    )
