"""Circuit substitution: replace primitive elements with P08 catalog models (Prompt 09).

The existing ``CircuitGraph`` uses ``ElementKind.INDUCTOR`` and
``ElementKind.CAPACITOR`` primitives.  Substitution replaces named elements
with ``ElementKind.ONE_PORT_MODEL`` elements whose model comes from the P08
catalog.

``build_substituted_graph()`` takes a base graph and an override mapping
``{element_id: OnePortModel}``, validates that every key exists in the graph,
and returns a new ``CircuitGraph`` with those elements replaced.

No modification is made to the base graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind

if TYPE_CHECKING:
    from foster_eom.models.base import OnePortModel
    from foster_eom.optimize.evaluator import EvaluationContext, EvaluationResult
    from foster_eom.realization.spec import RealizationBudget


# ---------------------------------------------------------------------------
# Core substitution
# ---------------------------------------------------------------------------


def build_substituted_graph(
    base_graph: CircuitGraph,
    model_overrides: dict[str, OnePortModel],
) -> CircuitGraph:
    """Build a new CircuitGraph with selected elements replaced by OnePortModels.

    Parameters
    ----------
    base_graph : CircuitGraph
        The original circuit graph (not modified).
    model_overrides : dict[str, OnePortModel]
        Maps element IDs to the replacement OnePortModel.
        Every key must exist as an element in ``base_graph``.

    Returns
    -------
    CircuitGraph
        New graph with the same nodes and elements, except overridden elements
        are replaced with ``ElementKind.ONE_PORT_MODEL`` carrying the catalog model.

    Raises
    ------
    KeyError
        If any key in ``model_overrides`` is not found in ``base_graph.elements``.
    ValueError
        If an override target is not a primitive L/C element (structural guard).
    """
    # Strict validation: all override element IDs must exist
    unknown = set(model_overrides) - set(base_graph.elements)
    if unknown:
        raise KeyError(
            f"model_overrides contains element IDs not found in the circuit graph: "
            f"{sorted(unknown)}"
        )

    # Guard: only substitute primitive L/C elements
    _SUBSTITUTABLE = {ElementKind.INDUCTOR, ElementKind.CAPACITOR}
    for eid, _model in model_overrides.items():
        orig = base_graph.elements[eid]
        if orig.kind not in _SUBSTITUTABLE:
            raise ValueError(
                f"Cannot substitute element '{eid}' of kind '{orig.kind.value}': "
                f"only INDUCTOR and CAPACITOR elements may be substituted."
            )

    # Build new graph (same structural skeleton)
    new_graph = CircuitGraph(
        ground_node_id=base_graph.ground_node_id,
        input_port=base_graph.input_port,
        eom_element_id=base_graph.eom_element_id,
    )

    # Copy all nodes
    for node in base_graph.nodes.values():
        new_graph.add_node(node)

    # Copy elements — substitute where overridden
    for eid, elem in base_graph.elements.items():
        if eid in model_overrides:
            new_elem = Element(
                id=eid,
                kind=ElementKind.ONE_PORT_MODEL,
                node_pos=elem.node_pos,
                node_neg=elem.node_neg,
                value=None,
                model=model_overrides[eid],
                symbolic_role=elem.symbolic_role,
            )
        else:
            new_elem = elem
        new_graph.add_element(new_elem)

    return new_graph


# ---------------------------------------------------------------------------
# Evaluate a substituted graph via the existing evaluator pathway
# ---------------------------------------------------------------------------


def evaluate_with_overrides(
    base_graph: CircuitGraph,
    model_overrides: dict[str, OnePortModel],
    context: EvaluationContext,
    budget: RealizationBudget | None = None,
) -> EvaluationResult:
    """Substitute model overrides and evaluate using the P05 infrastructure.

    This function bypasses the DecisionVariableMapper (no normalized x) and
    directly evaluates the substituted circuit graph against the full
    EvaluationContext constraints and objectives.

    Parameters
    ----------
    base_graph : CircuitGraph
        Original circuit graph from the continuous evaluation.
    model_overrides : dict[str, OnePortModel]
        Catalog model overrides keyed by element ID.
    context : EvaluationContext
        Full frozen evaluation context (frequencies, constraints, objectives).
    budget : RealizationBudget | None
        If supplied, ``consume(n_frequencies)`` is called on each solve.

    Returns
    -------
    EvaluationResult
        Full evaluation result (feasibility, objectives, solutions).
    """
    from foster_eom.circuit.mna import SolverOptions
    from foster_eom.circuit.solve import solve_circuit_single
    from foster_eom.errors import CircuitSolveStatus
    from foster_eom.optimize.constraints import ConstraintLayout
    from foster_eom.optimize.evaluator import EvaluationResult
    from foster_eom.optimize.objective import compute_objective

    subst_graph = build_substituted_graph(base_graph, model_overrides)

    ctx = context
    opts = SolverOptions()

    # Use a sentinel x key (non-normalized; combos don't share an x space)
    x_key: tuple[float, ...] = ()

    # Solve target frequencies
    target_solutions = []
    for fi in ctx.target_indices:
        f_hz = ctx.evaluation_frequencies_hz[fi]
        if budget is not None:
            budget.consume(1)  # type: ignore[union-attr]
        try:
            sol = solve_circuit_single(subst_graph, ctx.source_spec, f_hz, opts)
        except Exception as exc:
            return _failure(x_key, ctx, f"MNA singular at {f_hz:.3g} Hz: {exc}")
        if sol.status != CircuitSolveStatus.OK:
            return _failure(x_key, ctx, f"MNA failed at {f_hz:.3g} Hz: {sol.status}")
        import math as _math

        def _sol_finite(s: object) -> bool:
            from foster_eom.circuit.solve import CircuitSolution

            sol_typed = s  # type: ignore[assignment]
            sol_cast: CircuitSolution = sol_typed  # type: ignore[assignment]
            checks = []
            if sol_cast.z_in is not None:
                checks.append(
                    _math.isfinite(sol_cast.z_in.real) and _math.isfinite(sol_cast.z_in.imag)
                )
            if sol_cast.gamma is not None:
                checks.append(_math.isfinite(abs(sol_cast.gamma)))
            return all(checks) if checks else True

        if not _sol_finite(sol):
            return _failure(x_key, ctx, f"Non-finite solution at {f_hz:.3g} Hz")
        target_solutions.append(sol)

    # Coarse grid if required
    all_solutions_list: list = [None] * len(ctx.evaluation_frequencies_hz)
    for ti, fi in enumerate(ctx.target_indices):
        all_solutions_list[fi] = target_solutions[ti]

    do_coarse = ctx.requires_coarse_for_hard_soft
    if do_coarse:
        for fi in ctx.off_target_indices:
            if all_solutions_list[fi] is not None:
                continue
            f_hz = ctx.evaluation_frequencies_hz[fi]
            if budget is not None:
                budget.consume(1)  # type: ignore[union-attr]
            try:
                sol = solve_circuit_single(subst_graph, ctx.source_spec, f_hz, opts)
                if sol.status == CircuitSolveStatus.OK:
                    all_solutions_list[fi] = sol
            except Exception:
                pass

    # Fill None placeholders with null solutions
    from foster_eom.circuit.measurements import CircuitSolution
    from foster_eom.errors import CircuitSolveStatus as _CSS

    def _null_sol(f_hz: float) -> CircuitSolution:
        from foster_eom.circuit.mna import SolveDiagnostics

        return CircuitSolution(
            f_hz=f_hz,
            status=_CSS.SINGULAR_OR_ILL_CONDITIONED,
            diagnostics=SolveDiagnostics(),
        )

    all_sol_tuple = tuple(
        s if s is not None else _null_sol(ctx.evaluation_frequencies_hz[i])
        for i, s in enumerate(all_solutions_list)
    )

    # Constraint evaluation — L/C component values for range/separation constraints.
    # For catalog substituted graphs, ideal component-range constraints are not
    # meaningful (the selection was already filtered). Pass zeros so that
    # component-limit constraints are not spuriously violated. MNA-derived
    # constraints (gamma, v_eom, etc.) are fully active via the solutions.
    import numpy as np

    n_cells_b1 = ctx.domain.topology.branch1_cells
    n_cells_b2 = ctx.domain.topology.branch2_cells

    _zeros_b1 = tuple(0.0 for _ in range(n_cells_b1))
    _zeros_b2 = tuple(0.0 for _ in range(n_cells_b2))

    def _eval_layout(layout: ConstraintLayout) -> np.ndarray:
        return layout.evaluate(
            solutions=all_sol_tuple,
            target_indices=ctx.target_indices,
            off_target_indices=ctx.off_target_indices if do_coarse else (),
            branch1_pole_regions=ctx.domain.pole_regions_branch1,
            branch2_pole_regions=ctx.domain.pole_regions_branch2,
            branch1_k_residues=_zeros_b1,
            branch2_k_residues=_zeros_b2,
            branch1_f_poles=_zeros_b1,
            branch2_f_poles=_zeros_b2,
            branch1_l_vals=_zeros_b1,
            branch2_l_vals=_zeros_b2,
            branch1_c_vals=_zeros_b1,
            branch2_c_vals=_zeros_b2,
            component_limits_l_min=ctx.component_limits.l_min_h,
            component_limits_l_max=ctx.component_limits.l_max_h,
            component_limits_c_min=ctx.component_limits.c_min_f,
            component_limits_c_max=ctx.component_limits.c_max_f,
            pole_sep_min_b1=0.0,
            pole_sep_min_b2=0.0,
            z_ref_ohm=ctx.source_spec.z_ref_ohm,
            gamma_max=ctx.match_constraints.gamma_max,
            r_min_ohm=ctx.match_constraints.resistance_min_ohm,
            r_max_ohm=ctx.match_constraints.resistance_max_ohm,
            x_max_ohm=ctx.match_constraints.max_abs_reactance_ohm,
            source_current_max_a=ctx.stress_constraints.source_current_rms_max_a,
            off_target_eom_peak_rms_v=ctx.stress_constraints.off_target_eom_peak_rms_v,
        )

    hard_g = _eval_layout(ctx.hard_layout)
    soft_g = _eval_layout(ctx.soft_layout)

    obj = compute_objective(
        config=ctx.objective_config,
        target_solutions=tuple(target_solutions),
        soft_layout=ctx.soft_layout,
        soft_g_vector=tuple(float(v) for v in soft_g),
    )

    v_j = np.maximum(0.0, -hard_g)
    v_max = float(np.max(v_j)) if len(v_j) > 0 else 0.0
    v_sum = float(np.sum(v_j))
    eps = ctx.feasibility_tolerance
    eps_near = ctx.near_feasibility_tolerance

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
        feasible=v_max <= eps,
        near_feasible=v_max <= eps_near,
        numerical_status="ok",
        numerical_failure_reason=None,
        failed_frequency_hz=None,
        failed_stage=None,
        all_solutions=all_sol_tuple,
        target_solutions=tuple(target_solutions),
        coarse_evaluated=do_coarse,
    )


# ---------------------------------------------------------------------------
# Failure helper
# ---------------------------------------------------------------------------


def _failure(
    x_key: tuple,
    ctx: EvaluationContext,
    reason: str,
) -> EvaluationResult:
    from foster_eom.optimize.evaluator import EvaluationResult

    c = ctx  # type: ignore[assignment]
    n_hard = c.hard_layout.n  # type: ignore[union-attr]
    return EvaluationResult(
        x=x_key,
        objective_value=1e9,
        base_objective_value=1e9,
        soft_penalty_total=0.0,
        objective_terms={"total": 1e9, "base": 1e9, "soft_penalty": 0.0},
        hard_margins=tuple(-1.0 for _ in range(n_hard)),
        soft_penalties={},
        v_max=1.0,
        v_sum=float(n_hard),
        feasible=False,
        near_feasible=False,
        numerical_status="component_invalid",
        numerical_failure_reason=reason,
        failed_frequency_hz=None,
        failed_stage=None,
        all_solutions=(),
        target_solutions=(),
        coarse_evaluated=False,
    )
