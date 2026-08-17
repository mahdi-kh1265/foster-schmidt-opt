"""P10 end-to-end integration test.

Scenario:
  Synthetic 1-branch 1-cell Foster circuit: L=10nH, C=10pF.
  Ideal catalog models with tol_frac=0.10 (±10% for L) and 0.05 (±5% for C).

  gamma_max hard constraint is calibrated to be just above nominal |Γ|, so
  that tolerance spread (which shifts resonance) causes actual PHYSICAL_FAIL
  outcomes — NOT merely soft-constraint degradation.

  Specifically: at 10 MHz, nominal L=10nH resonates with C=10pF at f0≈503 MHz
  (ideal LC), so the circuit is far from self-resonance. The gamma constraint
  is set tight enough that ±10% L deviation produces measurable mismatch changes
  that violate the hard threshold.

  Key assertion: nominal_feasible=True, yield_evaluable<1.0, n_physical_fail>0.

  This proves P10 adds information not available from nominal P09 alone.
"""
from __future__ import annotations

import math
from pathlib import Path
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lib(tmp_path: Path):
    from foster_eom.catalog.library import ComponentLibrary

    return ComponentLibrary(tmp_path / "p10_test.fseom.db")


def _add_ideal_ind(lib, value_nom: float, tol_frac: float = 0.10, pn: str | None = None) -> str:
    from foster_eom.catalog.component import (
        ComponentKind,
        LibraryComponent,
        ModelCondition,
        ModelOrigin,
        ModelTier,
    )

    pn = pn or f"L{uuid4().hex[:6]}"
    c = LibraryComponent(
        id=str(uuid4()),
        kind=ComponentKind.INDUCTOR,
        vendor="SynVendor",
        part_number=pn,
        value_nom=value_nom,
        value_tol_frac=tol_frac,
        voltage_max_v=50.0,
    )
    cid = lib.add(c)
    mc = ModelCondition(
        id=str(uuid4()),
        component_id=cid,
        model_tier=ModelTier.IDEAL,
        model_origin=ModelOrigin.IDEAL,
    )
    lib.add_model_condition(mc)
    return cid


def _add_ideal_cap(lib, value_nom: float, tol_frac: float = 0.05, pn: str | None = None) -> str:
    from foster_eom.catalog.component import (
        ComponentKind,
        LibraryComponent,
        ModelCondition,
        ModelOrigin,
        ModelTier,
    )

    pn = pn or f"C{uuid4().hex[:6]}"
    c = LibraryComponent(
        id=str(uuid4()),
        kind=ComponentKind.CAPACITOR,
        vendor="SynVendor",
        part_number=pn,
        value_nom=value_nom,
        value_tol_frac=tol_frac,
        voltage_max_v=50.0,
    )
    cid = lib.add(c)
    mc = ModelCondition(
        id=str(uuid4()),
        component_id=cid,
        model_tier=ModelTier.IDEAL,
        model_origin=ModelOrigin.IDEAL,
    )
    lib.add_model_condition(mc)
    return cid


def _make_minimal_eval_result(
    feasible: bool = True,
    objective: float = 0.3,
    v_max: float = 0.0,
    hard_margins: tuple = (0.5,),
):
    from foster_eom.optimize.evaluator import EvaluationResult

    return EvaluationResult(
        x=(),
        objective_value=objective,
        base_objective_value=objective,
        soft_penalty_total=0.0,
        objective_terms={"total": objective},
        hard_margins=hard_margins,
        soft_penalties={},
        v_max=v_max,
        v_sum=v_max,
        feasible=feasible,
        near_feasible=feasible or v_max <= 0.05,
        numerical_status="ok",
        numerical_failure_reason=None,
        failed_frequency_hz=None,
        failed_stage=None,
        all_solutions=(),
        target_solutions=(),
        coarse_evaluated=False,
    )


def _build_synthetic_combo(lib, l_cid: str, c_cid: str) -> object:
    """Build a minimal CatalogCombo with one L and one C slot (parallel RLC)."""
    from foster_eom.catalog.component import ModelTier
    from foster_eom.realization.result import CatalogCombo
    from foster_eom.realization.spec import NeighborhoodEntry

    ne_l = NeighborhoodEntry(
        component_id=l_cid,
        model_condition_id="mc_l",
        vendor="SynVendor",
        part_number="L_TEST",
        value_nom=100e-9,   # 100 nH to match parallel RLC circuit
        value_tol_frac=0.10,
        model_tier=ModelTier.IDEAL,
        log_ratio=0.0,
    )
    ne_c = NeighborhoodEntry(
        component_id=c_cid,
        model_condition_id="mc_c",
        vendor="SynVendor",
        part_number="C_TEST",
        value_nom=250e-12,  # 250 pF to match parallel RLC circuit
        value_tol_frac=0.05,
        model_tier=ModelTier.IDEAL,
        log_ratio=0.0,
    )
    eval_result = _make_minimal_eval_result(
        feasible=True,
        objective=0.05,  # near zero (resonance: gamma≈0)
        v_max=0.0,
        hard_margins=(0.5,),
    )
    return CatalogCombo(
        slot_entries={"b1_L1": ne_l, "b1_C1": ne_c},
        eval_result=eval_result,
        deb_key=(False, 0.0, 0.0, 0.05),
        verify_passed=True,
    )



def _build_synthetic_context_and_graph(tmp_path: Path):
    """Build a minimal MNA EvaluationContext and CircuitGraph.

    Circuit: parallel L || C || R_load (shunt network), evaluated at the
    LC resonance frequency f_res = 1/(2π√(LC)).

    At resonance, Z_in = R_load = 50 Ω → gamma = 0. Nominal is perfectly
    matched.  With ±10%L or ±5%C tolerance, the resonance shifts, gamma rises.

    gamma_max = 0.07 is calibrated so that extreme tolerance draws violate
    the hard constraint, producing PHYSICAL_FAIL outcomes while the nominal
    passes.
    """
    from unittest.mock import MagicMock

    from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
    from foster_eom.circuit.mna import SourceSpec
    from foster_eom.domain.constraints import MatchConstraints, StressConstraints
    from foster_eom.domain.source import SourceMode
    from foster_eom.optimize.constraints import ConstraintSeverity, compile_constraint_layout

    L_nom = 100e-9
    C_nom = 250e-12
    R_load = 50.0

    # Resonance frequency
    f_res = 1.0 / (2.0 * math.pi * math.sqrt(L_nom * C_nom))  # ~31.83 MHz

    # Target and evaluation frequencies centred at resonance
    target_hz = (f_res,)
    eval_hz = (f_res * 0.8, f_res, f_res * 1.2)
    target_indices = (1,)
    off_target_indices = (0, 2)

    # Parallel L || C || R_load circuit (shunt to ground)
    graph = CircuitGraph(
        ground_node_id="gnd",
        input_port=Port("n_in", "gnd"),
        eom_element_id="R_load",
    )
    graph.add_node(Node(id="n_in", is_ground=False))
    graph.add_node(Node(id="gnd", is_ground=True))
    graph.add_element(Element(id="b1_L1", kind=ElementKind.INDUCTOR,
                              node_pos="n_in", node_neg="gnd", value=L_nom))
    graph.add_element(Element(id="b1_C1", kind=ElementKind.CAPACITOR,
                              node_pos="n_in", node_neg="gnd", value=C_nom))
    graph.add_element(Element(id="R_load", kind=ElementKind.RESISTOR,
                              node_pos="n_in", node_neg="gnd", value=R_load))

    z_ref = 50.0

    # gamma_max = 0.07: tight enough to fail for ±10%L draws (which reach ~0.11-0.14)
    # while nominal (gamma≈0) passes easily.
    gamma_max_calibrated = 0.07

    match_c = MatchConstraints(
        gamma_max=gamma_max_calibrated,
        resistance_max_ohm=1e6,
        max_abs_reactance_ohm=1e6,
    )
    stress_c = StressConstraints(
        source_current_rms_max_a=10.0,
        off_target_eom_peak_rms_v=100.0,
    )

    hard_layout = compile_constraint_layout(
        match_constraints=match_c,
        stress_constraints=stress_c,
        extra_records=[],
        target_frequencies_hz=target_hz,
        evaluation_frequencies_hz=eval_hz,
        target_indices=target_indices,
        off_target_indices=off_target_indices,
        severity_filter=ConstraintSeverity.HARD,
        n_cells_b1=1,
        n_cells_b2=0,
        z_ref_ohm=z_ref,
    )
    soft_layout = compile_constraint_layout(
        match_constraints=match_c,
        stress_constraints=stress_c,
        extra_records=[],
        target_frequencies_hz=target_hz,
        evaluation_frequencies_hz=eval_hz,
        target_indices=target_indices,
        off_target_indices=off_target_indices,
        severity_filter=ConstraintSeverity.SOFT,
        n_cells_b1=1,
        n_cells_b2=0,
        z_ref_ohm=z_ref,
    )

    source_spec = SourceSpec(
        mode=SourceMode.THEVENIN,
        z_source_real_ohm=z_ref,
        z_ref_ohm=z_ref,
        thevenin_vrms=1.0,
    )

    from foster_eom.optimize.evaluator import EvaluationContext

    ctx = MagicMock(spec=EvaluationContext)
    ctx.evaluation_frequencies_hz = eval_hz
    ctx.target_indices = target_indices
    ctx.off_target_indices = off_target_indices
    ctx.hard_layout = hard_layout
    ctx.soft_layout = soft_layout
    ctx.source_spec = source_spec
    ctx.eom_model = None
    ctx.p06_sweep_band_hz = None
    ctx.feasibility_tolerance = 1e-6
    ctx.near_feasibility_tolerance = 0.05
    ctx.requires_coarse_for_hard_soft = True

    domain_mock = MagicMock()
    domain_mock.topology.branch1_cells = 1
    domain_mock.topology.branch2_cells = 0
    domain_mock.pole_regions_branch1 = (None,)
    domain_mock.pole_regions_branch2 = ()
    ctx.domain = domain_mock

    cl_mock = MagicMock()
    cl_mock.l_min_h = 1e-12
    cl_mock.l_max_h = 1e-3
    cl_mock.c_min_f = 1e-15
    cl_mock.c_max_f = 1e-6
    ctx.component_limits = cl_mock
    ctx.match_constraints = match_c
    ctx.stress_constraints = stress_c

    from foster_eom.optimize.objective import ObjectiveConfig

    obj_cfg = ObjectiveConfig(
        z_ref_ohm=z_ref,
        w_gamma=1.0,
        w_voltage=0.0,
        w_loss=0.0,
        w_complexity=0.0,
        eom_element_id="R_load",
        n_reactive=2,  # L + C
    )
    ctx.objective_config = obj_cfg

    return ctx, graph, gamma_max_calibrated


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


class TestP10EndToEnd:
    """End-to-end P10 robustness test with synthetic circuit."""

    def test_nominal_passes_tolerance_spreads_produce_hard_failures(
        self, tmp_path: Path
    ) -> None:
        """Core P10 value proposition test.

        The nominal P09 combo is feasible.
        With ±10%L / ±5%C tolerance spread, some samples violate the
        hard gamma_max constraint → PHYSICAL_FAIL outcomes.
        P10 yields yield_evaluable < 1.0 and n_physical_fail > 0.
        """
        from foster_eom.robustness.runner import run_robustness
        from foster_eom.robustness.sampler import RobustnessSpec

        with _make_lib(tmp_path) as lib:
            l_cid = _add_ideal_ind(lib, 100e-9, tol_frac=0.10)
            c_cid = _add_ideal_cap(lib, 250e-12, tol_frac=0.05)

            combo = _build_synthetic_combo(lib, l_cid, c_cid)
            ctx, graph, gamma_max = _build_synthetic_context_and_graph(tmp_path)

            spec = RobustnessSpec(
                n_samples=200,
                seed=42,
                method="random",
                p06_diagnostic="none",
                n_retry=1,
            )

            result = run_robustness(
                combo=combo,
                base_graph=graph,
                context=ctx,
                library=lib,
                spec=spec,
            )

        # 1. Nominal is feasible (sanity)
        assert result.nominal_feasible is True

        # 2. Tolerance spread produces at least some PHYSICAL_FAIL outcomes
        ys = result.yield_stats
        assert ys.n_physical_fail > 0, (
            "Expected at least some PHYSICAL_FAIL samples from ±10%L / ±5%C tolerance "
            f"with gamma_max={gamma_max:.4f}. Got n_physical_fail=0. "
            "P10 is not providing information beyond the nominal result."
        )

        # 3. yield_evaluable < 1.0
        assert ys.yield_evaluable < 1.0, (
            f"yield_evaluable={ys.yield_evaluable:.4f} must be <1.0 when some samples fail."
        )

        # 4. yield bounds bracket yield_evaluable
        assert ys.yield_lower_bound <= ys.yield_evaluable, (
            f"yield_lower_bound={ys.yield_lower_bound:.4f} "
            f"> yield_evaluable={ys.yield_evaluable:.4f}"
        )
        assert ys.yield_evaluable <= ys.yield_upper_bound, (
            f"yield_evaluable={ys.yield_evaluable:.4f} "
            f"> yield_upper_bound={ys.yield_upper_bound:.4f}"
        )

        # 5. Wilson CI present for iid random
        assert ys.ci_method == "wilson"
        assert ys.ci_lo is not None and ys.ci_hi is not None
        assert 0.0 <= ys.ci_lo <= ys.ci_hi <= 1.0

        # 6. OAT sensitivity non-empty; dominant slot has positive sensitivity
        assert len(result.oat_sensitivity) > 0
        assert result.oat_sensitivity[0].sensitivity_J >= 0.0

        # 7. failure_association non-empty (since there are failures)
        assert len(result.failure_association) > 0

        # 8. No non-stochastic slots (all slots have tol_frac)
        assert result.non_stochastic_slots == []

        # 9. distributions.v_max.p95 > 0 (worst-case samples are infeasible)
        assert result.distributions.v_max.p95 > 0.0

        # 10. Determinism: same seed → identical sample draws
        with _make_lib(tmp_path / "det") as lib2:
            l_cid2 = _add_ideal_ind(lib2, 100e-9, tol_frac=0.10)
            c_cid2 = _add_ideal_cap(lib2, 250e-12, tol_frac=0.05)
            combo2 = _build_synthetic_combo(lib2, l_cid2, c_cid2)
            ctx2, graph2, _ = _build_synthetic_context_and_graph(tmp_path / "det")

            result2 = run_robustness(
                combo=combo2,
                base_graph=graph2,
                context=ctx2,
                library=lib2,
                spec=spec,
            )

        # Same draws → same outcomes
        assert len(result.samples) == len(result2.samples)
        for s1, s2 in zip(result.samples, result2.samples):
            for eid in s1.draw:
                assert s1.draw[eid] == pytest.approx(s2.draw.get(eid, float("nan")), rel=1e-10)

        # 11. No measured_residual notes for ideal-tier slots
        for note in result.perturbation_notes:
            assert "measured_residual" not in note or "ideal" in note, (
                f"Unexpected measured_residual note for ideal slot: {note}"
            )

    def test_non_stochastic_slot_warned_and_deterministic(self, tmp_path: Path) -> None:
        """Slot with no tol_frac is held at nominal and generates a warning."""
        import warnings as _warnings

        from foster_eom.robustness.uncertainty import build_slot_uncertainties

        with _make_lib(tmp_path / "nonstoch") as lib:
            # Add cap with NO tolerance
            from foster_eom.catalog.component import (
                ComponentKind,
                LibraryComponent,
                ModelCondition,
                ModelOrigin,
                ModelTier,
            )

            pn = "C_NOTOL"
            c = LibraryComponent(
                id=str(uuid4()),
                kind=ComponentKind.CAPACITOR,
                vendor="SynVendor",
                part_number=pn,
                value_nom=10e-12,
                value_tol_frac=None,  # ← no tolerance
                voltage_max_v=50.0,
            )
            cid = lib.add(c)
            mc = ModelCondition(
                id=str(uuid4()),
                component_id=cid,
                model_tier=ModelTier.IDEAL,
                model_origin=ModelOrigin.IDEAL,
            )
            lib.add_model_condition(mc)

            from foster_eom.realization.result import CatalogCombo
            from foster_eom.realization.spec import NeighborhoodEntry

            ne = NeighborhoodEntry(
                component_id=cid,
                model_condition_id="mc_notol",
                vendor="SynVendor",
                part_number=pn,
                value_nom=10e-12,
                value_tol_frac=None,
                model_tier=ModelTier.IDEAL,
                log_ratio=0.0,
            )
            eval_result = _make_minimal_eval_result()
            combo = CatalogCombo(
                slot_entries={"b1_C1": ne},
                eval_result=eval_result,
                deb_key=(False, 0.0, 0.0, 0.3),
            )

            with _warnings.catch_warnings(record=True) as w:
                _warnings.simplefilter("always")
                sus = build_slot_uncertainties(combo)
                assert len(w) == 1
                assert "no uncertainty data" in str(w[0].message)

            su = sus[0]
            assert not su.is_stochastic
            assert su.has_tol_frac is False
            assert su.catalog_tol_frac is None

    def test_yields_statistical_meaning_ci_width(self, tmp_path: Path) -> None:
        """For N=200, yield≈1.0: Wilson CI width should be reasonably small."""
        from foster_eom.robustness.stats import wilson_ci

        lo, hi = wilson_ci(196, 200, 0.95)
        width = hi - lo
        # At yield=0.98, N=200: CI ≈ ±1.9% → width < 0.05
        assert width < 0.05, f"CI width {width:.4f} unexpectedly large"
