"""Controller for catalog realization."""

from __future__ import annotations

from foster_eom.catalog.library import ComponentLibrary
from foster_eom.gui.adapter import state_to_spec
from foster_eom.gui.state import ProjectState
from foster_eom.realization.runner import realize


class RealizationCtrl:
    @staticmethod
    def run(state: ProjectState, opt_result: object) -> object:
        """Run realization using a fresh library connection."""
        from foster_eom.optimize.engine import OptimizationResult

        if not isinstance(opt_result, OptimizationResult):
            raise TypeError("Expected OptimizationResult")

        if not state.library_path:
            raise ValueError("No library selected.")

        if opt_result.best_feasible is None and opt_result.near_feasible_best is None:
            raise ValueError("No feasible or near-feasible solution to realize.")

        cand = opt_result.best_feasible or opt_result.near_feasible_best
        if cand is None:
            raise ValueError("Candidate is None")

        project_spec = state_to_spec(state)
        import math

        import numpy as np

        from foster_eom.domain.topology import LOrientation
        from foster_eom.foster.foster_form import FosterCell, FosterComponents
        from foster_eom.foster.network_builder import build_foster_circuit
        from foster_eom.foster.schmidt import BranchRealization
        from foster_eom.foster.seed import _domain_to_internal_pole_spec, generate_seeds
        from foster_eom.foster.sign_search import SignPattern
        from foster_eom.foster.topology_enum import TopologyCandidate
        from foster_eom.models.factory import build_eom_model
        from foster_eom.optimize.domain import group_seeds_into_domains
        from foster_eom.optimize.evaluator import EvaluationResult, build_evaluation_context
        from foster_eom.optimize.objective import ObjectiveConfig
        from foster_eom.optimize.variable_map import BranchCoordinates

        eom_model = build_eom_model(project_spec.eom)
        f_targets_hz = np.array([t.frequency_hz for t in project_spec.frequencies.enabled_targets])
        voltage_targets_rms_v = tuple(
            t.voltage_target_rms_v for t in project_spec.frequencies.enabled_targets
        )

        seed_result = generate_seeds(
            r_match_ohm=project_spec.source.z_source_real_ohm,
            source_spec=project_spec.source,
            eom_model=eom_model,
            f_targets_hz=f_targets_hz,
            topo_spec=project_spec.topology,
            component_limits=project_spec.components.continuous_limits,
        )

        pole_spec_b1 = _domain_to_internal_pole_spec(project_spec.topology.pole_spec_branch1)
        pole_spec_b2 = _domain_to_internal_pole_spec(project_spec.topology.pole_spec_branch2)

        all_domains = group_seeds_into_domains(
            seeds=seed_result.seeds,
            pole_spec_b1=pole_spec_b1,
            pole_spec_b2=pole_spec_b2,
            f_targets_hz=tuple(f_targets_hz),
            component_limits=project_spec.components.continuous_limits,
        )
        domain = next(d for d in all_domains if d.domain_id == cand.domain_id)

        obj_config = ObjectiveConfig(
            z_ref_ohm=project_spec.source.z_ref_ohm,
            w_gamma=project_spec.optimization.objective_weight_gamma,
            w_voltage=project_spec.optimization.objective_weight_voltage,
            w_loss=project_spec.optimization.objective_weight_loss,
            w_complexity=project_spec.optimization.objective_weight_complexity,
            voltage_targets_rms_v=voltage_targets_rms_v,
            voltage_target_weights=tuple(1.0 for _ in voltage_targets_rms_v),
        )

        context = build_evaluation_context(
            domain=domain,
            source_spec=project_spec.source,
            eom_model=eom_model,
            component_limits=project_spec.components.continuous_limits,
            match_constraints=project_spec.matching,
            stress_constraints=project_spec.stress,
            target_frequencies_hz=tuple(f_targets_hz),
            sweep_f_min_hz=project_spec.frequencies.sweep_f_min_hz,
            sweep_f_max_hz=project_spec.frequencies.sweep_f_max_hz,
            base_grid_points=11,
            objective_config=obj_config,
        )

        orient = LOrientation(cand.orientation) if cand.orientation else LOrientation.SHUNT_SERIES
        tc = TopologyCandidate(
            orientation=orient,
            branch1_cells=cand.branch1_cells,
            branch2_cells=cand.branch2_cells,
            branch1_has_c0=cand.branch1_has_c0,
            branch1_has_linf=cand.branch1_has_linf,
            branch2_has_c0=cand.branch2_has_c0,
            branch2_has_linf=cand.branch2_has_linf,
            branch1_n_coefficients=1,
            branch2_n_coefficients=1,
            n_reactive=1,
            structurally_valid=True,
            prune_reason=None,
        )
        br1 = (
            BranchRealization(cand.branch1_realization)
            if cand.branch1_realization
            else BranchRealization.FINITE_FOSTER
        )
        br2 = (
            BranchRealization(cand.branch2_realization)
            if cand.branch2_realization
            else BranchRealization.FINITE_FOSTER
        )
        sp = SignPattern(
            orientation=orient,
            signs=(),
            series_targets=(),
            shunt_targets=(),
            branch1_required_intervals=(),
            branch2_required_intervals=(),
            branch1_realization=br1,
            branch2_realization=br2,
        )

        def get_vals(cells, k_res, f_poles):
            l_h, c_f = [], []
            for i in range(cells):
                k = k_res[i]
                f = f_poles[i]
                w = 2.0 * math.pi * f
                c_f.append(1.0 / k if k > 0 else 0.0)
                l_h.append(k / (w * w) if w > 0 else 0.0)
            return tuple(l_h), tuple(c_f)

        b1_l, b1_c = get_vals(
            cand.branch1_cells, cand.k_residues_branch1, cand.pole_frequencies_branch1_hz
        )
        b1 = BranchCoordinates(
            l_values_h=b1_l,
            c_values_f=b1_c,
            k_residues=tuple(cand.k_residues_branch1),
            f_poles_hz=tuple(cand.pole_frequencies_branch1_hz),
            k0=cand.k0_branch1,
            k_inf=cand.k_inf_branch1,
        )

        b2_l, b2_c = get_vals(
            cand.branch2_cells, cand.k_residues_branch2, cand.pole_frequencies_branch2_hz
        )
        b2 = BranchCoordinates(
            l_values_h=b2_l,
            c_values_f=b2_c,
            k_residues=tuple(cand.k_residues_branch2),
            f_poles_hz=tuple(cand.pole_frequencies_branch2_hz),
            k0=cand.k0_branch2,
            k_inf=cand.k_inf_branch2,
        )

        def make_foster(b, cells_count, has_c0, has_linf):
            if cells_count == 0 and not has_c0 and not has_linf:
                return None
            cells = [
                FosterCell(l_h=b.l_values_h[i], c_f=b.c_values_f[i], f_pole_hz=b.f_poles_hz[i])
                for i in range(cells_count)
            ]
            c0 = (1.0 / b.k0) if has_c0 and b.k0 else None
            return FosterComponents(
                c0_f=c0, l_inf_h=b.k_inf if has_linf else None, cells=tuple(cells)
            )

        c1 = make_foster(b1, cand.branch1_cells, cand.branch1_has_c0, cand.branch1_has_linf)
        c2 = make_foster(b2, cand.branch2_cells, cand.branch2_has_c0, cand.branch2_has_linf)

        eval_res_adapter = EvaluationResult(
            x=tuple(cand.continuous_variables),
            objective_value=cand.objective_terms.get("total", 1e9),
            base_objective_value=cand.base_objective_value,
            soft_penalty_total=cand.soft_penalty_total,
            objective_terms=cand.objective_terms,
            hard_margins=cand.constraint_margins,
            soft_penalties={},
            v_max=cand.v_max,
            v_sum=cand.v_sum,
            feasible=cand.feasible,
            near_feasible=cand.near_feasible,
            numerical_status=cand.numerical_status,
            numerical_failure_reason=None,
            failed_frequency_hz=None,
            failed_stage=None,
            all_solutions=(),
            target_solutions=(),
            coarse_evaluated=True,
        )

        built = build_foster_circuit(tc, sp, c1, c2, eom_model)

        lib = ComponentLibrary(state.library_path)
        try:
            return realize(
                continuous_result=eval_res_adapter,
                context=context,
                b1=b1,
                b2=b2,
                base_graph=built.graph,
                library=lib,
            )
        finally:
            lib.close()
