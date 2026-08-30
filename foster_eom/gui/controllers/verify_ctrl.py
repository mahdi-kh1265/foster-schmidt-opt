"""Controller for verification."""

from __future__ import annotations

from foster_eom.analysis.q_factor import compute_q_metrics
from foster_eom.analysis.stress import compute_stress
from foster_eom.analysis.sweep import compute_adaptive_sweep
from foster_eom.gui.adapter import state_to_spec
from foster_eom.gui.state import ProjectState


class VerifyCtrl:
    @staticmethod
    def run(state: ProjectState, opt_result: object) -> tuple[object, list[object], object]:
        """Run verification suite and return (sweep_result, q_metrics, stress_summary)."""
        from foster_eom.optimize.engine import OptimizationResult

        if not isinstance(opt_result, OptimizationResult):
            raise TypeError("Expected OptimizationResult")

        if opt_result.best_feasible is None and opt_result.near_feasible_best is None:
            raise ValueError("No feasible or near-feasible solution to verify.")

        cand = opt_result.best_feasible or opt_result.near_feasible_best
        if cand is None:
            raise ValueError("Candidate is None")

        spec = state_to_spec(state)

        # Build graph from CandidateResult for P06
        import math

        from foster_eom.domain.topology import LOrientation
        from foster_eom.foster.foster_form import FosterCell, FosterComponents
        from foster_eom.foster.network_builder import build_foster_circuit
        from foster_eom.foster.schmidt import BranchRealization
        from foster_eom.foster.sign_search import SignPattern
        from foster_eom.foster.topology_enum import TopologyCandidate
        from foster_eom.models.factory import build_eom_model

        def _make_components(
            cells_count: int,
            has_c0: bool,
            has_linf: bool,
            k_residues: list[float],
            k0: float | None,
            k_inf: float | None,
            f_poles_hz: list[float],
        ) -> FosterComponents | None:
            if cells_count == 0 and not has_c0 and not has_linf:
                return None
            cells = []
            for i in range(cells_count):
                k = k_residues[i]
                f = f_poles_hz[i]
                w = 2.0 * math.pi * f
                c_f = 1.0 / k if k > 0 else 0.0
                l_h = k / (w * w) if w > 0 else 0.0
                cells.append(FosterCell(l_h=l_h, c_f=c_f, f_pole_hz=f))
            c0_f = (1.0 / k0) if has_c0 and k0 is not None and k0 > 0 else None
            return FosterComponents(c0_f=c0_f, l_inf_h=k_inf if has_linf else None, cells=tuple(cells))

        c1 = _make_components(
            cand.branch1_cells, cand.branch1_has_c0, cand.branch1_has_linf,
            cand.k_residues_branch1, cand.k0_branch1, cand.k_inf_branch1, cand.pole_frequencies_branch1_hz
        )
        c2 = _make_components(
            cand.branch2_cells, cand.branch2_has_c0, cand.branch2_has_linf,
            cand.k_residues_branch2, cand.k0_branch2, cand.k_inf_branch2, cand.pole_frequencies_branch2_hz
        )

        br1 = BranchRealization(cand.branch1_realization) if cand.branch1_realization else BranchRealization.FINITE_FOSTER
        br2 = BranchRealization(cand.branch2_realization) if cand.branch2_realization else BranchRealization.FINITE_FOSTER

        # P05 always uses orientation "shunt_series" or "series_shunt".
        # If empty, default to SHUNT_SERIES for safety.
        orient = LOrientation(cand.orientation) if cand.orientation else LOrientation.SHUNT_SERIES
        sp = SignPattern(
            orientation=orient,
            signs=(),
            series_targets=(),
            shunt_targets=(),
            branch1_required_intervals=(),
            branch2_required_intervals=(),
            branch1_realization=br1,
            branch2_realization=br2
        )

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
            prune_reason=None
        )
        eom_model = build_eom_model(spec.eom)
        built = build_foster_circuit(
            topology=tc,
            sign_pattern=sp,
            branch1_components=c1,
            branch2_components=c2,
            eom_model=eom_model
        )

        from foster_eom.analysis.sweep import SweepSpec
        target_hz = tuple(t.frequency_hz for t in spec.frequencies.targets)
        sweep_spec = SweepSpec(
            f_min_hz=spec.frequencies.sweep_f_min_hz,
            f_max_hz=spec.frequencies.sweep_f_max_hz,
        )

        sweep_res = compute_adaptive_sweep(
            graph=built.graph,
            source_spec=spec.source,
            eom_model=eom_model,
            spec=sweep_spec,
            target_hz=target_hz
        )
        stress_res = compute_stress(
            graph=built.graph,
            source_spec=spec.source,
            sweep_result=sweep_res,
            target_hz=target_hz
        )
        q_metrics = compute_q_metrics(
            sweep_result=sweep_res,
            target_hz=target_hz,
            graph=built.graph,
            source_spec=spec.source,
            usable_bw_eta=spec.q_bandwidth.voltage_fraction_for_bandwidth
        )

        from foster_eom.circuit.solve import solve_circuit
        sols = solve_circuit(built.graph, spec.source, sweep_res.frequencies_hz)
        z_in_sweep = [sol.z_in for sol in sols]

        return sweep_res, q_metrics, stress_res, z_in_sweep
