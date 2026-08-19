import numpy as np

from foster_eom.circuit.graph import CircuitGraph, ElementKind
from foster_eom.circuit.mna import assemble_mna, solve_mna_factorized
from foster_eom.errors import CircuitSolveStatus
from foster_eom.optimize.evaluator import (
    EvaluationContext,
    _build_graph,
    _get_pole_sep,
    _validate_components,
)
from foster_eom.optimize.variable_map import DecisionVariableMapper
from foster_eom.sensitivities.constraints import compute_layout_jacobian
from foster_eom.sensitivities.direct import compute_direct_state_sensitivities
from foster_eom.sensitivities.foster_mapping import dC0_dx, dCm_dxkm, dLinf_dx, dLm_dxfp, dLm_dxkm
from foster_eom.sensitivities.objective_gradient import (
    ObjectiveGradientResult,
    compute_objective_gradient,
)
from foster_eom.sensitivities.observables import (
    compute_observable_derivatives,
)
from foster_eom.sensitivities.off_target import compute_v_eom_adjoint_gradient
from foster_eom.sensitivities.stamps import stamp_capacitor_derivative, stamp_inductor_derivative


def build_y_p_list(
    graph: CircuitGraph,
    node_map: dict[str, int],
    mapper: DecisionVariableMapper,
    x: np.ndarray,
    f_hz: float,
) -> list[np.ndarray]:
    """Build the Y_p derivative matrices for each of the K parameters."""
    n_nodes = len(node_map)
    b1, b2 = mapper.unpack(x)

    # 1. Base element stamps
    dy_dc_elem = {}
    dy_dl_elem = {}

    for elem_id, elem in graph.elements.items():
        if elem.kind == ElementKind.CAPACITOR:
            y_base = np.zeros((n_nodes, n_nodes), dtype=np.complex128)
            stamp_capacitor_derivative(y_base, elem, node_map, graph.ground_node_id, f_hz)
            dy_dc_elem[elem_id] = y_base
        elif elem.kind == ElementKind.INDUCTOR:
            y_base = np.zeros((n_nodes, n_nodes), dtype=np.complex128)
            stamp_inductor_derivative(y_base, elem, node_map, graph.ground_node_id, f_hz)
            dy_dl_elem[elem_id] = y_base

    # 2. Map chain-rule parameter derivatives to Y_p_k
    y_p_list = []
    for desc in mapper.descriptors:
        y_p_k = np.zeros((n_nodes, n_nodes), dtype=np.complex128)
        phys = b1 if desc.branch == 1 else b2
        branch_str = f"b{desc.branch}"

        if desc.var_type == "logk0":
            elem_id = f"{branch_str}_C0"
            if elem_id in dy_dc_elem:
                c0_f = graph.elements[elem_id].value
                dc = dC0_dx(float(c0_f), desc) if c0_f is not None else 0.0
                y_p_k += dy_dc_elem[elem_id] * dc

        elif desc.var_type == "logkinf":
            elem_id = f"{branch_str}_Linf"
            if elem_id in dy_dl_elem:
                l_inf = graph.elements[elem_id].value
                dl = dLinf_dx(float(l_inf), desc) if l_inf is not None else 0.0
                y_p_k += dy_dl_elem[elem_id] * dl

        elif desc.var_type == "logkm":
            idx = (desc.cell_index + 1) if desc.cell_index is not None else 1
            cm_id = f"{branch_str}_C{idx}"
            lm_id = f"{branch_str}_L{idx}"

            if cm_id in dy_dc_elem:
                cm = graph.elements[cm_id].value
                dcm = dCm_dxkm(float(cm), desc) if cm is not None else 0.0
                y_p_k += dy_dc_elem[cm_id] * dcm

            if lm_id in dy_dl_elem:
                lm = graph.elements[lm_id].value
                dlm = dLm_dxkm(float(lm), desc) if lm is not None else 0.0
                y_p_k += dy_dl_elem[lm_id] * dlm

        elif desc.var_type == "fp":
            idx = (desc.cell_index + 1) if desc.cell_index is not None else 1
            lm_id = f"{branch_str}_L{idx}"
            if lm_id in dy_dl_elem:
                lm = graph.elements[lm_id].value
                fp = phys.f_poles_hz[desc.cell_index] if desc.cell_index is not None else 1.0
                dlm = dLm_dxfp(float(lm), float(fp), desc) if lm is not None else 0.0
                y_p_k += dy_dl_elem[lm_id] * dlm

        y_p_list.append(y_p_k)

    return y_p_list


class DerivativeTransaction:
    """Manages the lifecycle of sensitivity evaluations for a single parameter vector x."""

    def __init__(self, context: EvaluationContext):
        self.context = context
        self._x: np.ndarray | None = None

        self._j_base: np.ndarray | None = None
        self._j_constr: np.ndarray | None = None
        self._obj_grad_result: ObjectiveGradientResult | None = None

        self.metrics = {
            "factorizations": 0,
            "direct_substitutions": 0,
            "adjoint_substitutions": 0,
            "jacobian_evals": 0,
        }

    def _invalidate_cache(self, x: np.ndarray) -> None:
        self._x = x.copy()
        self._j_base = None
        self._j_constr = None
        self._obj_grad_result = None

    def evaluate_jacobians(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (j_base, j_constr) for x, caching appropriately."""
        if (
            self._x is not None
            and np.array_equal(self._x, x)
            and self._j_base is not None
            and self._j_constr is not None
        ):
            return self._j_base, self._j_constr

        self._invalidate_cache(x)
        self.metrics["jacobian_evals"] += 1

        mapper = self.context.domain.variable_mapper
        b1, b2 = mapper.unpack(x)
        _validate_components(b1, b2)
        graph = _build_graph(
            b1,
            b2,
            self.context.domain,
            self.context.eom_model,
            self.context.domain.canonical_sign_pattern,
        )

        target_observables = {}
        off_target_gradients = {}
        target_solutions = {}  # needed for jacobian

        # Target frequencies (Direct)
        for _ti, fi in enumerate(self.context.target_indices):
            f_hz = self.context.evaluation_frequencies_hz[fi]
            y_nom, b_nom, node_map = assemble_mna(graph, self.context.source_spec, f_hz)
            state, status, diagnostics = solve_mna_factorized(y_nom, b_nom)
            self.metrics["factorizations"] += 1

            if status != CircuitSolveStatus.OK or state is None:
                continue

            y_p_list = build_y_p_list(graph, node_map, mapper, x, f_hz)

            x_p = compute_direct_state_sensitivities(state, y_p_list)
            self.metrics["direct_substitutions"] += mapper.dimension

            # Construct enough state to compute observables
            v_port_nom = (
                state.V_nominal[node_map[graph.input_port.node_pos]]
                if graph.input_port.node_pos in node_map
                else 0.0j
            )
            if graph.input_port.node_neg in node_map:
                v_port_nom -= state.V_nominal[node_map[graph.input_port.node_neg]]

            i_port_nom = (
                self.context.source_spec.vth_phasor - v_port_nom
            ) / self.context.source_spec.z_source
            z_in_nom = v_port_nom / i_port_nom if abs(i_port_nom) > 0 else 0.0j

            obs = compute_observable_derivatives(
                graph,
                self.context.source_spec,
                node_map,
                state,
                x_p,
                v_port_nom,
                i_port_nom,
                z_in_nom,
            )
            target_observables[fi] = obs

            from foster_eom.circuit.measurements import (
                CircuitSolution,
                ElementMeasurement,
                _element_admittance,
            )

            z0 = self.context.source_spec.z_ref_ohm
            gamma = (z_in_nom - z0) / (z_in_nom + z0) if abs(z_in_nom + z0) > 0 else 0.0j

            # Compute v_eom_nom for the CircuitSolution
            v_eom_nom: complex | None = None
            if graph.eom_element_id is not None and graph.eom_element_id in graph.elements:
                eom_elem = graph.elements[graph.eom_element_id]
                eom_pos = eom_elem.node_pos
                eom_neg = eom_elem.node_neg
                v_eom_pos = (
                    0.0j if eom_pos == graph.ground_node_id else state.V_nominal[node_map[eom_pos]]
                )
                v_eom_neg = (
                    0.0j if eom_neg == graph.ground_node_id else state.V_nominal[node_map[eom_neg]]
                )
                v_eom_nom = v_eom_pos - v_eom_neg

            # Build element_measurements + p_source_delivered_w for J_loss support
            element_measurements: dict[str, ElementMeasurement] = {}
            for elem in graph.elements.values():
                v_ep = (
                    0.0j
                    if elem.node_pos == graph.ground_node_id
                    else complex(state.V_nominal[node_map[elem.node_pos]])
                )
                v_en = (
                    0.0j
                    if elem.node_neg == graph.ground_node_id
                    else complex(state.V_nominal[node_map[elem.node_neg]])
                )
                v_elem = v_ep - v_en
                y_elem = _element_admittance(elem, f_hz)
                i_elem = y_elem * v_elem
                s_elem = v_elem * np.conj(i_elem)
                element_measurements[elem.id] = ElementMeasurement(
                    element_id=elem.id,
                    element_kind=elem.kind,
                    voltage=v_elem,
                    current=i_elem,
                    complex_power=s_elem,
                    real_power_w=float(np.real(s_elem)),
                    reactive_power_var=float(np.imag(s_elem)),
                )
            p_delivered = float(np.real(v_port_nom * np.conj(i_port_nom)))

            sol = CircuitSolution(
                f_hz=f_hz,
                status=status,
                diagnostics=diagnostics,
                z_in=z_in_nom,
                i_source_droop=i_port_nom,
                gamma=gamma,
                v_eom=v_eom_nom,
                element_measurements=element_measurements,
                p_source_delivered_w=p_delivered,
            )
            target_solutions[fi] = sol

        # Off-target frequencies (Adjoint)
        for fi in self.context.off_target_indices:
            f_hz = self.context.evaluation_frequencies_hz[fi]
            y_nom, b_nom, node_map = assemble_mna(graph, self.context.source_spec, f_hz)
            state, status, _ = solve_mna_factorized(y_nom, b_nom)
            self.metrics["factorizations"] += 1

            if status != CircuitSolveStatus.OK or state is None:
                continue

            y_p_list = build_y_p_list(graph, node_map, mapper, x, f_hz)
            grad = compute_v_eom_adjoint_gradient(graph, node_map, state, y_p_list)
            self.metrics["adjoint_substitutions"] += 1
            off_target_gradients[fi] = grad

        # ---- Hard constraint Jacobian ----
        j_constr = compute_layout_jacobian(
            self.context.hard_layout,
            target_solutions,
            target_observables,
            off_target_gradients,
            mapper,
            x,
            self.context.component_limits.l_max_h,
            self.context.component_limits.c_max_f,
            self.context.domain.topology.pole_spec_branch1.delta_f_pole_min_hz
            if hasattr(self.context.domain.topology, "pole_spec_branch1")
            else 1e3,
            self.context.domain.topology.pole_spec_branch2.delta_f_pole_min_hz
            if hasattr(self.context.domain.topology, "pole_spec_branch2")
            else 1e3,
        )

        # ---- Self-contained soft constraint margins + Jacobian ----
        soft_g_vector: np.ndarray | None = None
        soft_jacobian: np.ndarray | None = None
        if self.context.soft_layout.n > 0:
            # Compute soft Jacobian using same infrastructure as hard constraints
            soft_jacobian = compute_layout_jacobian(
                self.context.soft_layout,
                target_solutions,
                target_observables,
                off_target_gradients,
                mapper,
                x,
                self.context.component_limits.l_max_h,
                self.context.component_limits.c_max_f,
                self.context.domain.topology.pole_spec_branch1.delta_f_pole_min_hz
                if hasattr(self.context.domain.topology, "pole_spec_branch1")
                else 1e3,
                self.context.domain.topology.pole_spec_branch2.delta_f_pole_min_hz
                if hasattr(self.context.domain.topology, "pole_spec_branch2")
                else 1e3,
            )
            # Compute soft margins from the production evaluator
            # Build all_solutions tuple aligned with evaluation_frequencies_hz

            all_sol_list: list[CircuitSolution] = []
            for i in range(len(self.context.evaluation_frequencies_hz)):
                if i in target_solutions:
                    all_sol_list.append(target_solutions[i])
                else:
                    all_sol_list.append(
                        CircuitSolution(
                            f_hz=self.context.evaluation_frequencies_hz[i],
                            status=CircuitSolveStatus.OK,
                            diagnostics=diagnostics,
                        )
                    )
            soft_g_vector = self.context.soft_layout.evaluate(
                solutions=tuple(all_sol_list),
                target_indices=self.context.target_indices,
                off_target_indices=self.context.off_target_indices,
                branch1_pole_regions=self.context.domain.pole_regions_branch1,
                branch2_pole_regions=self.context.domain.pole_regions_branch2,
                branch1_k_residues=b1.k_residues,
                branch2_k_residues=b2.k_residues,
                branch1_f_poles=b1.f_poles_hz,
                branch2_f_poles=b2.f_poles_hz,
                branch1_l_vals=b1.l_values_h,
                branch2_l_vals=b2.l_values_h,
                branch1_c_vals=b1.c_values_f,
                branch2_c_vals=b2.c_values_f,
                component_limits_l_min=self.context.component_limits.l_min_h,
                component_limits_l_max=self.context.component_limits.l_max_h,
                component_limits_c_min=self.context.component_limits.c_min_f,
                component_limits_c_max=self.context.component_limits.c_max_f,
                pole_sep_min_b1=_get_pole_sep(self.context.domain, 1),
                pole_sep_min_b2=_get_pole_sep(self.context.domain, 2),
                z_ref_ohm=self.context.source_spec.z_ref_ohm,
                gamma_max=self.context.match_constraints.gamma_max,
                r_min_ohm=self.context.match_constraints.resistance_min_ohm,
                r_max_ohm=self.context.match_constraints.resistance_max_ohm,
                x_max_ohm=self.context.match_constraints.max_abs_reactance_ohm,
                source_current_max_a=self.context.stress_constraints.source_current_rms_max_a,
                off_target_eom_peak_rms_v=self.context.stress_constraints.off_target_eom_peak_rms_v,
            )

        # ---- Actual production objective gradient ----
        obj_result = compute_objective_gradient(
            config=self.context.objective_config,
            target_solutions=target_solutions,
            target_observables=target_observables,
            target_indices=self.context.target_indices,
            soft_layout=self.context.soft_layout,
            soft_g_vector=soft_g_vector,
            soft_jacobian=soft_jacobian,
            n_params=mapper.dimension,
        )
        j_base = obj_result.gradient
        self._obj_grad_result = obj_result

        self._j_base = j_base
        self._j_constr = j_constr

        return j_base, j_constr
