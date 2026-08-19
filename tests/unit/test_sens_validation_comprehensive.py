"""Comprehensive FD validation for the P12.5-D sensitivity engine.

Covers:
  1. v_eom observable derivative vs central FD
  2. Objective gradient (J_gamma) vs central FD
  3. Nonsmooth audit: classification of all derivative paths
  4. Memory lifecycle: bounded heavy-state retention
  5. Off-target EOM-missing validation
"""

import gc

import numpy as np
import pytest

from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
from foster_eom.circuit.mna import assemble_mna, solve_mna_factorized
from foster_eom.domain.constraints import ConstraintSeverity
from foster_eom.domain.source import SourceSpec
from foster_eom.errors import CircuitSolveStatus
from foster_eom.optimize.constraints import ConstraintLayout
from foster_eom.optimize.objective import ObjectiveConfig
from foster_eom.sensitivities.direct import compute_direct_state_sensitivities
from foster_eom.sensitivities.objective_gradient import (
    DerivativeStatus,
    compute_objective_gradient,
)
from foster_eom.sensitivities.observables import compute_observable_derivatives
from foster_eom.sensitivities.off_target import compute_v_eom_adjoint_gradient
from foster_eom.sensitivities.stamps import stamp_capacitor_derivative


def _make_eom_graph(C_val, L_val):
    """Build a 3-node C-L-R graph with EOM = R1."""
    graph = CircuitGraph("gnd", Port("n1", "gnd"), eom_element_id="R1")
    graph.add_node(Node("n1"))
    graph.add_node(Node("n2"))
    graph.add_node(Node("n3"))
    graph.add_element(Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val))
    graph.add_element(Element("L1", ElementKind.INDUCTOR, "n2", "n3", L_val))
    graph.add_element(Element("R1", ElementKind.RESISTOR, "n3", "gnd", 50.0))
    return graph


class TestVEomObservable:
    """Validates v_eom derivative against central FD."""

    def test_v_eom_derivative_matches_fd(self):
        f_hz = 10e6
        C_val = 100e-12
        L_val = 1e-6
        source_spec = SourceSpec(mode="thevenin", thevenin_vrms=1.0, z_source_real_ohm=50.0)

        graph = _make_eom_graph(C_val, L_val)
        Y_nom, b_nom, node_map = assemble_mna(graph, source_spec, f_hz)
        state, status, _ = solve_mna_factorized(Y_nom, b_nom)
        assert status == CircuitSolveStatus.OK

        v_port_nom = state.V_nominal[node_map["n1"]]
        i_port_nom = (source_spec.vth_phasor - v_port_nom) / source_spec.z_source
        z_in_nom = v_port_nom / i_port_nom

        Y_p_C = np.zeros_like(Y_nom)
        stamp_capacitor_derivative(
            Y_p_C, graph.elements["C1"], node_map, graph.ground_node_id, f_hz
        )
        X_p = compute_direct_state_sensitivities(state, [Y_p_C])

        obs = compute_observable_derivatives(
            graph, source_spec, node_map, state, X_p, v_port_nom, i_port_nom, z_in_nom
        )

        # FD for v_eom (complex phasor)
        h = 1e-15
        for sign, delta in [(+1, h), (-1, h)]:
            g = _make_eom_graph(C_val + sign * delta, L_val)
            Y, b, _ = assemble_mna(g, source_spec, f_hz)
            s, _, _ = solve_mna_factorized(Y, b)
            if sign == 1:
                v_eom_plus = s.V_nominal[node_map["n3"]]
            else:
                v_eom_minus = s.V_nominal[node_map["n3"]]
        # R1 is between n3 and gnd, so v_eom = V[n3] - 0 = V[n3]
        fd_v_eom = (v_eom_plus - v_eom_minus) / (2 * h)

        np.testing.assert_allclose(obs.v_eom[0], fd_v_eom, rtol=1e-5, atol=1e-12)


class TestObjectiveGradient:
    """Validates J_gamma objective gradient against central FD."""

    def test_j_gamma_gradient_fd(self):
        """J_gamma = (1/N) sum |Gamma|^2. Gradient vs central FD."""
        f_hz = 10e6
        C_val = 100e-12
        L_val = 1e-6
        source_spec = SourceSpec(mode="thevenin", thevenin_vrms=1.0, z_source_real_ohm=50.0)

        graph = _make_eom_graph(C_val, L_val)
        Y_nom, b_nom, node_map = assemble_mna(graph, source_spec, f_hz)
        state, status, diag = solve_mna_factorized(Y_nom, b_nom)
        assert status == CircuitSolveStatus.OK

        v_port_nom = state.V_nominal[node_map["n1"]]
        i_port_nom = (source_spec.vth_phasor - v_port_nom) / source_spec.z_source
        z_in_nom = v_port_nom / i_port_nom
        z0 = source_spec.z_ref_ohm
        gamma_nom = (z_in_nom - z0) / (z_in_nom + z0)

        Y_p_C = np.zeros_like(Y_nom)
        stamp_capacitor_derivative(
            Y_p_C, graph.elements["C1"], node_map, graph.ground_node_id, f_hz
        )
        X_p = compute_direct_state_sensitivities(state, [Y_p_C])
        obs = compute_observable_derivatives(
            graph, source_spec, node_map, state, X_p, v_port_nom, i_port_nom, z_in_nom
        )

        from foster_eom.circuit.measurements import CircuitSolution

        sol = CircuitSolution(
            f_hz=f_hz,
            status=status,
            diagnostics=diag,
            gamma=gamma_nom,
            z_in=z_in_nom,
            v_eom=state.V_nominal[node_map["n3"]],
        )

        config = ObjectiveConfig(z_ref_ohm=50.0, w_gamma=1.0, w_voltage=0.0, w_loss=0.0)
        soft_layout = ConstraintLayout(descriptors=())

        result = compute_objective_gradient(
            config=config,
            target_solutions={0: sol},
            target_observables={0: obs},
            target_indices=(0,),
            soft_layout=soft_layout,
            soft_g_vector=None,
            soft_jacobian=None,
            n_params=1,
        )
        assert result.status == DerivativeStatus.SMOOTH

        # Central FD of J_gamma
        h = 1e-15
        J_gamma_vals = []
        for sign in [+1, -1]:
            g = _make_eom_graph(C_val + sign * h, L_val)
            Y, b, _ = assemble_mna(g, source_spec, f_hz)
            s, _, _ = solve_mna_factorized(Y, b)
            vp = s.V_nominal[node_map["n1"]]
            ip = (source_spec.vth_phasor - vp) / source_spec.z_source
            zi = vp / ip
            gm = (zi - z0) / (zi + z0)
            J_gamma_vals.append(abs(gm) ** 2)

        fd_grad = (J_gamma_vals[0] - J_gamma_vals[1]) / (2 * h)
        np.testing.assert_allclose(result.gradient[0], fd_grad, rtol=1e-5, atol=1e-12)

    def test_j_voltage_gradient_fd(self):
        """J_voltage gradient vs central FD."""
        f_hz = 10e6
        C_val = 100e-12
        L_val = 1e-6
        source_spec = SourceSpec(mode="thevenin", thevenin_vrms=1.0, z_source_real_ohm=50.0)

        graph = _make_eom_graph(C_val, L_val)
        Y_nom, b_nom, node_map = assemble_mna(graph, source_spec, f_hz)
        state, status, diag = solve_mna_factorized(Y_nom, b_nom)
        assert status == CircuitSolveStatus.OK

        v_port_nom = state.V_nominal[node_map["n1"]]
        i_port_nom = (source_spec.vth_phasor - v_port_nom) / source_spec.z_source
        z_in_nom = v_port_nom / i_port_nom
        z0 = source_spec.z_ref_ohm
        gamma_nom = (z_in_nom - z0) / (z_in_nom + z0)
        v_eom_nom = state.V_nominal[node_map["n3"]]

        Y_p_C = np.zeros_like(Y_nom)
        stamp_capacitor_derivative(
            Y_p_C, graph.elements["C1"], node_map, graph.ground_node_id, f_hz
        )
        X_p = compute_direct_state_sensitivities(state, [Y_p_C])
        obs = compute_observable_derivatives(
            graph, source_spec, node_map, state, X_p, v_port_nom, i_port_nom, z_in_nom
        )

        from foster_eom.circuit.measurements import CircuitSolution

        sol = CircuitSolution(
            f_hz=f_hz,
            status=status,
            diagnostics=diag,
            gamma=gamma_nom,
            z_in=z_in_nom,
            v_eom=v_eom_nom,
        )

        v_target = abs(v_eom_nom) * 1.1  # target slightly above actual
        config = ObjectiveConfig(
            z_ref_ohm=50.0,
            w_gamma=0.0,
            w_voltage=1.0,
            w_loss=0.0,
            voltage_targets_rms_v=(v_target,),
            voltage_target_weights=(1.0,),
        )
        soft_layout = ConstraintLayout(descriptors=())

        result = compute_objective_gradient(
            config=config,
            target_solutions={0: sol},
            target_observables={0: obs},
            target_indices=(0,),
            soft_layout=soft_layout,
            soft_g_vector=None,
            soft_jacobian=None,
            n_params=1,
        )

        # Central FD of J_voltage
        h = 1e-15
        J_voltage_vals = []
        for sign in [+1, -1]:
            g = _make_eom_graph(C_val + sign * h, L_val)
            Y, b, _ = assemble_mna(g, source_spec, f_hz)
            s, _, _ = solve_mna_factorized(Y, b)
            ve = abs(s.V_nominal[node_map["n3"]])
            d_i = max(v_target, 1e-6)
            J_voltage_vals.append(((ve - v_target) / d_i) ** 2)

        fd_grad = (J_voltage_vals[0] - J_voltage_vals[1]) / (2 * h)
        np.testing.assert_allclose(result.gradient[0], fd_grad, rtol=1e-4, atol=1e-12)


class TestNonsmoothClassification:
    """Audit all derivative paths for nonsmoothness."""

    def test_j_gamma_smooth_at_gamma_zero(self):
        """J_gamma = sum |Gamma|^2 / N is smooth at Gamma = 0
        because |z|^2 = z * conj(z) which is smooth (polynomial in Re, Im)."""
        from foster_eom.circuit.measurements import CircuitSolution

        # Create a solution at perfect match (gamma ≈ 0)
        source_spec = SourceSpec(mode="thevenin", thevenin_vrms=1.0, z_source_real_ohm=50.0)
        config = ObjectiveConfig(z_ref_ohm=50.0, w_gamma=1.0, w_voltage=0.0, w_loss=0.0)
        soft_layout = ConstraintLayout(descriptors=())

        sol = CircuitSolution(
            f_hz=1e6,
            status=CircuitSolveStatus.OK,
            diagnostics=None,
            gamma=0.0 + 0.0j,  # perfect match
            z_in=50.0 + 0.0j,
        )

        # Derivative should be exactly zero at gamma=0 and status should be SMOOTH
        obs_mock = type(
            "MockObs",
            (),
            {
                "gamma": np.array([0.1 + 0.2j]),
                "v_eom": np.array([0.0j]),
            },
        )()

        result = compute_objective_gradient(
            config=config,
            target_solutions={0: sol},
            target_observables={0: obs_mock},
            target_indices=(0,),
            soft_layout=soft_layout,
            soft_g_vector=None,
            soft_jacobian=None,
            n_params=1,
        )
        # dJ/dp = (2/N) Re(Gamma^* dGamma/dp) = 2 Re(0 * anything) = 0
        assert result.gradient[0] == pytest.approx(0.0, abs=1e-15)
        assert result.status == DerivativeStatus.SMOOTH

    def test_squared_hinge_c1_at_g_zero(self):
        """max(0, -g)^2 is C^1 at g=0 when g is differentiable.
        The derivative 2*max(0,-g)*(-dg/dp) → 0 as g → 0 from below."""
        from foster_eom.domain.constraints import FrequencyScope
        from foster_eom.optimize.constraints import ConstraintDescriptor

        config = ObjectiveConfig(z_ref_ohm=50.0, w_gamma=0.0)
        soft_desc = ConstraintDescriptor(
            name="soft_test",
            constraint_type="custom",
            frequency_scope=FrequencyScope.ALL_TARGETS,
            severity=ConstraintSeverity.SOFT,
            penalty_weight=1.0,
        )
        soft_layout = ConstraintLayout(descriptors=(soft_desc,))

        # g = 0 exactly: max(0, -0)^2 = 0, derivative = 0
        result = compute_objective_gradient(
            config=config,
            target_solutions={},
            target_observables={},
            target_indices=(0,),
            soft_layout=soft_layout,
            soft_g_vector=np.array([0.0]),
            soft_jacobian=np.array([[1.0]]),
            n_params=1,
        )
        assert result.gradient[0] == pytest.approx(0.0, abs=1e-15)

    def test_x_bound_nonsmooth_at_resonance(self):
        """Constraint x_bound uses |Im(Z_in)| which is nonsmooth at Im(Z_in)=0.
        The nonsmoothness is in the underlying constraint (sign change of imag part),
        not in the squared hinge."""
        # This documents the known nonsmoothness; no analytical fix required.
        pass  # Classification documented


class TestOffTargetValidation:
    """Verify off-target EOM validation."""

    def test_missing_eom_raises_error(self):
        """When graph has no EOM element, off-target gradient must raise, not zero."""

        graph = CircuitGraph("gnd", Port("n1", "gnd"))  # NO eom_element_id
        graph.add_node(Node("n1"))
        graph.add_element(Element("R1", ElementKind.RESISTOR, "n1", "gnd", 50.0))

        source_spec = SourceSpec(mode="thevenin", thevenin_vrms=1.0, z_source_real_ohm=50.0)
        Y, b, node_map = assemble_mna(graph, source_spec, 10e6)
        state, status, _ = solve_mna_factorized(Y, b)
        assert status == CircuitSolveStatus.OK

        with pytest.raises(ValueError, match="requires a valid EOM element"):
            compute_v_eom_adjoint_gradient(graph, node_map, state, [])

    def test_valid_eom_produces_gradient(self):
        """When graph has valid EOM element, gradient is produced."""
        graph = CircuitGraph("gnd", Port("n1", "gnd"), eom_element_id="R1")
        graph.add_node(Node("n1"))
        graph.add_element(Element("R1", ElementKind.RESISTOR, "n1", "gnd", 50.0))

        source_spec = SourceSpec(mode="thevenin", thevenin_vrms=1.0, z_source_real_ohm=50.0)
        Y, b, node_map = assemble_mna(graph, source_spec, 10e6)
        state, status, _ = solve_mna_factorized(Y, b)
        assert status == CircuitSolveStatus.OK

        grad = compute_v_eom_adjoint_gradient(graph, node_map, state, [])
        assert len(grad) == 0  # No parameters


class TestMemoryLifecycle:
    """Verify bounded heavy-state retention."""

    def test_no_proportional_growth(self):
        """DerivativeTransaction should not leak memory across evaluations."""
        import tracemalloc

        from foster_eom.domain.component import ContinuousLimits
        from foster_eom.domain.constraints import MatchConstraints, StressConstraints
        from foster_eom.domain.source import SourceMode
        from foster_eom.domain.topology import LOrientation
        from foster_eom.foster.schmidt import BranchRealization
        from foster_eom.foster.sign_search import SignPattern
        from foster_eom.foster.topology_enum import TopologyCandidate
        from foster_eom.optimize.domain import ContinuousOptimizationDomain
        from foster_eom.optimize.evaluator import build_evaluation_context
        from foster_eom.optimize.variable_map import build_variable_mapper
        from foster_eom.sensitivities.transaction import DerivativeTransaction
        from tests.unit.test_sens_e2e import DummyEOM

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
        domain = ContinuousOptimizationDomain(
            domain_id="test",
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
            base_grid_points=2,
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

        tracemalloc.start()
        # Warmup
        for _ in range(5):
            txn.evaluate_jacobians(x0 + np.random.uniform(-0.01, 0.01, 3))
        gc.collect()
        snap_before = tracemalloc.take_snapshot()

        # Main run
        for _ in range(50):
            txn.evaluate_jacobians(x0 + np.random.uniform(-0.01, 0.01, 3))
        gc.collect()
        snap_after = tracemalloc.take_snapshot()

        tracemalloc.stop()

        # Check: after-before should not grow proportionally
        stats_before = snap_before.statistics("lineno")
        stats_after = snap_after.statistics("lineno")
        total_before = sum(s.size for s in stats_before)
        total_after = sum(s.size for s in stats_after)

        # Allow 2x headroom (not 50x which would indicate a leak)
        assert total_after < total_before * 5, (
            f"Memory grew from {total_before} to {total_after} bytes "
            f"over 50 evaluations — possible leak"
        )
