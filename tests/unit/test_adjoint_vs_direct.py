import numpy as np
import pytest

from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
from foster_eom.circuit.mna import SolverOptions
from foster_eom.circuit.solve import solve_circuit_single
from foster_eom.domain.source import SourceSpec, SourceMode
from foster_eom.models.base import OnePortModel
from foster_eom.optimize.variable_map import build_variable_mapper
from foster_eom.domain.topology import LOrientation
from foster_eom.foster.topology_enum import TopologyCandidate
from foster_eom.foster.schmidt import BranchRealization

from foster_eom.sensitivities.transaction import build_y_p_list


class DummyEOM(OnePortModel):
    def _z_impl(self, f_hz: float | np.ndarray) -> complex | np.ndarray:
        return 50.0 - 1j / (2 * np.pi * f_hz * 1e-12)
        
    @property
    def metadata(self) -> dict:
        return {}

def test_direct_vs_adjoint_vs_fd():
    topo = TopologyCandidate(
        branch1_cells=1, branch2_cells=0,
        branch1_has_c0=True, branch1_has_linf=False,
        branch2_has_c0=False, branch2_has_linf=False,
        orientation=LOrientation.SCHMIDT_SHUNT_THEN_SERIES,
        branch1_n_coefficients=2, branch2_n_coefficients=0,
        n_reactive=1, structurally_valid=True, prune_reason=""
    )
    from foster_eom.foster.sign_search import SignPattern
    
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
        branch1_n_cells=1, branch1_has_c0=True, branch1_has_linf=False,
        branch1_pole_regions=((1e6, 10e6),), branch1_k_box_bounds=((1e9, 1e12),),
        branch1_k0_bounds=(1e9, 1e12), branch1_kinf_bounds=None,
        branch1_fixed_k0=None, branch1_fixed_kinf=None,
        branch1_fixed_k_residues=(None,), branch1_fixed_f_poles_hz=(None,),
        branch2_n_cells=0, branch2_has_c0=False, branch2_has_linf=False,
        branch2_pole_regions=(), branch2_k_box_bounds=(),
        branch2_k0_bounds=None, branch2_kinf_bounds=None,
        branch2_fixed_k0=None, branch2_fixed_kinf=None,
        branch2_fixed_k_residues=(), branch2_fixed_f_poles_hz=()
    )
    
    from foster_eom.optimize.domain import ContinuousOptimizationDomain
    domain = ContinuousOptimizationDomain(
        domain_id="test", orientation=LOrientation.SCHMIDT_SHUNT_THEN_SERIES, topology=topo,
        branch1_realization=BranchRealization.FINITE_FOSTER, branch2_realization=BranchRealization.ZERO_IMPEDANCE,
        seed_indices=(0,), pole_regions_branch1=((1e6, 10e6),), pole_regions_branch2=(),
        k_box_bounds_branch1=((1e9, 1e12),), k_box_bounds_branch2=(),
        k0_bounds_b1=(1e9, 1e12), k0_bounds_b2=None, k_inf_bounds_b1=None, k_inf_bounds_b2=None,
        n_movable_poles_branch1=1, n_movable_poles_branch2=0, variable_mapper=mapper,
        dimension=3, structurally_feasible=True, infeasibility_reason=None, canonical_sign_pattern=sp
    )

    x0 = np.array([0.5, 0.5, 0.5])
    f_hz = 1.0e6
    source = SourceSpec(mode=SourceMode.THEVENIN, thevenin_vrms=1.0, z_source_real_ohm=50.0, z_ref_ohm=50.0)
    eom = DummyEOM()
    
    from foster_eom.foster.network_builder import build_foster_circuit
    from foster_eom.circuit.mna import assemble_mna
    
    # helper to evaluate V_eom magnitude
    def eval_v_eom(x_val):
        b1, b2 = mapper.unpack(x_val)
        from foster_eom.optimize.evaluator import _build_graph
        graph = _build_graph(b1, b2, domain, eom, sp)
        sol = solve_circuit_single(graph, source, f_hz, SolverOptions())
        return abs(sol.v_eom)
        
    v0 = eval_v_eom(x0)
    
    # 1. Finite Differences
    eps = 1e-6
    n_dim = len(x0)
    fd_grad = np.zeros(n_dim)
    for i in range(n_dim):
        x_p = x0.copy()
        x_m = x0.copy()
        x_p[i] += eps
        x_m[i] -= eps
        fd_grad[i] = (eval_v_eom(x_p) - eval_v_eom(x_m)) / (2 * eps)
        
    # 2. Setup Direct and Adjoint
    b1, b2 = mapper.unpack(x0)
    from foster_eom.optimize.evaluator import _build_graph
    graph = _build_graph(b1, b2, domain, eom, sp)
    sol = solve_circuit_single(graph, source, f_hz, SolverOptions())
    
    # Re-build matrices for sensitivities
    from foster_eom.circuit.mna import solve_mna, assemble_mna
    Y, I_vec, node_map = assemble_mna(graph, source, f_hz)
    
    y_prime_list = build_y_p_list(graph, node_map, mapper, x0, f_hz)
    # The EOM node index
    v_eom = sol.v_eom
    eom_node_name = "load"
    try:
        eom_idx = graph.get_node_index(eom_node_name)
    except KeyError:
        eom_idx = 1 # heuristic
        
    # We need the full x_state vector which MNA solver solved for
    V, status, _ = solve_mna(Y, I_vec, SolverOptions())
    x_state = V
    
    # Adjoint
    # the objective is |V_eom|. So f(x) = |v_eom|. df/dv_eom = 0.5 * conj(v_eom) / |v_eom|
    df_dv = 0.5 * np.conj(v_eom) / abs(v_eom) if abs(v_eom) > 0 else 0.0
    # Wait, the v is complex, the function is f(v) = sqrt(v.real^2 + v.imag^2)
    # df/dv.real = v.real / |v|
    # df/dv.imag = v.imag / |v|
    # adjoint rhs wants to map complex state directly.
    # The derivative of |v| wrt parameter p: d|v|/dp = Re(conj(v) / |v| * dv/dp)
    adjoint_rhs = np.zeros(Y.shape[0], dtype=np.complex128)
    adjoint_rhs[eom_idx - 1] = np.conj(v_eom) / abs(v_eom) if abs(v_eom) > 0 else 0.0
    lam = np.linalg.solve(Y.T, adjoint_rhs)
    
    adj_grad = np.zeros(n_dim)
    for i in range(n_dim):
        adj_grad[i] = -np.vdot(lam, y_prime_list[i] @ x_state).real
        
    # Direct
    dir_grad = np.zeros(n_dim)
    for i in range(n_dim):
        b_prime = -(y_prime_list[i] @ x_state)
        dx_state = np.linalg.solve(Y, b_prime)
        dv_eom = dx_state[eom_idx - 1]
        dir_grad[i] = np.real(np.conj(v_eom) / abs(v_eom) * dv_eom)
        
    print(f"FD:      {fd_grad}")
    print(f"Adjoint: {adj_grad}")
    print(f"Direct:  {dir_grad}")
    
    np.testing.assert_allclose(adj_grad, fd_grad, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(dir_grad, fd_grad, rtol=1e-4, atol=1e-6)
