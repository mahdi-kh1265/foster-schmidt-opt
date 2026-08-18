import numpy as np
import pytest
from unittest.mock import MagicMock

from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
from foster_eom.circuit.measurements import CircuitSolution
from foster_eom.errors import CircuitSolveStatus
from foster_eom.optimize.evaluator import EvaluationContext
from foster_eom.optimize.variable_map import build_variable_mapper
from foster_eom.sensitivities.transaction import DerivativeTransaction

def test_derivative_transaction_caching(monkeypatch):
    """E2E test for caching behavior in DerivativeTransaction."""
    # 1. Setup mock context and mapper
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
    
    ctx = MagicMock()
    ctx.domain = MagicMock()
    ctx.domain.variable_mapper = mapper
    ctx.target_indices = (0,)
    ctx.off_target_indices = ()
    ctx.evaluation_frequencies_hz = (10e6,)
    
    source_spec = MagicMock()
    source_spec.vth_phasor = 1.0 + 0.0j
    source_spec.z_source = 50.0
    source_spec.z_ref_ohm = 50.0
    ctx.source_spec = source_spec
    
    hard_layout = MagicMock()
    hard_layout.n = 0
    hard_layout.descriptors = []
    ctx.hard_layout = hard_layout
    
    # Mock _build_graph to return a simple LC network
    def mock_build_graph(b1, b2, domain, eom_model, sign_pattern):
        graph = CircuitGraph("gnd", Port("n1", "gnd"), "R1")
        graph.add_node(Node("n1"))
        graph.add_element(Element("b1_C0", ElementKind.CAPACITOR, "n1", "gnd", b1.c_values_f[-1]))
        graph.add_element(Element("R1", ElementKind.RESISTOR, "n1", "gnd", 50.0))
        return graph
        
    monkeypatch.setattr("foster_eom.sensitivities.transaction._build_graph", mock_build_graph)
    monkeypatch.setattr("foster_eom.sensitivities.transaction._validate_components", lambda b1, b2: None)
    
    txn = DerivativeTransaction(ctx)
    x_val = np.array([0.5, 0.5, 0.5]) # logk0, logkm, fp
    
    # Evaluate
    j_base, j_constr = txn.evaluate_jacobians(x_val)
    
    assert txn.metrics["jacobian_evals"] == 1
    assert txn.metrics["factorizations"] == 1
    
    # Second evaluation with same x should hit cache
    j_base_2, j_constr_2 = txn.evaluate_jacobians(x_val)
    assert txn.metrics["jacobian_evals"] == 1 # unchanged!
    assert txn.metrics["factorizations"] == 1
    
    # New x triggers re-evaluation
    x_new = np.array([0.6, 0.5, 0.5])
    j_base_3, j_constr_3 = txn.evaluate_jacobians(x_new)
    assert txn.metrics["jacobian_evals"] == 2
    assert txn.metrics["factorizations"] == 2
