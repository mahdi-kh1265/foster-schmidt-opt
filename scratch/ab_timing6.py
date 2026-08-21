import foster_eom.optimize.local_polish as lp
from foster_eom.optimize.nominal_state import NominalStateExchange, NominalStateIdentity

old_matches = NominalStateIdentity.matches
def trace_matches(self, other):
    print(f"MATCH CALLED. self={self.x_key}, other={other.x_key}")
    return old_matches(self, other)

NominalStateIdentity.matches = trace_matches

old_lookup = NominalStateExchange.lookup
def trace_lookup(self, x_key, context):
    print(f"LOOKUP CALLED with x_key={x_key}, committed={self._committed is not None}")
    return old_lookup(self, x_key, context)
NominalStateExchange.lookup = trace_lookup

old_begin = NominalStateExchange.begin
def trace_begin(self, x_key, context, graph):
    print(f"BEGIN CALLED with x_key={x_key}")
    return old_begin(self, x_key, context, graph)
NominalStateExchange.begin = trace_begin

old_settle = NominalStateExchange.settle
def trace_settle(self, ok):
    print(f"SETTLE CALLED ok={ok}")
    return old_settle(self, ok)
NominalStateExchange.settle = trace_settle


# Run 2 iterations of pathological F
import numpy as np

import foster_eom.optimize.engine as engine_mod
from foster_eom.domain.objectives import DerivativeMode
from foster_eom.foster.seed import generate_seeds
from foster_eom.optimize.evaluator import DomainEvaluatorCache
from scripts.p12_5_e_equivalence_benchmark import _build_args, _case_specs

specs = _case_specs()
spec, label, notes, cap = specs['pathological']
args = _build_args(spec)
seed_res = generate_seeds(
    r_match_ohm=args["source_spec"].z_source_real_ohm,
    source_spec=args["source_spec"],
    eom_model=args["eom_model"],
    f_targets_hz=np.array(args["target_frequencies_hz"]),
    topo_spec=args["topology_spec"],
    component_limits=args["component_limits"],
)

def harness(basins, context, cache, opt_spec):
    s = opt_spec.model_copy(
        update={
            "local_derivative_mode": DerivativeMode.ANALYTICAL,
            "local_max_iterations": 2,
            "polish_top_k": 1,
        }
    )
    print("starting polish")
    return lp.polish_top_k(basins, context, DomainEvaluatorCache(), s)

engine_mod.polish_top_k = harness
engine_mod.run_optimization(
    seed_result=seed_res,
    opt_spec=args["opt_spec"],
    source_spec=args["source_spec"],
    eom_model=args["eom_model"],
    component_limits=args["component_limits"],
    match_constraints=args["match_constraints"],
    stress_constraints=args["stress_constraints"],
    target_frequencies_hz=args["target_frequencies_hz"],
    sweep_f_min_hz=args["sweep_f_min_hz"],
    sweep_f_max_hz=args["sweep_f_max_hz"],
    base_grid_points=args["base_grid_points"],
    voltage_targets_rms_v=args["voltage_targets_rms_v"],
)
