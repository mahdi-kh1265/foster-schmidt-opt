import foster_eom.optimize.local_polish as lp
from foster_eom.optimize.nominal_state import NominalStateIdentity

old_matches = NominalStateIdentity.matches

def trace_matches(self, other):
    res = old_matches(self, other)
    if not res:
        print("MATCH FAILED!")
        if self.context is not other.context: print("context differs")
        if self.x_key != other.x_key: print(f"x_key differs!\nself : {self.x_key}\nother: {other.x_key}")
        if self.domain_id != other.domain_id: print("domain_id differs")
        if self.frequencies_hz != other.frequencies_hz: print("frequencies_hz differs")
    return res

NominalStateIdentity.matches = trace_matches

# Run 1 iteration of pathological F
import numpy as np

import foster_eom.optimize.engine as engine_mod
from foster_eom.domain.objectives import DerivativeMode
from foster_eom.foster.seed import generate_seeds
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
            "local_max_iterations": 1,
            "polish_top_k": 1,
        }
    )
    print("starting polish")
    return lp.polish_top_k(basins, context, cache, s)

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
