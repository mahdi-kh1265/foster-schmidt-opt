import cProfile
import io
import pstats

import numpy as np

import foster_eom.optimize.engine as engine_mod
from foster_eom.domain.objectives import DerivativeMode
from foster_eom.foster.seed import generate_seeds
from foster_eom.optimize.derivative_provider import AnalyticalDerivativeProvider
from foster_eom.optimize.evaluator import DomainEvaluatorCache
from foster_eom.optimize.local_polish import polish_top_k as _real_polish_top_k
from scripts.p12_5_e_equivalence_benchmark import _build_args, _case_specs

orig_init = AnalyticalDerivativeProvider.__init__
def mocked_init(self, context, cache=None):
    orig_init(self, context, cache)
    if hasattr(cache, "_force_disable_exchange") and cache._force_disable_exchange:
        if self.exchange is not None:
            self.exchange.disable()
        self.exchange = None
        self.transaction._exchange = None

AnalyticalDerivativeProvider.__init__ = mocked_init

def profile_pathological(mode: str):
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

    # We'll profile ONLY the polish_top_k phase
    def harness(basins, context, cache, opt_spec):
        s = opt_spec.model_copy(
            update={
                "local_derivative_mode": DerivativeMode.ANALYTICAL,
                "local_max_iterations": cap,
                "polish_top_k": 1,
            }
        )
        fresh_cache = DomainEvaluatorCache()
        if mode == 'E':
            fresh_cache._force_disable_exchange = True
        else:
            fresh_cache._force_disable_exchange = False

        pr = cProfile.Profile()
        pr.enable()
        results = _real_polish_top_k(basins, context, fresh_cache, s)
        pr.disable()

        s_io = io.StringIO()
        ps = pstats.Stats(pr, stream=s_io).sort_stats('tottime')
        ps.print_stats(30)
        print(f"--- PROFILE {mode} ---")
        print(s_io.getvalue())
        return results

    original = engine_mod.polish_top_k
    engine_mod.polish_top_k = harness
    try:
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
    finally:
        engine_mod.polish_top_k = original

profile_pathological('E')
profile_pathological('F')
