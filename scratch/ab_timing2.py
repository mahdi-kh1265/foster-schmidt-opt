import os
import time

import numpy as np
import psutil

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
        # Must also override transaction.exchange
        self.transaction._exchange = None

AnalyticalDerivativeProvider.__init__ = mocked_init

# We'll override polish_top_k temporarily to intercept and run A/B
class Harness:
    def __init__(self, mode: str, iters: int):
        self.mode = mode
        self.iters = iters
        self.dt = 0.0
        self.rss_peak = 0.0
        self.telemetry = None

    def __call__(self, basins, context, cache, opt_spec):
        spec = opt_spec.model_copy(
            update={
                "local_derivative_mode": DerivativeMode.ANALYTICAL,
                "local_max_iterations": self.iters,
                "polish_top_k": len(basins),
            }
        )
        fresh_cache = DomainEvaluatorCache()
        if self.mode == 'E':
            fresh_cache._force_disable_exchange = True
        else:
            fresh_cache._force_disable_exchange = False

        proc = psutil.Process(os.getpid())
        rss_before = proc.memory_info().rss / 1e6
        t0 = time.perf_counter()
        results = _real_polish_top_k(basins, context, fresh_cache, spec)
        self.dt = time.perf_counter() - t0
        rss_after = proc.memory_info().rss / 1e6
        self.rss_peak = max(rss_after, rss_before)

        # Save telemetry
        if len(results) > 0:
            self.telemetry = results[0].telemetry
        return results

def run_case(case_id: str, order: list[str]):
    specs = _case_specs()
    if case_id not in specs:
        return
    spec, label, notes, cap = specs[case_id]
    args = _build_args(spec)

    seed_res = generate_seeds(
        r_match_ohm=args["source_spec"].z_source_real_ohm,
        source_spec=args["source_spec"],
        eom_model=args["eom_model"],
        f_targets_hz=np.array(args["target_frequencies_hz"]),
        topo_spec=args["topology_spec"],
        component_limits=args["component_limits"],
    )

    times = {'E': [], 'F': []}

    for mode in order:
        harness = Harness(mode, cap)
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
        times[mode].append(harness.dt)
        print(f"{case_id} {mode}: {harness.dt:.3f}s, reuse={getattr(harness.telemetry, 'transaction_nominal_states_reused', 0)}, sweeps={getattr(harness.telemetry, 'transaction_nominal_sweep_solves', 0)}, eval_solves={getattr(harness.telemetry, 'evaluator_target_freq_solves', 0)}, fact={getattr(harness.telemetry, 'factorizations', 0)}, obj={getattr(harness.telemetry, 'final_objective', 0):.4f}, vmax={getattr(harness.telemetry, 'final_v_max', 0):.4f}, peak_rss={harness.rss_peak:.1f}MB")

    return times

print("SMALL")
t_small = run_case("small", ['E', 'F', 'F', 'E'])
print(f"E: {t_small['E']}")
print(f"F: {t_small['F']}")

print("PATHOLOGICAL")
t_patho = run_case("pathological", ['E', 'F', 'F', 'E'])
print(f"E: {t_patho['E']}")
print(f"F: {t_patho['F']}")
