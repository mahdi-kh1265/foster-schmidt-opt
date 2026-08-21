import gc
import platform
import time

import numpy as np
import scipy

from foster_eom.domain.objectives import DerivativeMode
from foster_eom.optimize.evaluator import DomainEvaluatorCache
from foster_eom.optimize.local_polish import polish_basin
from foster_eom.optimize.nominal_state import NominalStateExchange
from tests.unit.test_p12_5_e_analytical_polish import _basin, _spec
from tests.unit.test_p12_5_g3_scientific_equivalence import _build_custom_case


def print_env():
    print("=== ENVIRONMENT ===")
    print(f"Platform: {platform.platform()}")
    print(f"CPU: {platform.processor()}")
    print(f"Python: {platform.python_version()}")
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print("BLAS/LAPACK (scipy): ", scipy.show_config(mode="dicts").get("lapack_opt_info", "unknown"))


def run_benchmark(mode, ctx, x0, spec):
    gc.collect()
    cache = DomainEvaluatorCache()

    if mode == DerivativeMode.ANALYTICAL:
        exchange = NominalStateExchange()
        exchange.enable()
        cache.nominal_exchange = exchange

    t0 = time.perf_counter()
    pr = polish_basin(_basin(ctx, cache, x0), 0, ctx, cache, spec)
    t1 = time.perf_counter()

    telemetry = pr.telemetry

    nom_hits = 0
    reuse_hits = 0
    fallback = telemetry.fallback_reason

    if mode == DerivativeMode.ANALYTICAL:
        exchange = cache.nominal_exchange
        nom_hits = getattr(exchange, "lookup_hits", 0)
        # We can also check transaction reuse if tracked

    return {
        "wall_time": t1 - t0,
        "nfev": pr.n_evaluations,
        "njev": pr.n_iterations,
        "pr": pr,
        "fallback": fallback,
        "nom_hits": nom_hits
    }

def main():
    print_env()
    print("\n--- WARM-UP ---")
    ctx_warm = _build_custom_case(n_cells=1, base_grid_points=10)
    x0_warm = np.array([0.5, 0.5, 0.5])
    run_benchmark(DerivativeMode.REFERENCE_FD, ctx_warm, x0_warm, _spec(DerivativeMode.REFERENCE_FD, max_iter=2))
    run_benchmark(DerivativeMode.ANALYTICAL, ctx_warm, x0_warm, _spec(DerivativeMode.ANALYTICAL, max_iter=2))

    print("\n--- REPRESENTATIVE BENCHMARK ---")
    # Representative: 2 cells, 100 points
    ctx_rep = _build_custom_case(n_cells=2, base_grid_points=100)
    Np_rep = ctx_rep.domain.dimension
    Ng_rep = ctx_rep.hard_layout.n
    x0_rep = np.array([0.4]*Np_rep)
    spec_fd = _spec(DerivativeMode.REFERENCE_FD, max_iter=50)
    spec_an = _spec(DerivativeMode.ANALYTICAL, max_iter=50)

    print(f"Np: {Np_rep}, Ng: {Ng_rep}, target: 1, off-target rows: 99")

    order = [
        (DerivativeMode.REFERENCE_FD, spec_fd, "FD"),
        (DerivativeMode.ANALYTICAL, spec_an, "ANALYTICAL"),
        (DerivativeMode.ANALYTICAL, spec_an, "ANALYTICAL"),
        (DerivativeMode.REFERENCE_FD, spec_fd, "FD"),
        (DerivativeMode.REFERENCE_FD, spec_fd, "FD"),
        (DerivativeMode.ANALYTICAL, spec_an, "ANALYTICAL"),
    ]

    rep_results = []
    for mode, spec, name in order:
        res = run_benchmark(mode, ctx_rep, x0_rep.copy(), spec)
        print(f"[{name}] Wall: {res['wall_time']:.4f}s, nfev: {res['nfev']}, njev: {res['njev']}")
        res['name'] = name
        rep_results.append(res)

    print("\n--- PATHOLOGICAL BENCHMARK ---")
    # Pathological: 6 cells, 1200 points
    ctx_patho = _build_custom_case(n_cells=6, base_grid_points=1200)
    Np_patho = ctx_patho.domain.dimension
    Ng_patho = ctx_patho.hard_layout.n
    x0_patho = np.array([0.4]*Np_patho)

    # Cap pathological to 15 iterations so it doesn't take 10 minutes for FD
    spec_fd_p = _spec(DerivativeMode.REFERENCE_FD, max_iter=15)
    spec_an_p = _spec(DerivativeMode.ANALYTICAL, max_iter=15)

    print(f"Np: {Np_patho}, Ng: {Ng_patho}, target: 1, off-target rows: 1199")

    order_patho = [
        (DerivativeMode.REFERENCE_FD, spec_fd_p, "FD"),
        (DerivativeMode.ANALYTICAL, spec_an_p, "ANALYTICAL"),
        (DerivativeMode.ANALYTICAL, spec_an_p, "ANALYTICAL"),
        (DerivativeMode.REFERENCE_FD, spec_fd_p, "FD"),
    ]

    patho_results = []
    for mode, spec, name in order_patho:
        res = run_benchmark(mode, ctx_patho, x0_patho.copy(), spec)
        print(f"[{name}] Wall: {res['wall_time']:.4f}s, nfev: {res['nfev']}, njev: {res['njev']}")
        res['name'] = name
        patho_results.append(res)

def compare(pr_fd, pr_an):
    x_fd = pr_fd.post_polish.x
    x_an = pr_an.post_polish.x
    if np.allclose(x_fd, x_an, rtol=1e-5, atol=1e-5):
        return "EQUIVALENT_SAME_ENDPOINT"
    # check objective
    obj_fd = pr_fd.post_polish.metrics.objective_value
    obj_an = pr_an.post_polish.metrics.objective_value
    if np.isclose(obj_fd, obj_an, rtol=1e-3):
        return "EQUIVALENT_DIFFERENT_ENDPOINT"
    if obj_an < obj_fd:
        return "ANALYTICAL_BETTER"
    return "NOT_EQUIVALENT"

    # Scientific Equivalence
    print("\n--- SCIENTIFIC EQUIVALENCE ---")
    pr_fd = rep_results[0]['pr']
    pr_an = rep_results[1]['pr']
    print(f"Representative Equivalence: {compare(pr_fd, pr_an)}")

    pr_fd_p = patho_results[0]['pr']
    pr_an_p = patho_results[1]['pr']
    print(f"Pathological Equivalence: {compare(pr_fd_p, pr_an_p)}")

    # Rerun rep for reproducibility
    print("\n--- REPRODUCIBILITY RERUN ---")
    res_repro = run_benchmark(DerivativeMode.ANALYTICAL, ctx_rep, x0_rep.copy(), spec_an)
    print(f"ANALYTICAL Wall: {res_repro['wall_time']:.4f}s (vs {rep_results[1]['wall_time']:.4f}s)")

if __name__ == "__main__":
    main()
