import gc
import time

import numpy as np

from foster_eom.domain.objectives import DerivativeMode
from foster_eom.optimize.evaluator import DomainEvaluatorCache
from foster_eom.optimize.local_polish import polish_basin
from foster_eom.optimize.nominal_state import NominalStateExchange
from tests.unit.test_p12_5_e_analytical_polish import _basin, _spec
from tests.unit.test_p12_5_g3_scientific_equivalence import _build_custom_case


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
    return t1 - t0, pr

def compare(pr_fd, pr_an):
    x_fd = pr_fd.post_polish.x
    x_an = pr_an.post_polish.x
    if np.allclose(x_fd, x_an, rtol=1e-5, atol=1e-5):
        return "EQUIVALENT_SAME_ENDPOINT"
    obj_fd = pr_fd.retained.objective_value
    obj_an = pr_an.retained.objective_value
    if np.isclose(obj_fd, obj_an, rtol=1e-3):
        return "EQUIVALENT_DIFFERENT_ENDPOINT"
    if obj_an < obj_fd:
        return "ANALYTICAL_BETTER"
    return "NOT_EQUIVALENT"

def main():
    # Representative
    ctx_rep = _build_custom_case(n_cells=2, base_grid_points=100)
    x0_rep = np.array([0.4]*ctx_rep.domain.dimension)
    spec_fd = _spec(DerivativeMode.REFERENCE_FD, max_iter=50)
    spec_an = _spec(DerivativeMode.ANALYTICAL, max_iter=50)

    t_fd, pr_fd = run_benchmark(DerivativeMode.REFERENCE_FD, ctx_rep, x0_rep.copy(), spec_fd)
    t_an, pr_an = run_benchmark(DerivativeMode.ANALYTICAL, ctx_rep, x0_rep.copy(), spec_an)

    print(f"Representative Equivalence: {compare(pr_fd, pr_an)}")

    # Rerun for reproducibility
    t_an2, _ = run_benchmark(DerivativeMode.ANALYTICAL, ctx_rep, x0_rep.copy(), spec_an)
    print(f"REPRODUCIBILITY: ANALYTICAL Wall: {t_an2:.4f}s")

    # Pathological
    ctx_patho = _build_custom_case(n_cells=6, base_grid_points=1200)
    x0_patho = np.array([0.4]*ctx_patho.domain.dimension)
    spec_fd_p = _spec(DerivativeMode.REFERENCE_FD, max_iter=15)
    spec_an_p = _spec(DerivativeMode.ANALYTICAL, max_iter=15)

    _, pr_fd_p = run_benchmark(DerivativeMode.REFERENCE_FD, ctx_patho, x0_patho.copy(), spec_fd_p)
    _, pr_an_p = run_benchmark(DerivativeMode.ANALYTICAL, ctx_patho, x0_patho.copy(), spec_an_p)

    print(f"Pathological Equivalence: {compare(pr_fd_p, pr_an_p)}")

if __name__ == "__main__":
    main()
