import numpy as np

from foster_eom.optimize.evaluator import DomainEvaluatorCache
from tests.unit.test_p12_5_g3_scientific_equivalence import _build_custom_case, _run_pair

ctx_c = _build_custom_case(n_cells=3)
pr_fd_c, pr_an_c = _run_pair(ctx_c, np.full(7, 0.5))
print(f'G3-C FD: {pr_fd_c.retained.objective_value:.12e}, AN: {pr_an_c.retained.objective_value:.12e}')

ctx_f = _build_custom_case(n_cells=2)
pr_fd_f, pr_an_f = _run_pair(ctx_f, np.array([0.5, 0.5, 0.49, 0.5, 0.51]))
print(f'G3-F FD: {pr_fd_f.retained.objective_value:.12e}, AN: {pr_an_f.retained.objective_value:.12e}')

ctx_j = _build_custom_case(n_cells=6, base_grid_points=1201)
pr_fd_j, _ = _run_pair(ctx_j, np.full(13, 0.5), max_iter=2)
cache = DomainEvaluatorCache()
val = ctx_j.evaluator.evaluate(pr_fd_j.retained.x, cache)
print(f'G3-J Ng: {len(val.hard_constraints)}')
