"""P10 robustness analysis package.

Provides manufacturing / parameter robustness analysis for a frozen P09 CatalogCombo.

Workflow::

    from foster_eom.robustness.runner import run_robustness
    from foster_eom.robustness.sampler import RobustnessSpec

    spec = RobustnessSpec(n_samples=500, seed=42, method="random")
    result = run_robustness(combo, base_graph, context, library, spec=spec)
    print(f"yield_evaluable = {result.yield_stats.yield_evaluable:.3f}")
"""
