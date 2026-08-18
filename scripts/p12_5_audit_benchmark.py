"""P12.5-A Benchmark Script.

Runs the five representative optimization workloads with instrumentation.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import scipy

from foster_eom.domain.project import ProjectSpec
from foster_eom.foster.seed import generate_seeds
from foster_eom.optimize.engine import run_optimization
from foster_eom.optimize.perf import perf_context


def _print_env_info():
    print("=" * 80)
    print("P12.5-A Environment Audit")
    print("=" * 80)
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"CPU: {platform.processor()}")
    print(f"Logical Cores: {psutil.cpu_count(logical=True)}")
    print(f"Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Python: {sys.version.replace(chr(10), ' ')}")
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {scipy.__version__}")

    # Thread settings
    for var in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print("-" * 80)


def _build_args(spec: ProjectSpec):
    # Map ProjectSpec to optimization args
    from foster_eom.models.factory import build_eom_model

    eom_model = build_eom_model(spec.eom)

    targets_hz = tuple(t.frequency_hz for t in spec.frequencies.targets if t.enabled)
    v_targets = tuple(t.voltage_target_rms_v for t in spec.frequencies.targets if t.enabled)

    return {
        "opt_spec": spec.optimization,
        "source_spec": spec.source,
        "eom_model": eom_model,
        "component_limits": spec.components.continuous_limits,
        "match_constraints": spec.matching,
        "stress_constraints": spec.stress,
        "target_frequencies_hz": targets_hz,
        "sweep_f_min_hz": spec.frequencies.sweep_f_min_hz,
        "sweep_f_max_hz": spec.frequencies.sweep_f_max_hz,
        "base_grid_points": spec.frequencies.base_grid_points,
        "voltage_targets_rms_v": v_targets,
        "topology_spec": spec.topology,
    }


def run_case(name: str, spec: ProjectSpec, n_runs: int = 1):
    print(f"\n--- Running {name} ---")
    args = _build_args(spec)

    for run_idx in range(n_runs):
        print(f"  Run {run_idx + 1}/{n_runs}:")

        # 1. Generate seeds
        t0 = time.perf_counter()
        cpu0 = time.process_time()

        seed_res = generate_seeds(
            r_match_ohm=args["source_spec"].z_source_real_ohm,
            source_spec=args["source_spec"],
            eom_model=args["eom_model"],
            f_targets_hz=np.array(args["target_frequencies_hz"]),
            topo_spec=args["topology_spec"],
            component_limits=args["component_limits"],
        )

        # 2. Run optimization with perf_context ON
        with perf_context() as perf:
            res = run_optimization(
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

        t1 = time.perf_counter()
        cpu1 = time.process_time()

        wall_time = t1 - t0
        cpu_time = cpu1 - cpu0

        # Count duplicates
        n_unique_x = len(perf.eval_counts_by_x)
        total_evals = sum(perf.eval_counts_by_x.values())
        duplicates = total_evals - n_unique_x
        dup_frac = duplicates / total_evals if total_evals > 0 else 0.0

        print(
            f"    Wall time: {wall_time:.3f}s (CPU: {cpu_time:.3f}s, Util: {(cpu_time / wall_time) * 100:.1f}%)"
        )
        print(f"    Perf: Polish time={perf.polish_time:.3f}s, DE={perf.de_time:.3f}s")
        print(
            f"    Total Polish Evals: {total_evals}, Unique X: {n_unique_x}, Duplicates: {duplicates} ({dup_frac * 100:.1f}%)"
        )
        print(f"    MNA Solves: {perf.mna_solves}, Frequencies: {perf.frequencies_solved}")
        print(f"    Peak RSS: {perf.peak_rss_mb:.1f} MB")

        best = res.best_feasible or res.near_feasible_best
        if best:
            print(
                f"    Best Feasible: {res.best_feasible is not None}, Obj={best.objective_terms.get('total', 0):.4f}"
            )
        else:
            print("    NO BEST FOUND.")

        print("    Per-basin breakdown:")
        for b, pt in perf.basin_polish_time.items():
            print(
                f"      {b}: {pt:.3f}s, {perf.basin_nit[b]} iter, {perf.basin_nfev[b]} f_ev, {perf.basin_njev[b]} j_ev, {perf.basin_mna_solves[b]} MNA solves, Status={perf.basin_status[b]}"
            )

        if name == "PATHOLOGICAL_GUI_SURROGATE":
            print("    RSS Trace:")
            for lbl, r in perf.rss_history:
                print(f"      {lbl}: {r:.1f} MB")


def main():
    _print_env_info()

    # Base YAML path
    base_yaml = Path(__file__).parent.parent / "fs-theo" / "examples" / "design_spec.example.yaml"
    import yaml

    from foster_eom.persistence.yaml_io import _dict_to_spec

    with open(base_yaml, encoding="utf-8") as f:
        base_dict = yaml.safe_load(f)

    # Construct 5 cases
    import copy

    # 1. PATHOLOGICAL_GUI_SURROGATE
    dict_patho = copy.deepcopy(base_dict)
    dict_patho["optimization"]["global"]["workers"] = 1
    spec_patho = _dict_to_spec(dict_patho)

    # 2. FROZEN_SMALL
    dict_small = copy.deepcopy(base_dict)
    dict_small["frequencies"]["targets"] = dict_small["frequencies"]["targets"][:1]
    dict_small["topology"]["branch1_cells"]["max"] = 1
    dict_small["topology"]["branch2_cells"]["max"] = 1
    dict_small["optimization"]["global"]["method"] = "differential_evolution"
    dict_small["optimization"]["global"]["max_evaluations"] = 1000
    dict_small["optimization"]["global"]["workers"] = 1
    dict_small["optimization"]["local"]["polish_top_k"] = 1
    spec_small = _dict_to_spec(dict_small)

    # 3. FROZEN_TYPICAL
    dict_typ = copy.deepcopy(base_dict)
    dict_typ["frequencies"]["targets"] = dict_typ["frequencies"]["targets"][:2]
    dict_typ["topology"]["branch1_cells"]["max"] = 2
    dict_typ["optimization"]["global"]["max_evaluations"] = 5000
    dict_typ["optimization"]["global"]["workers"] = 1
    dict_typ["optimization"]["local"]["polish_top_k"] = 3
    spec_typ = _dict_to_spec(dict_typ)

    # 4. FROZEN_LARGE
    dict_large = copy.deepcopy(base_dict)
    dict_large["topology"]["branch1_cells"]["min"] = 3
    dict_large["topology"]["branch1_cells"]["max"] = 3
    dict_large["topology"]["branch2_cells"]["min"] = 3
    dict_large["topology"]["branch2_cells"]["max"] = 3
    dict_large["optimization"]["global"]["max_evaluations"] = 2000
    dict_large["optimization"]["global"]["workers"] = 1
    dict_large["optimization"]["local"]["polish_top_k"] = 3
    spec_large = _dict_to_spec(dict_large)

    # 5. MULTI_FREQUENCY
    dict_multi = copy.deepcopy(base_dict)
    dict_multi["frequencies"]["targets"] = [
        {"frequency_hz": 8e6, "voltage_target_rms_v": 20.0, "label": "f1"},
        {"frequency_hz": 9e6, "voltage_target_rms_v": 20.0, "label": "f2"},
        {"frequency_hz": 10e6, "voltage_target_rms_v": 20.0, "label": "f3"},
        {"frequency_hz": 11e6, "voltage_target_rms_v": 20.0, "label": "f4"},
        {"frequency_hz": 12e6, "voltage_target_rms_v": 20.0, "label": "f5"},
    ]
    dict_multi["optimization"]["global"]["max_evaluations"] = 2000
    dict_multi["optimization"]["global"]["workers"] = 1
    dict_multi["optimization"]["local"]["polish_top_k"] = 2
    spec_multi = _dict_to_spec(dict_multi)

    # Construct 5 cases
    import copy

    print("Executing Benchmarks...")
    run_case("FROZEN_SMALL", spec_small, n_runs=3)
    run_case("FROZEN_TYPICAL", spec_typ, n_runs=1)
    run_case("FROZEN_LARGE", spec_large, n_runs=1)
    run_case("MULTI_FREQUENCY", spec_multi, n_runs=1)
    run_case("PATHOLOGICAL_GUI_SURROGATE", spec_patho, n_runs=1)


if __name__ == "__main__":
    main()
