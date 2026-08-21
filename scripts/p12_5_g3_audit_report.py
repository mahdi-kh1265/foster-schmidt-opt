import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tests.unit.test_p12_5_g3_scientific_equivalence import X0, _build_custom_case, _run_pair


def run_case(name, ctx, x_start, max_iter=None):
    print(f"Running {name}...")
    try:
        t0 = time.time()
        pr_fd, pr_an = _run_pair(ctx, x_start, max_iter=max_iter)
        t1 = time.time()
        print(f"  Done in {t1-t0:.2f}s")
        return {
            "name": name,
            "pr_fd": pr_fd,
            "pr_an": pr_an,
            "ctx": ctx,
            "x_start": x_start
        }
    except Exception as e:
        print(f"  Failed: {e}")
        return None

def main():
    cases = []

    # G3-A
    ctx = _build_custom_case()
    cases.append(run_case("G3-A Small deterministic", ctx, X0))

    # G3-B
    ctx = _build_custom_case(base_grid_points=15)
    x_rep = np.array([0.2, 0.4, 0.6])
    cases.append(run_case("G3-B Representative", ctx, x_rep))

    # G3-C
    ctx = _build_custom_case(n_cells=3) # dimension = 7
    x_start = np.full(7, 0.5)
    cases.append(run_case("G3-C Multipole", ctx, x_start))

    # G3-D
    ctx = _build_custom_case(n_targets=4)
    x_start = np.array([0.5, 0.5, 0.5])
    cases.append(run_case("G3-D Multi-frequency", ctx, x_start))

    # G3-E
    ctx = _build_custom_case()
    x_bound = np.array([-0.05, 1.05, 0.5])
    cases.append(run_case("G3-E Boundary clipped", ctx, x_bound))

    # G3-F
    ctx = _build_custom_case(n_cells=2) # dimension = 5
    x_start = np.array([0.5, 0.5, 0.49, 0.5, 0.51])
    cases.append(run_case("G3-F Pole separation", ctx, x_start))

    # G3-G
    ctx = _build_custom_case(w_loss=1.0)
    cases.append(run_case("G3-G Hard + soft", ctx, X0))

    # G3-H
    ctx = _build_custom_case(lossy_eom=True)
    cases.append(run_case("G3-H Loss enabled", ctx, X0))

    # G3-I
    ctx = _build_custom_case()
    starts = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.5, 0.5, 0.5]),
        np.array([0.9, 0.8, 0.7]),
    ]
    for i, start_x in enumerate(starts):
        cases.append(run_case(f"G3-I Multi-start {i+1}", ctx, start_x))

    # G3-J
    ctx = _build_custom_case(n_cells=6, base_grid_points=1201)
    x_start = np.full(13, 0.5)
    cases.append(run_case("G3-J Pathological", ctx, x_start, max_iter=2))

    output_file = os.path.join(os.path.dirname(__file__), '..', 'scratch', 'g3_report.md')
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n--- RESULTS ---\n")
        header = "| Case | Start | Np | Ng | FD feasible | Analytical feasible | FD hard violation | Analytical hard violation | FD objective | Analytical objective | Same endpoint? | Max |Δu| | Unexpected fallback | Verdict |"
        f.write(header + "\n")
        f.write("|" + "|".join(["---"] * 15) + "|\n")

        for res in cases:
            if not res:
                continue
            name = res["name"]
            pr_fd = res["pr_fd"]
            pr_an = res["pr_an"]
            ctx = res["ctx"]

            start_str = str(list(np.round(res["x_start"], 2)))
            Np = ctx.domain.dimension
            Ng = "N/A"
            fd_feas = pr_fd.retained.feasible
            an_feas = pr_an.retained.feasible
            fd_viol = pr_fd.retained.v_max
            an_viol = pr_an.retained.v_max
            fd_obj = pr_fd.retained.objective_value
            an_obj = pr_an.retained.objective_value

            diff = np.max(np.abs(np.array(pr_fd.retained.x) - np.array(pr_an.retained.x)))
            same_ep = diff < 1e-5

            fallback = "Yes" if pr_an.telemetry.fallback_reason else "No"

            if same_ep:
                verdict = "EQUIVALENT_SAME_ENDPOINT"
            else:
                if (an_feas and not fd_feas) or an_obj < fd_obj - 1e-6:
                    verdict = "ANALYTICAL_BETTER"
                else:
                    verdict = "EQUIVALENT_DIFFERENT_ENDPOINT"

            f.write(f"| {name} | {start_str} | {Np} | {Ng} | {fd_feas} | {an_feas} | {fd_viol:.3e} | {an_viol:.3e} | {fd_obj:.3e} | {an_obj:.3e} | {same_ep} | {diff:.3e} | {fallback} | {verdict} |\n")
    print(f"Report written to {output_file}")

if __name__ == "__main__":
    main()
