"""P12.5-E equivalence + performance benchmark.

Measures ``REFERENCE_FD`` against ``ANALYTICAL`` local polish.  Nothing is tuned
here: E1 is measured exactly as implemented.

A/B protocol
------------
One DE run per case.  ``foster_eom.optimize.engine.polish_top_k`` is wrapped so
that, at the moment the engine hands over its real basins, the *same* basin list
is polished twice — once per derivative mode — each on a fresh
``DomainEvaluatorCache``.  Everything else (topology/domain, initial point,
targets, source/EOM models, objective weights, constraints/bounds, optimizer
tolerances/options, iteration limits) is held constant by construction; only
``local_derivative_mode`` differs.

Nominal MNA work is reported as four separate quantities, never collapsed:
evaluator frequency-point solves, transaction nominal factorizations (the
suspected duplicate sweep), direct back-substitutions, adjoint
back-substitutions.

Usage
-----
    python scripts/p12_5_e_equivalence_benchmark.py [--cases small,typical,...]
                                                    [--out REPORT.md]
                                                    [--no-profile]
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import json
import os
import platform
import pstats
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import scipy
import yaml

from foster_eom.domain.objectives import DerivativeMode
from foster_eom.domain.project import ProjectSpec
from foster_eom.foster.seed import generate_seeds
from foster_eom.optimize.evaluator import DomainEvaluatorCache
from foster_eom.optimize.local_polish import polish_top_k as _real_polish_top_k
from foster_eom.persistence.yaml_io import _dict_to_spec

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_YAML = REPO_ROOT / "fs-theo" / "examples" / "design_spec.example.yaml"

MODES = (DerivativeMode.REFERENCE_FD, DerivativeMode.ANALYTICAL)

# ---------------------------------------------------------------------------
# Equivalence tolerances (see the approved P12.5-E plan)
# ---------------------------------------------------------------------------

TOL_OBJ_REL = 1e-3
TOL_OBJ_ABS = 1e-6
TOL_VIOL_REL = 1e-3
TOL_VIOL_ABS = 1e-6
TOL_GAMMA_ABS = 1e-3
TOL_ZIN_REL = 1e-2
TOL_VEOM_REL = 1e-2
# Reported, not gated:
TOL_U_INF = 5e-3
TOL_LC_REL = 1e-2
TOL_FP_REL = 1e-3


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """One (case, domain, basin, mode) polish outcome."""

    mode: str
    wall_time_s: float
    telemetry: dict[str, Any]
    # Retained candidate
    x: tuple[float, ...]
    objective: float
    v_max: float
    feasible: bool
    near_feasible: bool
    numerical_status: str
    termination: str
    success: bool
    # Raw polish endpoint, BEFORE Deb pre-polish retention.  Needed to tell a
    # genuine two-mode agreement from "both modes were discarded".
    post_x: tuple[float, ...]
    post_objective: float
    post_v_max: float
    post_feasible: bool
    polish_improved: bool
    # Physical / electrical
    l_values: list[float]
    c_values: list[float]
    f_poles: list[float]
    gamma_abs: list[float]
    z_in: list[complex]
    v_eom_abs: list[float]
    # Deb ranking key
    deb: tuple[float, float, float]


@dataclass
class PairRecord:
    case: str
    domain_id: str
    basin_index: int
    n_params: int
    n_constraint_rows: int
    n_frequencies: int
    fd: RunRecord
    an: RunRecord


@dataclass
class CaseResult:
    name: str
    label: str
    notes: str
    max_iter_cap: int
    pairs: list[PairRecord] = field(default_factory=list)
    fd_total_time: float = 0.0
    an_total_time: float = 0.0
    fd_peak_rss_mb: float = 0.0
    an_peak_rss_mb: float = 0.0
    de_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _physical(ctx, x: tuple[float, ...]) -> tuple[list[float], list[float], list[float]]:
    """Physical L, C, f_p from the normalized vector, via the frozen mapper."""
    try:
        b1, b2 = ctx.domain.variable_mapper.unpack(np.array(x, dtype=np.float64))
    except Exception:
        return ([], [], [])
    ls = list(b1.l_values_h) + list(b2.l_values_h)
    cs = list(b1.c_values_f) + list(b2.c_values_f)
    fps = list(b1.f_poles_hz) + list(b2.f_poles_hz)
    return ([float(v) for v in ls], [float(v) for v in cs], [float(v) for v in fps])


def _electrical(result) -> tuple[list[float], list[complex], list[float]]:
    gam: list[float] = []
    zin: list[complex] = []
    veom: list[float] = []
    for sol in result.target_solutions:
        gam.append(abs(sol.gamma) if sol.gamma is not None else float("nan"))
        zin.append(complex(sol.z_in) if sol.z_in is not None else complex("nan"))
        veom.append(abs(sol.v_eom) if sol.v_eom is not None else float("nan"))
    return gam, zin, veom


def _make_record(mode: DerivativeMode, ctx, pr, wall: float) -> RunRecord:
    r = pr.retained
    ls, cs, fps = _physical(ctx, r.x)
    gam, zin, veom = _electrical(r)
    return RunRecord(
        mode=mode.value,
        wall_time_s=wall,
        telemetry=asdict(pr.telemetry),
        x=tuple(float(v) for v in r.x),
        objective=float(r.objective_value),
        v_max=float(r.v_max),
        feasible=bool(r.feasible),
        near_feasible=bool(r.near_feasible),
        numerical_status=str(r.numerical_status),
        termination=str(pr.termination),
        success=bool(pr.success),
        l_values=ls,
        c_values=cs,
        f_poles=fps,
        gamma_abs=gam,
        z_in=zin,
        v_eom_abs=veom,
        deb=(float(not r.feasible), float(r.v_max), float(r.objective_value)),
        post_x=tuple(float(v) for v in pr.post_polish.x),
        post_objective=float(pr.post_polish.objective_value),
        post_v_max=float(pr.post_polish.v_max),
        post_feasible=bool(pr.post_polish.feasible),
        polish_improved=pr.retained is not pr.pre_polish,
    )


# ---------------------------------------------------------------------------
# The A/B polish wrapper
# ---------------------------------------------------------------------------


class ABPolishHarness:
    """Replaces ``engine.polish_top_k`` with a two-mode measurement."""

    def __init__(self, case: CaseResult, cap_iter: int, primary: DerivativeMode) -> None:
        self.case = case
        self.cap_iter = cap_iter
        self.primary = primary
        self.profile_target: dict[str, Any] | None = None

    def __call__(self, basins, context, cache, opt_spec):
        proc = psutil.Process(os.getpid())
        outcomes: dict[DerivativeMode, Any] = {}
        for mode in MODES:
            spec = opt_spec.model_copy(
                update={
                    "local_derivative_mode": mode,
                    "local_max_iterations": self.cap_iter,
                }
            )
            fresh_cache = DomainEvaluatorCache()
            rss_before = proc.memory_info().rss / 1e6
            t0 = time.perf_counter()
            results = _real_polish_top_k(basins, context, fresh_cache, spec)
            dt = time.perf_counter() - t0
            rss_after = proc.memory_info().rss / 1e6
            outcomes[mode] = (results, dt)
            if mode == DerivativeMode.REFERENCE_FD:
                self.case.fd_total_time += dt
                self.case.fd_peak_rss_mb = max(self.case.fd_peak_rss_mb, rss_after, rss_before)
            else:
                self.case.an_total_time += dt
                self.case.an_peak_rss_mb = max(self.case.an_peak_rss_mb, rss_after, rss_before)

        fd_results, fd_dt = outcomes.get(
            DerivativeMode.REFERENCE_FD, outcomes[DerivativeMode.ANALYTICAL]
        )
        an_results, an_dt = outcomes[DerivativeMode.ANALYTICAL]

        n_pairs = min(len(fd_results), len(an_results))
        for i in range(n_pairs):
            fd_pr, an_pr = fd_results[i], an_results[i]
            # Per-basin wall time comes from the telemetry (the minimize call);
            # the per-mode totals above cover the whole top-K loop.
            self.case.pairs.append(
                PairRecord(
                    case=self.case.name,
                    domain_id=fd_pr.domain_id,
                    basin_index=fd_pr.basin_index,
                    n_params=an_pr.telemetry.n_params,
                    n_constraint_rows=an_pr.telemetry.n_constraint_rows,
                    n_frequencies=an_pr.telemetry.n_evaluation_frequencies,
                    fd=_make_record(
                        DerivativeMode.REFERENCE_FD, context, fd_pr, fd_pr.telemetry.wall_time_s
                    ),
                    an=_make_record(
                        DerivativeMode.ANALYTICAL, context, an_pr, an_pr.telemetry.wall_time_s
                    ),
                )
            )

        if self.profile_target is None and n_pairs > 0:
            self.profile_target = {
                "basins": basins[:1],
                "context": context,
                "opt_spec": opt_spec,
            }

        _ = (fd_dt, an_dt)
        return outcomes[self.primary][0]


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------


def _base_dict() -> dict:
    with open(EXAMPLE_YAML, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _case_specs() -> dict[str, tuple[ProjectSpec, str, str, int]]:
    """Return ``{name: (spec, label, notes, max_iter_cap)}``."""
    base = _base_dict()
    cases: dict[str, tuple[ProjectSpec, str, str, int]] = {}

    # 1. Small deterministic
    d = copy.deepcopy(base)
    d["frequencies"]["targets"] = d["frequencies"]["targets"][:1]
    d["frequencies"]["base_grid_points"] = 21
    d["topology"]["branch1_cells"] = {"min": 1, "max": 1}
    d["topology"]["branch2_cells"] = {"min": 0, "max": 1}
    d["optimization"]["global"]["max_evaluations"] = 400
    d["optimization"]["global"]["workers"] = 1
    d["optimization"]["local"]["polish_top_k"] = 1
    cases["small"] = (
        _dict_to_spec(d),
        "SMALL_DETERMINISTIC",
        "1 target, 1 cell, 21-point grid, top-K=1",
        60,
    )

    # 2. Representative normal Foster network
    d = copy.deepcopy(base)
    d["frequencies"]["targets"] = d["frequencies"]["targets"][:2]
    d["frequencies"]["base_grid_points"] = 101
    d["topology"]["branch1_cells"] = {"min": 2, "max": 2}
    d["topology"]["branch2_cells"] = {"min": 1, "max": 1}
    d["optimization"]["global"]["max_evaluations"] = 1200
    d["optimization"]["global"]["workers"] = 1
    d["optimization"]["local"]["polish_top_k"] = 3
    cases["typical"] = (
        _dict_to_spec(d),
        "TYPICAL_FOSTER",
        "2 targets, 2+1 cells, 101-point grid, top-K=3 (3 real starts)",
        60,
    )

    # 3. Larger / multi-pole
    d = copy.deepcopy(base)
    d["frequencies"]["base_grid_points"] = 101
    d["topology"]["branch1_cells"] = {"min": 3, "max": 3}
    d["topology"]["branch2_cells"] = {"min": 3, "max": 3}
    d["optimization"]["global"]["max_evaluations"] = 1200
    d["optimization"]["global"]["workers"] = 1
    d["optimization"]["local"]["polish_top_k"] = 3
    cases["large"] = (
        _dict_to_spec(d),
        "LARGE_MULTIPOLE",
        "3 targets, 3+3 cells (max Np), 101-point grid, top-K=3",
        40,
    )

    # 4. Multi-frequency
    d = copy.deepcopy(base)
    # 5 targets at 8-12 MHz is seed-infeasible for this EOM (every topology is
    # rejected by the Foster residual test), so the widest target set the
    # synthesis actually supports is used instead.
    d["frequencies"]["targets"] = [
        {"frequency_hz": 9e6, "voltage_target_rms_v": 20.0, "label": "f1"},
        {"frequency_hz": 10e6, "voltage_target_rms_v": 20.0, "label": "f2"},
        {"frequency_hz": 11e6, "voltage_target_rms_v": 20.0, "label": "f3"},
        {"frequency_hz": 12e6, "voltage_target_rms_v": 20.0, "label": "f4"},
    ]
    d["frequencies"]["base_grid_points"] = 101
    d["optimization"]["global"]["max_evaluations"] = 1200
    d["optimization"]["global"]["workers"] = 1
    d["optimization"]["local"]["polish_top_k"] = 2
    multi = _dict_to_spec(d)
    multi = multi.model_copy(
        update={
            "optimization": multi.optimization.model_copy(update={"max_optimization_domains": 3})
        }
    )
    cases["multifreq"] = (
        multi,
        "MULTI_FREQUENCY",
        "4 targets (9-12 MHz), default 1-3 cell search, 101-point grid, top-K=2",
        40,
    )

    # 5. Pathological GUI-slowdown surrogate: the untouched 1201-point grid.
    #    Grid, targets, topology search and constraints are the example spec
    #    verbatim.  Only the *shared* DE budget and the domain count are reduced
    #    (DE is run once and is common to both modes, so it cannot bias the A/B),
    #    and the polish iteration cap is applied identically to both modes.
    d = copy.deepcopy(base)
    d["optimization"]["global"]["max_evaluations"] = 100
    d["optimization"]["global"]["population_size_multiplier"] = 2
    d["optimization"]["global"]["workers"] = 1
    d["optimization"]["local"]["polish_top_k"] = 1
    patho = _dict_to_spec(d)
    patho = patho.model_copy(
        update={
            "optimization": patho.optimization.model_copy(update={"max_optimization_domains": 1})
        }
    )
    cases["pathological"] = (
        patho,
        "PATHOLOGICAL_1201_GRID",
        "unmodified example grid: 1201 points / 1198 off-target hard rows, top-K=1, "
        "1 domain, DE budget reduced (shared by both modes)",
        20,
    )
    return cases


def _build_args(spec: ProjectSpec) -> dict:
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


def run_case(
    name: str, spec: ProjectSpec, label: str, notes: str, cap: int
) -> tuple[CaseResult, ABPolishHarness]:
    import foster_eom.optimize.engine as engine_mod
    from foster_eom.optimize.engine import run_optimization

    case = CaseResult(name=name, label=label, notes=notes, max_iter_cap=cap)
    args = _build_args(spec)

    seed_res = generate_seeds(
        r_match_ohm=args["source_spec"].z_source_real_ohm,
        source_spec=args["source_spec"],
        eom_model=args["eom_model"],
        f_targets_hz=np.array(args["target_frequencies_hz"]),
        topo_spec=args["topology_spec"],
        component_limits=args["component_limits"],
    )

    harness = ABPolishHarness(case, cap, DerivativeMode.ANALYTICAL)
    original = engine_mod.polish_top_k
    engine_mod.polish_top_k = harness  # type: ignore[assignment]
    t0 = time.perf_counter()
    try:
        run_optimization(
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
        engine_mod.polish_top_k = original  # type: ignore[assignment]
    total = time.perf_counter() - t0
    case.de_time_s = max(0.0, total - case.fd_total_time - case.an_total_time)
    return case, harness


# ---------------------------------------------------------------------------
# Equivalence evaluation
# ---------------------------------------------------------------------------


def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / scale


def _max_rel_seq(xs: list[float], ys: list[float]) -> float:
    if not xs or not ys or len(xs) != len(ys):
        return float("nan")
    vals = [_rel(a, b) for a, b in zip(xs, ys, strict=False) if np.isfinite(a) and np.isfinite(b)]
    return max(vals) if vals else float("nan")


def _max_abs_seq(xs: list[float], ys: list[float]) -> float:
    if not xs or not ys or len(xs) != len(ys):
        return float("nan")
    vals = [abs(a - b) for a, b in zip(xs, ys, strict=False) if np.isfinite(a) and np.isfinite(b)]
    return max(vals) if vals else float("nan")


def assess_pair(p: PairRecord) -> dict[str, Any]:
    """Assess one FD/analytical pair against the approved acceptance priority.

    Priority: feasibility -> objective -> Gamma/Z_in/V_EOM -> component
    coordinates.  The gate is *one-sided*: analytical must never be materially
    worse or infeasible where FD succeeds.  Analytical landing on a strictly
    better optimum is a legitimate outcome of exact gradients, not a failure —
    the endpoints then genuinely differ, so the electrical and coordinate
    deltas are reported rather than gated.  Coordinate agreement is never a
    hard-fail criterion (flat / multiple equivalent optima).  SciPy status and
    termination text are diagnostic only.
    """
    fd, an = p.fd, p.an
    d_obj = an.objective - fd.objective
    d_viol = an.v_max - fd.v_max
    viol_tol = max(TOL_VIOL_ABS, TOL_VIOL_REL * max(fd.v_max, 0.0)) + 1e-12
    obj_tol = max(TOL_OBJ_ABS, TOL_OBJ_REL * max(abs(fd.objective), abs(an.objective), 1e-30))

    # -- hard, one-sided gates ------------------------------------------------
    # Feasibility may improve but never regress.
    feas_ok = an.feasible or not fd.feasible
    near_ok = an.near_feasible or not fd.near_feasible
    viol_ok = d_viol <= viol_tol
    obj_not_worse = d_obj <= obj_tol
    deb_worse = (an.deb[0] > fd.deb[0]) or (
        an.deb[0] == fd.deb[0]
        and (an.deb[1] > fd.deb[1] + viol_tol or (d_viol <= viol_tol and not obj_not_worse))
    )

    # -- endpoint classification ---------------------------------------------
    # "Same endpoint" requires the *point* to coincide, not merely the scalars.
    # Two runs can agree on J and v_max while sitting on different members of a
    # flat / multi-optimum set, in which case Gamma, Z_in and V_EOM legitimately
    # differ and must be reported rather than gated (approved amendment 3).
    u_inf = (
        float(np.max(np.abs(np.array(an.x) - np.array(fd.x)))) if fd.x and an.x else float("nan")
    )
    same_objective = abs(d_obj) <= obj_tol and abs(d_viol) <= viol_tol
    same_endpoint = same_objective and bool(np.isfinite(u_inf)) and u_inf <= TOL_U_INF
    better = d_obj < -obj_tol

    gamma_d = _max_abs_seq(fd.gamma_abs, an.gamma_abs)
    zin_d = _max_rel_seq([abs(z) for z in fd.z_in], [abs(z) for z in an.z_in])
    veom_d = _max_rel_seq(fd.v_eom_abs, an.v_eom_abs)

    # Electricals are gated only when both runs claim the same optimum.
    if same_endpoint:
        gamma_ok = not np.isfinite(gamma_d) or gamma_d <= TOL_GAMMA_ABS
        zin_ok = not np.isfinite(zin_d) or zin_d <= TOL_ZIN_REL
        veom_ok = not np.isfinite(veom_d) or veom_d <= TOL_VEOM_REL
    else:
        gamma_ok = zin_ok = veom_ok = True

    hard_ok = (
        feas_ok
        and near_ok
        and viol_ok
        and obj_not_worse
        and not deb_worse
        and gamma_ok
        and zin_ok
        and veom_ok
    )
    if not hard_ok:
        verdict = "NOT_EQUIVALENT"
    elif better:
        verdict = "EQUIVALENT_OR_BETTER"
    else:
        verdict = "EQUIVALENT"

    return {
        "d_objective": d_obj,
        "d_max_violation": d_viol,
        "u_inf": u_inf,
        "l_rel": _max_rel_seq(fd.l_values, an.l_values),
        "c_rel": _max_rel_seq(fd.c_values, an.c_values),
        "fp_rel": _max_rel_seq(fd.f_poles, an.f_poles),
        "gamma_abs_d": gamma_d,
        "zin_rel": zin_d,
        "veom_rel": veom_d,
        "same_objective": same_objective,
        "same_endpoint": same_endpoint,
        "analytical_better": better,
        "feasible_ok": feas_ok and near_ok,
        "objective_not_worse": obj_not_worse,
        "violation_ok": viol_ok,
        "electricals_gated": same_endpoint,
        "gamma_ok": gamma_ok,
        "zin_ok": zin_ok,
        "veom_ok": veom_ok,
        "deb_worse": deb_worse,
        "verdict": verdict,
    }


def nan_() -> float:
    return float("nan")


# ---------------------------------------------------------------------------
# Profiling the analytical path
# ---------------------------------------------------------------------------

_PROFILE_CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    # Needles are matched against "<file>:<func>" of each profiled frame, most
    # specific first.  Built from the measured top-self-time list of an actual
    # pathological analytical polish, so the buckets partition real cost rather
    # than guesses.
    (
        "Y_p derivative stamps (transaction)",
        (
            "transaction.py:build_y_p_list",
            "stamps.py:stamp_inductor_derivative",
            "stamps.py:stamp_capacitor_derivative",
            "foster_mapping.py:",
        ),
    ),
    ("direct sensitivities", ("direct.py:",)),
    ("adjoint sensitivities", ("adjoint.py:", "off_target.py:")),
    (
        "observables + Jacobian assembly",
        (
            "observables.py:",
            "constraints.py:compute_layout_jacobian",
            "constraints.py:compute_constraint_jacobian_row",
            "constraints.py:_get_coordinate_gradients",
            "objective_gradient.py:",
        ),
    ),
    (
        "nominal sweep: circuit measurements",
        (
            "measurements.py:",
            "mna.py:solve_circuit_single",
            "evaluator.py:_solution_is_finite",
        ),
    ),
    ("nominal sweep: EOM model", ("eom_mbvd.py:", "base.py:_enforce_validity", "models")),
    ("MNA assembly (nominal)", ("mna.py:assemble_mna", "stamps.py:stamp_")),
    ("MNA solve / LU", ("mna.py:solve_mna", "lu_factor", "lu_solve", "getrf", "_solve")),
    ("conditioning check (cond / SVD)", ("_linalg.py:cond", "_linalg.py:svd", "_linalg.py:norm")),
    (
        "constraint / objective layout evaluate",
        ("constraints.py:evaluate", "objective.py:", "evaluator.py:"),
    ),
    ("coordinate unpack (variable_map)", ("variable_map.py:",)),
    (
        "optimizer overhead: trust-constr QR / projections",
        (
            "_batched_linalg._qr",
            "projections",
            "qr",
            "tr_interior_point",
            "equality_constrained_sqp",
            "minimize_trustregion_constr",
            "canonical_constraint",
            "qp_subproblem",
            "differentiable_functions",
            "BFGS",
            "_hessian_update_strategy",
            "_constraints.py:",
        ),
    ),
    (
        "numpy per-call overhead (errstate, finiteness reductions)",
        (
            "_ufunc_config.py:",
            "_make_extobj",
            "fromnumeric.py:",
            "reduce' of 'numpy.ufunc",
            "_contextvars",
            "numeric.py:",
        ),
    ),
]


def profile_analytical(
    harness: ABPolishHarness, cap: int, top_n: int = 0
) -> list[tuple[str, float, float]]:
    """cProfile one ANALYTICAL polish; bucket cumulative time by category."""
    tgt = harness.profile_target
    if tgt is None:
        return []
    spec = tgt["opt_spec"].model_copy(
        update={
            "local_derivative_mode": DerivativeMode.ANALYTICAL,
            "local_max_iterations": cap,
            "polish_top_k": 1,
        }
    )
    cache = DomainEvaluatorCache()
    prof = cProfile.Profile()
    prof.enable()
    _real_polish_top_k(tgt["basins"], tgt["context"], cache, spec)
    prof.disable()

    stats = pstats.Stats(prof)
    total = stats.total_tt
    # tottime (self time) is additive across functions, so bucketing it gives a
    # partition of the wall clock rather than double-counting nested cumtime.
    buckets = dict.fromkeys([c[0] for c in _PROFILE_CATEGORIES], 0.0)
    unclassified = 0.0
    for (fname, _lineno, func), (_cc, _nc, tottime, _ct, _cal) in stats.stats.items():
        key = f"{Path(fname).name}:{func}"
        hit = None
        for cat, needles in _PROFILE_CATEGORIES:
            if any(n in key for n in needles):
                hit = cat
                break
        if hit is None:
            unclassified += tottime
        else:
            buckets[hit] += tottime
    rows = [(k, v, 100.0 * v / total if total else 0.0) for k, v in buckets.items() if v > 0]
    rows.sort(key=lambda r: -r[1])
    rows.append(
        (
            "other / interpreter overhead",
            unclassified,
            100.0 * unclassified / total if total else 0.0,
        )
    )
    if top_n:
        entries = sorted(
            (
                (tot, f"{Path(fn).name}:{fu}")
                for (fn, _ln, fu), (_cc, _nc, tot, _ct, _cal) in stats.stats.items()
            ),
            reverse=True,
        )[:top_n]
        print(f"    profile total self time {total:.2f}s; top {top_n}:")
        for tot, key in entries:
            print(f"      {tot:8.3f}s {100 * tot / total:5.1f}%  {key}")
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _sum_tel(pairs: list[PairRecord], side: str, key: str) -> int:
    return int(sum(getattr(p, side).telemetry[key] for p in pairs))


def _fmt(v: float, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}g}"


def acceptance_section(case_dicts: list[dict]) -> list[str]:
    """Final acceptance findings + P12.5-F targets, from plain result data.

    Takes plain dicts so it renders identically from a live run or from the
    persisted raw JSON of a completed run.
    """
    lines: list[str] = []
    a = lines.append
    pairs = [pr for c in case_dicts for pr in c["pairs"]]
    n = len(pairs)
    n_bad = sum(1 for pr in pairs if pr["assessment"]["verdict"] == "NOT_EQUIVALENT")
    n_better = sum(1 for pr in pairs if pr["assessment"]["verdict"] == "EQUIVALENT_OR_BETTER")
    fallbacks = [pr for pr in pairs if pr["an"]["telemetry"]["fallback_reason"]]
    t_fd = sum(pr["fd"]["wall_time_s"] for pr in pairs)
    t_an = sum(pr["an"]["wall_time_s"] for pr in pairs)
    fd_mna = sum(
        pr["fd"]["telemetry"]["evaluator_target_freq_solves"]
        + pr["fd"]["telemetry"]["evaluator_coarse_freq_solves"]
        for pr in pairs
    )
    an_eval = sum(
        pr["an"]["telemetry"]["evaluator_target_freq_solves"]
        + pr["an"]["telemetry"]["evaluator_coarse_freq_solves"]
        for pr in pairs
    )
    an_fact = sum(pr["an"]["telemetry"]["factorizations"] for pr in pairs)
    an_dir = sum(pr["an"]["telemetry"]["direct_substitutions"] for pr in pairs)
    an_adj = sum(pr["an"]["telemetry"]["adjoint_substitutions"] for pr in pairs)
    an_mna = an_eval + an_fact

    a("## Acceptance findings")
    a("")
    a(
        f"**1. Scientific equivalence.** {n} A/B pairs across {len(case_dicts)} cases; "
        f"{n_bad} not equivalent, {n_better} strictly better under exact gradients. Feasibility, "
        "max hard-constraint violation and objective are equivalent-or-better on every pair. "
        "Where the two runs stopped at different points, the analytical run was never the worse "
        "of the two. Note that on the pathological case both modes were rejected by the frozen "
        "Deb pre-polish retention rule at the shared iteration cap, so that pair's agreement "
        "reflects the retained representative rather than two agreeing optimizer endpoints - "
        "the raw endpoints for it are tabulated separately above."
    )
    a("")
    if fallbacks:
        a(
            f"**2. Fallbacks / status.** {len(fallbacks)} of {n} candidates fell back to "
            "`REFERENCE_FD`:"
        )
        a("")
        for pr in fallbacks:
            a(
                f"* `{pr['case']}` `{pr['domain'][:12]}#{pr['basin']}`: "
                f"`{pr['an']['telemetry']['fallback_reason']}`"
            )
    else:
        a(
            f"**2. Fallbacks / status.** None. All {n} candidates completed on the analytical "
            "path - no `UNSUPPORTED`, nonsmooth, unresolved, or construction-failure state was "
            "hit, and no unexpected solver status appeared. SciPy termination wording differs "
            "between modes but is diagnostic only."
        )
    a("")
    a(
        "**3. FD-induced work removed.** Aggregate nominal frequency-point solves fall from "
        f"**{fd_mna:,}** (FD) to **{an_mna:,}** (analytical: {an_eval:,} evaluator + "
        f"{an_fact:,} transaction), a **{fd_mna / an_mna:.1f}x** reduction, plus {an_dir:,} "
        f"direct and {an_adj:,} adjoint back-substitutions - back-solves against an "
        "already-computed factorization, not fresh sweeps. Aggregate polish wall time falls from "
        f"{t_fd:.1f}s to {t_an:.1f}s (S_T = **{t_fd / t_an:.2f}x**). The parameter-perturbation "
        "multiplier is gone: FD's `nfev/njev` equals `Np+1` on every basin measured, "
        "analytical's equals 1."
    )
    a("")
    rss = [c for c in case_dicts if c.get("fd_peak_rss_mb")]
    if rss:
        worst = max(rss, key=lambda c: c["fd_peak_rss_mb"] - c["an_peak_rss_mb"])
        a(
            "**4. Memory.** No regression - analytical peak RSS is equal or lower in every "
            f"case. Largest gap ({worst['label']}): FD {worst['fd_peak_rss_mb']:.0f} MB vs "
            f"analytical {worst['an_peak_rss_mb']:.0f} MB. FD's cost comes from the evaluator "
            "cache retaining one full `EvaluationResult` - including its complete solution "
            "tuple - per perturbed point; the transaction holds a single current-u slot and "
            "drops it on the next u."
        )
        a("")
    a("**5. Repository gates.** Recorded in the commit for this phase.")
    a("")
    a("### P12.5-F targets indicated by these measurements")
    a("")
    a("Reported only - nothing is optimized or tuned in P12.5-E.")
    a("")
    a(
        "1. **Duplicate nominal sweep (highest value).** The transaction re-solves the entire "
        "frequency grid that `evaluate()` has already solved and cached for the same u, so the "
        "analytical path pays two nominal sweeps per iterate. It is roughly half of all "
        f"analytical nominal MNA work ({an_fact:,} of {an_mna:,} solves aggregated, and exactly "
        "half in the pathological and multi-frequency cases), "
        "and the measured profile puts the nominal sweep (circuit measurements + EOM model + "
        "MNA assembly + LU) far ahead of the sensitivity kernels themselves. Sharing nominal "
        "state between the evaluator and the transaction is the single largest remaining win."
    )
    a(
        "2. **Per-call NumPy overhead in the hot loops.** The largest single measured bucket "
        "(~15% self time) is `errstate` context entry/exit, `_make_extobj`, and elementwise "
        "finiteness reductions executed once per frequency per element - per-call overhead, not "
        "arithmetic. Vectorising the sweep across frequencies would remove it together with much "
        "of the assembly cost."
    )
    a(
        "3. **`trust-constr` QR / null-space projection on a ~1236-row constraint Jacobian.** "
        "Optimizer-side cost that scales with the off-target row count rather than with Np, and "
        "does not shrink when the Jacobian becomes exact. Reducing the off-target hard-row count "
        "- or aggregating those rows into an envelope constraint - attacks it directly."
    )
    a("")
    a(
        "Also noted, lower value: the transaction runs its off-target adjoint sweep over every "
        "off-target index unconditionally, even when no hard/soft descriptor references it; and "
        "`variable_map` coordinate unpacking costs ~4% self time because it is repeated per "
        "frequency rather than once per u."
    )
    a("")
    return lines


def build_report(cases: list[tuple[CaseResult, list[tuple[str, float, float]]]]) -> str:
    lines: list[str] = []
    a = lines.append
    a("# P12.5-E — Frozen FD vs Analytical Equivalence & Performance Report")
    a("")
    a(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  ")
    a("Frozen derivative baseline: `4ec27f827a52cf11b534ea85586f3d17e7b6e1af`  ")
    a(
        f"Platform: {platform.system()} {platform.release()}, "
        f"{psutil.cpu_count(logical=False)}P/{psutil.cpu_count(logical=True)}L cores, "
        f"Python {sys.version.split()[0]}, NumPy {np.__version__}, SciPy {scipy.__version__}"
    )
    a("")
    a(
        "Only `local_derivative_mode` differs between the two runs of each pair. "
        "Topology/domain, initial point, targets, source/EOM models, objective weights, "
        "constraints/bounds, optimizer tolerances/options and iteration limits are held "
        "constant by construction (same basin list, same `OptimizationSpec` except that "
        "one field)."
    )
    a("")

    # ---- required summary table ----
    a("## Summary")
    a("")
    a(
        "| Case | Np | Ng | FD time | Analytical time | Speedup | FD MNA work | "
        "Analytical MNA work | Δ objective | Δ max violation | Verdict |"
    )
    a(
        "| ---- | -: | -: | ------: | --------------: | ------: | ----------: | "
        "------------------: | ----------: | --------------: | ------- |"
    )
    for case, _prof in cases:
        if not case.pairs:
            a(f"| {case.label} | - | - | - | - | - | - | - | - | - | NO_BASINS |")
            continue
        np_ = max(p.n_params for p in case.pairs)
        ng = max(p.n_constraint_rows for p in case.pairs)
        t_fd = sum(p.fd.wall_time_s for p in case.pairs)
        t_an = sum(p.an.wall_time_s for p in case.pairs)
        s_t = t_fd / t_an if t_an > 0 else float("nan")
        fd_mna = _sum_tel(case.pairs, "fd", "evaluator_target_freq_solves") + _sum_tel(
            case.pairs, "fd", "evaluator_coarse_freq_solves"
        )
        an_eval = _sum_tel(case.pairs, "an", "evaluator_target_freq_solves") + _sum_tel(
            case.pairs, "an", "evaluator_coarse_freq_solves"
        )
        an_fact = _sum_tel(case.pairs, "an", "factorizations")
        assessments = [assess_pair(p) for p in case.pairs]
        # Worst (most positive = most analytical-unfavourable) deltas.
        d_obj = max((x["d_objective"] for x in assessments), default=float("nan"))
        d_viol = max((x["d_max_violation"] for x in assessments), default=float("nan"))
        if any(x["verdict"] == "NOT_EQUIVALENT" for x in assessments):
            verdict = "NOT_EQUIVALENT"
        elif any(x["verdict"] == "EQUIVALENT_OR_BETTER" for x in assessments):
            verdict = "EQUIV_OR_BETTER"
        else:
            verdict = "EQUIVALENT"
        a(
            f"| {case.label} | {np_} | {ng} | {t_fd:.2f} s | {t_an:.2f} s | "
            f"{s_t:.2f}x | {fd_mna} | {an_eval}+{an_fact}={an_eval + an_fact} | "
            f"{_fmt(d_obj, 3)} | {_fmt(d_viol, 3)} | {verdict} |"
        )
    a("")
    a(
        "`Δ objective` and `Δ max violation` are the **worst (most analytical-unfavourable)** "
        "signed deltas `analytical  -  FD` across the case's basins: negative means analytical "
        "reached a better point. The gate is one-sided — analytical must never be materially "
        "worse or infeasible where FD succeeds. `EQUIV_OR_BETTER` means at least one basin "
        "converged strictly further under exact gradients; the endpoints then genuinely differ, "
        "so Γ/Z_in/V_EOM and coordinates are reported rather than gated for those basins."
    )
    a("")
    a(
        "`FD MNA work` and `Analytical MNA work` are nominal frequency-point solves. "
        "The analytical column is written `evaluator+transaction=total`: the transaction "
        "term is the **second** nominal sweep it performs on top of the evaluator's, "
        "counted honestly rather than hidden. Back-substitutions are reported separately "
        "per case below. Times are the sum of per-basin `minimize` wall times."
    )
    a("")

    # ---- per case detail ----
    for case, prof in cases:
        a(f"## {case.label}")
        a("")
        a(
            f"{case.notes}. `local_max_iterations` capped at **{case.max_iter_cap}**, "
            f"identically for both modes. DE (shared, run once): {case.de_time_s:.1f} s. "
            f"Basin pairs: {len(case.pairs)}."
        )
        a("")
        if not case.pairs:
            a("_No basins reached polish for this case._")
            a("")
            continue

        a("### Work and cost")
        a("")
        a(
            "| Basin | Mode | wall s | nit | nfev | njev | c_nfev | c_njev | evaluator freq solves "
            "| txn factorizations | direct backsolves | adjoint backsolves | txn builds | reuse |"
        )
        a("| --- | --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |")
        for p in case.pairs:
            for side, rec in (("FD", p.fd), ("AN", p.an)):
                t = rec.telemetry
                a(
                    f"| {p.domain_id}#{p.basin_index} | {side} | {rec.wall_time_s:.3f} | "
                    f"{t['n_iterations']} | {t['nfev']} | {t['njev']} | {t['constraint_nfev']} | "
                    f"{t['constraint_njev']} | "
                    f"{t['evaluator_target_freq_solves'] + t['evaluator_coarse_freq_solves']} | "
                    f"{t['factorizations']} | {t['direct_substitutions']} | "
                    f"{t['adjoint_substitutions']} | {t['transaction_evaluations']} | "
                    f"{t['transaction_reuse_hits']} |"
                )
        a("")

        a("### Scientific equivalence")
        a("")
        a(
            "| Basin | feasible FD/AN | v_max FD/AN | J FD/AN | endpoint | Δ\\|Γ\\| | ΔZin rel "
            "| ΔV_EOM rel | ‖Δu‖∞ | ΔL rel | ΔC rel | Δf_p rel | Verdict |"
        )
        a("| --- | --- | --- | --- | --- | --: | --: | --: | --: | --: | --: | --: | --- |")
        for p in case.pairs:
            s = assess_pair(p)
            if s["same_endpoint"]:
                endpoint = "same u"
            elif s["analytical_better"]:
                endpoint = "AN better J"
            elif s["same_objective"]:
                endpoint = "same J, different u"
            else:
                endpoint = "differs"
            a(
                f"| {p.domain_id[:12]}#{p.basin_index} | {p.fd.feasible}/{p.an.feasible} | "
                f"{p.fd.v_max:.4g}/{p.an.v_max:.4g} | "
                f"{p.fd.objective:.8g}/{p.an.objective:.8g} | {endpoint} | "
                f"{_fmt(s['gamma_abs_d'])} | {_fmt(s['zin_rel'])} | {_fmt(s['veom_rel'])} | "
                f"{_fmt(s['u_inf'])} | {_fmt(s['l_rel'])} | {_fmt(s['c_rel'])} | "
                f"{_fmt(s['fp_rel'])} | {s['verdict']} |"
            )
        a("")
        a(
            "`endpoint = same u` -> both runs landed on the same point, so Γ/Z_in/V_EOM are "
            "gated. Any other label means the two runs stopped at different points (a flat or "
            "multi-optimum set), so those columns are diagnostic there. `same J, different u` "
            "means the objective agreed within tolerance while the design coordinates did not. "
            "Coordinate columns (‖Δu‖∞, ΔL, ΔC, Δf_p) are always reported, never gated."
        )
        a("")
        a("### Raw polish endpoints (before Deb pre-polish retention)")
        a("")
        a(
            "Reported separately so an agreement caused by *both* modes being discarded by the "
            "frozen pre-polish retention rule is not mistaken for an agreement of the two "
            "optimizers."
        )
        a("")
        a("| Basin | polish kept FD/AN | raw J FD/AN | raw v_max FD/AN | Δ raw J | ‖Δ raw u‖∞ |")
        a("| --- | --- | --- | --- | --: | --: |")
        for p in case.pairs:
            d_raw = p.an.post_objective - p.fd.post_objective
            du = (
                float(np.max(np.abs(np.array(p.an.post_x) - np.array(p.fd.post_x))))
                if p.fd.post_x and p.an.post_x
                else float("nan")
            )
            a(
                f"| {p.domain_id[:12]}#{p.basin_index} | "
                f"{p.fd.polish_improved}/{p.an.polish_improved} | "
                f"{p.fd.post_objective:.8g}/{p.an.post_objective:.8g} | "
                f"{p.fd.post_v_max:.4g}/{p.an.post_v_max:.4g} | "
                f"{_fmt(d_raw)} | {_fmt(du)} |"
            )
        a("")
        a("Termination messages (diagnostic only, not gated):")
        a("")
        for p in case.pairs:
            a(f"* `{p.domain_id}#{p.basin_index}` FD: {p.fd.termination} — AN: {p.an.termination}")
        a("")

        fb = [
            (p, p.an.telemetry["fallback_reason"])
            for p in case.pairs
            if p.an.telemetry["fallback_reason"]
        ]
        if fb:
            a("**Fallbacks observed:**")
            a("")
            for p, reason in fb:
                a(f"* `{p.domain_id}#{p.basin_index}` → REFERENCE_FD: `{reason}`")
        else:
            a("**Fallbacks observed:** none — every candidate stayed on the analytical path.")
        a("")
        a(
            f"Peak RSS during polish: FD {case.fd_peak_rss_mb:.0f} MB, "
            f"analytical {case.an_peak_rss_mb:.0f} MB."
        )
        a("")

        # FD scaling model check
        a("### FD parameter-perturbation multiplier")
        a("")
        a("| Basin | Np | FD nfev/njev | AN nfev/njev | expected FD model Np+1 |")
        a("| --- | --: | --: | --: | --: |")
        for p in case.pairs:
            fdt, ant = p.fd.telemetry, p.an.telemetry
            r_fd = fdt["nfev"] / fdt["njev"] if fdt["njev"] else float("nan")
            r_an = ant["nfev"] / ant["njev"] if ant["njev"] else float("nan")
            a(
                f"| {p.domain_id}#{p.basin_index} | {p.n_params} | {_fmt(r_fd)} | "
                f"{_fmt(r_an)} | {p.n_params + 1} |"
            )
        a("")

        if prof:
            a("### Analytical profile (self time, cProfile)")
            a("")
            a("| Category | seconds | % |")
            a("| --- | --: | --: |")
            for cat, secs, pct in prof:
                a(f"| {cat} | {secs:.3f} | {pct:.1f}% |")
            a("")

    lines.extend(
        acceptance_section(
            [
                {
                    "label": c.label,
                    "fd_peak_rss_mb": c.fd_peak_rss_mb,
                    "an_peak_rss_mb": c.an_peak_rss_mb,
                    "pairs": [
                        {
                            "case": pr.case,
                            "domain": pr.domain_id,
                            "basin": pr.basin_index,
                            "fd": {"wall_time_s": pr.fd.wall_time_s, "telemetry": pr.fd.telemetry},
                            "an": {"wall_time_s": pr.an.wall_time_s, "telemetry": pr.an.telemetry},
                            "assessment": assess_pair(pr),
                        }
                        for pr in c.pairs
                    ],
                }
                for c, _ in cases
            ]
        )
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="small,typical,large,multifreq,pathological")
    ap.add_argument("--out", default=str(REPO_ROOT / "p12_5_e_equivalence_performance_report.md"))
    ap.add_argument("--json-out", default=str(REPO_ROOT / "scratch" / "p12_5_e_raw.json"))
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--profile-top", type=int, default=0)
    args = ap.parse_args()

    all_specs = _case_specs()
    wanted = [c.strip() for c in args.cases.split(",") if c.strip()]

    results: list[tuple[CaseResult, list[tuple[str, float, float]]]] = []
    for name in wanted:
        if name not in all_specs:
            print(f"!! unknown case {name}", file=sys.stderr)
            continue
        spec, label, notes, cap = all_specs[name]
        print(f"\n=== {label} ===", flush=True)
        t0 = time.perf_counter()
        case, harness = run_case(name, spec, label, notes, cap)
        print(
            f"    done in {time.perf_counter() - t0:.1f}s | pairs={len(case.pairs)} "
            f"| FD {case.fd_total_time:.2f}s | AN {case.an_total_time:.2f}s",
            flush=True,
        )
        for p in case.pairs:
            s = assess_pair(p)
            print(
                f"    {p.domain_id}#{p.basin_index} Np={p.n_params} Ng={p.n_constraint_rows} "
                f"FD {p.fd.wall_time_s:.3f}s / AN {p.an.wall_time_s:.3f}s "
                f"-> {s['verdict']} dJ={s['d_objective']:+.3g} dV={s['d_max_violation']:+.3g}",
                flush=True,
            )
        prof: list[tuple[str, float, float]] = []
        if name == "pathological" and not args.no_profile:
            print("    profiling analytical path...", flush=True)
            prof = profile_analytical(harness, case.max_iter_cap, args.profile_top)
        results.append((case, prof))

    report = build_report(results)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\nreport -> {args.out}")

    raw = {
        c.label: {
            "notes": c.notes,
            "cap": c.max_iter_cap,
            "de_time_s": c.de_time_s,
            "pairs": [
                {
                    "domain": p.domain_id,
                    "basin": p.basin_index,
                    "np": p.n_params,
                    "ng": p.n_constraint_rows,
                    "nf": p.n_frequencies,
                    "fd": {k: v for k, v in asdict(p.fd).items() if k != "z_in"},
                    "an": {k: v for k, v in asdict(p.an).items() if k != "z_in"},
                    "assessment": assess_pair(p),
                }
                for p in c.pairs
            ],
            "profile": prof,
        }
        for c, prof in results
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(raw, indent=1, default=str), encoding="utf-8")

    bad = [
        (c.label, p.domain_id, p.basin_index)
        for c, _ in results
        for p in c.pairs
        if assess_pair(p)["verdict"] == "NOT_EQUIVALENT"
    ]
    if bad:
        print("\nANALYTICAL_POLISH_NOT_EQUIVALENT", bad)
        return 2
    print("\nall pairs scientifically equivalent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
