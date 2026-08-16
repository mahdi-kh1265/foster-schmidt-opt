"""Continuous optimization domain construction and selection (Prompt 05).

A ``ContinuousOptimizationDomain`` is one fixed topology/orientation/pole-region
assignment.  All 04B seeds that share the same canonical attributes belong to
the same domain and share a ``DecisionVariableMapper``.

Domain identity is a SHA-256 hash of canonical JSON — deterministic across
processes and platforms, no Python built-in hash().
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from foster_eom.domain.component import ContinuousLimits
from foster_eom.foster.poles import (
    PoleMode,
    PoleSpec,
    compute_pole_legal_region,
)
from foster_eom.foster.schmidt import BranchRealization
from foster_eom.foster.sign_search import SignPattern
from foster_eom.foster.topology_enum import TopologyCandidate
from foster_eom.optimize.variable_map import (
    DecisionVariableMapper,
    build_variable_mapper,
)

if TYPE_CHECKING:
    from foster_eom.domain.topology import LOrientation
    from foster_eom.foster.seed import SeedCandidate


# ---------------------------------------------------------------------------
# Domain object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuousOptimizationDomain:
    """One fixed continuous search space derived from compatible 04B seeds.

    All seeds in ``seed_indices`` share the same normalized vector semantics.

    Parameters
    ----------
    domain_id : str
        Canonical SHA-256 hex digest uniquely identifying this search space.
    orientation : LOrientation
    topology : TopologyCandidate
    branch1_realization : BranchRealization
    branch2_realization : BranchRealization
    pole_regions_branch1, pole_regions_branch2 : tuple[(f_lo, f_hi), ...]
        Legal connected interval per cell.  Point interval ``(f, f)`` means FIXED.
    k_box_bounds_branch1, k_box_bounds_branch2 : tuple[(k_min, k_max), ...]
        Outer log-k envelope per cell (derived from pole-region extremes and
        component limits).
    k0_bounds_b1, k_inf_bounds_b1, k0_bounds_b2, k_inf_bounds_b2 :
        Endpoint coefficient bounds (or None).
    n_movable_poles_branch1, n_movable_poles_branch2 : int
        Cells where ``f_hi > f_lo``.
    variable_mapper : DecisionVariableMapper
    seed_indices : tuple[int, ...]
        Indices into ``SeedGenerationResult.seeds``.
    dimension : int
        Total optimizer decision-vector length.
    structurally_feasible : bool
        False if any ``k_box_min > k_box_max`` or FIXED-FIXED pole separation
        is violated.
    infeasibility_reason : str | None
        Human-readable reason if ``structurally_feasible == False``.
    """

    domain_id: str
    orientation: LOrientation
    topology: TopologyCandidate
    branch1_realization: BranchRealization
    branch2_realization: BranchRealization
    pole_regions_branch1: tuple[tuple[float, float], ...]
    pole_regions_branch2: tuple[tuple[float, float], ...]
    k_box_bounds_branch1: tuple[tuple[float, float], ...]
    k_box_bounds_branch2: tuple[tuple[float, float], ...]
    k0_bounds_b1: tuple[float, float] | None
    k_inf_bounds_b1: tuple[float, float] | None
    k0_bounds_b2: tuple[float, float] | None
    k_inf_bounds_b2: tuple[float, float] | None
    n_movable_poles_branch1: int
    n_movable_poles_branch2: int
    variable_mapper: DecisionVariableMapper
    seed_indices: tuple[int, ...]
    dimension: int
    structurally_feasible: bool
    infeasibility_reason: str | None
    #: Canonical sign pattern from the first seed (used to reconstruct the graph).
    canonical_sign_pattern: SignPattern


# ---------------------------------------------------------------------------
# Domain identity hash
# ---------------------------------------------------------------------------


def _domain_hash(
    orientation_value: str,
    topology: TopologyCandidate,
    branch1_realization: BranchRealization,
    branch2_realization: BranchRealization,
    pole_regions_branch1: tuple[tuple[float, float], ...],
    pole_regions_branch2: tuple[tuple[float, float], ...],
    n_movable_b1: int,
    n_movable_b2: int,
) -> str:
    """Compute a canonical SHA-256 domain identifier."""
    payload = {
        "orientation": orientation_value,
        "topology": {
            "branch1_cells": topology.branch1_cells,
            "branch2_cells": topology.branch2_cells,
            "branch1_has_c0": topology.branch1_has_c0,
            "branch1_has_linf": topology.branch1_has_linf,
            "branch2_has_c0": topology.branch2_has_c0,
            "branch2_has_linf": topology.branch2_has_linf,
        },
        "branch1_realization": branch1_realization.value,
        "branch2_realization": branch2_realization.value,
        "pole_regions_b1": [[repr(lo), repr(hi)] for lo, hi in pole_regions_branch1],
        "pole_regions_b2": [[repr(lo), repr(hi)] for lo, hi in pole_regions_branch2],
        "n_movable_b1": n_movable_b1,
        "n_movable_b2": n_movable_b2,
        "variable_ordering": "v3",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pole-region reconstruction for one branch
# ---------------------------------------------------------------------------


def _reconstruct_pole_regions(
    n_cells: int,
    f_poles_seed_hz: tuple[float, ...],
    pole_spec: PoleSpec,
    f_targets_hz: np.ndarray,
    delta_f_min: float,
) -> tuple[tuple[float, float], ...]:
    """Reconstruct the connected legal interval for each cell pole.

    Uses ``compute_pole_legal_region`` from the frozen 04B poles module.
    For FIXED-mode specs, all regions are point intervals.
    """
    regions: list[tuple[float, float]] = []
    for m in range(n_cells):
        fp_seed = f_poles_seed_hz[m] if m < len(f_poles_seed_hz) else 0.0
        prev = f_poles_seed_hz[m - 1] if m > 0 else -math.inf
        nxt = f_poles_seed_hz[m + 1] if m + 1 < len(f_poles_seed_hz) else None
        f_lo, f_hi = compute_pole_legal_region(
            cell_index=m,
            f_pole_seed_hz=fp_seed,
            pole_spec=pole_spec,
            f_targets_hz=f_targets_hz,
            n_cells=n_cells,
            prev_pole_hz=prev,
            next_pole_hz=nxt,
        )
        regions.append((f_lo, f_hi))
    return tuple(regions)


# ---------------------------------------------------------------------------
# k-box bounds for one branch
# ---------------------------------------------------------------------------


def _compute_k_box_bounds(
    n_cells: int,
    pole_regions: tuple[tuple[float, float], ...],
    component_limits: ContinuousLimits,
) -> tuple[tuple[tuple[float, float], ...], bool, str | None]:
    """Compute outer k_m envelope and detect structural infeasibility.

    Returns (bounds_tuple, structurally_feasible, reason).
    """
    c_min = component_limits.c_min_f
    c_max = component_limits.c_max_f
    l_min = component_limits.l_min_h
    l_max = component_limits.l_max_h

    bounds: list[tuple[float, float]] = []
    for m in range(n_cells):
        f_lo, f_hi = pole_regions[m] if m < len(pole_regions) else (1.0, 1e9)
        q_lo = (_TWO_PI * f_lo) ** 2
        q_hi = (_TWO_PI * f_hi) ** 2
        k_box_min = max(1.0 / c_max, q_lo * l_min) if q_lo > 0 else 1.0 / c_max
        k_box_max = min(1.0 / c_min, q_hi * l_max) if q_hi > 0 else 1.0 / c_min
        bounds.append((k_box_min, k_box_max))
        if k_box_min > k_box_max:
            return (
                tuple(bounds),
                False,
                f"Cell {m}: k_box_min ({k_box_min:.3g}) > k_box_max ({k_box_max:.3g})",
            )
    return tuple(bounds), True, None


_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Fixed-fixed pole separation pre-check
# ---------------------------------------------------------------------------


def _check_fixed_fixed_separation(
    f_poles: tuple[float, ...],
    pole_regions: tuple[tuple[float, float], ...],
    delta_f_min: float,
) -> tuple[bool, str | None]:
    """Check FIXED-FIXED adjacency violations in the full pole vector."""
    for m in range(len(f_poles) - 1):
        # Only FIXED-FIXED pairs (both point intervals)
        lo_m, hi_m = pole_regions[m] if m < len(pole_regions) else (0.0, 1.0)
        lo_n, hi_n = pole_regions[m + 1] if m + 1 < len(pole_regions) else (0.0, 1.0)
        is_fixed_m = abs(hi_m - lo_m) < 1.0
        is_fixed_n = abs(hi_n - lo_n) < 1.0
        if is_fixed_m and is_fixed_n:
            sep = f_poles[m + 1] - f_poles[m]
            if sep < delta_f_min:
                return (
                    False,
                    f"FIXED-FIXED pole separation at cells {m}/{m+1}: "
                    f"{sep:.3g} Hz < min {delta_f_min:.3g} Hz",
                )
    return True, None


# ---------------------------------------------------------------------------
# Build domain from one seed
# ---------------------------------------------------------------------------


def build_domain_from_seed(
    seed: SeedCandidate,
    seed_index: int,
    pole_spec_b1: PoleSpec,
    pole_spec_b2: PoleSpec,
    f_targets_hz: np.ndarray,
    component_limits: ContinuousLimits,
) -> ContinuousOptimizationDomain:
    """Construct a ``ContinuousOptimizationDomain`` from one accepted seed.

    The domain may be structurally infeasible (e.g. empty k-box).  Callers
    must check ``domain.structurally_feasible`` before submitting to DE.
    """
    topology = seed.topology
    orientation = seed.orientation

    # ---- Branch realizations ----
    b1_real = seed.sign_pattern.branch1_realization
    b2_real = seed.sign_pattern.branch2_realization

    # ---- Pole regions ----
    if b1_real == BranchRealization.FINITE_FOSTER and seed.branch1_solve is not None:
        f_poles_b1 = seed.branch1_solve.f_poles_hz
    else:
        f_poles_b1 = ()

    if b2_real == BranchRealization.FINITE_FOSTER and seed.branch2_solve is not None:
        f_poles_b2 = seed.branch2_solve.f_poles_hz
    else:
        f_poles_b2 = ()

    n_cells_b1 = topology.branch1_cells
    n_cells_b2 = topology.branch2_cells

    pole_regions_b1 = _reconstruct_pole_regions(
        n_cells_b1, f_poles_b1, pole_spec_b1, f_targets_hz,
        pole_spec_b1.delta_f_pole_min_hz,
    ) if b1_real == BranchRealization.FINITE_FOSTER else ()

    pole_regions_b2 = _reconstruct_pole_regions(
        n_cells_b2, f_poles_b2, pole_spec_b2, f_targets_hz,
        pole_spec_b2.delta_f_pole_min_hz,
    ) if b2_real == BranchRealization.FINITE_FOSTER else ()

    # ---- k-box bounds ----
    k_box_b1: tuple[tuple[float, float], ...] = ()
    k_box_b2: tuple[tuple[float, float], ...] = ()
    infeasible_reason: str | None = None
    struct_feasible = True

    if b1_real == BranchRealization.FINITE_FOSTER and n_cells_b1 > 0:
        k_box_b1, sf, reason = _compute_k_box_bounds(n_cells_b1, pole_regions_b1, component_limits)
        if not sf:
            struct_feasible = False
            infeasible_reason = f"branch1: {reason}"

    if b2_real == BranchRealization.FINITE_FOSTER and n_cells_b2 > 0:
        k_box_b2, sf, reason = _compute_k_box_bounds(n_cells_b2, pole_regions_b2, component_limits)
        if not sf and struct_feasible:
            struct_feasible = False
            infeasible_reason = f"branch2: {reason}"

    # ---- FIXED-FIXED separation check ----
    if struct_feasible and b1_real == BranchRealization.FINITE_FOSTER:
        ok, reason = _check_fixed_fixed_separation(
            f_poles_b1, pole_regions_b1, pole_spec_b1.delta_f_pole_min_hz
        )
        if not ok:
            struct_feasible = False
            infeasible_reason = f"branch1 FIXED-FIXED: {reason}"

    if struct_feasible and b2_real == BranchRealization.FINITE_FOSTER:
        ok, reason = _check_fixed_fixed_separation(
            f_poles_b2, pole_regions_b2, pole_spec_b2.delta_f_pole_min_hz
        )
        if not ok:
            struct_feasible = False
            infeasible_reason = f"branch2 FIXED-FIXED: {reason}"

    # ---- Endpoint bounds ----
    k0_bounds_b1: tuple[float, float] | None = None
    k_inf_bounds_b1: tuple[float, float] | None = None
    k0_bounds_b2: tuple[float, float] | None = None
    k_inf_bounds_b2: tuple[float, float] | None = None

    c_min = component_limits.c_min_f
    c_max = component_limits.c_max_f
    l_min = component_limits.l_min_h
    l_max = component_limits.l_max_h

    if topology.branch1_has_c0:
        k0_bounds_b1 = (1.0 / c_max, 1.0 / c_min)
    if topology.branch1_has_linf:
        k_inf_bounds_b1 = (l_min, l_max)
    if topology.branch2_has_c0:
        k0_bounds_b2 = (1.0 / c_max, 1.0 / c_min)
    if topology.branch2_has_linf:
        k_inf_bounds_b2 = (l_min, l_max)

    # ---- Fixed/movable pole classification ----
    fixed_kr_b1: list[float | None] = []
    fixed_fp_b1: list[float | None] = []
    fixed_kr_b2: list[float | None] = []
    fixed_fp_b2: list[float | None] = []

    def _classify_poles(
        n_cells: int,
        pole_regions: tuple[tuple[float, float], ...],
        f_poles: tuple[float, ...],
        k_residues: tuple[float, ...],
        pole_spec: PoleSpec,
        fixed_kr_out: list[float | None],
        fixed_fp_out: list[float | None],
    ) -> None:
        for m in range(n_cells):
            f_lo, f_hi = pole_regions[m] if m < len(pole_regions) else (0.0, 0.0)
            is_fixed_fp = abs(f_hi - f_lo) < 1.0 or pole_spec.mode == PoleMode.FIXED
            is_fixed_kr = False  # k_m is always a variable unless degenerate bounds

            fixed_fp_out.append(f_poles[m] if (is_fixed_fp and m < len(f_poles)) else None)
            fixed_kr_out.append(k_residues[m] if (is_fixed_kr and m < len(k_residues)) else None)

    if b1_real == BranchRealization.FINITE_FOSTER and seed.branch1_solve is not None:
        _classify_poles(
            n_cells_b1, pole_regions_b1,
            seed.branch1_solve.f_poles_hz,
            seed.branch1_solve.k_residues,
            pole_spec_b1, fixed_kr_b1, fixed_fp_b1,
        )
    else:
        fixed_kr_b1 = []
        fixed_fp_b1 = []

    if b2_real == BranchRealization.FINITE_FOSTER and seed.branch2_solve is not None:
        _classify_poles(
            n_cells_b2, pole_regions_b2,
            seed.branch2_solve.f_poles_hz,
            seed.branch2_solve.k_residues,
            pole_spec_b2, fixed_kr_b2, fixed_fp_b2,
        )
    else:
        fixed_kr_b2 = []
        fixed_fp_b2 = []

    # ---- Build mapper ----
    fixed_k0_b1 = seed.branch1_solve.k0 if seed.branch1_solve else None
    fixed_ki_b1 = seed.branch1_solve.k_inf if seed.branch1_solve else None
    fixed_k0_b2 = seed.branch2_solve.k0 if seed.branch2_solve else None
    fixed_ki_b2 = seed.branch2_solve.k_inf if seed.branch2_solve else None

    mapper = build_variable_mapper(
        branch1_n_cells=n_cells_b1,
        branch1_has_c0=topology.branch1_has_c0,
        branch1_has_linf=topology.branch1_has_linf,
        branch1_pole_regions=pole_regions_b1,
        branch1_k_box_bounds=k_box_b1,
        branch1_k0_bounds=k0_bounds_b1,
        branch1_kinf_bounds=k_inf_bounds_b1,
        branch1_fixed_k0=fixed_k0_b1,
        branch1_fixed_kinf=fixed_ki_b1,
        branch1_fixed_k_residues=tuple(fixed_kr_b1),
        branch1_fixed_f_poles_hz=tuple(fixed_fp_b1),
        branch2_n_cells=n_cells_b2,
        branch2_has_c0=topology.branch2_has_c0,
        branch2_has_linf=topology.branch2_has_linf,
        branch2_pole_regions=pole_regions_b2,
        branch2_k_box_bounds=k_box_b2,
        branch2_k0_bounds=k0_bounds_b2,
        branch2_kinf_bounds=k_inf_bounds_b2,
        branch2_fixed_k0=fixed_k0_b2,
        branch2_fixed_kinf=fixed_ki_b2,
        branch2_fixed_k_residues=tuple(fixed_kr_b2),
        branch2_fixed_f_poles_hz=tuple(fixed_fp_b2),
    )

    # ---- Movable-pole counts ----
    n_mov_b1 = sum(1 for lo, hi in pole_regions_b1 if hi - lo > 1.0)
    n_mov_b2 = sum(1 for lo, hi in pole_regions_b2 if hi - lo > 1.0)

    # ---- Domain ID ----
    did = _domain_hash(
        orientation.value,
        topology,
        b1_real,
        b2_real,
        pole_regions_b1,
        pole_regions_b2,
        n_mov_b1,
        n_mov_b2,
    )

    return ContinuousOptimizationDomain(
        domain_id=did,
        orientation=orientation,
        topology=topology,
        branch1_realization=b1_real,
        branch2_realization=b2_real,
        pole_regions_branch1=pole_regions_b1,
        pole_regions_branch2=pole_regions_b2,
        k_box_bounds_branch1=k_box_b1,
        k_box_bounds_branch2=k_box_b2,
        k0_bounds_b1=k0_bounds_b1,
        k_inf_bounds_b1=k_inf_bounds_b1,
        k0_bounds_b2=k0_bounds_b2,
        k_inf_bounds_b2=k_inf_bounds_b2,
        n_movable_poles_branch1=n_mov_b1,
        n_movable_poles_branch2=n_mov_b2,
        variable_mapper=mapper,
        seed_indices=(seed_index,),
        dimension=mapper.dimension,
        structurally_feasible=struct_feasible,
        infeasibility_reason=infeasible_reason,
        canonical_sign_pattern=seed.sign_pattern,
    )


# ---------------------------------------------------------------------------
# Group seeds into domains
# ---------------------------------------------------------------------------


def group_seeds_into_domains(
    seeds: tuple[SeedCandidate, ...],
    pole_spec_b1: PoleSpec,
    pole_spec_b2: PoleSpec,
    f_targets_hz: np.ndarray,
    component_limits: ContinuousLimits,
) -> list[ContinuousOptimizationDomain]:
    """Group all accepted 04B seeds into ``ContinuousOptimizationDomain`` objects.

    Seeds with identical ``domain_id`` are merged into one domain (their
    indices combined).  Structurally infeasible domains are included in the
    return list (callers filter by ``domain.structurally_feasible``).
    """
    domain_map: dict[str, ContinuousOptimizationDomain] = {}

    for idx, seed in enumerate(seeds):
        d = build_domain_from_seed(
            seed, idx, pole_spec_b1, pole_spec_b2, f_targets_hz, component_limits
        )
        if d.domain_id in domain_map:
            existing = domain_map[d.domain_id]
            # Merge seed index into existing domain
            merged_indices = existing.seed_indices + (idx,)
            domain_map[d.domain_id] = ContinuousOptimizationDomain(
                **{
                    **existing.__dict__,
                    "seed_indices": merged_indices,
                }
            )
        else:
            domain_map[d.domain_id] = d

    return list(domain_map.values())
