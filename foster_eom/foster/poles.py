"""Pole validation, deterministic generation, and interval matching (Prompt 04A).

All public APIs accept and return frequencies in Hz.  No ω or q_m exposed.
Deterministic: no RNG in Prompt 04.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from foster_eom.domain.topology import PoleMode
from foster_eom.foster.foster_form import RequiredPoleIntervalHz

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoleLayoutHz:
    """A concrete set of pole frequencies in Hz."""

    f_poles_hz: tuple[float, ...]


@dataclass(frozen=True)
class PoleValidation:
    """Result of pole layout validation."""

    valid: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class RequiredIntervalFeasibility:
    """Result of bipartite matching for required-pole-interval coverage."""

    feasible: bool
    n_required: int
    n_available: int
    matching: tuple[tuple[int, int], ...] | None  # (R_k idx, A_j idx)
    uncovered_intervals: tuple[RequiredPoleIntervalHz, ...]
    reason: str | None


@dataclass(frozen=True)
class PoleSpec:
    """Pole specification — mirrors domain topology but Hz-only API.

    mode : PoleMode
        FIXED, INTERVALS, AUTO, SCHMIDT_SEED.
    fixed_poles_hz : tuple[float, ...] | None
        For FIXED mode.  Must be strictly increasing.
    intervals_hz : tuple[tuple[float, float], ...] | None
        For INTERVALS mode.
    allowed_band_hz : tuple[float, float] | None
        Overall allowed band.
    delta_f_target_min_hz : float
        Minimum distance from a pole to a target frequency.
    delta_f_pole_min_hz : float
        Minimum distance between poles.
    """

    mode: PoleMode
    fixed_poles_hz: tuple[float, ...] | None = None
    intervals_hz: tuple[tuple[float, float], ...] | None = None
    allowed_band_hz: tuple[float, float] | None = None
    delta_f_target_min_hz: float = 100.0
    delta_f_pole_min_hz: float = 100.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_poles(
    poles: PoleLayoutHz,
    f_targets_hz: np.ndarray,
    pole_spec: PoleSpec,
) -> PoleValidation:
    """Validate a pole layout.  All inputs in Hz.

    FIXED poles validated exactly as given.  Never reordered.
    """
    f_t = np.asarray(f_targets_hz, dtype=np.float64).ravel()
    fp = list(poles.f_poles_hz)
    violations: list[str] = []

    # Strict ordering (FIXED must be exactly as given)
    for i in range(len(fp) - 1):
        if fp[i] >= fp[i + 1]:
            violations.append(
                f"Poles not strictly increasing: f[{i}]={fp[i]:.6g} >= f[{i + 1}]={fp[i + 1]:.6g}"
            )

    # Target exclusion
    for i, fpi in enumerate(fp):
        for j, ftj in enumerate(f_t):
            if abs(fpi - ftj) < pole_spec.delta_f_target_min_hz:
                violations.append(f"Pole f[{i}]={fpi:.6g} too close to target f[{j}]={ftj:.6g}")

    # Pole-pole separation
    for i in range(len(fp) - 1):
        if abs(fp[i + 1] - fp[i]) < pole_spec.delta_f_pole_min_hz:
            violations.append(
                f"Pole separation f[{i}]→f[{i + 1}] = {abs(fp[i + 1] - fp[i]):.6g} "
                f"< min {pole_spec.delta_f_pole_min_hz:.6g}"
            )

    # Band constraint
    if pole_spec.allowed_band_hz is not None:
        lo, hi = pole_spec.allowed_band_hz
        for i, fpi in enumerate(fp):
            if fpi < lo or fpi > hi:
                violations.append(
                    f"Pole f[{i}]={fpi:.6g} outside allowed band [{lo:.6g}, {hi:.6g}]"
                )

    return PoleValidation(valid=len(violations) == 0, violations=tuple(violations))


# ---------------------------------------------------------------------------
# Required-interval feasibility — augmenting-path bipartite matching
# ---------------------------------------------------------------------------


def check_required_interval_feasibility(
    required_intervals: list[RequiredPoleIntervalHz],
    n_cells: int,
    pole_spec: PoleSpec,
    f_targets_hz: np.ndarray,
) -> RequiredIntervalFeasibility:
    """Deterministic augmenting-path bipartite matching.

    Constructs edges R_k → A_j when slot A_j can place a legal pole
    inside R_k.  Finds maximum matching.  Feasible iff all K required
    intervals are matched.

    After matching, verifies that actual legal pole placement exists
    in each matched intersection (not just overlap).

    All inputs in Hz.
    """
    f_t = np.asarray(f_targets_hz, dtype=np.float64).ravel()
    k = len(required_intervals)
    m = n_cells

    if k > m:
        return RequiredIntervalFeasibility(
            feasible=False,
            n_required=k,
            n_available=m,
            matching=None,
            uncovered_intervals=tuple(required_intervals),
            reason=f"K={k} required intervals > M={m} available poles",
        )

    if k == 0:
        return RequiredIntervalFeasibility(
            feasible=True,
            n_required=0,
            n_available=m,
            matching=(),
            uncovered_intervals=(),
            reason=None,
        )

    # Build available slots
    slots = _build_slots(pole_spec, m)

    # Build adjacency: R_k → list of A_j indices
    adjacency: list[list[int]] = []
    for ri in required_intervals:
        adj_j: list[int] = []
        for j, slot in enumerate(slots):
            if _slot_can_cover_interval(slot, ri, f_t, pole_spec):
                adj_j.append(j)
        adjacency.append(adj_j)

    # Find maximum bipartite matching via augmenting paths (Hopcroft-Karp)
    match_r = [-1] * k  # R_k → matched A_j
    match_a = [-1] * len(slots)  # A_j → matched R_k

    def _augment(rk: int, visited: set[int]) -> bool:
        for aj in adjacency[rk]:
            if aj in visited:
                continue
            visited.add(aj)
            if match_a[aj] == -1 or _augment(match_a[aj], visited):
                match_r[rk] = aj
                match_a[aj] = rk
                return True
        return False

    for rk in range(k):
        _augment(rk, set())

    matched_pairs = [(rk, match_r[rk]) for rk in range(k) if match_r[rk] != -1]
    uncovered = [required_intervals[rk] for rk in range(k) if match_r[rk] == -1]

    if len(matched_pairs) < k:
        return RequiredIntervalFeasibility(
            feasible=False,
            n_required=k,
            n_available=m,
            matching=tuple(matched_pairs),
            uncovered_intervals=tuple(uncovered),
            reason=f"{len(uncovered)} required interval(s) could not be matched",
        )

    # Post-matching: a perfect matching exists.
    # Exhaustively search perfect matchings for one that admits legal placement.
    valid_matching: list[tuple[int, int]] | None = None

    def _search_placements(rk: int, current_match_r: list[int], used_a: set[int]) -> bool:
        nonlocal valid_matching
        if rk == k:
            candidate_pairs = [(i, current_match_r[i]) for i in range(k)]
            if (
                _place_poles_from_matching(
                    candidate_pairs, required_intervals, slots, f_t, pole_spec
                )
                is not None
            ):
                valid_matching = candidate_pairs
                return True
            return False

        for aj in adjacency[rk]:
            if aj not in used_a:
                current_match_r[rk] = aj
                used_a.add(aj)
                if _search_placements(rk + 1, current_match_r, used_a):
                    return True
                used_a.remove(aj)
                current_match_r[rk] = -1
        return False

    if _search_placements(0, [-1] * k, set()):
        return RequiredIntervalFeasibility(
            feasible=True,
            n_required=k,
            n_available=m,
            matching=tuple(valid_matching),  # type: ignore[arg-type]
            uncovered_intervals=(),
            reason=None,
        )

    return RequiredIntervalFeasibility(
        feasible=False,
        n_required=k,
        n_available=m,
        matching=tuple(matched_pairs),  # Return the first one found as diagnostic
        uncovered_intervals=(),
        reason="Matching(s) exist but legal pole placement infeasible "
        "(ordering/separation/exclusion constraints)",
    )


def _build_slots(pole_spec: PoleSpec, n_cells: int) -> list[tuple[float, float]]:
    """Build available pole slots as Hz intervals."""
    if pole_spec.mode == PoleMode.FIXED:
        if pole_spec.fixed_poles_hz is None:
            return []
        # Each fixed pole is a point slot
        return [(fp, fp) for fp in pole_spec.fixed_poles_hz]

    if pole_spec.mode == PoleMode.INTERVALS:
        if pole_spec.intervals_hz is None:
            return []
        return list(pole_spec.intervals_hz)

    # AUTO / SCHMIDT_SEED: allowed band as one big slot per cell
    if pole_spec.allowed_band_hz is not None:
        lo, hi = pole_spec.allowed_band_hz
        return [(lo, hi)] * n_cells
    # Fallback: very wide band
    return [(1.0, 1e12)] * n_cells


def _slot_can_cover_interval(
    slot: tuple[float, float],
    req: RequiredPoleIntervalHz,
    f_targets: np.ndarray,
    spec: PoleSpec,
) -> bool:
    """Can at least one legal pole position in slot cover req interval?"""
    is_point_slot = abs(slot[1] - slot[0]) < 1e-6

    if is_point_slot:
        # Point slot (FIXED pole): check if the pole is inside the
        # required interval and satisfies target exclusion.
        fp = slot[0]
        if fp < req.f_lo_hz or fp > req.f_hi_hz:
            return False
        excl = spec.delta_f_target_min_hz
        return all(abs(fp - ft) >= excl for ft in f_targets)

    # Interval slot: compute intersection
    lo = max(slot[0], req.f_lo_hz)
    hi = min(slot[1], req.f_hi_hz)
    if lo >= hi:
        return False

    # Check if any point in [lo, hi] satisfies target exclusion
    excl = spec.delta_f_target_min_hz
    available = _subtract_exclusion_zones(lo, hi, f_targets, excl)
    return len(available) > 0


def _subtract_exclusion_zones(
    lo: float, hi: float, f_targets: np.ndarray, excl: float
) -> list[tuple[float, float]]:
    """Return sub-intervals of [lo, hi] after removing target exclusion zones."""
    if lo >= hi:
        return []
    # Sort targets within or near [lo, hi]
    nearby = sorted(ft for ft in f_targets if ft - excl < hi and ft + excl > lo)
    intervals: list[tuple[float, float]] = []
    current_lo = lo
    for ft in nearby:
        zone_lo = ft - excl
        zone_hi = ft + excl
        if zone_lo > current_lo:
            intervals.append((current_lo, min(zone_lo, hi)))
        current_lo = max(current_lo, zone_hi)
    if current_lo < hi:
        intervals.append((current_lo, hi))
    return [(a, b) for a, b in intervals if b > a]


def _place_poles_from_matching(
    matched_pairs: list[tuple[int, int]],
    required_intervals: list[RequiredPoleIntervalHz],
    slots: list[tuple[float, float]],
    f_targets: np.ndarray,
    spec: PoleSpec,
) -> list[float] | None:
    """Place poles in matched intersections, respecting ordering/separation.

    Returns pole frequencies or None if constraints can't be satisfied.
    """

    def _intersection_key(pair: tuple[int, int]) -> tuple[float, float]:
        rk, aj = pair
        req = required_intervals[rk]
        slot = slots[aj]
        if abs(slot[1] - slot[0]) < 1e-6:
            return (slot[0], slot[0])  # point slot
        lo = max(slot[0], req.f_lo_hz)
        hi = min(slot[1], req.f_hi_hz)
        return (lo, hi)

    # Sort by (intersection_lo, intersection_hi) — narrower intervals first
    sorted_pairs = sorted(matched_pairs, key=_intersection_key)

    poles: list[float] = []
    prev_pole = -math.inf

    for rk, aj in sorted_pairs:
        req = required_intervals[rk]
        slot = slots[aj]
        is_point = abs(slot[1] - slot[0]) < 1e-6

        if is_point:
            # FIXED pole: use directly
            fp = slot[0]
            if fp < prev_pole + spec.delta_f_pole_min_hz:
                return None
            if not all(abs(fp - ft) >= spec.delta_f_target_min_hz for ft in f_targets):
                return None
            poles.append(fp)
            prev_pole = fp
            continue

        # Interval slot: find intersection
        lo = max(slot[0], req.f_lo_hz)
        hi = min(slot[1], req.f_hi_hz)
        if lo >= hi:
            return None

        # Remove target exclusion zones
        available = _subtract_exclusion_zones(lo, hi, f_targets, spec.delta_f_target_min_hz)
        if not available:
            return None

        # Find valid placement respecting separation from previous
        placed = False
        for sub_lo, sub_hi in available:
            candidate_lo = max(sub_lo, prev_pole + spec.delta_f_pole_min_hz)
            if candidate_lo < sub_hi:
                # Place at geometric midpoint of available sub-interval
                fp = (
                    math.sqrt(candidate_lo * sub_hi)
                    if candidate_lo > 0
                    else (candidate_lo + sub_hi) / 2
                )
                fp = max(fp, candidate_lo)
                fp = min(fp, sub_hi - 1e-10)  # stay strictly inside
                poles.append(fp)
                prev_pole = fp
                placed = True
                break
        if not placed:
            return None

    return poles


# ---------------------------------------------------------------------------
# Deterministic pole generation
# ---------------------------------------------------------------------------


def generate_pole_candidates(
    pole_spec: PoleSpec,
    f_targets_hz: np.ndarray,
    n_cells: int,
    required_intervals: list[RequiredPoleIntervalHz] | None = None,
) -> list[PoleLayoutHz]:
    """Generate candidate pole layouts.  Deterministic.  All Hz.

    Modes
    -----
    FIXED : validate and return single layout.
    INTERVALS : geometric midpoint of each interval.
    AUTO : required intervals → geometric midpoints, extras → log-space.
    SCHMIDT_SEED : geometric midpoints of required + AUTO for extras.
    """
    f_t = np.asarray(f_targets_hz, dtype=np.float64).ravel()

    if pole_spec.mode == PoleMode.FIXED:
        if pole_spec.fixed_poles_hz is None:
            return []
        fp = list(pole_spec.fixed_poles_hz)
        # Validate ordering
        for i in range(len(fp) - 1):
            if fp[i] >= fp[i + 1]:
                raise ValueError(
                    f"FIXED poles must be strictly increasing: "
                    f"f[{i}]={fp[i]:.6g} >= f[{i + 1}]={fp[i + 1]:.6g}.  "
                    f"Poles are validated exactly as given; never reordered."
                )
        layout = PoleLayoutHz(f_poles_hz=tuple(fp))
        return [layout]

    if pole_spec.mode == PoleMode.INTERVALS:
        if pole_spec.intervals_hz is None:
            return []
        # One pole per interval at geometric midpoint
        poles: list[float] = []
        for lo, hi in pole_spec.intervals_hz:
            if lo <= 0 or hi <= 0:
                poles.append((lo + hi) / 2.0)
            else:
                poles.append(math.sqrt(lo * hi))
        return [PoleLayoutHz(f_poles_hz=tuple(sorted(poles)))]

    # AUTO / SCHMIDT_SEED
    req = required_intervals or []
    # Place one pole per required interval
    assigned: list[float] = []
    for ri in req:
        if ri.f_lo_hz > 0 and ri.f_hi_hz > 0:
            assigned.append(math.sqrt(ri.f_lo_hz * ri.f_hi_hz))
        else:
            assigned.append((ri.f_lo_hz + ri.f_hi_hz) / 2.0)

    # Extra poles in free gaps
    n_extra = n_cells - len(assigned)
    if n_extra > 0:
        all_boundaries = sorted(set(list(f_t) + assigned))
        # Add band boundaries
        if pole_spec.allowed_band_hz is not None:
            band_lo, band_hi = pole_spec.allowed_band_hz
        else:
            band_lo = min(all_boundaries) * 0.5 if all_boundaries else 1e3
            band_hi = max(all_boundaries) * 2.0 if all_boundaries else 1e9

        # Build gaps
        gaps: list[tuple[float, float]] = []
        points_sorted = sorted(set([band_lo, *all_boundaries, band_hi]))
        for i in range(len(points_sorted) - 1):
            g_lo = points_sorted[i]
            g_hi = points_sorted[i + 1]
            if g_hi - g_lo > pole_spec.delta_f_pole_min_hz:
                gaps.append((g_lo, g_hi))

        # Sort by gap width descending, place extras at midpoints
        gaps.sort(key=lambda g: g[1] - g[0], reverse=True)
        for gi in range(min(n_extra, len(gaps))):
            g_lo, g_hi = gaps[gi]
            if g_lo > 0 and g_hi > 0:
                assigned.append(math.sqrt(g_lo * g_hi))
            else:
                assigned.append((g_lo + g_hi) / 2.0)

    assigned_sorted = sorted(assigned)
    return [PoleLayoutHz(f_poles_hz=tuple(assigned_sorted))]
