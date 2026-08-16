"""Sign-pattern enumeration for Schmidt target vectors (Prompt 04B).

Enumerates all feasible sign-choice patterns (+1 / -1) across N target
frequencies. For N <= 8, exhaustive search is used. For N > 8, a
deterministic beam search is employed (honestly incomplete).

All orientation invariants are enforced by explicit ``ValueError``,
never ``assert``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from itertools import product as itertools_product

import numpy as np

from foster_eom.domain.topology import LOrientation
from foster_eom.foster.foster_form import RequiredPoleIntervalHz, find_required_pole_intervals
from foster_eom.foster.poles import PoleSpec, check_required_interval_feasibility
from foster_eom.foster.schmidt import (
    BranchRealization,
    FosterBranchTolerances,
    ReactanceTarget,
    ReactanceTargetState,
    SchmidtResult,
    classify_branch_realization,
    validate_branch_realization_legality,
)

# ---------------------------------------------------------------------------
# Enums / data structures
# ---------------------------------------------------------------------------

_DEFAULT_BRANCH_TOL = FosterBranchTolerances()


class SignPruneCode(enum.StrEnum):
    """Reason for structural pruning of a sign pattern."""

    MIXED_OPEN_FINITE = "mixed_open_finite"
    REQUIRED_POLES_EXCEED_ALL_CELL_COUNTS = "required_poles_exceed_all_cell_counts"
    FIXED_POLE_INCOMPATIBLE = "fixed_pole_incompatible"
    INTERVAL_POLE_INCOMPATIBLE = "interval_pole_incompatible"
    ILLEGAL_FINAL_BRANCH_REALIZATION = "illegal_final_branch_realization"


@dataclass(frozen=True)
class SignPattern:
    """A complete sign pattern for one orientation."""

    orientation: LOrientation
    signs: tuple[int, ...]
    series_targets: tuple[ReactanceTarget, ...]  # branch2
    shunt_targets: tuple[ReactanceTarget, ...]  # branch1
    branch1_required_intervals: tuple[RequiredPoleIntervalHz, ...]
    branch2_required_intervals: tuple[RequiredPoleIntervalHz, ...]
    branch1_realization: BranchRealization
    branch2_realization: BranchRealization


@dataclass(frozen=True)
class SignPatternInfo:
    """A complete sign pattern with structural feasibility diagnostics."""

    pattern: SignPattern
    n_required_poles_branch1: int
    n_required_poles_branch2: int
    max_abs_series_ohm: float | None
    max_abs_shunt_ohm: float | None
    has_open_targets_branch1: bool
    has_open_targets_branch2: bool
    branch1_structurally_feasible: bool
    branch2_structurally_feasible: bool
    overall_structurally_feasible: bool


@dataclass(frozen=True)
class SignSearchConstraints:
    """Structural constraints from TopologySearchSpec for pruning."""

    branch1_min_cells: int
    branch1_max_cells: int
    branch2_min_cells: int
    branch2_max_cells: int
    pole_spec_branch1: PoleSpec
    pole_spec_branch2: PoleSpec


@dataclass(frozen=True)
class SignSearchDiagnostics:
    """Per-orientation sign-search diagnostics."""

    orientation: LOrientation
    search_exhaustive: bool
    search_truncated: bool
    n_total_evaluated: int
    n_pruned_structural: int
    structural_prune_counts: dict[SignPruneCode, int]
    n_discarded_by_beam_cap: int
    n_discarded_by_final_cap: int


@dataclass(frozen=True)
class SignSearchResult:
    """Result of sign-pattern enumeration for one orientation."""

    patterns: list[SignPatternInfo]
    diagnostics: SignSearchDiagnostics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_branch_targets(
    schmidt_result: SchmidtResult,
    signs: tuple[int, ...],
) -> tuple[tuple[ReactanceTarget, ...], tuple[ReactanceTarget, ...]]:
    """Extract (shunt_targets, series_targets) for a sign choice.

    Returns (branch1=shunt, branch2=series).
    """
    shunt_list: list[ReactanceTarget] = []
    series_list: list[ReactanceTarget] = []
    for pt, sign in zip(schmidt_result.points, signs, strict=True):
        if sign >= 0:
            shunt_list.append(pt.x_shunt_plus)
            series_list.append(pt.x_series_for_plus)
        else:
            shunt_list.append(pt.x_shunt_minus)
            series_list.append(pt.x_series_for_minus)
    return tuple(shunt_list), tuple(series_list)


def _finite_values(targets: tuple[ReactanceTarget, ...]) -> np.ndarray:
    """Extract FINITE target values as an array."""
    return np.array(
        [t.value_ohm for t in targets if t.state == ReactanceTargetState.FINITE],
        dtype=np.float64,
    )


def _finite_freqs(targets: tuple[ReactanceTarget, ...]) -> np.ndarray:
    """Extract FINITE target frequencies as an array."""
    return np.array(
        [t.f_hz for t in targets if t.state == ReactanceTargetState.FINITE],
        dtype=np.float64,
    )


def _has_mixed_open_finite(targets: tuple[ReactanceTarget, ...]) -> bool:
    """Check if targets contain both OPEN and FINITE."""
    has_f = any(t.state == ReactanceTargetState.FINITE for t in targets)
    has_o = any(t.state == ReactanceTargetState.OPEN_CIRCUIT for t in targets)
    return has_f and has_o


def _compute_required_intervals(
    targets: tuple[ReactanceTarget, ...],
    r_match_ohm: float,
    branch_tol: FosterBranchTolerances,
) -> list[RequiredPoleIntervalHz]:
    """Compute required pole intervals using frozen 04A whole-prefix semantics.

    Delegates to find_required_pole_intervals on the full prefix of
    FINITE targets (in frequency order). If the whole prefix is trivial
    zero under 04A semantics, returns empty. If a later nonzero target
    makes the prefix nontrivial, the recomputation recovers any
    previously hidden equal-pair interval requirements.
    """
    finite_targets = [t for t in targets if t.state == ReactanceTargetState.FINITE]
    if len(finite_targets) < 2:
        return []
    f_arr = np.array([t.f_hz for t in finite_targets], dtype=np.float64)
    x_arr = np.array([t.value_ohm for t in finite_targets], dtype=np.float64)
    return find_required_pole_intervals(f_arr, x_arr, r_match_ohm, branch_tol)


def _branch_feasible_for_any_cell_count(
    required_intervals: list[RequiredPoleIntervalHz],
    min_cells: int,
    max_cells: int,
    pole_spec: PoleSpec,
    f_targets_hz: np.ndarray,
) -> tuple[bool, SignPruneCode | None]:
    """Check if any allowed cell count can accommodate the required intervals.

    Returns (feasible, prune_code_if_not).
    """
    from foster_eom.foster.poles import PoleMode

    if not required_intervals:
        return True, None

    k = len(required_intervals)

    # For AUTO/SCHMIDT_SEED: simple lower-bound test
    if pole_spec.mode in (PoleMode.AUTO, PoleMode.SCHMIDT_SEED):
        if k > max_cells:
            return False, SignPruneCode.REQUIRED_POLES_EXCEED_ALL_CELL_COUNTS
        return True, None

    # For FIXED/INTERVAL: try each allowed cell count
    for m in range(min_cells, max_cells + 1):
        if m < k:
            continue
        result = check_required_interval_feasibility(
            required_intervals,
            n_cells=m,
            pole_spec=pole_spec,
            f_targets_hz=f_targets_hz,
        )
        if result.feasible:
            return True, None

    # All cell counts infeasible
    if pole_spec.mode == PoleMode.FIXED:
        return False, SignPruneCode.FIXED_POLE_INCOMPATIBLE
    elif pole_spec.mode == PoleMode.INTERVALS:
        return False, SignPruneCode.INTERVAL_POLE_INCOMPATIBLE
    return False, SignPruneCode.REQUIRED_POLES_EXCEED_ALL_CELL_COUNTS


def _classify_and_check_pattern(
    schmidt_result: SchmidtResult,
    signs: tuple[int, ...],
    constraints: SignSearchConstraints,
    branch_tol: FosterBranchTolerances,
) -> tuple[SignPatternInfo | None, SignPruneCode | None]:
    """Classify a complete sign pattern. Returns (info, prune_code)."""
    r_match = schmidt_result.r_match_ohm
    orientation = schmidt_result.orientation

    shunt_targets, series_targets = _extract_branch_targets(schmidt_result, signs)

    # Mixed OPEN + FINITE
    if _has_mixed_open_finite(shunt_targets):
        return None, SignPruneCode.MIXED_OPEN_FINITE
    if _has_mixed_open_finite(series_targets):
        return None, SignPruneCode.MIXED_OPEN_FINITE

    # Classify branch realizations
    try:
        b1_real = classify_branch_realization(
            shunt_targets, r_match, is_series=False, branch_tol=branch_tol
        )
    except ValueError:
        return None, SignPruneCode.MIXED_OPEN_FINITE
    try:
        b2_real = classify_branch_realization(
            series_targets, r_match, is_series=True, branch_tol=branch_tol
        )
    except ValueError:
        return None, SignPruneCode.MIXED_OPEN_FINITE

    # Legality
    b1_legal, _b1_reason = validate_branch_realization_legality(b1_real, is_series=False)
    b2_legal, _b2_reason = validate_branch_realization_legality(b2_real, is_series=True)
    if not b1_legal or not b2_legal:
        return None, SignPruneCode.ILLEGAL_FINAL_BRANCH_REALIZATION

    # Required intervals (whole-prefix)
    b1_intervals: list[RequiredPoleIntervalHz] = []
    b2_intervals: list[RequiredPoleIntervalHz] = []
    b1_feasible = True
    b2_feasible = True
    prune_code: SignPruneCode | None = None

    if b1_real == BranchRealization.FINITE_FOSTER:
        b1_intervals = _compute_required_intervals(shunt_targets, r_match, branch_tol)
        b1_feasible, prune_code = _branch_feasible_for_any_cell_count(
            b1_intervals,
            constraints.branch1_min_cells,
            constraints.branch1_max_cells,
            constraints.pole_spec_branch1,
            _finite_freqs(shunt_targets),
        )
        if not b1_feasible:
            return None, prune_code

    if b2_real == BranchRealization.FINITE_FOSTER:
        b2_intervals = _compute_required_intervals(series_targets, r_match, branch_tol)
        b2_feasible, prune_code = _branch_feasible_for_any_cell_count(
            b2_intervals,
            constraints.branch2_min_cells,
            constraints.branch2_max_cells,
            constraints.pole_spec_branch2,
            _finite_freqs(series_targets),
        )
        if not b2_feasible:
            return None, prune_code

    # Build info
    has_open_b1 = any(t.state == ReactanceTargetState.OPEN_CIRCUIT for t in shunt_targets)
    has_open_b2 = any(t.state == ReactanceTargetState.OPEN_CIRCUIT for t in series_targets)

    shunt_vals = _finite_values(shunt_targets)
    series_vals = _finite_values(series_targets)

    pattern = SignPattern(
        orientation=orientation,
        signs=signs,
        series_targets=series_targets,
        shunt_targets=shunt_targets,
        branch1_required_intervals=tuple(b1_intervals),
        branch2_required_intervals=tuple(b2_intervals),
        branch1_realization=b1_real,
        branch2_realization=b2_real,
    )
    info = SignPatternInfo(
        pattern=pattern,
        n_required_poles_branch1=len(b1_intervals),
        n_required_poles_branch2=len(b2_intervals),
        max_abs_series_ohm=float(np.max(np.abs(series_vals))) if len(series_vals) > 0 else None,
        max_abs_shunt_ohm=float(np.max(np.abs(shunt_vals))) if len(shunt_vals) > 0 else None,
        has_open_targets_branch1=has_open_b1,
        has_open_targets_branch2=has_open_b2,
        branch1_structurally_feasible=b1_feasible,
        branch2_structurally_feasible=b2_feasible,
        overall_structurally_feasible=b1_feasible and b2_feasible,
    )
    return info, None


def _beam_ranking_key(
    partial_signs: tuple[int, ...],
    shunt_targets: tuple[ReactanceTarget, ...],
    series_targets: tuple[ReactanceTarget, ...],
    n_required_b1: int,
    n_required_b2: int,
) -> tuple[int, float, tuple[int, ...]]:
    """Deterministic beam ranking key.

    (total_required_poles, max_reactance_magnitude, lexicographic_signs)
    Lower is better.
    """
    total_required = n_required_b1 + n_required_b2

    # max_reactance_magnitude over FINITE targets only
    all_finite_vals = [
        abs(t.value_ohm)
        for t in (*shunt_targets, *series_targets)
        if t.state == ReactanceTargetState.FINITE and t.value_ohm is not None
    ]
    max_react = max(all_finite_vals) if all_finite_vals else 0.0

    return (total_required, max_react, partial_signs)


# ---------------------------------------------------------------------------
# Prefix pruning for beam search
# ---------------------------------------------------------------------------


def _prefix_prune(
    shunt_targets: tuple[ReactanceTarget, ...],
    series_targets: tuple[ReactanceTarget, ...],
    constraints: SignSearchConstraints,
    r_match_ohm: float,
    branch_tol: FosterBranchTolerances,
    all_f_targets_hz: np.ndarray,
) -> SignPruneCode | None:
    """Check if a partial prefix has an irreversible contradiction.

    Returns a prune code if prunable, None otherwise.
    Only irreversible conditions are tested.
    """
    # Mixed OPEN + FINITE (irreversible)
    if _has_mixed_open_finite(shunt_targets):
        return SignPruneCode.MIXED_OPEN_FINITE
    if _has_mixed_open_finite(series_targets):
        return SignPruneCode.MIXED_OPEN_FINITE

    # Required intervals from whole current prefix
    # Branch 1 (shunt)
    b1_intervals = _compute_required_intervals(shunt_targets, r_match_ohm, branch_tol)
    if b1_intervals:
        feasible, prune_code = _branch_feasible_for_any_cell_count(
            b1_intervals,
            constraints.branch1_min_cells,
            constraints.branch1_max_cells,
            constraints.pole_spec_branch1,
            _finite_freqs(shunt_targets)
            if any(t.state == ReactanceTargetState.FINITE for t in shunt_targets)
            else all_f_targets_hz,
        )
        if not feasible:
            return prune_code

    # Branch 2 (series)
    b2_intervals = _compute_required_intervals(series_targets, r_match_ohm, branch_tol)
    if b2_intervals:
        feasible, prune_code = _branch_feasible_for_any_cell_count(
            b2_intervals,
            constraints.branch2_min_cells,
            constraints.branch2_max_cells,
            constraints.pole_spec_branch2,
            _finite_freqs(series_targets)
            if any(t.state == ReactanceTargetState.FINITE for t in series_targets)
            else all_f_targets_hz,
        )
        if not feasible:
            return prune_code

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enumerate_sign_patterns(
    schmidt_result: SchmidtResult,
    constraints: SignSearchConstraints,
    branch_tol: FosterBranchTolerances | None = None,
    max_patterns: int = 256,
    beam_width: int = 1000,
) -> SignSearchResult:
    """Enumerate feasible sign patterns for a Schmidt result.

    Parameters
    ----------
    schmidt_result : SchmidtResult
        Schmidt target result for one orientation.
    constraints : SignSearchConstraints
        Branch-specific cell count ranges and pole specs.
    branch_tol : FosterBranchTolerances | None
        Zero-reactance classification tolerances.
    max_patterns : int
        Maximum complete patterns to return (caps final result).
    beam_width : int
        Maximum partial patterns retained per level in beam search (N > 8).

    Returns
    -------
    SignSearchResult
    """
    bt = branch_tol or _DEFAULT_BRANCH_TOL
    orientation = schmidt_result.orientation
    n = len(schmidt_result.points)

    # Validate all points are valid
    if not schmidt_result.all_valid:
        return SignSearchResult(
            patterns=[],
            diagnostics=SignSearchDiagnostics(
                orientation=orientation,
                search_exhaustive=True,
                search_truncated=False,
                n_total_evaluated=0,
                n_pruned_structural=0,
                structural_prune_counts={},
                n_discarded_by_beam_cap=0,
                n_discarded_by_final_cap=0,
            ),
        )

    all_f_targets_hz = np.array([pt.f_hz for pt in schmidt_result.points], dtype=np.float64)

    if n <= 8:
        # Exhaustive search
        return _exhaustive_search(schmidt_result, constraints, bt, all_f_targets_hz)
    else:
        # Beam search
        return _beam_search(
            schmidt_result,
            constraints,
            bt,
            all_f_targets_hz,
            max_patterns=max_patterns,
            beam_width=beam_width,
        )


def _exhaustive_search(
    schmidt_result: SchmidtResult,
    constraints: SignSearchConstraints,
    branch_tol: FosterBranchTolerances,
    all_f_targets_hz: np.ndarray,
) -> SignSearchResult:
    """Exhaustive 2^N search for N <= 8."""
    n = len(schmidt_result.points)
    patterns: list[SignPatternInfo] = []
    prune_counts: dict[SignPruneCode, int] = {}
    n_total = 0

    for combo in itertools_product([1, -1], repeat=n):
        signs = tuple(combo)
        n_total += 1

        info, prune_code = _classify_and_check_pattern(
            schmidt_result,
            signs,
            constraints,
            branch_tol,
        )
        if prune_code is not None:
            prune_counts[prune_code] = prune_counts.get(prune_code, 0) + 1
            continue
        if info is not None:
            patterns.append(info)

    n_pruned = sum(prune_counts.values())
    return SignSearchResult(
        patterns=patterns,
        diagnostics=SignSearchDiagnostics(
            orientation=schmidt_result.orientation,
            search_exhaustive=True,
            search_truncated=False,
            n_total_evaluated=n_total,
            n_pruned_structural=n_pruned,
            structural_prune_counts=prune_counts,
            n_discarded_by_beam_cap=0,
            n_discarded_by_final_cap=0,
        ),
    )


# Internal type for partial patterns during beam search
@dataclass
class _PartialPattern:
    signs: list[int]
    shunt_targets: list[ReactanceTarget]
    series_targets: list[ReactanceTarget]
    n_required_b1: int
    n_required_b2: int


def _beam_search(
    schmidt_result: SchmidtResult,
    constraints: SignSearchConstraints,
    branch_tol: FosterBranchTolerances,
    all_f_targets_hz: np.ndarray,
    max_patterns: int,
    beam_width: int,
) -> SignSearchResult:
    """Deterministic beam search for N > 8."""
    n = len(schmidt_result.points)
    r_match = schmidt_result.r_match_ohm

    prune_counts: dict[SignPruneCode, int] = {}
    n_total_evaluated = 0
    n_discarded_beam = 0

    # Start with a single empty partial pattern
    beam: list[_PartialPattern] = [
        _PartialPattern(
            signs=[], shunt_targets=[], series_targets=[], n_required_b1=0, n_required_b2=0
        )
    ]

    for i in range(n):
        pt = schmidt_result.points[i]
        new_beam: list[tuple[tuple[int, float, tuple[int, ...]], _PartialPattern]] = []

        for partial in beam:
            for sign in [1, -1]:
                n_total_evaluated += 1

                if sign >= 0:
                    shunt_t = pt.x_shunt_plus
                    series_t = pt.x_series_for_plus
                else:
                    shunt_t = pt.x_shunt_minus
                    series_t = pt.x_series_for_minus

                new_signs = [*partial.signs, sign]
                new_shunt = [*partial.shunt_targets, shunt_t]
                new_series = [*partial.series_targets, series_t]

                # Prefix pruning
                prune_code = _prefix_prune(
                    tuple(new_shunt),
                    tuple(new_series),
                    constraints,
                    r_match,
                    branch_tol,
                    all_f_targets_hz,
                )
                if prune_code is not None:
                    prune_counts[prune_code] = prune_counts.get(prune_code, 0) + 1
                    continue

                # Compute required intervals for ranking
                b1_intervals = _compute_required_intervals(tuple(new_shunt), r_match, branch_tol)
                b2_intervals = _compute_required_intervals(tuple(new_series), r_match, branch_tol)

                new_partial = _PartialPattern(
                    signs=new_signs,
                    shunt_targets=new_shunt,
                    series_targets=new_series,
                    n_required_b1=len(b1_intervals),
                    n_required_b2=len(b2_intervals),
                )
                key = _beam_ranking_key(
                    tuple(new_signs),
                    tuple(new_shunt),
                    tuple(new_series),
                    new_partial.n_required_b1,
                    new_partial.n_required_b2,
                )
                new_beam.append((key, new_partial))

        # Sort by ranking key and apply beam width
        new_beam.sort(key=lambda x: x[0])
        if len(new_beam) > beam_width:
            n_discarded_beam += len(new_beam) - beam_width
            new_beam = new_beam[:beam_width]

        beam = [p for _, p in new_beam]

    # Complete patterns — classify them
    patterns: list[SignPatternInfo] = []
    for partial in beam:
        signs = tuple(partial.signs)
        info, prune_code = _classify_and_check_pattern(
            schmidt_result,
            signs,
            constraints,
            branch_tol,
        )
        if prune_code is not None:
            prune_counts[prune_code] = prune_counts.get(prune_code, 0) + 1
            continue
        if info is not None:
            patterns.append(info)

    # Final cap
    n_discarded_final = 0
    if len(patterns) > max_patterns:
        n_discarded_final = len(patterns) - max_patterns
        # Sort by a deterministic key before truncation
        patterns.sort(
            key=lambda p: (
                p.n_required_poles_branch1 + p.n_required_poles_branch2,
                p.max_abs_series_ohm or 0.0,
                p.max_abs_shunt_ohm or 0.0,
                p.pattern.signs,
            )
        )
        patterns = patterns[:max_patterns]

    n_pruned = sum(prune_counts.values())
    return SignSearchResult(
        patterns=patterns,
        diagnostics=SignSearchDiagnostics(
            orientation=schmidt_result.orientation,
            search_exhaustive=False,
            search_truncated=(n_discarded_beam > 0) or (n_discarded_final > 0),
            n_total_evaluated=n_total_evaluated,
            n_pruned_structural=n_pruned,
            structural_prune_counts=prune_counts,
            n_discarded_by_beam_cap=n_discarded_beam,
            n_discarded_by_final_cap=n_discarded_final,
        ),
    )
