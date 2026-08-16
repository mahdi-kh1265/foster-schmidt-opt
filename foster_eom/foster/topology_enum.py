"""Topology enumeration for Foster network synthesis (Prompt 04B).

Enumerates ``TopologyCandidate`` objects for a given ``SignPatternInfo``,
applying structural pruning rules (cell counts, DOF, required poles).

Orientation invariants are enforced by explicit ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass

from foster_eom.domain.topology import LOrientation, TopologySearchSpec
from foster_eom.foster.schmidt import BranchRealization
from foster_eom.foster.sign_search import SignPatternInfo

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopologyCandidate:
    """A candidate topology for a Foster L-network."""

    orientation: LOrientation
    branch1_cells: int
    branch2_cells: int
    branch1_has_c0: bool
    branch1_has_linf: bool
    branch2_has_c0: bool
    branch2_has_linf: bool
    branch1_n_coefficients: int
    branch2_n_coefficients: int
    n_reactive: int
    structurally_valid: bool
    prune_reason: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_reactive(
    cells: int,
    has_c0: bool,
    has_linf: bool,
) -> int:
    """Count reactive components: 2 * cells + endpoint caps/inductors."""
    return 2 * cells + (1 if has_c0 else 0) + (1 if has_linf else 0)


def _count_coefficients(
    cells: int,
    has_c0: bool,
    has_linf: bool,
) -> int:
    """Count Foster coefficients: cells + endpoint flags."""
    return cells + (1 if has_c0 else 0) + (1 if has_linf else 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enumerate_topologies(
    topo_spec: TopologySearchSpec,
    sign_info: SignPatternInfo,
) -> list[TopologyCandidate]:
    """Enumerate structurally valid topologies for a sign pattern.

    Parameters
    ----------
    topo_spec : TopologySearchSpec
        Global topology search constraints.
    sign_info : SignPatternInfo
        Sign pattern with branch realization and required-interval info.

    Returns
    -------
    list[TopologyCandidate]
        All non-pruned candidates.

    Raises
    ------
    ValueError
        If ``sign_info.pattern.orientation`` does not match an allowed
        orientation in ``topo_spec``.
    """
    orientation = sign_info.pattern.orientation
    if orientation not in topo_spec.orientations:
        raise ValueError(
            f"Orientation mismatch: sign pattern orientation "
            f"{orientation!r} is not in topo_spec.orientations "
            f"{topo_spec.orientations!r}"
        )

    b1_real = sign_info.pattern.branch1_realization
    b2_real = sign_info.pattern.branch2_realization
    k1 = sign_info.n_required_poles_branch1
    k2 = sign_info.n_required_poles_branch2

    candidates: list[TopologyCandidate] = []

    # Determine cell/endpoint ranges per branch
    if b1_real == BranchRealization.FINITE_FOSTER:
        b1_cells_range = list(range(topo_spec.branch1_cells_min, topo_spec.branch1_cells_max + 1))
        b1_c0_options = [False, True] if topo_spec.endpoint_series_cap_branch1 else [False]
        b1_linf_options = [False, True] if topo_spec.endpoint_series_ind_branch1 else [False]
    else:
        # Trivial branch: exactly one canonical representation
        b1_cells_range = [0]
        b1_c0_options = [False]
        b1_linf_options = [False]

    if b2_real == BranchRealization.FINITE_FOSTER:
        b2_cells_range = list(range(topo_spec.branch2_cells_min, topo_spec.branch2_cells_max + 1))
        b2_c0_options = [False, True] if topo_spec.endpoint_series_cap_branch2 else [False]
        b2_linf_options = [False, True] if topo_spec.endpoint_series_ind_branch2 else [False]
    else:
        b2_cells_range = [0]
        b2_c0_options = [False]
        b2_linf_options = [False]

    for m1 in b1_cells_range:
        for m2 in b2_cells_range:
            for c0_1 in b1_c0_options:
                for linf_1 in b1_linf_options:
                    for c0_2 in b2_c0_options:
                        for linf_2 in b2_linf_options:
                            n1 = _count_reactive(m1, c0_1, linf_1)
                            n2 = _count_reactive(m2, c0_2, linf_2)
                            total_reactive = n1 + n2
                            p1 = _count_coefficients(m1, c0_1, linf_1)
                            p2 = _count_coefficients(m2, c0_2, linf_2)

                            # Pruning
                            valid = True

                            # Total reactive cap
                            if total_reactive > topo_spec.max_total_reactive_components:
                                valid = False

                            # Required poles vs cells
                            if valid and b1_real == BranchRealization.FINITE_FOSTER and m1 < k1:
                                valid = False
                            if valid and b2_real == BranchRealization.FINITE_FOSTER and m2 < k2:
                                valid = False

                            # Zero DOF for FINITE_FOSTER
                            if valid and b1_real == BranchRealization.FINITE_FOSTER and p1 == 0:
                                valid = False
                            if valid and b2_real == BranchRealization.FINITE_FOSTER and p2 == 0:
                                valid = False

                            if valid:
                                candidates.append(
                                    TopologyCandidate(
                                        orientation=orientation,
                                        branch1_cells=m1,
                                        branch2_cells=m2,
                                        branch1_has_c0=c0_1,
                                        branch1_has_linf=linf_1,
                                        branch2_has_c0=c0_2,
                                        branch2_has_linf=linf_2,
                                        branch1_n_coefficients=p1,
                                        branch2_n_coefficients=p2,
                                        n_reactive=total_reactive,
                                        structurally_valid=True,
                                        prune_reason=None,
                                    )
                                )

    return candidates
