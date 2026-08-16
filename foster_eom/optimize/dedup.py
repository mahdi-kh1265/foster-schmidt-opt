"""Basin deduplication for domain-local candidates (Prompt 05).

Deterministic nearest-representative clustering using RMS distance in the
normalized [0,1]^n decision space.  Representatives are Deb-best members of
each basin.  Deduplication is strictly domain-local; cross-domain clustering
is never performed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from foster_eom.optimize.evaluator import EvaluationResult

# ---------------------------------------------------------------------------
# Deb comparator
# ---------------------------------------------------------------------------


def deb_key(r: EvaluationResult) -> tuple:
    """Return a comparison tuple (lower is better) following Deb ordering.

    1. Infeasible comes after feasible.
    2. Among infeasible: lower v_max wins.
    3. Tie v_max: lower v_sum wins.
    4. Among feasible (or final tie): lower objective_value wins.
    """
    return (
        not r.feasible,
        r.v_max,
        r.v_sum,
        r.objective_value,
    )


def deb_better(a: EvaluationResult, b: EvaluationResult) -> bool:
    """Return True if ``a`` is strictly Deb-better than ``b``."""
    return deb_key(a) < deb_key(b)


# ---------------------------------------------------------------------------
# RMS distance in normalized space
# ---------------------------------------------------------------------------


def rms_distance(x: tuple[float, ...], y: tuple[float, ...]) -> float:
    """Dimension-normalized RMS distance in [0,1]^n space."""
    n = len(x)
    if n == 0:
        return 0.0
    diff = np.array(x, dtype=np.float64) - np.array(y, dtype=np.float64)
    return float(math.sqrt(float(np.mean(diff ** 2))))


# ---------------------------------------------------------------------------
# Basin
# ---------------------------------------------------------------------------


@dataclass
class Basin:
    """A deduplication basin with one representative and zero or more members."""

    representative: EvaluationResult
    members: list[EvaluationResult]

    def add(self, result: EvaluationResult) -> None:
        self.members.append(result)
        if deb_better(result, self.representative):
            self.representative = result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate_basins(
    candidates: list[EvaluationResult],
    radius: float,
) -> list[Basin]:
    """Group candidates into basins by nearest-representative clustering.

    Algorithm:
    1. Sort candidates by Deb key (best first).
    2. For each candidate, find the existing basin representative closest to it.
    3. If ``d_RMS <= radius``, add to that basin.
    4. Otherwise, start a new basin with this candidate as representative.

    Parameters
    ----------
    candidates : list[EvaluationResult]
        Candidates to cluster (may include infeasible).
    radius : float
        Clustering radius in normalized RMS distance.

    Returns
    -------
    list[Basin]
        Basins, sorted by Deb key of their representative (best first).
    """
    if not candidates:
        return []

    # Sort by Deb key (best first)
    sorted_cands = sorted(candidates, key=deb_key)

    basins: list[Basin] = []
    for result in sorted_cands:
        best_basin: Basin | None = None
        best_dist = math.inf
        for basin in basins:
            d = rms_distance(result.x, basin.representative.x)
            if d <= radius and d < best_dist:
                best_dist = d
                best_basin = basin
        if best_basin is not None:
            best_basin.add(result)
        else:
            basins.append(Basin(representative=result, members=[result]))

    # Sort basins by representative Deb key (best first)
    basins.sort(key=lambda b: deb_key(b.representative))
    return basins
