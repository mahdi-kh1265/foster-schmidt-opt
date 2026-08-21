"""Progress reporting for optimization runs.

Pure-data module — no Qt imports.  The GUI layer bridges ``ProgressUpdate``
to Qt signals at the worker boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressUpdate:
    """Snapshot of optimization progress at a single point in time.

    Attributes
    ----------
    phase : str
        One of ``"SEEDING"``, ``"DE"``, ``"LOCAL_POLISH"``,
        ``"COMPLETE"``, ``"CANCELLED"``, ``"ERROR"``.
    de_evals : int
        Cumulative unique DE evaluations consumed so far.
    de_budget : int
        Whole-run DE budget (``max_global_evaluations`` minus seed evals).
    domain_index : int
        Index of the domain currently being optimized (0-based).
    domain_count : int
        Total number of domains being optimized.
    polish_candidate_index : int
        Index of the current candidate being polished (0-based).
    polish_top_k : int
        Total candidates to polish.
    polish_iteration : int
        Current trust-constr iteration within the active candidate.
    polish_max_iterations : int
        Configured ``local_max_iterations``.
    elapsed_s : float
        Wall-clock seconds since the run started.
    derivative_mode : str
        ``"analytical"`` or ``"reference_fd"``.
    fallback_occurred : bool
        Whether an analytical-to-FD fallback has fired in this run.
    overall_percent : int
        Budget-based progress estimate across all phases (0–100).
        Labelled as *estimated* in the UI; does not predict wall-clock time.
    phase_percent : int
        Progress within the current phase (0–100).
    """

    phase: str = "SEEDING"
    de_evals: int = 0
    de_budget: int = 0
    domain_index: int = 0
    domain_count: int = 0
    polish_candidate_index: int = 0
    polish_top_k: int = 0
    polish_iteration: int = 0
    polish_max_iterations: int = 0
    elapsed_s: float = 0.0
    derivative_mode: str = "analytical"
    fallback_occurred: bool = False
    overall_percent: int = 0
    phase_percent: int = 0


# Type alias for progress callbacks.
ProgressCallback = Callable[[ProgressUpdate], None]
