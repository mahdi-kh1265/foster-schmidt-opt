"""Performance instrumentation for P12.5-A profiling.

Safe, opt-in context-based telemetry. Off by default.
"""

from __future__ import annotations

import contextvars
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class PerfStats:
    # High-level phases
    seed_time: float = 0.0
    domain_time: float = 0.0
    de_time: float = 0.0
    dedup_time: float = 0.0
    polish_time: float = 0.0
    total_time: float = 0.0
    total_cpu_time: float = 0.0

    # Counts
    de_evals: int = 0
    polish_evals: int = 0
    polish_iterations: int = 0
    mna_assemblies: int = 0
    mna_solves: int = 0
    frequencies_solved: int = 0

    # Memory
    peak_rss_mb: float = 0.0
    rss_history: list[tuple[str, float]] = field(default_factory=list)

    # Granular tracking
    # (domain, x_hash) -> evaluation count
    eval_counts_by_x: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Basin tracking
    basin_polish_time: dict[str, float] = field(default_factory=dict)
    basin_mna_solves: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    basin_nit: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    basin_status: dict[str, str] = field(default_factory=dict)
    basin_njev: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    basin_nfev: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    basin_success: dict[str, bool] = field(default_factory=dict)

    # Active context for attribution
    current_phase: str | None = None
    current_domain: str | None = None
    current_basin: str | None = None
    current_callback: str | None = None

    def record_mna_solve(self, n_freqs: int) -> None:
        self.mna_solves += 1
        self.frequencies_solved += n_freqs
        if self.current_basin:
            self.basin_mna_solves[self.current_basin] += 1

    def record_assembly(self) -> None:
        self.mna_assemblies += 1

    def record_x_eval(self, domain: str, x: tuple[float, ...]) -> None:
        key = f"{domain}_{hash(x)}"
        self.eval_counts_by_x[key] += 1

    def record_memory(self, label: str) -> None:
        import os

        import psutil  # type: ignore[import-untyped]

        process = psutil.Process(os.getpid())
        rss_mb = process.memory_info().rss / (1024 * 1024)
        self.rss_history.append((label, rss_mb))
        if rss_mb > self.peak_rss_mb:
            self.peak_rss_mb = rss_mb


_perf_context: contextvars.ContextVar[PerfStats | None] = contextvars.ContextVar(
    "perf_context", default=None
)


@contextmanager
def perf_context() -> Iterator[PerfStats]:
    """Enable performance tracking for this context."""
    stats = PerfStats()
    token = _perf_context.set(stats)
    try:
        yield stats
    finally:
        _perf_context.reset(token)


def get_perf_stats() -> PerfStats | None:
    """Get the active performance stats object, if any."""
    return _perf_context.get()
