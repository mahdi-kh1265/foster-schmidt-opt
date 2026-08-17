"""Adaptive frequency sweep and resonance detection (Prompt 06, spec §15).

Performs a midpoint/triplet-based adaptive refinement over a user-specified
verification band.  Every evaluated frequency runs the full MNA solve so that
power-balance and numerical-status checks are preserved.

Public API
----------
SweepSpec              frozen configuration (band + refinement parameters)
ResonancePeak          one detected local extremum
SweepResult            complete sweep outcome
compute_adaptive_sweep(graph, source_spec, eom_model, spec, target_hz)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from foster_eom.circuit.graph import CircuitGraph, ElementKind
from foster_eom.circuit.solve import solve_circuit_single
from foster_eom.domain.source import SourceSpec
from foster_eom.models.base import OnePortModel

_EPS = 1e-30  # division-by-zero guard


# ---------------------------------------------------------------------------
# SweepSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepSpec:
    """Resolved verification band and adaptive-sweep parameters.

    Parameters
    ----------
    f_min_hz, f_max_hz : float
        Explicit resolved band; never invisible.  Use ``from_targets()``
        to derive these from target frequencies.
    n_base_points : int
        Background log-spaced points on the base grid.
    max_depth : int
        Maximum recursive depth for the response-error criterion.
    pole_min_refinement_depth : int
        Minimum depth applied unconditionally to pole-neighbourhood intervals,
        independent of ``max_depth``.  Reaching this depth does NOT by itself
        mark an interval unresolved.
    min_interval_width_ratio : float
        Floor on interval width as a fraction of the full band.  Intervals
        narrower than this are resolved at the stated frequency resolution.
    curvature_tol : float
        Relative interpolation-error threshold that triggers subdivision.
    off_target_v_eom_db_threshold : float
        dB above nearest-target |V_EOM|; off-target peaks above this are
        flagged as potential violations.
    include_pole_neighborhoods : bool
        Insert bracketing points around each identified pole.
    pole_neighborhood_half_width_ratio : float
        Half-width of the mandatory refinement window around each pole,
        as a fraction of the pole frequency.
    power_balance_rtol : float
        Relative tolerance for the power-balance residual check.
    """

    f_min_hz: float
    f_max_hz: float
    n_base_points: int = 200
    max_depth: int = 5
    pole_min_refinement_depth: int = 3
    min_interval_width_ratio: float = 1e-3
    curvature_tol: float = 0.02
    off_target_v_eom_db_threshold: float = 6.0
    include_pole_neighborhoods: bool = True
    pole_neighborhood_half_width_ratio: float = 0.02
    power_balance_rtol: float = 1e-3

    @classmethod
    def from_targets(
        cls,
        target_hz: Sequence[float],
        margin_lo: float = 0.5,
        margin_hi: float = 2.0,
        validity_range: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> SweepSpec:
        """Derive ``f_min_hz`` / ``f_max_hz`` from target frequencies.

        Derived values are always stored explicitly and are reproducible.
        ``validity_range`` clips the band to the model validity interval.
        """
        targets = sorted(target_hz)
        f_min = targets[0] * margin_lo
        f_max = targets[-1] * margin_hi
        if validity_range is not None:
            f_min = max(f_min, validity_range[0])
            f_max = min(f_max, validity_range[1])
        return cls(f_min_hz=float(f_min), f_max_hz=float(f_max), **kwargs)


# ---------------------------------------------------------------------------
# ResonancePeak
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResonancePeak:
    """One detected local maximum or minimum in a monitored quantity.

    Attributes
    ----------
    frequency_hz : float
    quantity_name : str
        ``"v_eom"`` | ``"gamma"`` | ``"i_source"``.
    amplitude : float
        |value| at the peak.
    is_local_maximum : bool
        True for a peak, False for an antiresonance trough.
    nearest_target_hz : float
    distance_to_nearest_target_hz : float
    target_associated : bool
        Within the off-target classification band of a target.
    constraint_severity : str
        ``"safe"`` | ``"warning"`` | ``"violation"``.
    """

    frequency_hz: float
    quantity_name: str
    amplitude: float
    is_local_maximum: bool
    nearest_target_hz: float
    distance_to_nearest_target_hz: float
    target_associated: bool
    constraint_severity: str


# ---------------------------------------------------------------------------
# SweepResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepResult:
    """Complete adaptive sweep result.

    All per-frequency arrays parallel ``frequencies_hz``; ``None`` marks
    MNA failures.

    Attributes
    ----------
    spec : SweepSpec
    frequencies_hz : tuple[float, ...]
    v_eom_mag, gamma_mag, i_source_mag : tuple[float | None, ...]
    unwrapped_phase_rad : tuple[float | None, ...]
        Unwrapped phase of V_EOM (rad).
    resonance_list : tuple[ResonancePeak, ...]
    off_target_unsafe : bool
    failed_frequencies_hz : tuple[float, ...]
    worst_power_balance_residual : float
        ``max |P_del - sum_P_diss| / |P_del|`` over successful solves.
    power_balance_ok : bool
    verification_complete : bool
    unresolved_intervals : tuple[tuple[float, float], ...]
    declared_frequency_resolution_hz : float
        ``min_interval_width_ratio x (f_max - f_min)``.
    """

    spec: SweepSpec
    frequencies_hz: tuple[float, ...]
    v_eom_mag: tuple[float | None, ...]
    gamma_mag: tuple[float | None, ...]
    i_source_mag: tuple[float | None, ...]
    unwrapped_phase_rad: tuple[float | None, ...]
    resonance_list: tuple[ResonancePeak, ...]
    off_target_unsafe: bool
    failed_frequencies_hz: tuple[float, ...]
    worst_power_balance_residual: float
    power_balance_ok: bool
    verification_complete: bool
    unresolved_intervals: tuple[tuple[float, float], ...]
    declared_frequency_resolution_hz: float


# ---------------------------------------------------------------------------
# Internal solve cache
# ---------------------------------------------------------------------------

_CacheEntry = tuple[float, "float | None", "float | None", "float | None", "float | None"]


class _SolveCache:
    """Memoised circuit solves keyed by frequency (1 mHz resolution)."""

    def __init__(
        self,
        graph: CircuitGraph,
        source_spec: SourceSpec,
        power_balance_rtol: float,
    ) -> None:
        self._graph = graph
        self._source = source_spec
        self._tol = power_balance_rtol
        self._cache: dict[int, _CacheEntry] = {}
        self.failed_hz: list[float] = []
        self.worst_pb_residual: float = 0.0

    @staticmethod
    def _key(f: float) -> int:
        return round(f * 1e3)

    def query(self, f: float) -> _CacheEntry:
        """Return (f, v_eom, gamma, i_source, phase_rad); None on failure."""
        k = self._key(f)
        if k in self._cache:
            return self._cache[k]

        from foster_eom.errors import CircuitSolveStatus

        try:
            sol = solve_circuit_single(self._graph, self._source, f)
        except Exception:
            self.failed_hz.append(f)
            result: _CacheEntry = (f, None, None, None, None)
            self._cache[k] = result
            return result

        if sol.status != CircuitSolveStatus.OK:
            self.failed_hz.append(f)
            result = (f, None, None, None, None)
            self._cache[k] = result
            return result

        # Power-balance residual
        p_del = sol.p_source_delivered_w
        p_dis = sol.p_dissipated_total_w
        if p_del is not None and abs(p_del) > _EPS:
            pb_res = abs((p_del or 0.0) - (p_dis or 0.0)) / abs(p_del)
            self.worst_pb_residual = max(self.worst_pb_residual, pb_res)

        v_eom = abs(sol.v_eom) if sol.v_eom is not None else None
        gamma = abs(sol.gamma) if sol.gamma is not None else None
        i_src = abs(sol.i_source_droop) if sol.i_source_droop is not None else None
        phase = float(np.angle(sol.v_eom)) if sol.v_eom is not None else None

        result = (f, v_eom, gamma, i_src, phase)
        self._cache[k] = result
        return result


# ---------------------------------------------------------------------------
# Pole extraction
# ---------------------------------------------------------------------------


def _extract_poles(graph: CircuitGraph) -> list[float]:
    """Estimate resonant frequencies from parallel L-C pairs in the graph."""
    poles: list[float] = []
    for elem_c in graph.elements.values():
        if elem_c.kind != ElementKind.CAPACITOR:
            continue
        if not elem_c.value or elem_c.value <= 0:
            continue
        nodes_c = frozenset([elem_c.node_pos, elem_c.node_neg])
        for elem_l in graph.elements.values():
            if elem_l.kind != ElementKind.INDUCTOR:
                continue
            if not elem_l.value or elem_l.value <= 0:
                continue
            if frozenset([elem_l.node_pos, elem_l.node_neg]) == nodes_c:
                f_pole = 1.0 / (2.0 * math.pi * math.sqrt(elem_l.value * elem_c.value))
                poles.append(f_pole)
    return poles


# ---------------------------------------------------------------------------
# Base grid
# ---------------------------------------------------------------------------


def _build_base_grid(spec: SweepSpec, graph: CircuitGraph) -> list[float]:
    lo, hi = spec.f_min_hz, spec.f_max_hz
    grid: list[float] = list(np.geomspace(lo, hi, spec.n_base_points))
    if spec.include_pole_neighborhoods:
        hw = spec.pole_neighborhood_half_width_ratio
        for fp in _extract_poles(graph):
            if lo <= fp <= hi:
                grid.extend([fp * (1.0 - hw), fp, fp * (1.0 + hw)])
    arr = np.unique(np.clip(np.array(grid, dtype=float), lo, hi))
    return list(arr)


# ---------------------------------------------------------------------------
# Adaptive refinement
# ---------------------------------------------------------------------------


def _in_pole_nbhd(f_a: float, f_b: float, poles: list[float], hw: float) -> bool:
    for fp in poles:
        half = fp * hw
        if f_a <= fp + half and f_b >= fp - half:
            return True
    return False


def _criterion_fires(
    v_a: float | None,
    v_m: float | None,
    v_b: float | None,
    g_a: float | None,
    g_m: float | None,
    g_b: float | None,
    i_a: float | None,
    i_m: float | None,
    i_b: float | None,
    ph_a: float | None,
    ph_m: float | None,
    ph_b: float | None,
    tol: float,
) -> bool:
    for qa, qm, qb in [(v_a, v_m, v_b), (g_a, g_m, g_b), (i_a, i_m, i_b)]:
        if qa is None or qm is None or qb is None:
            return True
        lin = (qa + qb) * 0.5
        if abs(qm - lin) / max(abs(qm), _EPS) > tol:
            return True
        if qm > max(qa, qb) or qm < min(qa, qb):
            return True
    for pa, pb in [(ph_a, ph_m), (ph_m, ph_b)]:
        if pa is not None and pb is not None and abs(pb - pa) > math.pi / 2.0:
            return True
    return False


def _refine(
    f_a: float,
    f_b: float,
    depth: int,
    cache: _SolveCache,
    spec: SweepSpec,
    poles: list[float],
    unresolved: list[tuple[float, float]],
) -> list[float]:
    """Recursively subdivide (f_a, f_b); return list of added interior freqs."""
    band = spec.f_max_hz - spec.f_min_hz
    min_w = spec.min_interval_width_ratio * band

    if f_b - f_a <= min_w:
        return []  # at declared minimum frequency resolution -- resolved

    f_m = (f_a + f_b) * 0.5
    _, v_a, g_a, i_a, ph_a = cache.query(f_a)
    _, v_m, g_m, i_m, ph_m = cache.query(f_m)
    _, v_b, g_b, i_b, ph_b = cache.query(f_b)

    in_nbhd = _in_pole_nbhd(f_a, f_b, poles, spec.pole_neighborhood_half_width_ratio)
    mandatory = spec.pole_min_refinement_depth if in_nbhd else 0
    needs_subdiv = _criterion_fires(
        v_a, v_m, v_b, g_a, g_m, g_b, i_a, i_m, i_b, ph_a, ph_m, ph_b, spec.curvature_tol
    )
    must_recurse = needs_subdiv or (depth < mandatory)

    added = [f_m]
    if must_recurse:
        if depth >= spec.max_depth:
            if needs_subdiv:
                unresolved.append((f_a, f_b))
        else:
            added += _refine(f_a, f_m, depth + 1, cache, spec, poles, unresolved)
            added += _refine(f_m, f_b, depth + 1, cache, spec, poles, unresolved)
    return added


# ---------------------------------------------------------------------------
# Resonance detection
# ---------------------------------------------------------------------------


def _detect_peaks(
    freqs: np.ndarray,
    values: np.ndarray,
    qty: str,
    targets: Sequence[float],
    spec: SweepSpec,
    ref_level: float,
) -> list[ResonancePeak]:
    peaks: list[ResonancePeak] = []
    n = len(freqs)
    if n < 3:
        return peaks

    t_arr = np.array(sorted(targets), dtype=float) if targets else np.array([], dtype=float)
    assoc_band: float
    if len(t_arr) > 1:
        assoc_band = 0.1 * float(np.min(np.diff(t_arr)))
    elif len(t_arr) == 1:
        assoc_band = 0.1 * float(t_arr[0])
    else:
        assoc_band = 0.0

    for i in range(1, n - 1):
        left, mid, right = float(values[i - 1]), float(values[i]), float(values[i + 1])
        if not (math.isfinite(left) and math.isfinite(mid) and math.isfinite(right)):
            continue
        d_l, d_r = mid - left, right - mid
        is_max = d_l > 0 and d_r < 0
        is_min = d_l < 0 and d_r > 0
        if not (is_max or is_min):
            continue

        f_pk = float(freqs[i])

        if len(t_arr) > 0:
            dists = np.abs(t_arr - f_pk)
            ni = int(np.argmin(dists))
            nearest_t = float(t_arr[ni])
            dist = float(dists[ni])
        else:
            nearest_t = f_pk
            dist = 0.0

        target_assoc = dist <= assoc_band
        severity = "safe"
        if is_max and not target_assoc and qty == "v_eom" and ref_level > _EPS:
            db_above = 20.0 * math.log10(max(mid / ref_level, _EPS))
            if db_above > spec.off_target_v_eom_db_threshold:
                severity = "violation"
            elif db_above > 0:
                severity = "warning"

        peaks.append(
            ResonancePeak(
                frequency_hz=f_pk,
                quantity_name=qty,
                amplitude=mid,
                is_local_maximum=is_max,
                nearest_target_hz=nearest_t,
                distance_to_nearest_target_hz=dist,
                target_associated=target_assoc,
                constraint_severity=severity,
            )
        )
    return peaks


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def compute_adaptive_sweep(
    graph: CircuitGraph,
    source_spec: SourceSpec,
    eom_model: OnePortModel,
    spec: SweepSpec,
    target_hz: Sequence[float] = (),
) -> SweepResult:
    """Run an adaptive frequency sweep over the resolved verification band.

    Parameters
    ----------
    graph : CircuitGraph
        Assembled passive circuit (source + matcher + EOM already stamped).
    source_spec : SourceSpec
    eom_model : OnePortModel
        Reserved; EOM is already in ``graph``.
    spec : SweepSpec
    target_hz : Sequence[float]
        Target frequencies for resonance classification.

    Returns
    -------
    SweepResult
    """
    band = spec.f_max_hz - spec.f_min_hz
    declared_res = spec.min_interval_width_ratio * band
    poles = _extract_poles(graph)
    cache = _SolveCache(graph, source_spec, spec.power_balance_rtol)

    # 1. Base grid + warm cache
    base_grid = sorted(_build_base_grid(spec, graph))
    for f in base_grid:
        cache.query(f)

    # 2. Adaptive refinement
    unresolved: list[tuple[float, float]] = []
    extra: list[float] = []
    for i in range(len(base_grid) - 1):
        extra += _refine(base_grid[i], base_grid[i + 1], 0, cache, spec, poles, unresolved)

    # 3. All evaluated frequencies
    all_freqs = sorted(set(round(f, 6) for f in base_grid + extra))

    # 4. Build quantity arrays
    v_arr: list[float | None] = []
    g_arr: list[float | None] = []
    i_arr: list[float | None] = []
    ph_arr: list[float | None] = []
    for f in all_freqs:
        _, v, g, i_s, ph = cache.query(f)
        v_arr.append(v)
        g_arr.append(g)
        i_arr.append(i_s)
        ph_arr.append(ph)

    # 5. Unwrap phase across full sorted sequence
    valid_idx = [j for j, p in enumerate(ph_arr) if p is not None]
    ph_out: list[float | None] = list(ph_arr)
    if valid_idx:
        raw = np.array([ph_arr[j] for j in valid_idx], dtype=float)
        unwrapped = np.unwrap(raw)
        for j, val in zip(valid_idx, unwrapped, strict=True):
            ph_out[j] = float(val)

    # 6. Resonance detection
    freqs_arr = np.array(all_freqs, dtype=float)
    ref_v: float = 0.0
    for ft in target_hz:
        _, vt, _, _, _ = cache.query(ft)
        if vt is not None:
            ref_v = max(ref_v, vt)

    all_peaks: list[ResonancePeak] = []
    for qty, arr in [("v_eom", v_arr), ("gamma", g_arr), ("i_source", i_arr)]:
        arr_f = np.array([x if x is not None else float("nan") for x in arr], dtype=float)
        all_peaks += _detect_peaks(freqs_arr, arr_f, qty, target_hz, spec, ref_v)

    off_target_unsafe = any(
        p.constraint_severity == "violation" and not p.target_associated for p in all_peaks
    )

    verification_complete = not cache.failed_hz and not unresolved

    return SweepResult(
        spec=spec,
        frequencies_hz=tuple(all_freqs),
        v_eom_mag=tuple(v_arr),
        gamma_mag=tuple(g_arr),
        i_source_mag=tuple(i_arr),
        unwrapped_phase_rad=tuple(ph_out),
        resonance_list=tuple(all_peaks),
        off_target_unsafe=off_target_unsafe,
        failed_frequencies_hz=tuple(sorted(set(cache.failed_hz))),
        worst_power_balance_residual=cache.worst_pb_residual,
        power_balance_ok=cache.worst_pb_residual < spec.power_balance_rtol,
        verification_complete=verification_complete,
        unresolved_intervals=tuple(unresolved),
        declared_frequency_resolution_hz=declared_res,
    )
