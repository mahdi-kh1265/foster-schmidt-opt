"""Q-factor and bandwidth extraction (Prompt 06, spec ss17).

Energy-Q convention (loaded)
----------------------------
With RMS phasors::

    W_stored(omega) = 0.5 * sum_L L |I_L_rms|^2  +  0.5 * sum_C C |V_C_rms|^2
    P_loss(omega)   = sum_R R |I_R_rms|^2    (all resistors in graph)
    Q_energy(omega) = omega * W_stored(omega) / P_loss(omega)

This is the **loaded Q** -- source resistance is included in P_loss.
Unloaded Q is not computed in V1.

Public API
----------
QStatus            enum of structured result codes
ResonanceQMetrics  per-target Q metrics
QResult            aggregate output
compute_q_metrics(sweep_result, target_hz, graph, source_spec, ...) -> QResult
"""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from foster_eom.analysis.sweep import SweepResult
from foster_eom.circuit.graph import CircuitGraph, ElementKind
from foster_eom.domain.source import SourceSpec

_EPS = 1e-30


# ---------------------------------------------------------------------------
# QStatus
# ---------------------------------------------------------------------------


class QStatus(enum.StrEnum):
    """Structured result code for Q extraction."""

    OK = "ok"
    NO_LOCAL_PEAK = "no_local_peak"
    MULTIPLE_NEARBY_PEAKS = "multiple_nearby_peaks"
    TARGET_ON_SHOULDER = "target_on_shoulder"
    CROSSINGS_MISSING_LOW = "crossings_missing_low"
    CROSSINGS_MISSING_HIGH = "crossings_missing_high"
    CROSSINGS_MISSING_BOTH = "crossings_missing_both"
    PEAK_AT_BAND_BOUNDARY = "peak_at_band_boundary"
    UNRESOLVED_REGION = "unresolved_region"


# ---------------------------------------------------------------------------
# ResonanceQMetrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResonanceQMetrics:
    """Q metrics for a single target frequency.

    f0_hz is NON-NONE for MULTIPLE_NEARBY_PEAKS and TARGET_ON_SHOULDER
    (peaks were found).  It is None only for NO_LOCAL_PEAK and
    UNRESOLVED_REGION (no peak could be identified).

    candidate_peaks_hz lists all detected peaks within the search window
    sorted by descending amplitude; it is empty iff f0_hz is None.
    """

    target_hz: float
    f0_hz: float | None
    candidate_peaks_hz: tuple[float, ...]
    target_on_peak: bool
    nearest_peak_hz: float | None  # alias: candidate_peaks_hz[0] or None
    f_low_hz: float | None
    f_high_hz: float | None
    q_voltage: float | None
    usable_bandwidth_hz: float | None
    q_energy: float | None
    q_energy_available: bool
    q_energy_unavailable_reason: str | None
    status: QStatus


# ---------------------------------------------------------------------------
# QResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QResult:
    """Aggregate Q-factor extraction result."""

    per_target: tuple[ResonanceQMetrics, ...]
    sweep_used: SweepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_crossing(
    freqs: np.ndarray,
    values: np.ndarray,
    threshold: float,
    start_idx: int,
    direction: str,
) -> float | None:
    """Find the first values==threshold crossing from start_idx.

    Linear interpolation between bracketing points; no extrapolation.
    direction: 'left' or 'right'.
    """
    n = len(freqs)
    idx_range = range(start_idx - 1, -1, -1) if direction == "left" else range(start_idx + 1, n)

    prev_f = float(freqs[start_idx])
    prev_v = float(values[start_idx])
    for j in idx_range:
        f_j, v_j = float(freqs[j]), float(values[j])
        if not math.isfinite(v_j):
            break
        if (prev_v - threshold) * (v_j - threshold) < 0:
            alpha = (threshold - prev_v) / (v_j - prev_v)
            return prev_f + alpha * (f_j - prev_f)
        prev_f, prev_v = f_j, v_j
    return None


def _usable_bandwidth(
    freqs: np.ndarray,
    values: np.ndarray,
    target_hz: float,
    v_target: float,
    eta: float,
) -> float | None:
    thresh = eta * v_target
    idx = int(np.argmin(np.abs(freqs - target_hz)))
    lo = float(freqs[idx])
    for j in range(idx, -1, -1):
        if not math.isfinite(values[j]) or values[j] < thresh:
            break
        lo = float(freqs[j])
    hi = float(freqs[idx])
    for j in range(idx, len(freqs)):
        if not math.isfinite(values[j]) or values[j] < thresh:
            break
        hi = float(freqs[j])
    if hi <= lo:
        return None
    return hi - lo


# ---------------------------------------------------------------------------
# Energy-Q
# ---------------------------------------------------------------------------


def _compute_energy_q(
    graph: CircuitGraph,
    source_spec: SourceSpec,
    f_hz: float,
) -> tuple[float | None, bool, str | None]:
    """Loaded energy Q at f_hz from native R/L/C elements only.

    Returns (q_energy, available, unavailable_reason).
    """
    for elem in graph.elements.values():
        if elem.kind == ElementKind.ONE_PORT_MODEL:
            return None, False, "ONE_PORT_MODEL elements present"

    from foster_eom.circuit.measurements import compute_measurements
    from foster_eom.circuit.mna import SolverOptions, assemble_mna, solve_mna
    from foster_eom.errors import CircuitSolveStatus

    try:
        Y, I_vec, node_map = assemble_mna(graph, source_spec, f_hz)
        V_vec, _status, diag = solve_mna(Y, I_vec, SolverOptions())
        if V_vec is None:
            return None, False, "MNA singular at energy-Q frequency"
        sol = compute_measurements(graph, source_spec, V_vec, node_map, f_hz, diag)
        if sol.status != CircuitSolveStatus.OK or sol.element_measurements is None:
            return None, False, "MNA solve failed"
    except Exception as exc:
        return None, False, f"Exception during energy-Q solve: {exc}"

    omega = 2.0 * math.pi * f_hz
    w_stored = 0.0
    p_loss = 0.0

    for elem in graph.elements.values():
        meas = sol.element_measurements.get(elem.id)
        if meas is None:
            continue
        if elem.kind == ElementKind.INDUCTOR and elem.value:
            w_stored += 0.5 * elem.value * abs(meas.current) ** 2
        elif elem.kind == ElementKind.CAPACITOR and elem.value:
            w_stored += 0.5 * elem.value * abs(meas.voltage) ** 2
        elif elem.kind == ElementKind.RESISTOR and elem.value:
            p_loss += elem.value * abs(meas.current) ** 2

    if p_loss <= _EPS:
        return None, True, "zero dissipation (lossless circuit)"

    return omega * w_stored / p_loss, True, None


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def compute_q_metrics(
    sweep_result: SweepResult,
    target_hz: Sequence[float],
    graph: CircuitGraph | None = None,
    source_spec: SourceSpec | None = None,
    search_window_ratio: float = 0.3,
    multiple_peak_threshold_ratio: float = 0.8,
    usable_bw_eta: float = 0.9,
    peak_on_target_rtol: float = 0.02,
) -> QResult:
    """Extract Q metrics from a completed adaptive sweep.

    Parameters
    ----------
    sweep_result : SweepResult
    target_hz : Sequence[float]
    graph, source_spec : optional
        Required for energy-Q computation.
    search_window_ratio : float
        Look for peaks within +/-this fraction of each target frequency.
    multiple_peak_threshold_ratio : float
        Secondary peaks within this fraction of the highest are co-reported.
    usable_bw_eta : float
        |V_EOM| >= eta * V_target defines the usable bandwidth.
    peak_on_target_rtol : float
        Relative tolerance to declare target == peak.

    Returns
    -------
    QResult
    """
    freqs = np.array(sweep_result.frequencies_hz, dtype=float)
    v_arr = np.array(
        [x if x is not None else float("nan") for x in sweep_result.v_eom_mag],
        dtype=float,
    )

    metrics: list[ResonanceQMetrics] = []

    for tgt in target_hz:
        # 1. Unresolved interval check
        in_unresolved = any(f_lo <= tgt <= f_hi for f_lo, f_hi in sweep_result.unresolved_intervals)
        if in_unresolved:
            metrics.append(
                ResonanceQMetrics(
                    target_hz=tgt,
                    f0_hz=None,
                    candidate_peaks_hz=(),
                    target_on_peak=False,
                    nearest_peak_hz=None,
                    f_low_hz=None,
                    f_high_hz=None,
                    q_voltage=None,
                    usable_bandwidth_hz=None,
                    q_energy=None,
                    q_energy_available=False,
                    q_energy_unavailable_reason="unresolved adaptive interval near target",
                    status=QStatus.UNRESOLVED_REGION,
                )
            )
            continue

        # 2. Find peaks within search window
        window = search_window_ratio * tgt
        mask = (freqs >= tgt - window) & (freqs <= tgt + window)
        w_freqs = freqs[mask]
        w_vals = v_arr[mask]

        candidate_peaks: list[tuple[float, float]] = []
        n_w = len(w_freqs)
        for j in range(1, n_w - 1):
            left, mid, right = float(w_vals[j - 1]), float(w_vals[j]), float(w_vals[j + 1])
            if math.isfinite(mid) and mid > max(left, right):
                candidate_peaks.append((float(w_freqs[j]), mid))

        candidate_peaks.sort(key=lambda x: x[1], reverse=True)

        if not candidate_peaks:
            at_boundary = len(freqs[mask]) > 0 and (
                float(freqs[mask][0]) <= sweep_result.spec.f_min_hz + _EPS
                or float(freqs[mask][-1]) >= sweep_result.spec.f_max_hz - _EPS
            )
            status = QStatus.PEAK_AT_BAND_BOUNDARY if at_boundary else QStatus.NO_LOCAL_PEAK
            metrics.append(
                ResonanceQMetrics(
                    target_hz=tgt,
                    f0_hz=None,
                    candidate_peaks_hz=(),
                    target_on_peak=False,
                    nearest_peak_hz=None,
                    f_low_hz=None,
                    f_high_hz=None,
                    q_voltage=None,
                    usable_bandwidth_hz=None,
                    q_energy=None,
                    q_energy_available=False,
                    q_energy_unavailable_reason="no peak found in search window",
                    status=status,
                )
            )
            continue

        best_f, best_amp = candidate_peaks[0]
        multiple = len(candidate_peaks) > 1 and (
            candidate_peaks[1][1] >= multiple_peak_threshold_ratio * best_amp
        )
        on_shoulder = abs(tgt - best_f) > search_window_ratio * 0.3 * tgt
        candidate_freqs = tuple(f for f, _ in candidate_peaks)

        # 3. f0 location in global array
        f0_idx = int(np.argmin(np.abs(freqs - best_f)))
        at_boundary = f0_idx == 0 or f0_idx == len(freqs) - 1
        if at_boundary:
            metrics.append(
                ResonanceQMetrics(
                    target_hz=tgt,
                    f0_hz=best_f,
                    candidate_peaks_hz=candidate_freqs,
                    target_on_peak=abs(tgt - best_f) <= peak_on_target_rtol * tgt,
                    nearest_peak_hz=best_f,
                    f_low_hz=None,
                    f_high_hz=None,
                    q_voltage=None,
                    usable_bandwidth_hz=None,
                    q_energy=None,
                    q_energy_available=False,
                    q_energy_unavailable_reason="peak at band boundary",
                    status=QStatus.PEAK_AT_BAND_BOUNDARY,
                )
            )
            continue

        # 4. -3 dB crossings
        thresh = best_amp / math.sqrt(2.0)
        f_low = _linear_crossing(freqs, v_arr, thresh, f0_idx, "left")
        f_high = _linear_crossing(freqs, v_arr, thresh, f0_idx, "right")

        if f_low is None and f_high is None:
            cross_status = QStatus.CROSSINGS_MISSING_BOTH
        elif f_low is None:
            cross_status = QStatus.CROSSINGS_MISSING_LOW
        elif f_high is None:
            cross_status = QStatus.CROSSINGS_MISSING_HIGH
        else:
            cross_status = QStatus.OK

        q_voltage: float | None = (
            best_f / (f_high - f_low)  # type: ignore[operator]
            if cross_status == QStatus.OK
            else None
        )

        # 5. Final status
        if multiple:
            final_status = QStatus.MULTIPLE_NEARBY_PEAKS
        elif on_shoulder:
            final_status = QStatus.TARGET_ON_SHOULDER
        elif cross_status != QStatus.OK:
            final_status = cross_status
        else:
            final_status = QStatus.OK

        # 6. Usable bandwidth
        v_at_target_idx = int(np.argmin(np.abs(freqs - tgt)))
        v_at_target = float(v_arr[v_at_target_idx])
        usable_bw = _usable_bandwidth(freqs, v_arr, tgt, v_at_target, usable_bw_eta)

        # 7. Energy Q
        if graph is not None and source_spec is not None:
            q_e, qe_avail, qe_reason = _compute_energy_q(graph, source_spec, best_f)
        else:
            q_e, qe_avail, qe_reason = None, False, "graph/source_spec not provided"

        metrics.append(
            ResonanceQMetrics(
                target_hz=tgt,
                f0_hz=best_f,
                candidate_peaks_hz=candidate_freqs,
                target_on_peak=abs(tgt - best_f) <= peak_on_target_rtol * tgt,
                nearest_peak_hz=best_f,
                f_low_hz=f_low,
                f_high_hz=f_high,
                q_voltage=q_voltage,
                usable_bandwidth_hz=usable_bw,
                q_energy=q_e,
                q_energy_available=qe_avail,
                q_energy_unavailable_reason=qe_reason,
                status=final_status,
            )
        )

    return QResult(per_target=tuple(metrics), sweep_used=sweep_result)
