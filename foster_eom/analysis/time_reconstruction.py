"""Multi-tone time-domain reconstruction (Prompt 06, spec ss19).

Reconstructs simultaneous signals from complex RMS phasors::

    x(t) = sqrt(2) . Re{ sum_k X_k exp(j 2pi f_k t + j phi_k) }

Phase modes
-----------
SPECIFIED          Use user-supplied phases.
ALL_ZERO           All phases = 0.
RANDOM_MC          Seeded RNG; mc_draws independent uniform [0, 2pi) draws.
WORST_CASE         Analytical bound: sqrt(2) . sum_k |X_k|.
CONSERVATIVE_BOUND Same formula; different label for clarity.

Free-phase worst case is analytically exact -- no numerical optimizer needed.
A constrained-phase DE-based hook is reserved for future use.

Public API
----------
TonePhase           resolved per-tone (frequency, amplitude, phase)
ReconstructedSignal reconstructed waveform + peak/RMS/crest metrics
TimeDomainResult    aggregate result
compute_time_domain(tones, phase_mode, ...) -> TimeDomainResult
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import gcd

import numpy as np

from foster_eom.domain.objectives import TimeDomainPhaseMode

_SQRT2 = math.sqrt(2.0)
_EPS = 1e-30


# ---------------------------------------------------------------------------
# TonePhase
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TonePhase:
    """Resolved phase for one tone."""

    frequency_hz: float
    amplitude_rms: float  # |X_k| (RMS phasor magnitude)
    phase_rad: float  # resolved phase angle (rad)


# ---------------------------------------------------------------------------
# ReconstructedSignal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconstructedSignal:
    """Reconstructed waveform for one monitored quantity.

    conservative_bound = sqrt(2) . sum_k |X_k| -- always computed
    regardless of phase mode.
    """

    element_id: str
    peak_val: float
    rms_val: float
    crest_factor: float
    time_of_peak_s: float
    phase_mode_used: str
    conservative_bound: float
    t_array_s: np.ndarray
    x_array: np.ndarray


# ---------------------------------------------------------------------------
# TimeDomainResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeDomainResult:
    """Aggregate multi-tone time-domain reconstruction result."""

    tone_phases: tuple[TonePhase, ...]
    phase_mode: str
    time_window_s: float
    common_period_found: bool
    window_description: str
    dt_s: float
    n_points: int
    point_count_capped: bool
    rng_seed_used: int | None  # None unless RANDOM_MC

    eom_signal: ReconstructedSignal
    element_signals: tuple[ReconstructedSignal, ...]

    mc_draws: int | None
    mc_peak_mean: float | None
    mc_peak_std: float | None
    mc_peak_max: float | None


# ---------------------------------------------------------------------------
# Commensurate-frequency detection
# ---------------------------------------------------------------------------


def _find_common_period(
    freqs: Sequence[float],
    rtol: float,
    max_denom: int,
    max_T: float,
) -> tuple[float | None, str]:
    """Return (common_period_s, description) or (None, fallback_reason)."""
    if not freqs:
        return None, "no tones"
    if len(freqs) == 1:
        T = 1.0 / freqs[0]
        if max_T >= T:
            return T, f"single tone T = 1 / {freqs[0]:.6g} Hz"
        return None, f"single-tone period {T:.3e} s > max_common_period_s {max_T:.3e} s"

    f0 = freqs[0]
    denoms: list[int] = [1]
    for fi in freqs[1:]:
        ratio = fi / f0
        frac = Fraction(ratio).limit_denominator(max_denom)
        approx = frac.numerator / frac.denominator
        if abs(approx - ratio) / max(abs(ratio), _EPS) > rtol:
            return None, (
                f"ratio {fi:.6g}/{f0:.6g}={ratio:.6g} not rationalisable "
                f"to rtol {rtol} with denominator <= {max_denom}"
            )
        denoms.append(frac.denominator)

    lcm_d = denoms[0]
    for d in denoms[1:]:
        lcm_d = lcm_d * d // gcd(lcm_d, d)

    T = float(lcm_d) / f0
    if max_T < T:
        return None, (
            f"inferred common period {T:.3e} s > max_common_period_s {max_T:.3e} s"
            " -- using finite observation window"
        )
    return T, f"commensurate: T = {T:.3e} s (lcm denominator {lcm_d})"


# ---------------------------------------------------------------------------
# Waveform reconstruction
# ---------------------------------------------------------------------------


def _reconstruct(
    t: np.ndarray,
    amps: Sequence[float],
    phases: Sequence[float],
    freqs: Sequence[float],
) -> np.ndarray:
    x = np.zeros_like(t)
    for A, phi, f in zip(amps, phases, freqs, strict=True):
        x += _SQRT2 * A * np.cos(2.0 * math.pi * f * t + phi)
    return x


def _signal_metrics(
    t: np.ndarray,
    x: np.ndarray,
    amps: Sequence[float],
    element_id: str,
    phase_mode: str,
) -> ReconstructedSignal:
    abs_x = np.abs(x)
    pk_idx = int(np.argmax(abs_x))
    peak = float(abs_x[pk_idx])
    rms = float(np.sqrt(np.mean(x**2)))
    crest = peak / max(rms, _EPS)
    conservative_bound = _SQRT2 * sum(abs(A) for A in amps)
    return ReconstructedSignal(
        element_id=element_id,
        peak_val=peak,
        rms_val=rms,
        crest_factor=crest,
        time_of_peak_s=float(t[pk_idx]),
        phase_mode_used=phase_mode,
        conservative_bound=conservative_bound,
        t_array_s=t,
        x_array=x,
    )


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def compute_time_domain(
    tones: Sequence[tuple[float, complex]],
    phase_mode: TimeDomainPhaseMode = TimeDomainPhaseMode.ALL_ZERO,
    specified_phases_rad: Sequence[float] | None = None,
    element_phasors: dict[str, Sequence[complex]] | None = None,
    min_samples_per_cycle: int = 20,
    max_n_points: int = 50_000,
    commensurate_rtol: float = 1e-4,
    commensurate_max_denominator: int = 1000,
    max_common_period_s: float = 1e-3,
    n_cycles_fallback: int = 10,
    mc_draws: int = 500,
    rng_seed: int | None = None,
) -> TimeDomainResult:
    """Reconstruct multi-tone time-domain signals.

    Parameters
    ----------
    tones : Sequence[tuple[float, complex]]
        [(f_hz, X_k_rms_phasor), ...] -- one entry per commanded tone.
    phase_mode : TimeDomainPhaseMode
    specified_phases_rad : Sequence[float] | None
        Required when phase_mode == SPECIFIED.
    element_phasors : dict[str, Sequence[complex]] | None
        {element_id: [X_k_rms_phasor_per_tone]} for element waveforms.
    min_samples_per_cycle : int
    max_n_points : int
    commensurate_rtol : float
    commensurate_max_denominator : int
    max_common_period_s : float
    n_cycles_fallback : int
    mc_draws : int
    rng_seed : int | None
        RNG seed for RANDOM_MC.

    Returns
    -------
    TimeDomainResult
    """
    freqs = [f for f, _ in tones]
    amps_rms = [abs(X) for _, X in tones]

    if not freqs:
        raise ValueError("tones must be non-empty")

    # Time window
    T_common, window_desc = _find_common_period(
        freqs, commensurate_rtol, commensurate_max_denominator, max_common_period_s
    )
    common_period_found = T_common is not None
    if T_common is not None:
        time_window = T_common
    else:
        time_window = n_cycles_fallback / min(freqs)
        window_desc = window_desc + (
            f"; using {n_cycles_fallback} cycles of lowest tone ({min(freqs):.6g} Hz)"
        )

    # Time resolution
    f_max = max(freqs)
    dt = 1.0 / (min_samples_per_cycle * f_max)
    n_pts = math.ceil(time_window / dt) + 1
    point_count_capped = False
    if n_pts > max_n_points:
        n_pts = max_n_points
        dt = time_window / (n_pts - 1)
        point_count_capped = True
        window_desc += f" [CAPPED at {max_n_points} pts; dt coarsened to {dt:.3e} s]"

    t = np.linspace(0.0, time_window, n_pts)

    # Phase resolution
    rng_seed_used: int | None = None
    mc_peak_mean: float | None = None
    mc_peak_std: float | None = None
    mc_peak_max: float | None = None

    if phase_mode == TimeDomainPhaseMode.SPECIFIED:
        if specified_phases_rad is None or len(specified_phases_rad) != len(freqs):
            raise ValueError("specified_phases_rad must match tones length for SPECIFIED mode")
        phases = list(specified_phases_rad)

    elif phase_mode == TimeDomainPhaseMode.ALL_ZERO:
        phases = [0.0] * len(freqs)

    elif phase_mode in (TimeDomainPhaseMode.WORST_CASE, TimeDomainPhaseMode.CONSERVATIVE_BOUND):
        # Analytical: free-phase worst case = sqrt(2) * sum|X_k|, achieved at t=0 with phi=0
        phases = [0.0] * len(freqs)

    elif phase_mode == TimeDomainPhaseMode.RANDOM_MC:
        rng_seed_used = rng_seed
        rng = np.random.default_rng(rng_seed)
        mc_peaks: list[float] = []
        for _ in range(mc_draws):
            ph = list(rng.uniform(0.0, 2.0 * math.pi, size=len(freqs)))
            x_draw = _reconstruct(t, amps_rms, ph, freqs)
            mc_peaks.append(float(np.max(np.abs(x_draw))))
        mc_peak_mean = float(np.mean(mc_peaks))
        mc_peak_std = float(np.std(mc_peaks))
        mc_peak_max = float(np.max(mc_peaks))
        phases = [0.0] * len(freqs)  # representative waveform uses zero phases

    else:
        raise ValueError(f"Unsupported phase_mode: {phase_mode}")

    # EOM waveform
    x_eom = _reconstruct(t, amps_rms, phases, freqs)
    eom_signal = _signal_metrics(t, x_eom, amps_rms, "eom", str(phase_mode))

    # TonePhase records
    tone_phases = tuple(
        TonePhase(frequency_hz=f, amplitude_rms=A, phase_rad=ph)
        for f, A, ph in zip(freqs, amps_rms, phases, strict=True)
    )

    # Element signals
    element_signals_list: list[ReconstructedSignal] = []
    if element_phasors:
        for eid, pk in element_phasors.items():
            if len(pk) != len(freqs):
                continue
            el_amps = [abs(p) for p in pk]
            x_el = _reconstruct(t, el_amps, phases, freqs)
            element_signals_list.append(_signal_metrics(t, x_el, el_amps, eid, str(phase_mode)))

    return TimeDomainResult(
        tone_phases=tone_phases,
        phase_mode=str(phase_mode),
        time_window_s=time_window,
        common_period_found=common_period_found,
        window_description=window_desc,
        dt_s=dt,
        n_points=n_pts,
        point_count_capped=point_count_capped,
        rng_seed_used=rng_seed_used,
        eom_signal=eom_signal,
        element_signals=tuple(element_signals_list),
        mc_draws=mc_draws if phase_mode == TimeDomainPhaseMode.RANDOM_MC else None,
        mc_peak_mean=mc_peak_mean,
        mc_peak_std=mc_peak_std,
        mc_peak_max=mc_peak_max,
    )
