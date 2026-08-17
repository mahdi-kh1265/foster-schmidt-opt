"""P11 MNA vs. SPICE comparison engine.

Error convention
----------------
* ``err_rel = |delta| / max(|mna|, atol_floor)`` (combined absolute+relative).
* Phase: ``angle(spice_val * conj(mna_val))`` in radians per frequency.
  Masked where ``|mna_val| < mag_floor_for_phase``.
  Unwrapping applied only for display; the per-point ``angle()`` is the
  canonical discrepancy measure.
* Resonance: peak of ``|Z_in(f)|`` found independently in MNA and SPICE.
"""

from __future__ import annotations

import math

import numpy as np

from foster_eom.spice.result import (
    QuantityComparison,
    ValidationThresholds,
)


def compute_quantity_comparison(
    name: str,
    frequencies_hz: np.ndarray,
    mna_vals: np.ndarray,
    spice_vals: np.ndarray,
    thresholds: ValidationThresholds,
    compute_resonance: bool = False,
) -> QuantityComparison:
    """Compute error metrics for one electrical quantity.

    Parameters
    ----------
    name : str
    frequencies_hz : np.ndarray  (N,)
    mna_vals : np.ndarray  complex (N,)
    spice_vals : np.ndarray  complex (N,); already scaled by vth_phasor
    thresholds : ValidationThresholds
    compute_resonance : bool
        If True, find peak of |Z_in| in both arrays.
    """
    mna = np.asarray(mna_vals, dtype=np.complex128)
    spice = np.asarray(spice_vals, dtype=np.complex128)

    delta = spice - mna
    abs_errs = np.abs(delta)
    mna_mag = np.abs(mna)

    # Combined relative error: |delta| / max(|mna|, floor)
    denom = np.maximum(mna_mag, thresholds.atol_floor)
    rel_errs = abs_errs / denom

    max_abs_err = float(np.max(abs_errs))
    rms_abs_err = float(np.sqrt(np.mean(abs_errs**2)))
    max_rel_err = float(np.max(rel_errs))
    rms_rel_err = float(np.sqrt(np.mean(rel_errs**2)))

    # Phase: angle(spice * conj(mna)), masked below mag_floor
    phase_mask = mna_mag >= thresholds.mag_floor_for_phase
    n_masked = int(np.sum(~phase_mask))

    if np.any(phase_mask):
        # angle(spice * conj(mna)) = angle(spice) - angle(mna)
        phase_discrepancy = np.angle(spice[phase_mask] * np.conj(mna[phase_mask]))
        max_phase_err_deg = float(np.max(np.abs(phase_discrepancy)) * 180.0 / math.pi)
    else:
        max_phase_err_deg = float("nan")

    # Resonance
    res_mna: float | None = None
    res_spice: float | None = None
    res_shift: float | None = None
    if compute_resonance and len(frequencies_hz) > 1:
        i_mna = int(np.argmax(mna_mag))
        i_spice = int(np.argmax(np.abs(spice)))
        res_mna = float(frequencies_hz[i_mna])
        res_spice = float(frequencies_hz[i_spice])
        res_shift = res_spice - res_mna

    return QuantityComparison(
        quantity_name=name,
        frequencies_hz=frequencies_hz,
        mna_values=mna,
        spice_values=spice,
        max_abs_err=max_abs_err,
        rms_abs_err=rms_abs_err,
        max_rel_err=max_rel_err,
        rms_rel_err=rms_rel_err,
        max_phase_err_deg=max_phase_err_deg,
        n_phase_masked=n_masked,
        resonance_mna_hz=res_mna,
        resonance_spice_hz=res_spice,
        resonance_shift_hz=res_shift,
    )


def classify_status(
    comparisons: list[QuantityComparison],
    thresholds: ValidationThresholds,
) -> tuple[str, str | None]:
    """Classify overall pass/warn/fail from a list of comparisons.

    Returns
    -------
    (status, fail_reason)
    """
    worst_rel = 0.0
    worst_phase = 0.0
    worst_name = ""

    for cmp in comparisons:
        if cmp.max_rel_err > worst_rel:
            worst_rel = cmp.max_rel_err
            worst_name = cmp.quantity_name
        phase = cmp.max_phase_err_deg
        if not math.isnan(phase) and phase > worst_phase:
            worst_phase = phase

    if worst_rel <= thresholds.pass_max_rel_err and worst_phase <= thresholds.pass_max_phase_deg:
        return "pass", None

    if worst_rel <= thresholds.warn_max_rel_err and worst_phase <= thresholds.warn_max_phase_deg:
        return "warn", None

    fail_parts = []
    if worst_rel > thresholds.warn_max_rel_err:
        fail_parts.append(
            f"max_rel_err={worst_rel:.3e} > warn threshold {thresholds.warn_max_rel_err:.3e}"
            f" (quantity: {worst_name})"
        )
    if worst_phase > thresholds.warn_max_phase_deg:
        fail_parts.append(
            f"max_phase_err={worst_phase:.3f} deg > warn threshold {thresholds.warn_max_phase_deg:.3f} deg"
        )
    return "fail", "; ".join(fail_parts)
