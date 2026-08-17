"""P11 SPICE validation result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Comparison thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationThresholds:
    """Pass/warn/fail thresholds for MNA vs. SPICE comparison.

    Parameters
    ----------
    pass_max_rel_err : float
        Max combined relative error for PASS (default 1e-4 = 0.01%).
    pass_max_phase_deg : float
        Max phase discrepancy for PASS in degrees.
    warn_max_rel_err : float
        Max combined relative error for WARN.
    warn_max_phase_deg : float
        Max phase discrepancy for WARN in degrees.
    atol_floor : float
        Absolute magnitude floor used in combined relative error:
        ``|delta| / max(|A_mna|, atol_floor)``.
    mag_floor_for_phase : float
        Magnitude threshold below which phase comparison is masked.
    """

    pass_max_rel_err: float = 1e-4
    pass_max_phase_deg: float = 0.01
    warn_max_rel_err: float = 1e-2
    warn_max_phase_deg: float = 1.0
    atol_floor: float = 1e-30
    mag_floor_for_phase: float = 1e-20


# ---------------------------------------------------------------------------
# Per-quantity comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantityComparison:
    """Comparison of one electrical quantity between MNA and SPICE.

    Convention notes
    ----------------
    * ``mna_values`` and ``spice_values`` are in the same units (complex
      phasors in the native MNA amplitude convention).  SPICE values come
      from a unit-phasor AC run and are scaled by ``source_spec.vth_phasor``
      in Python before populating this struct.  No sqrt(2) factor is used;
      ngspice AC analysis returns complex small-signal phasors directly.
    * Phase discrepancy: ``angle(spice_val * conj(mna_val))`` in radians,
      masked where ``|mna_val| < mag_floor_for_phase``.
    * Unwrapping is applied only for presentation in error reports.

    Parameters
    ----------
    quantity_name : str
        Human-readable name, e.g. ``"Z_in"``, ``"I_source"``, ``"V_eom"``,
        ``"branch_I_b1_L1"``.
    frequencies_hz : np.ndarray
        Evaluation frequencies in Hz.
    mna_values : np.ndarray
        Complex MNA values.
    spice_values : np.ndarray
        Complex SPICE values (unit-source scaled; same units as mna).
    max_abs_err : float
    rms_abs_err : float
    max_rel_err : float
        Combined relative: ``|delta| / max(|mna|, atol_floor)``.
    rms_rel_err : float
    max_phase_err_deg : float
        Max ``|angle(spice * conj(mna))|`` in degrees over unmasked freqs.
        NaN if all frequencies masked.
    n_phase_masked : int
        Number of frequency points masked from phase comparison.
    resonance_mna_hz : float | None
        Frequency of peak ``|Z_in|`` in MNA; None if not applicable.
    resonance_spice_hz : float | None
        Frequency of peak ``|Z_in|`` in SPICE; None if not applicable.
    resonance_shift_hz : float | None
        ``resonance_spice_hz - resonance_mna_hz``; None if not applicable.
    """

    quantity_name: str
    frequencies_hz: np.ndarray
    mna_values: np.ndarray
    spice_values: np.ndarray
    max_abs_err: float
    rms_abs_err: float
    max_rel_err: float
    rms_rel_err: float
    max_phase_err_deg: float
    n_phase_masked: int
    resonance_mna_hz: float | None = None
    resonance_spice_hz: float | None = None
    resonance_shift_hz: float | None = None


# ---------------------------------------------------------------------------
# Measurement plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasurementPlan:
    """Explicit reference-plane and branch measurement specification.

    All element IDs refer to elements in the ``CircuitGraph``.  A dedicated
    SPICE sense source (``Vsense_<eid> DC 0``) is inserted in series with
    each branch for which an independent current comparison is requested.

    Parameters
    ----------
    eom_element_id : str | None
        Graph element ID of the EOM load; overrides ``graph.eom_element_id``
        when set.  ``V_eom = V(pos) - V(neg)`` of this element.
    branch_element_ids : tuple[str, ...]
        Additional element IDs for branch-current comparisons.
    """

    eom_element_id: str | None = None
    branch_element_ids: tuple[str, ...] = field(default_factory=tuple)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch_element_ids", tuple(self.branch_element_ids))


# ---------------------------------------------------------------------------
# Top-level validation report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpiceValidationReport:
    """Complete MNA-vs-SPICE cross-check report.

    Status values
    -------------
    ``"pass"``
        All quantities within pass thresholds.
    ``"warn"``
        All quantities within warn thresholds; some exceed pass thresholds.
    ``"fail"``
        At least one quantity exceeds warn thresholds.
    ``"unsupported"``
        One or more elements use models with no validated SPICE representation
        (e.g. tabular/measured ``TabularImpedanceComponent``).  SPICE run not
        attempted.  This is distinct from solver unavailability.  No partial
        topology is run with those elements commented out.
    ``"solver_unavailable"``
        ngspice not found on PATH.

    Convention fields
    -----------------
    ``source_convention``
        Always ``"spice=AC_1_0_unit_phasor,scale=vth_phasor_in_python"``.
        Documents that SPICE emits unit-source complex outputs and Python
        multiplies by ``source_spec.vth_phasor`` before comparison.
    ``current_direction_convention``
        Always ``"Vsense_oriented:I(Vsense)>0_into_DUT"``.
        A series 0-V sense source is inserted at the DUT reference plane
        after R_s; positive current flows into the DUT.
    ``phase_convention``
        Always ``"angle(spice_conj_mna),masked_below_mag_floor"``.
    """

    title: str
    status: Literal["pass", "warn", "fail", "unsupported", "solver_unavailable"]
    solver_version: str | None
    netlist_sha256: str | None
    source_vth_phasor: complex
    source_phase_deg: float
    source_convention: str
    current_direction_convention: str
    phase_convention: str
    frequencies_hz: np.ndarray | None
    comparisons: list[QuantityComparison]
    unsupported_elements: list[str]
    unsupported_model_reasons: dict[str, str]
    fail_reason: str | None
    thresholds: ValidationThresholds
