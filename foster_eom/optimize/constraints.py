"""Constraint layout compilation and evaluation (Prompt 05).

Compiles typed ``ConstraintDescriptor`` lists from schema objects into frozen
``ConstraintLayout`` containers.  At evaluation time, ``g_vector()`` returns
the constraint-margin vector in a deterministic, fixed-length order.

Constraint conventions:
    g_j >= 0  →  constraint satisfied
    g_j <  0  →  constraint violated; v_j = -g_j > 0

All margins are normalized (dimensionless).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from foster_eom.domain.constraints import (
    ConstraintSeverity,
    FrequencyScope,
    MatchConstraints,
    StressConstraints,
)

if TYPE_CHECKING:
    from foster_eom.circuit.measurements import CircuitSolution
    from foster_eom.domain.constraints import ConstraintRecord


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstraintDescriptor:
    """Metadata for one entry in a ConstraintLayout.

    ``name`` is unique within its layout.  ``normalization_scale`` is the
    denominator in the normalized margin formula so that g_j is dimensionless.
    """

    name: str
    constraint_type: str  # "gamma"|"r_max"|"r_min"|"x_bound"|"v_min"|"v_max"|
    # "i_source"|"cap_v"|"ind_i"|"pole_sep"|
    # "comp_L_hi"|"comp_L_lo"|"comp_C_hi"|"comp_C_lo"|"offtarget"
    frequency_scope: FrequencyScope
    severity: ConstraintSeverity
    target_index: int | None = None  # index into EvaluationContext.target_indices
    freq_index: int | None = None  # index into evaluation_frequencies_hz
    branch: int | None = None  # 1 or 2
    cell_index: int | None = None
    element_id: str | None = None
    normalization_scale: float = 1.0
    penalty_weight: float = 1.0  # for SOFT constraints
    validation_only: bool = False


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstraintLayout:
    """Ordered, immutable collection of ConstraintDescriptors.

    ``g_vector()`` evaluates all descriptors in deterministic order and
    returns a float64 array of the same length.  Numerical failures are
    represented by ``(-1.0,) * len(descriptors)`` (see ``EvaluationResult``).
    """

    descriptors: tuple[ConstraintDescriptor, ...]

    @property
    def n(self) -> int:
        return len(self.descriptors)

    def evaluate(
        self,
        solutions: tuple[CircuitSolution, ...],
        target_indices: tuple[int, ...],
        off_target_indices: tuple[int, ...],
        branch1_pole_regions: tuple[tuple[float, float], ...],
        branch2_pole_regions: tuple[tuple[float, float], ...],
        branch1_k_residues: tuple[float, ...],
        branch2_k_residues: tuple[float, ...],
        branch1_f_poles: tuple[float, ...],
        branch2_f_poles: tuple[float, ...],
        branch1_l_vals: tuple[float, ...],
        branch2_l_vals: tuple[float, ...],
        branch1_c_vals: tuple[float, ...],
        branch2_c_vals: tuple[float, ...],
        component_limits_l_min: float,
        component_limits_l_max: float,
        component_limits_c_min: float,
        component_limits_c_max: float,
        pole_sep_min_b1: float,
        pole_sep_min_b2: float,
        z_ref_ohm: float,
        gamma_max: float | None,
        r_min_ohm: float | None,
        r_max_ohm: float | None,
        x_max_ohm: float | None,
        source_current_max_a: float | None,
        off_target_eom_peak_rms_v: float | None,
    ) -> np.ndarray:
        """Return constraint margin vector in descriptor order."""
        g = np.empty(len(self.descriptors), dtype=np.float64)
        for i, desc in enumerate(self.descriptors):
            g[i] = _eval_one(
                desc=desc,
                solutions=solutions,
                target_indices=target_indices,
                off_target_indices=off_target_indices,
                b1_pole_regions=branch1_pole_regions,
                b2_pole_regions=branch2_pole_regions,
                b1_kr=branch1_k_residues,
                b2_kr=branch2_k_residues,
                b1_fp=branch1_f_poles,
                b2_fp=branch2_f_poles,
                b1_lv=branch1_l_vals,
                b2_lv=branch2_l_vals,
                b1_cv=branch1_c_vals,
                b2_cv=branch2_c_vals,
                l_min=component_limits_l_min,
                l_max=component_limits_l_max,
                c_min=component_limits_c_min,
                c_max=component_limits_c_max,
                sep_b1=pole_sep_min_b1,
                sep_b2=pole_sep_min_b2,
                z_ref=z_ref_ohm,
                gamma_max=gamma_max,
                r_min=r_min_ohm,
                r_max=r_max_ohm,
                x_max=x_max_ohm,
                i_max=source_current_max_a,
                ot_v_max=off_target_eom_peak_rms_v,
            )
        return g


# ---------------------------------------------------------------------------
# Single-descriptor evaluator
# ---------------------------------------------------------------------------


def _eval_one(
    desc: ConstraintDescriptor,
    solutions: tuple[CircuitSolution, ...],
    target_indices: tuple[int, ...],
    off_target_indices: tuple[int, ...],
    b1_pole_regions: tuple[tuple[float, float], ...],
    b2_pole_regions: tuple[tuple[float, float], ...],
    b1_kr: tuple[float, ...],
    b2_kr: tuple[float, ...],
    b1_fp: tuple[float, ...],
    b2_fp: tuple[float, ...],
    b1_lv: tuple[float, ...],
    b2_lv: tuple[float, ...],
    b1_cv: tuple[float, ...],
    b2_cv: tuple[float, ...],
    l_min: float,
    l_max: float,
    c_min: float,
    c_max: float,
    sep_b1: float,
    sep_b2: float,
    z_ref: float,
    gamma_max: float | None,
    r_min: float | None,
    r_max: float | None,
    x_max: float | None,
    i_max: float | None,
    ot_v_max: float | None,
) -> float:
    ct = desc.constraint_type

    if ct == "gamma":
        if gamma_max is None or desc.freq_index is None:
            return 1.0
        sol = solutions[desc.freq_index]
        if sol.gamma is None:
            return -1.0
        gm = abs(sol.gamma)
        scale = max(gamma_max, 1e-6)
        return float((gamma_max - gm) / scale)

    if ct == "r_max":
        if r_max is None or desc.freq_index is None:
            return 1.0
        sol = solutions[desc.freq_index]
        if sol.z_in is None:
            return -1.0
        r_in = sol.z_in.real
        scale = max(r_max, z_ref, 1.0)
        return float((r_max - r_in) / scale)

    if ct == "r_min":
        if r_min is None or desc.freq_index is None:
            return 1.0
        sol = solutions[desc.freq_index]
        if sol.z_in is None:
            return -1.0
        r_in = sol.z_in.real
        scale = max(r_max or z_ref, z_ref, 1.0)
        return float((r_in - r_min) / scale)

    if ct == "x_bound":
        if x_max is None or desc.freq_index is None:
            return 1.0
        sol = solutions[desc.freq_index]
        if sol.z_in is None:
            return -1.0
        x_in = abs(sol.z_in.imag)
        r_scale = max(r_max or z_ref, z_ref, 1.0)
        x_scale = max(x_max, r_scale)
        return float((x_max - x_in) / x_scale)

    if ct == "v_min":
        if desc.freq_index is None:
            return 1.0
        sol = solutions[desc.freq_index]
        if sol.v_eom is None:
            return -1.0
        v_mag = abs(sol.v_eom)
        v_tgt = desc.normalization_scale  # stored as normalization_scale
        return float((v_mag - v_tgt) / max(v_tgt, 1e-6))

    if ct == "v_max":
        if desc.freq_index is None:
            return 1.0
        sol = solutions[desc.freq_index]
        if sol.v_eom is None:
            return -1.0
        v_mag = abs(sol.v_eom)
        v_lim = desc.normalization_scale
        return float((v_lim - v_mag) / max(v_lim, 1e-6))

    if ct == "i_source":
        if i_max is None or desc.freq_index is None:
            return 1.0
        sol = solutions[desc.freq_index]
        if sol.i_source_droop is None:
            return -1.0
        i_rms = abs(sol.i_source_droop)
        return float((i_max - i_rms) / max(i_max, 1e-9))

    if ct == "offtarget":
        if ot_v_max is None or desc.freq_index is None:
            return 1.0
        sol = solutions[desc.freq_index]
        if sol.v_eom is None:
            return -1.0
        v_mag = abs(sol.v_eom)
        return float((ot_v_max - v_mag) / max(ot_v_max, 1e-6))

    if ct in ("comp_L_hi", "comp_L_lo", "comp_C_hi", "comp_C_lo"):
        branch = desc.branch
        m = desc.cell_index
        if branch is None or m is None:
            return 1.0
        lv = b1_lv if branch == 1 else b2_lv
        cv = b1_cv if branch == 1 else b2_cv
        if ct == "comp_L_hi":
            if m >= len(lv):
                return 1.0
            return float((l_max - lv[m]) / max(l_max, 1e-20))
        if ct == "comp_L_lo":
            if m >= len(lv):
                return 1.0
            return float((lv[m] - l_min) / max(l_max, 1e-20))
        if ct == "comp_C_hi":
            if m >= len(cv):
                return 1.0
            return float((c_max - cv[m]) / max(c_max, 1e-20))
        if ct == "comp_C_lo":
            if m >= len(cv):
                return 1.0
            return float((cv[m] - c_min) / max(c_max, 1e-20))

    if ct == "pole_sep":
        branch = desc.branch
        m = desc.cell_index
        if branch is None or m is None:
            return 1.0
        fp = b1_fp if branch == 1 else b2_fp
        sep = sep_b1 if branch == 1 else sep_b2
        if m + 1 >= len(fp):
            return 1.0
        actual_sep = fp[m + 1] - fp[m]
        return float((actual_sep - sep) / max(sep, 1.0))

    return 1.0  # Unknown type: pass


# ---------------------------------------------------------------------------
# Descriptor sort key (for deterministic ordering)
# ---------------------------------------------------------------------------


def _descriptor_sort_key(d: ConstraintDescriptor) -> tuple:
    return (
        d.constraint_type,
        d.target_index if d.target_index is not None else -1,
        d.freq_index if d.freq_index is not None else -1,
        d.branch if d.branch is not None else -1,
        d.cell_index if d.cell_index is not None else -1,
        d.element_id or "",
        d.name,
    )


# ---------------------------------------------------------------------------
# Human-readable constraint labels (diagnostic API — no evaluator coupling)
# ---------------------------------------------------------------------------


def _format_freq(f_hz: float) -> str:
    """Format a frequency with SI prefix for display."""
    if f_hz >= 1e9:
        return f"{f_hz / 1e9:.2f} GHz"
    if f_hz >= 1e6:
        return f"{f_hz / 1e6:.2f} MHz"
    if f_hz >= 1e3:
        return f"{f_hz / 1e3:.1f} kHz"
    return f"{f_hz:.1f} Hz"


def human_label(
    desc: ConstraintDescriptor,
    evaluation_frequencies_hz: tuple[float, ...] = (),
) -> str:
    """Convert a ConstraintDescriptor into a physics-readable label.

    This is a pure diagnostic function — it does NOT affect constraint
    evaluation, ordering, normalization, or any optimizer behavior.

    Parameters
    ----------
    desc : ConstraintDescriptor
        The descriptor to label.
    evaluation_frequencies_hz : tuple[float, ...]
        Evaluation frequency grid (needed to resolve ``freq_index``).

    Returns
    -------
    str
        A human-readable label such as ``"Γ ≤ limit @ 1.00 MHz"``.
    """
    ct = desc.constraint_type
    fi = desc.freq_index

    # Resolve frequency string
    freq_str = ""
    if fi is not None and 0 <= fi < len(evaluation_frequencies_hz):
        freq_str = f" @ {_format_freq(evaluation_frequencies_hz[fi])}"

    _LABELS: dict[str, str] = {
        "gamma": f"Γ ≤ limit{freq_str}",
        "r_max": f"R_in ≤ limit{freq_str}",
        "r_min": f"R_in ≥ limit{freq_str}",
        "x_bound": f"|X_in| ≤ limit{freq_str}",
        "i_source": f"I_source ≤ limit{freq_str}",
        "v_min": f"V_EOM ≥ target{freq_str}",
        "v_max": f"V_EOM ≤ limit{freq_str}",
        "offtarget": f"Off-target V_EOM ≤ limit{freq_str}",
    }

    if ct in _LABELS:
        return _LABELS[ct]

    branch = desc.branch
    m = desc.cell_index

    if ct == "comp_L_hi" and branch is not None and m is not None:
        return f"L_b{branch}[{m}] ≤ L_max"
    if ct == "comp_L_lo" and branch is not None and m is not None:
        return f"L_b{branch}[{m}] ≥ L_min"
    if ct == "comp_C_hi" and branch is not None and m is not None:
        return f"C_b{branch}[{m}] ≤ C_max"
    if ct == "comp_C_lo" and branch is not None and m is not None:
        return f"C_b{branch}[{m}] ≥ C_min"
    if ct == "pole_sep" and branch is not None and m is not None:
        return f"Pole separation b{branch}[{m}-{m + 1}]"

    # Custom or unknown — use the descriptor name as-is
    return desc.name


def layout_human_labels(
    layout: ConstraintLayout,
    evaluation_frequencies_hz: tuple[float, ...] = (),
) -> tuple[str, ...]:
    """Return human-readable labels for all descriptors in layout order.

    This is a pure diagnostic convenience function.  It does NOT change
    the constraint layout, order, or evaluation semantics.
    """
    return tuple(
        human_label(d, evaluation_frequencies_hz) for d in layout.descriptors
    )


# ---------------------------------------------------------------------------
# Layout compiler
# ---------------------------------------------------------------------------


def compile_constraint_layout(
    match_constraints: MatchConstraints,
    stress_constraints: StressConstraints,
    extra_records: list[ConstraintRecord],
    target_frequencies_hz: tuple[float, ...],
    evaluation_frequencies_hz: tuple[float, ...],
    target_indices: tuple[int, ...],
    off_target_indices: tuple[int, ...],
    severity_filter: ConstraintSeverity,
    n_cells_b1: int,
    n_cells_b2: int,
    z_ref_ohm: float,
) -> ConstraintLayout:
    """Build a sorted, frozen ``ConstraintLayout`` from schema objects.

    Parameters
    ----------
    severity_filter : ConstraintSeverity
        Include only descriptors of this severity.
    """
    descs: list[ConstraintDescriptor] = []

    def _add(d: ConstraintDescriptor) -> None:
        if d.severity == severity_filter and not d.validation_only:
            descs.append(d)

    # ---- MatchConstraints (applied to ALL_TARGETS) ----
    r_scale = max(match_constraints.resistance_max_ohm, z_ref_ohm, 1.0)
    x_scale = max(match_constraints.max_abs_reactance_ohm, r_scale)

    for ti, fi in enumerate(target_indices):
        f_hz = evaluation_frequencies_hz[fi]
        name_base = f"f{f_hz:.0f}Hz"

        _add(
            ConstraintDescriptor(
                name=f"gamma_{name_base}",
                constraint_type="gamma",
                frequency_scope=FrequencyScope.ALL_TARGETS,
                severity=ConstraintSeverity.HARD,
                target_index=ti,
                freq_index=fi,
                normalization_scale=max(match_constraints.gamma_max, 1e-6),
            )
        )
        _add(
            ConstraintDescriptor(
                name=f"r_max_{name_base}",
                constraint_type="r_max",
                frequency_scope=FrequencyScope.ALL_TARGETS,
                severity=ConstraintSeverity.HARD,
                target_index=ti,
                freq_index=fi,
                normalization_scale=r_scale,
            )
        )
        _add(
            ConstraintDescriptor(
                name=f"r_min_{name_base}",
                constraint_type="r_min",
                frequency_scope=FrequencyScope.ALL_TARGETS,
                severity=ConstraintSeverity.HARD,
                target_index=ti,
                freq_index=fi,
                normalization_scale=r_scale,
            )
        )
        _add(
            ConstraintDescriptor(
                name=f"x_bound_{name_base}",
                constraint_type="x_bound",
                frequency_scope=FrequencyScope.ALL_TARGETS,
                severity=ConstraintSeverity.HARD,
                target_index=ti,
                freq_index=fi,
                normalization_scale=x_scale,
            )
        )
        _add(
            ConstraintDescriptor(
                name=f"i_source_{name_base}",
                constraint_type="i_source",
                frequency_scope=FrequencyScope.ALL_TARGETS,
                severity=ConstraintSeverity.HARD,
                target_index=ti,
                freq_index=fi,
                normalization_scale=max(stress_constraints.source_current_rms_max_a, 1e-9),
            )
        )

    # ---- Off-target EOM envelope ----
    for fi in off_target_indices:
        f_hz = evaluation_frequencies_hz[fi]
        _add(
            ConstraintDescriptor(
                name=f"offtarget_veom_{f_hz:.0f}Hz",
                constraint_type="offtarget",
                frequency_scope=FrequencyScope.OFF_TARGET,
                severity=ConstraintSeverity.HARD,
                freq_index=fi,
                normalization_scale=max(stress_constraints.off_target_eom_peak_rms_v, 1e-6),
            )
        )

    # ---- Component bounds (per cell, both branches) ----
    for branch in (1, 2):
        n_cells = n_cells_b1 if branch == 1 else n_cells_b2
        for m in range(n_cells):
            for ct in ("comp_L_hi", "comp_L_lo", "comp_C_hi", "comp_C_lo"):
                _add(
                    ConstraintDescriptor(
                        name=f"{ct}_b{branch}_m{m}",
                        constraint_type=ct,
                        frequency_scope=FrequencyScope.ALL_TARGETS,
                        severity=ConstraintSeverity.HARD,
                        branch=branch,
                        cell_index=m,
                        normalization_scale=1.0,
                    )
                )

    # ---- Pole separation (per adjacent pair, both branches) ----
    for branch in (1, 2):
        n_cells = n_cells_b1 if branch == 1 else n_cells_b2
        for m in range(n_cells - 1):
            _add(
                ConstraintDescriptor(
                    name=f"pole_sep_b{branch}_m{m}m{m + 1}",
                    constraint_type="pole_sep",
                    frequency_scope=FrequencyScope.ALL_TARGETS,
                    severity=ConstraintSeverity.HARD,
                    branch=branch,
                    cell_index=m,
                    normalization_scale=1.0,
                )
            )

    # ---- Extra ConstraintRecord entries ----
    for rec in extra_records:
        if rec.validation_only:
            continue
        sev = ConstraintSeverity.HARD if rec.severity.value == "hard" else ConstraintSeverity.SOFT
        if sev != severity_filter:
            continue
        scope = FrequencyScope(rec.frequency_scope.value)
        if scope == FrequencyScope.ALL_TARGETS:
            for ti, fi in enumerate(target_indices):
                f_hz = evaluation_frequencies_hz[fi]
                _add(
                    ConstraintDescriptor(
                        name=f"{rec.name}_f{f_hz:.0f}Hz",
                        constraint_type="custom",
                        frequency_scope=scope,
                        severity=sev,
                        target_index=ti,
                        freq_index=fi,
                        normalization_scale=max(abs(rec.limit), 1e-6),
                        penalty_weight=rec.penalty_weight,
                    )
                )
        elif scope == FrequencyScope.SPECIFIC:
            for sf in rec.specific_frequencies_hz:
                # Find closest index in evaluation_frequencies_hz
                eval_arr = np.array(evaluation_frequencies_hz)
                fi = int(np.argmin(np.abs(eval_arr - sf)))
                _add(
                    ConstraintDescriptor(
                        name=f"{rec.name}_f{sf:.0f}Hz",
                        constraint_type="custom",
                        frequency_scope=scope,
                        severity=sev,
                        freq_index=fi,
                        normalization_scale=max(abs(rec.limit), 1e-6),
                        penalty_weight=rec.penalty_weight,
                    )
                )

    # Sort deterministically
    descs.sort(key=_descriptor_sort_key)

    return ConstraintLayout(descriptors=tuple(descs))
