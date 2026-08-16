"""Decision-variable mapper for continuous Foster optimization (Prompt 05).

Maps between normalized ``x ∈ [0, 1]^n`` and physical Foster coordinates
``(k_m, f_poles_hz)`` for one ``ContinuousOptimizationDomain``.

Coordinate convention (nonredundant — no simultaneous L/C variables):

Per branch:
  log(k_0)       → endpoint capacitor coefficient  C_0 = 1/k_0
  log(k_inf)     → endpoint inductor coefficient   L_inf = k_inf
  log(k_m)       → cell m residue coefficient      C_m = 1/k_m
  f_p,m          → cell m pole frequency  (only for movable poles)

Physical reconstruction at every evaluation:
  C_0   = 1/k_0
  L_inf = k_inf
  C_m   = 1/k_m
  L_m   = k_m / (2*pi*f_p,m)^2

Box-normalization for log(k) variables:
  x = (log(k) - log(k_box_min)) / (log(k_box_max/k_box_min))

Box-normalization for pole variables:
  x = (f_p - f_lo) / (f_hi - f_lo)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Per-variable descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariableDescriptor:
    """Metadata for one component of the normalized decision vector.

    Attributes
    ----------
    name : str
        Human-readable name, e.g. ``"b1_logk0"``, ``"b2_fp_1"``.
    branch : int
        1 or 2.
    var_type : str
        ``"logk0"`` | ``"logkinf"`` | ``"logkm"`` | ``"fp"``.
    cell_index : int | None
        For ``"logkm"`` and ``"fp"`` variables.
    log_k_box_min : float | None
        For log-k variables: ``log(k_box_min)``.
    log_k_box_range : float | None
        For log-k variables: ``log(k_box_max/k_box_min)`` (positive).
    f_lo_hz : float | None
        For pole variables: lower bound of legal interval.
    f_hi_hz : float | None
        For pole variables: upper bound of legal interval.  Equal to
        ``f_lo_hz`` for FIXED (point) poles — should not appear in vector.
    """

    name: str
    branch: int
    var_type: str
    cell_index: int | None = None
    log_k_box_min: float | None = None
    log_k_box_range: float | None = None
    f_lo_hz: float | None = None
    f_hi_hz: float | None = None


# ---------------------------------------------------------------------------
# Branch unpacked result
# ---------------------------------------------------------------------------


class BranchCoordinates(NamedTuple):
    """Unpacked Foster coordinates for one branch."""

    k0: float | None  # None if branch has no C_0
    k_inf: float | None  # None if branch has no L_inf
    k_residues: tuple[float, ...]  # one per cell (even if FIXED)
    f_poles_hz: tuple[float, ...]  # one per cell (even if FIXED)
    l_values_h: tuple[float, ...]  # derived: L_m = k_m / q_m
    c_values_f: tuple[float, ...]  # derived: C_m = 1/k_m, C_0 = 1/k0


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionVariableMapper:
    """Frozen mapper between normalized ``x`` and Foster coordinates.

    Constructed once per ``ContinuousOptimizationDomain``.  All methods
    are pure functions (no mutable state).

    Parameters
    ----------
    descriptors : tuple[VariableDescriptor, ...]
        Ordered variable descriptors (defines vector layout).
    branch1_n_cells : int
    branch1_has_c0 : bool
    branch1_has_linf : bool
    branch1_fixed_k0 : float | None
        If C_0 exists but is not a variable (degenerate case), its value.
    branch1_fixed_kinf : float | None
    branch1_fixed_k_residues : tuple[float | None, ...]
        None = variable; float = FIXED value for that cell.
    branch1_fixed_f_poles_hz : tuple[float | None, ...]
        None = movable; float = FIXED pole value.
    (same pattern for branch2)
    """

    descriptors: tuple[VariableDescriptor, ...]
    dimension: int

    # Branch 1
    branch1_n_cells: int
    branch1_has_c0: bool
    branch1_has_linf: bool
    branch1_fixed_k0: float | None
    branch1_fixed_kinf: float | None
    branch1_fixed_k_residues: tuple[float | None, ...]  # None = variable
    branch1_fixed_f_poles_hz: tuple[float | None, ...]  # None = movable

    # Branch 2
    branch2_n_cells: int
    branch2_has_c0: bool
    branch2_has_linf: bool
    branch2_fixed_k0: float | None
    branch2_fixed_kinf: float | None
    branch2_fixed_k_residues: tuple[float | None, ...]
    branch2_fixed_f_poles_hz: tuple[float | None, ...]

    def pack(
        self,
        k0_b1: float | None,
        k_inf_b1: float | None,
        k_residues_b1: tuple[float, ...],
        f_poles_b1: tuple[float, ...],
        k0_b2: float | None,
        k_inf_b2: float | None,
        k_residues_b2: tuple[float, ...],
        f_poles_b2: tuple[float, ...],
    ) -> np.ndarray:
        """Pack physical Foster coordinates into normalized ``x ∈ [0,1]^n``.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        x = np.empty(self.dimension, dtype=np.float64)
        for idx, d in enumerate(self.descriptors):
            if d.var_type == "logk0":
                k = k0_b1 if d.branch == 1 else k0_b2
                assert k is not None
                x[idx] = self._pack_logk(k, d)
            elif d.var_type == "logkinf":
                k = k_inf_b1 if d.branch == 1 else k_inf_b2
                assert k is not None
                x[idx] = self._pack_logk(k, d)
            elif d.var_type == "logkm":
                assert d.cell_index is not None
                kr = k_residues_b1 if d.branch == 1 else k_residues_b2
                x[idx] = self._pack_logk(kr[d.cell_index], d)
            elif d.var_type == "fp":
                assert d.cell_index is not None
                fp = f_poles_b1 if d.branch == 1 else f_poles_b2
                x[idx] = self._pack_fp(fp[d.cell_index], d)
        return np.clip(x, 0.0, 1.0)

    def unpack(self, x: np.ndarray) -> tuple[BranchCoordinates, BranchCoordinates]:
        """Unpack normalized ``x`` into physical Foster coordinates.

        Returns
        -------
        (branch1, branch2) : tuple[BranchCoordinates, BranchCoordinates]
        """
        x = np.asarray(x, dtype=np.float64)
        # Collect variable values from x
        var_vals: dict[str, float] = {}
        for i, d in enumerate(self.descriptors):
            var_vals[d.name] = float(x[i])

        b1 = self._unpack_branch(1, var_vals)
        b2 = self._unpack_branch(2, var_vals)
        return b1, b2

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_logk(k: float, d: VariableDescriptor) -> float:
        assert d.log_k_box_min is not None and d.log_k_box_range is not None
        log_k = math.log(k)
        return (log_k - d.log_k_box_min) / d.log_k_box_range

    @staticmethod
    def _unpack_logk(x_val: float, d: VariableDescriptor) -> float:
        assert d.log_k_box_min is not None and d.log_k_box_range is not None
        log_k = d.log_k_box_min + x_val * d.log_k_box_range
        return math.exp(log_k)

    @staticmethod
    def _pack_fp(f: float, d: VariableDescriptor) -> float:
        assert d.f_lo_hz is not None and d.f_hi_hz is not None
        span = d.f_hi_hz - d.f_lo_hz
        if span <= 0.0:
            return 0.0
        return (f - d.f_lo_hz) / span

    @staticmethod
    def _unpack_fp(x_val: float, d: VariableDescriptor) -> float:
        assert d.f_lo_hz is not None and d.f_hi_hz is not None
        return d.f_lo_hz + x_val * (d.f_hi_hz - d.f_lo_hz)

    def _unpack_branch(self, branch: int, var_vals: dict[str, float]) -> BranchCoordinates:
        n_cells = self.branch1_n_cells if branch == 1 else self.branch2_n_cells
        has_c0 = self.branch1_has_c0 if branch == 1 else self.branch2_has_c0
        has_linf = self.branch1_has_linf if branch == 1 else self.branch2_has_linf
        fixed_k0 = self.branch1_fixed_k0 if branch == 1 else self.branch2_fixed_k0
        fixed_kinf = self.branch1_fixed_kinf if branch == 1 else self.branch2_fixed_kinf
        fixed_kr = self.branch1_fixed_k_residues if branch == 1 else self.branch2_fixed_k_residues
        fixed_fp = self.branch1_fixed_f_poles_hz if branch == 1 else self.branch2_fixed_f_poles_hz

        # k0
        k0: float | None = None
        if has_c0:
            name = f"b{branch}_logk0"
            if name in var_vals:
                desc = next(d for d in self.descriptors if d.name == name)
                k0 = self._unpack_logk(var_vals[name], desc)
            else:
                k0 = fixed_k0

        # k_inf
        k_inf: float | None = None
        if has_linf:
            name = f"b{branch}_logkinf"
            if name in var_vals:
                desc = next(d for d in self.descriptors if d.name == name)
                k_inf = self._unpack_logk(var_vals[name], desc)
            else:
                k_inf = fixed_kinf

        # cells
        k_residues: list[float] = []
        f_poles: list[float] = []
        l_vals: list[float] = []
        c_vals: list[float] = []

        for m in range(n_cells):
            # k_m
            km_name = f"b{branch}_logkm_{m}"
            if km_name in var_vals:
                desc = next(d for d in self.descriptors if d.name == km_name)
                km = self._unpack_logk(var_vals[km_name], desc)
            else:
                km_fixed = fixed_kr[m] if m < len(fixed_kr) else None
                assert km_fixed is not None, f"Missing k_m for branch {branch} cell {m}"
                km = km_fixed

            # f_p,m
            fp_name = f"b{branch}_fp_{m}"
            if fp_name in var_vals:
                desc = next(d for d in self.descriptors if d.name == fp_name)
                fp = self._unpack_fp(var_vals[fp_name], desc)
            else:
                fp_fixed = fixed_fp[m] if m < len(fixed_fp) else None
                assert fp_fixed is not None, f"Missing f_p for branch {branch} cell {m}"
                fp = fp_fixed

            q_m = (_TWO_PI * fp) ** 2
            l_m = km / q_m if q_m > 0 else 0.0
            c_m = 1.0 / km if km > 0 else math.inf

            k_residues.append(km)
            f_poles.append(fp)
            l_vals.append(l_m)
            c_vals.append(c_m)

        # C_0 and L_inf derived values
        if k0 is not None and k0 > 0:
            c_vals.append(1.0 / k0)
        if k_inf is not None:
            l_vals.append(k_inf)

        return BranchCoordinates(
            k0=k0,
            k_inf=k_inf,
            k_residues=tuple(k_residues),
            f_poles_hz=tuple(f_poles),
            l_values_h=tuple(l_vals),
            c_values_f=tuple(c_vals),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_variable_mapper(
    branch1_n_cells: int,
    branch1_has_c0: bool,
    branch1_has_linf: bool,
    branch1_pole_regions: tuple[tuple[float, float], ...],
    branch1_k_box_bounds: tuple[tuple[float, float], ...],
    branch1_k0_bounds: tuple[float, float] | None,
    branch1_kinf_bounds: tuple[float, float] | None,
    branch1_fixed_k0: float | None,
    branch1_fixed_kinf: float | None,
    branch1_fixed_k_residues: tuple[float | None, ...],
    branch1_fixed_f_poles_hz: tuple[float | None, ...],
    branch2_n_cells: int,
    branch2_has_c0: bool,
    branch2_has_linf: bool,
    branch2_pole_regions: tuple[tuple[float, float], ...],
    branch2_k_box_bounds: tuple[tuple[float, float], ...],
    branch2_k0_bounds: tuple[float, float] | None,
    branch2_kinf_bounds: tuple[float, float] | None,
    branch2_fixed_k0: float | None,
    branch2_fixed_kinf: float | None,
    branch2_fixed_k_residues: tuple[float | None, ...],
    branch2_fixed_f_poles_hz: tuple[float | None, ...],
) -> DecisionVariableMapper:
    """Construct a ``DecisionVariableMapper`` from branch parameters.

    ``branch*_k_box_bounds[m] = (k_box_min_m, k_box_max_m)`` is the outer
    envelope for the log-k variable of movable-pole cell m.
    ``branch*_fixed_k_residues[m] = None`` means that cell m's k is a variable.
    ``branch*_fixed_f_poles_hz[m] = None`` means that cell m's pole is movable.
    """
    descriptors: list[VariableDescriptor] = []

    def _add_branch(
        branch: int,
        n_cells: int,
        has_c0: bool,
        has_linf: bool,
        pole_regions: tuple[tuple[float, float], ...],
        k_box_bounds: tuple[tuple[float, float], ...],
        k0_bounds: tuple[float, float] | None,
        kinf_bounds: tuple[float, float] | None,
        fixed_kr: tuple[float | None, ...],
        fixed_fp: tuple[float | None, ...],
    ) -> None:
        # Endpoint capacitor
        if has_c0 and k0_bounds is not None:
            k0_min, k0_max = k0_bounds
            if k0_min > 0 and k0_max > k0_min:
                log_min = math.log(k0_min)
                log_range = math.log(k0_max / k0_min)
                descriptors.append(
                    VariableDescriptor(
                        name=f"b{branch}_logk0",
                        branch=branch,
                        var_type="logk0",
                        log_k_box_min=log_min,
                        log_k_box_range=log_range,
                    )
                )

        # Endpoint inductor
        if has_linf and kinf_bounds is not None:
            ki_min, ki_max = kinf_bounds
            if ki_min > 0 and ki_max > ki_min:
                log_min = math.log(ki_min)
                log_range = math.log(ki_max / ki_min)
                descriptors.append(
                    VariableDescriptor(
                        name=f"b{branch}_logkinf",
                        branch=branch,
                        var_type="logkinf",
                        log_k_box_min=log_min,
                        log_k_box_range=log_range,
                    )
                )

        # Cells
        for m in range(n_cells):
            is_fixed_k = m < len(fixed_kr) and fixed_kr[m] is not None
            is_fixed_fp = m < len(fixed_fp) and fixed_fp[m] is not None

            # log(k_m) — variable if not fixed
            if not is_fixed_k:
                if m < len(k_box_bounds):
                    kmin, kmax = k_box_bounds[m]
                else:
                    # Fallback: use pole-region bounds to derive outer box
                    f_lo, f_hi = pole_regions[m] if m < len(pole_regions) else (1e3, 1e9)
                    kmin, kmax = 1e-15, 1e15  # degenerate; domain will be infeasible

                if kmin > 0 and kmax > kmin:
                    log_min = math.log(kmin)
                    log_range = math.log(kmax / kmin)
                    descriptors.append(
                        VariableDescriptor(
                            name=f"b{branch}_logkm_{m}",
                            branch=branch,
                            var_type="logkm",
                            cell_index=m,
                            log_k_box_min=log_min,
                            log_k_box_range=log_range,
                        )
                    )

            # f_p,m — variable if movable (f_lo < f_hi)
            if not is_fixed_fp and m < len(pole_regions):
                f_lo, f_hi = pole_regions[m]
                if f_hi > f_lo:
                    descriptors.append(
                        VariableDescriptor(
                            name=f"b{branch}_fp_{m}",
                            branch=branch,
                            var_type="fp",
                            cell_index=m,
                            f_lo_hz=f_lo,
                            f_hi_hz=f_hi,
                        )
                    )

    _add_branch(
        1,
        branch1_n_cells,
        branch1_has_c0,
        branch1_has_linf,
        branch1_pole_regions,
        branch1_k_box_bounds,
        branch1_k0_bounds,
        branch1_kinf_bounds,
        branch1_fixed_k_residues,
        branch1_fixed_f_poles_hz,
    )
    _add_branch(
        2,
        branch2_n_cells,
        branch2_has_c0,
        branch2_has_linf,
        branch2_pole_regions,
        branch2_k_box_bounds,
        branch2_k0_bounds,
        branch2_kinf_bounds,
        branch2_fixed_k_residues,
        branch2_fixed_f_poles_hz,
    )

    return DecisionVariableMapper(
        descriptors=tuple(descriptors),
        dimension=len(descriptors),
        branch1_n_cells=branch1_n_cells,
        branch1_has_c0=branch1_has_c0,
        branch1_has_linf=branch1_has_linf,
        branch1_fixed_k0=branch1_fixed_k0,
        branch1_fixed_kinf=branch1_fixed_kinf,
        branch1_fixed_k_residues=branch1_fixed_k_residues,
        branch1_fixed_f_poles_hz=branch1_fixed_f_poles_hz,
        branch2_n_cells=branch2_n_cells,
        branch2_has_c0=branch2_has_c0,
        branch2_has_linf=branch2_has_linf,
        branch2_fixed_k0=branch2_fixed_k0,
        branch2_fixed_kinf=branch2_fixed_kinf,
        branch2_fixed_k_residues=branch2_fixed_k_residues,
        branch2_fixed_f_poles_hz=branch2_fixed_f_poles_hz,
    )
