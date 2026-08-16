"""Seed synthesis pipeline (Prompt 04B).

Generates deterministic, analytically derived seed candidates from
Schmidt/Foster math. Does NOT perform nonlinear optimization.

All orientation invariants are enforced by explicit ``ValueError``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from itertools import product as itertools_product
from typing import TYPE_CHECKING

import numpy as np

from foster_eom.circuit.solve import solve_circuit_single
from foster_eom.domain.component import ContinuousLimits
from foster_eom.domain.source import SourceSpec
from foster_eom.domain.topology import LOrientation, TopologySearchSpec
from foster_eom.foster.foster_form import (
    FosterComponents,
    coefficients_to_components,
    compute_coefficient_bounds,
)
from foster_eom.foster.foster_solve import (
    FosterSolveResult,
    build_foster_linear_system,
    solve_foster_system,
)
from foster_eom.foster.network_builder import BuiltFosterCircuit, build_foster_circuit
from foster_eom.foster.poles import PoleMode as InternalPoleMode
from foster_eom.foster.poles import PoleSpec as InternalPoleSpec
from foster_eom.foster.poles import generate_pole_candidates
from foster_eom.foster.schmidt import (
    BranchRealization,
    ReactanceTargetState,
    schmidt_dual_targets,
    schmidt_standard_targets,
)
from foster_eom.foster.sign_search import (
    SignPattern,
    SignPatternInfo,
    SignPruneCode,
    SignSearchConstraints,
    SignSearchDiagnostics,
    enumerate_sign_patterns,
)
from foster_eom.foster.topology_enum import TopologyCandidate, enumerate_topologies
from foster_eom.models.base import OnePortModel
from foster_eom.units import s11_db_from_gamma, z_to_gamma

if TYPE_CHECKING:
    from foster_eom.domain.topology import PoleSpec as DomainPoleSpec


# ---------------------------------------------------------------------------
# Domain → internal PoleSpec conversion
# ---------------------------------------------------------------------------


def _domain_to_internal_pole_spec(
    domain_ps: DomainPoleSpec,
) -> InternalPoleSpec:
    """Convert domain Pydantic PoleSpec to internal dataclass PoleSpec."""
    from foster_eom.domain.topology import PoleMode as DomainPoleMode

    mode_map = {
        DomainPoleMode.AUTO: InternalPoleMode.AUTO,
        DomainPoleMode.FIXED: InternalPoleMode.FIXED,
        DomainPoleMode.INTERVALS: InternalPoleMode.INTERVALS,
        DomainPoleMode.SCHMIDT_SEED: InternalPoleMode.SCHMIDT_SEED,
    }
    return InternalPoleSpec(
        mode=mode_map[domain_ps.mode],
        fixed_poles_hz=tuple(domain_ps.fixed_poles_hz) if domain_ps.fixed_poles_hz else None,
        intervals_hz=(
            tuple((iv.min_hz, iv.max_hz) for iv in domain_ps.intervals)
            if domain_ps.intervals
            else None
        ),
        allowed_band_hz=domain_ps.allowed_band_hz,
        delta_f_target_min_hz=domain_ps.min_distance_from_target_hz,
        delta_f_pole_min_hz=domain_ps.min_separation_hz,
    )


# ---------------------------------------------------------------------------
# Search options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignSearchOptions:
    """Controllable budget for sign-pattern search."""

    beam_width: int = 1000
    max_patterns: int = 256

    def __post_init__(self) -> None:
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {self.beam_width}")
        if self.max_patterns < 1:
            raise ValueError(f"max_patterns must be >= 1, got {self.max_patterns}")


# ---------------------------------------------------------------------------
# Failure codes
# ---------------------------------------------------------------------------


class SeedFailureCode(enum.StrEnum):
    SCHMIDT_INFEASIBLE = "schmidt_infeasible"
    ILLEGAL_BRANCH_REALIZATION = "illegal_branch_realization"
    SIGN_POLE_INCOMPATIBILITY = "sign_pole_incompatibility"
    INSUFFICIENT_REQUIRED_POLES = "insufficient_required_poles"
    POLE_LAYOUT_FAILURE = "pole_layout_failure"
    COEFFICIENT_BOUND_INFEASIBLE = "coefficient_bound_infeasible"
    FOSTER_NON_CONVERGED = "foster_non_converged"
    FOSTER_RESIDUAL_UNACCEPTABLE = "foster_residual_unacceptable"
    GRAPH_CONSTRUCTION_FAILURE = "graph_construction_failure"
    MNA_FAILURE = "mna_failure"
    POWER_BALANCE_FAILURE = "power_balance_failure"
    RMATCH_TOLERANCE_FAILURE = "rmatch_tolerance_failure"


@dataclass(frozen=True)
class SeedFailureRecord:
    code: SeedFailureCode
    reason: str
    orientation: LOrientation | None
    sign_pattern: tuple[int, ...] | None
    topology: TopologyCandidate | None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedValidation:
    match_error_at_targets: tuple[float, ...]
    max_match_error: float
    rms_match_error: float
    all_rmatch_satisfied: bool
    match_tolerance: float

    z_in_at_targets: tuple[complex, ...]
    gamma_at_targets: tuple[complex, ...]
    s11_db_at_targets: tuple[float, ...]
    perfect_match_flags: tuple[bool, ...]

    power_balance_ok_at_targets: tuple[bool, ...]
    power_balance_error_at_targets: tuple[float, ...]
    all_power_balance_ok: bool


# ---------------------------------------------------------------------------
# SeedCandidate — accepted-seed-only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedCandidate:
    """An accepted, fully validated seed candidate.

    Only instances that passed all validation stages (Foster solve,
    graph build, MNA, power balance, R_match tolerance) are
    constructed. There is no 'feasible' flag — existence of the
    object implies acceptance. Failures are represented solely
    through SeedFailureRecord and rejection diagnostics.
    """

    orientation: LOrientation
    sign_pattern: SignPattern
    topology: TopologyCandidate
    branch1_solve: FosterSolveResult | None
    branch2_solve: FosterSolveResult | None
    branch1_components: FosterComponents | None
    branch2_components: FosterComponents | None
    built_circuit: BuiltFosterCircuit
    validation: SeedValidation

    @property
    def shunt_components(self) -> FosterComponents | None:
        return self.branch1_components

    @property
    def series_components(self) -> FosterComponents | None:
        return self.branch2_components


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedGenerationDiagnostics:
    n_orientation_attempts: int
    n_sign_patterns: int
    n_topologies: int
    n_pole_layouts_branch1: int
    n_pole_layouts_branch2: int
    n_pole_layout_pairs: int
    n_solver_attempts: int
    n_mna_attempts: int

    rejection_counts: dict[SeedFailureCode, int]
    representative_failures: tuple[SeedFailureRecord, ...]
    max_failure_records_per_code: int

    sign_search_by_orientation: dict[LOrientation, SignSearchDiagnostics]
    sign_search_exhaustive: bool
    sign_search_truncated: bool

    sign_beam_width: int
    sign_max_patterns: int


@dataclass(frozen=True)
class SeedGenerationResult:
    seeds: tuple[SeedCandidate, ...]
    diagnostics: SeedGenerationDiagnostics


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def seed_sort_key(seed: SeedCandidate) -> tuple:
    """Deterministic lexicographic ranking. Lower is better."""
    v = seed.validation
    # Worst branch residual (trivial branches → 0.0)
    b1_resid = seed.branch1_solve.normalized_residual if seed.branch1_solve else 0.0
    b2_resid = seed.branch2_solve.normalized_residual if seed.branch2_solve else 0.0
    worst_residual = max(b1_resid, b2_resid)
    # Worst branch condition number
    b1_cond = seed.branch1_solve.scaled_condition_number if seed.branch1_solve else 0.0
    b2_cond = seed.branch2_solve.scaled_condition_number if seed.branch2_solve else 0.0
    worst_cond = max(b1_cond, b2_cond)

    return (
        v.max_match_error,
        v.rms_match_error,
        worst_residual,
        worst_cond,
        seed.topology.n_reactive,
        seed.topology.branch1_cells + seed.topology.branch2_cells,
        seed.sign_pattern.signs,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _DiagAccumulator:
    """Mutable accumulator for pipeline diagnostics."""

    def __init__(self, max_failure_records_per_code: int = 3) -> None:
        self.n_orientation_attempts = 0
        self.n_sign_patterns = 0
        self.n_topologies = 0
        self.n_pole_layouts_b1 = 0
        self.n_pole_layouts_b2 = 0
        self.n_pole_layout_pairs = 0
        self.n_solver_attempts = 0
        self.n_mna_attempts = 0
        self.rejection_counts: dict[SeedFailureCode, int] = {}
        self._failure_records: dict[SeedFailureCode, list[SeedFailureRecord]] = {}
        self.max_per_code = max_failure_records_per_code
        self.sign_search_by_orientation: dict[LOrientation, SignSearchDiagnostics] = {}

    def record_failure(self, record: SeedFailureRecord) -> None:
        code = record.code
        self.rejection_counts[code] = self.rejection_counts.get(code, 0) + 1
        if code not in self._failure_records:
            self._failure_records[code] = []
        if len(self._failure_records[code]) < self.max_per_code:
            self._failure_records[code].append(record)

    def propagate_sign_prune(
        self,
        sign_diag: SignSearchDiagnostics,
    ) -> None:
        """Map sign-search prune codes to seed failure codes."""

        mapping = {
            SignPruneCode.MIXED_OPEN_FINITE: SeedFailureCode.ILLEGAL_BRANCH_REALIZATION,
            SignPruneCode.REQUIRED_POLES_EXCEED_ALL_CELL_COUNTS: SeedFailureCode.INSUFFICIENT_REQUIRED_POLES,
            SignPruneCode.FIXED_POLE_INCOMPATIBLE: SeedFailureCode.SIGN_POLE_INCOMPATIBILITY,
            SignPruneCode.INTERVAL_POLE_INCOMPATIBLE: SeedFailureCode.SIGN_POLE_INCOMPATIBILITY,
            SignPruneCode.ILLEGAL_FINAL_BRANCH_REALIZATION: SeedFailureCode.ILLEGAL_BRANCH_REALIZATION,
        }
        for prune_code, count in sign_diag.structural_prune_counts.items():
            seed_code = mapping.get(prune_code)
            if seed_code is not None:
                self.rejection_counts[seed_code] = self.rejection_counts.get(seed_code, 0) + count

    @property
    def representative_failures(self) -> tuple[SeedFailureRecord, ...]:
        all_recs: list[SeedFailureRecord] = []
        for recs in self._failure_records.values():
            all_recs.extend(recs)
        return tuple(all_recs)


def _solve_branch(
    targets: tuple,
    f_targets_hz: np.ndarray,
    pole_freqs_hz: np.ndarray,
    has_c0: bool,
    has_linf: bool,
    component_limits: ContinuousLimits,
) -> tuple[FosterSolveResult | None, FosterComponents | None, SeedFailureCode | None, str | None]:
    """Solve one FINITE_FOSTER branch. Returns (solve_result, components, failure_code, reason)."""
    # Extract finite target values
    x_targets = np.array(
        [t.value_ohm for t in targets if t.state == ReactanceTargetState.FINITE],
        dtype=np.float64,
    )
    f_finite = np.array(
        [t.f_hz for t in targets if t.state == ReactanceTargetState.FINITE],
        dtype=np.float64,
    )

    # Compute coefficient bounds
    bounds = compute_coefficient_bounds(
        pole_freqs_hz,
        enable_k0=has_c0,
        enable_kinf=has_linf,
        component_limits=component_limits,
    )
    if bounds.any_infeasible:
        return (
            None,
            None,
            SeedFailureCode.COEFFICIENT_BOUND_INFEASIBLE,
            (f"Infeasible coefficient bounds at cells {bounds.infeasible_cells}"),
        )

    # Build and solve the linear system
    system = build_foster_linear_system(
        f_finite,
        x_targets,
        pole_freqs_hz,
        enable_k0=has_c0,
        enable_kinf=has_linf,
        coefficient_bounds=bounds,
    )
    result = solve_foster_system(system, pole_freqs_hz)

    if not result.feasible:
        if "bounds" in result.reason:
            code = SeedFailureCode.COEFFICIENT_BOUND_INFEASIBLE
        elif "max target error" in result.reason:
            code = SeedFailureCode.FOSTER_RESIDUAL_UNACCEPTABLE
        else:
            code = SeedFailureCode.FOSTER_NON_CONVERGED
        return result, None, code, result.reason

    # Convert to components
    components = coefficients_to_components(
        result.k0,
        result.k_inf,
        np.array(result.k_residues),
        np.array(result.f_poles_hz),
    )
    return result, components, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_seeds(
    r_match_ohm: float,
    source_spec: SourceSpec,
    eom_model: OnePortModel,
    f_targets_hz: np.ndarray,
    topo_spec: TopologySearchSpec,
    component_limits: ContinuousLimits,
    match_tolerance: float = 0.01,
    max_seeds: int = 20,
    sign_search_options: SignSearchOptions | None = None,
    max_failure_records_per_code: int = 3,
) -> SeedGenerationResult:
    """Generate seed candidates via Schmidt/Foster synthesis.

    Parameters
    ----------
    r_match_ohm : Desired real input match resistance (> 0).
    source_spec : Thévenin source for MNA validation.
    eom_model : EOM load model.
    f_targets_hz : Strictly increasing positive target frequencies.
    topo_spec : Topology search constraints.
    component_limits : L/C physical limits.
    match_tolerance : Max allowable relative match error.
    max_seeds : Maximum seeds to return.
    sign_search_options : Controllable sign-search budget.
    max_failure_records_per_code : Max stored failure samples per code.

    Returns
    -------
    SeedGenerationResult

    Raises
    ------
    ValueError
        On invalid inputs.
    """
    # --- Input validation ---
    if r_match_ohm <= 0:
        raise ValueError("r_match_ohm must be positive")
    if max_seeds <= 0:
        raise ValueError("max_seeds must be positive")
    if match_tolerance < 0:
        raise ValueError("match_tolerance must be non-negative")

    f_arr = np.asarray(f_targets_hz, dtype=np.float64).ravel()
    if len(f_arr) < 1:
        raise ValueError("At least one target frequency required")
    if np.any(~np.isfinite(f_arr)):
        raise ValueError("Target frequencies must be finite")
    if np.any(f_arr <= 0):
        raise ValueError("All target frequencies must be positive")
    # Strictly increasing
    if len(f_arr) > 1 and np.any(np.diff(f_arr) <= 0):
        raise ValueError("Target frequencies must be strictly increasing")

    sso = sign_search_options or SignSearchOptions()
    diag = _DiagAccumulator(max_failure_records_per_code)
    seeds: list[SeedCandidate] = []

    # Compute EOM load impedances at target frequencies
    z_load = np.array([complex(eom_model.z(float(f))) for f in f_arr], dtype=np.complex128)

    # --- Main pipeline loop ---
    for orientation in topo_spec.orientations:
        diag.n_orientation_attempts += 1

        # Schmidt targets
        if orientation == LOrientation.SCHMIDT_SHUNT_THEN_SERIES:
            schmidt = schmidt_standard_targets(r_match_ohm, z_load, f_arr)
        elif orientation == LOrientation.ALTERNATE_L_ORIENTATION:
            schmidt = schmidt_dual_targets(r_match_ohm, z_load, f_arr)
        else:
            raise ValueError(f"Unknown orientation: {orientation!r}")

        if not schmidt.all_valid:
            diag.record_failure(
                SeedFailureRecord(
                    code=SeedFailureCode.SCHMIDT_INFEASIBLE,
                    reason="Not all frequencies feasible for this orientation",
                    orientation=orientation,
                    sign_pattern=None,
                    topology=None,
                )
            )
            continue

        # Sign search — convert domain PoleSpec → internal PoleSpec
        internal_ps_b1 = _domain_to_internal_pole_spec(topo_spec.pole_spec_branch1)
        internal_ps_b2 = _domain_to_internal_pole_spec(topo_spec.pole_spec_branch2)
        constraints = SignSearchConstraints(
            branch1_min_cells=topo_spec.branch1_cells_min,
            branch1_max_cells=topo_spec.branch1_cells_max,
            branch2_min_cells=topo_spec.branch2_cells_min,
            branch2_max_cells=topo_spec.branch2_cells_max,
            pole_spec_branch1=internal_ps_b1,
            pole_spec_branch2=internal_ps_b2,
        )
        sign_result = enumerate_sign_patterns(
            schmidt,
            constraints,
            max_patterns=sso.max_patterns,
            beam_width=sso.beam_width,
        )
        diag.sign_search_by_orientation[orientation] = sign_result.diagnostics
        diag.propagate_sign_prune(sign_result.diagnostics)
        diag.n_sign_patterns += len(sign_result.patterns)

        for sign_info in sign_result.patterns:
            # Topology enumeration
            topologies = enumerate_topologies(topo_spec, sign_info)
            diag.n_topologies += len(topologies)

            for topology in topologies:
                # Pole layouts per branch
                p1_layouts = _generate_branch_pole_layouts(
                    sign_info,
                    topology,
                    topo_spec,
                    f_arr,
                    branch=1,
                )
                p2_layouts = _generate_branch_pole_layouts(
                    sign_info,
                    topology,
                    topo_spec,
                    f_arr,
                    branch=2,
                )
                diag.n_pole_layouts_b1 += len(p1_layouts)
                diag.n_pole_layouts_b2 += len(p2_layouts)
                diag.n_pole_layout_pairs += len(p1_layouts) * len(p2_layouts)

                for p1, p2 in itertools_product(p1_layouts, p2_layouts):
                    # Solve branches
                    b1_solve: FosterSolveResult | None = None
                    b1_comp: FosterComponents | None = None
                    b2_solve: FosterSolveResult | None = None
                    b2_comp: FosterComponents | None = None

                    failed = False
                    if sign_info.pattern.branch1_realization == BranchRealization.FINITE_FOSTER:
                        diag.n_solver_attempts += 1
                        b1_solve, b1_comp, fail_code, fail_reason = _solve_branch(
                            sign_info.pattern.shunt_targets,
                            f_arr,
                            p1,
                            topology.branch1_has_c0,
                            topology.branch1_has_linf,
                            component_limits,
                        )
                        if fail_code is not None:
                            diag.record_failure(
                                SeedFailureRecord(
                                    code=fail_code,
                                    reason=fail_reason or "",
                                    orientation=orientation,
                                    sign_pattern=sign_info.pattern.signs,
                                    topology=topology,
                                )
                            )
                            failed = True

                    if (
                        not failed
                        and sign_info.pattern.branch2_realization == BranchRealization.FINITE_FOSTER
                    ):
                        diag.n_solver_attempts += 1
                        b2_solve, b2_comp, fail_code, fail_reason = _solve_branch(
                            sign_info.pattern.series_targets,
                            f_arr,
                            p2,
                            topology.branch2_has_c0,
                            topology.branch2_has_linf,
                            component_limits,
                        )
                        if fail_code is not None:
                            diag.record_failure(
                                SeedFailureRecord(
                                    code=fail_code,
                                    reason=fail_reason or "",
                                    orientation=orientation,
                                    sign_pattern=sign_info.pattern.signs,
                                    topology=topology,
                                )
                            )
                            failed = True

                    if failed:
                        continue

                    # Build circuit
                    try:
                        built = build_foster_circuit(
                            topology,
                            sign_info.pattern,
                            b1_comp,
                            b2_comp,
                            eom_model,
                        )
                    except (ValueError, RuntimeError) as e:
                        diag.record_failure(
                            SeedFailureRecord(
                                code=SeedFailureCode.GRAPH_CONSTRUCTION_FAILURE,
                                reason=str(e),
                                orientation=orientation,
                                sign_pattern=sign_info.pattern.signs,
                                topology=topology,
                            )
                        )
                        continue

                    # MNA validation
                    diag.n_mna_attempts += 1
                    validation = _validate_seed(
                        built,
                        source_spec,
                        r_match_ohm,
                        f_arr,
                        match_tolerance,
                    )

                    if validation is None:
                        diag.record_failure(
                            SeedFailureRecord(
                                code=SeedFailureCode.MNA_FAILURE,
                                reason="MNA solve failed at one or more targets",
                                orientation=orientation,
                                sign_pattern=sign_info.pattern.signs,
                                topology=topology,
                            )
                        )
                        continue

                    # Power balance — hard failure
                    if not validation.all_power_balance_ok:
                        diag.record_failure(
                            SeedFailureRecord(
                                code=SeedFailureCode.POWER_BALANCE_FAILURE,
                                reason="Power balance failed at one or more targets",
                                orientation=orientation,
                                sign_pattern=sign_info.pattern.signs,
                                topology=topology,
                            )
                        )
                        continue

                    # R_match tolerance
                    if not validation.all_rmatch_satisfied:
                        diag.record_failure(
                            SeedFailureRecord(
                                code=SeedFailureCode.RMATCH_TOLERANCE_FAILURE,
                                reason=f"Max match error {validation.max_match_error:.6e} > tolerance {match_tolerance:.6e}",
                                orientation=orientation,
                                sign_pattern=sign_info.pattern.signs,
                                topology=topology,
                            )
                        )
                        continue

                    # Accept seed
                    seeds.append(
                        SeedCandidate(
                            orientation=orientation,
                            sign_pattern=sign_info.pattern,
                            topology=topology,
                            branch1_solve=b1_solve,
                            branch2_solve=b2_solve,
                            branch1_components=b1_comp,
                            branch2_components=b2_comp,
                            built_circuit=built,
                            validation=validation,
                        )
                    )

    # Sort and cap
    seeds.sort(key=seed_sort_key)
    if len(seeds) > max_seeds:
        seeds = seeds[:max_seeds]

    # Aggregate sign-search diagnostics
    all_exhaustive = (
        all(d.search_exhaustive for d in diag.sign_search_by_orientation.values())
        if diag.sign_search_by_orientation
        else True
    )
    any_truncated = (
        any(d.search_truncated for d in diag.sign_search_by_orientation.values())
        if diag.sign_search_by_orientation
        else False
    )

    return SeedGenerationResult(
        seeds=tuple(seeds),
        diagnostics=SeedGenerationDiagnostics(
            n_orientation_attempts=diag.n_orientation_attempts,
            n_sign_patterns=diag.n_sign_patterns,
            n_topologies=diag.n_topologies,
            n_pole_layouts_branch1=diag.n_pole_layouts_b1,
            n_pole_layouts_branch2=diag.n_pole_layouts_b2,
            n_pole_layout_pairs=diag.n_pole_layout_pairs,
            n_solver_attempts=diag.n_solver_attempts,
            n_mna_attempts=diag.n_mna_attempts,
            rejection_counts=dict(diag.rejection_counts),
            representative_failures=diag.representative_failures,
            max_failure_records_per_code=max_failure_records_per_code,
            sign_search_by_orientation=dict(diag.sign_search_by_orientation),
            sign_search_exhaustive=all_exhaustive,
            sign_search_truncated=any_truncated,
            sign_beam_width=sso.beam_width,
            sign_max_patterns=sso.max_patterns,
        ),
    )


# ---------------------------------------------------------------------------
# Internal pipeline helpers
# ---------------------------------------------------------------------------


def _generate_branch_pole_layouts(
    sign_info: SignPatternInfo,
    topology: TopologyCandidate,
    topo_spec: TopologySearchSpec,
    f_targets_hz: np.ndarray,
    branch: int,
) -> list[np.ndarray]:
    """Generate pole layout candidates for one branch.

    Returns [None-wrapped-as-empty] for trivial branches.
    """
    if branch == 1:
        realization = sign_info.pattern.branch1_realization
        n_cells = topology.branch1_cells
        required_intervals = list(sign_info.pattern.branch1_required_intervals)
        pole_spec = _domain_to_internal_pole_spec(topo_spec.pole_spec_branch1)
    else:
        realization = sign_info.pattern.branch2_realization
        n_cells = topology.branch2_cells
        required_intervals = list(sign_info.pattern.branch2_required_intervals)
        pole_spec = _domain_to_internal_pole_spec(topo_spec.pole_spec_branch2)

    if realization != BranchRealization.FINITE_FOSTER:
        return [np.array([], dtype=np.float64)]

    if n_cells == 0:
        return [np.array([], dtype=np.float64)]

    candidates = generate_pole_candidates(
        pole_spec=pole_spec,
        f_targets_hz=f_targets_hz,
        n_cells=n_cells,
        required_intervals=required_intervals,
    )
    if not candidates:
        return []
    return [np.array(c.f_poles_hz, dtype=np.float64) for c in candidates]


def _validate_seed(
    built: BuiltFosterCircuit,
    source_spec: SourceSpec,
    r_match_ohm: float,
    f_targets_hz: np.ndarray,
    match_tolerance: float,
) -> SeedValidation | None:
    """Run MNA at all target frequencies and compute validation metrics."""
    from foster_eom.errors import CircuitSolveStatus

    z_in_list: list[complex] = []
    gamma_list: list[complex] = []
    s11_list: list[float] = []
    match_errors: list[float] = []
    power_ok_list: list[bool] = []
    power_err_list: list[float] = []
    perfect_flags: list[bool] = []

    for f in f_targets_hz:
        try:
            sol = solve_circuit_single(built.graph, source_spec, float(f))
        except Exception:
            return None

        if sol.status != CircuitSolveStatus.OK:
            return None

        z_in = sol.z_in
        if z_in is None:
            return None

        z_in_list.append(z_in)

        gamma = z_to_gamma(z_in, source_spec.z_ref_ohm)
        gamma_list.append(gamma)
        s11_val = s11_db_from_gamma(gamma)
        s11_list.append(s11_val)

        perfect = abs(gamma) == 0.0
        perfect_flags.append(perfect)

        match_err = abs(z_in - r_match_ohm) / r_match_ohm
        match_errors.append(match_err)

        # Power balance
        pb_ok = sol.power_balance_ok
        pb_residual = sol.power_balance_residual
        pb_err = abs(pb_residual) if pb_residual is not None else 0.0
        power_ok_list.append(pb_ok)
        power_err_list.append(pb_err)

    max_err = max(match_errors)
    rms_err = float(np.sqrt(np.mean(np.array(match_errors) ** 2)))
    all_rmatch = max_err <= match_tolerance

    return SeedValidation(
        match_error_at_targets=tuple(match_errors),
        max_match_error=max_err,
        rms_match_error=rms_err,
        all_rmatch_satisfied=all_rmatch,
        match_tolerance=match_tolerance,
        z_in_at_targets=tuple(z_in_list),
        gamma_at_targets=tuple(gamma_list),
        s11_db_at_targets=tuple(s11_list),
        perfect_match_flags=tuple(perfect_flags),
        power_balance_ok_at_targets=tuple(power_ok_list),
        power_balance_error_at_targets=tuple(power_err_list),
        all_power_balance_ok=all(power_ok_list),
    )
