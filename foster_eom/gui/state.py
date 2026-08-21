"""GUI Project State (MVC Model).

Holds inputs, tracks revisions, and caches results. No Qt imports.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import uuid
from dataclasses import dataclass, field


@dataclass
class SourceParams:
    mode: str = "thevenin"
    vth_rms: float = 1.0
    z_source_ohm: float = 50.0
    available_power_w: float = 0.0


@dataclass
class MotionalBranchParams:
    rm_ohm: float = 0.0
    lm_h: float = 0.0
    cm_f: float = 0.0


@dataclass
class EOMParams:
    model_type: str = "ideal_capacitor"
    c0_f: float | None = 10e-12
    rs_ohm: float | None = None
    ls_h: float | None = None
    g0_s: float | None = None
    motional_branches: list[MotionalBranchParams] = field(default_factory=list)
    tabular_file: str | None = None
    tabular_format: str | None = None
    validity_hz: tuple[float, float] | None = None


@dataclass
class TopologyParams:
    n_branches: int = 1
    n_cells_per_branch: int = 1
    fixed_poles: list[str] = field(default_factory=list)


@dataclass
class OptimizationPresetParams:
    """GUI-level optimization preset selection and custom overrides."""

    preset: str = "BALANCED"  # FAST, BALANCED, THOROUGH, CUSTOM
    custom_max_global_evaluations: int = 2500
    custom_polish_top_k: int = 2
    custom_local_max_iterations: int = 100


@dataclass
class MatchParams:
    """GUI-level matching constraint parameters."""

    gamma_max: float = 0.25
    resistance_min_ohm: float = 35.0
    resistance_max_ohm: float = 70.0
    max_abs_reactance_ohm: float = 20.0


@dataclass
class StressParams:
    """GUI-level stress constraint parameters."""

    source_current_rms_max_a: float = 0.5
    off_target_eom_peak_rms_v: float = 50.0
    default_cap_peak_voltage_v: float = 100.0
    default_ind_peak_current_a: float = 1.0


@dataclass
class ComponentLimitParams:
    """GUI-level L/C component range parameters."""

    l_min_h: float = 10.0e-9
    l_max_h: float = 100.0e-6
    c_min_f: float = 0.2e-12
    c_max_f: float = 20.0e-9


@dataclass
class ObjectiveWeightParams:
    """GUI-level objective weight parameters (Advanced)."""

    weight_gamma: float = 1.0
    weight_voltage: float = 1.0
    weight_loss: float = 0.0
    weight_complexity: float = 0.0


@dataclass
class ProjectState:
    # --- inputs ---
    name: str = ""
    frequencies_hz: list[float] = field(default_factory=list)
    voltage_targets_rms_v: list[float | None] = field(default_factory=list)
    sweep_f_min_hz: float = 1e6
    sweep_f_max_hz: float = 30e6
    source: SourceParams = field(default_factory=SourceParams)
    eom: EOMParams = field(default_factory=EOMParams)
    topology: TopologyParams = field(default_factory=TopologyParams)
    optimization_preset: OptimizationPresetParams = field(
        default_factory=OptimizationPresetParams
    )
    match_params: MatchParams = field(default_factory=MatchParams)
    stress_params: StressParams = field(default_factory=StressParams)
    component_limits: ComponentLimitParams = field(default_factory=ComponentLimitParams)
    objective_weights: ObjectiveWeightParams = field(default_factory=ObjectiveWeightParams)


    # GUI-level context
    library_path: str | None = None
    library_sha: str | None = None  # Snapshot of current library

    # --- revision tracking ---
    revision: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_sha256: str = ""

    # --- committed results (no Qt opaque objects, must be serializable) ---
    optimize_result_path: str | None = None
    optimize_revision: str | None = None

    verify_result_path: str | None = None
    verify_revision: str | None = None

    realization_result_path: str | None = None
    realization_revision: str | None = None
    realization_library_sha: str | None = None

    robustness_result_path: str | None = None
    robustness_revision: str | None = None
    robustness_library_sha: str | None = None

    spice_result_path: str | None = None
    spice_revision: str | None = None

    modified: bool = False

    def compute_input_sha(self) -> str:
        """Compute canonical hash of inputs to detect external modifications."""
        data = {
            "name": self.name,
            "frequencies_hz": self.frequencies_hz,
            "source": dataclasses.asdict(self.source),
            "eom": dataclasses.asdict(self.eom),
            "topology": dataclasses.asdict(self.topology),
        }
        s = json.dumps(data, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def bump_revision(self) -> None:
        """Invalidate all downstream results because inputs changed."""
        self.revision = str(uuid.uuid4())
        self.input_sha256 = self.compute_input_sha()
        self.optimize_result_path = None
        self.verify_result_path = None
        self.realization_result_path = None
        self.robustness_result_path = None
        self.spice_result_path = None
        self.modified = True

    def invalidate_library(self) -> None:
        """Invalidate P09/P10/SPICE when library changes."""
        self.realization_result_path = None
        self.robustness_result_path = None
        # SPICE result is invalidated because it depends on catalog realization
        self.spice_result_path = None
        self.modified = True
