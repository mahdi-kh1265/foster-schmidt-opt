"""Adapter between GUI ProjectState and backend ProjectSpec."""

from __future__ import annotations

from pathlib import Path

import yaml

from foster_eom.domain.component import ComponentPolicy, ContinuousLimits
from foster_eom.domain.constraints import MatchConstraints, StressConstraints
from foster_eom.domain.eom import EOMModelSpec, EOMModelType, MotionalBranch
from foster_eom.domain.frequency_plan import FrequencyPlan, FrequencyTarget
from foster_eom.domain.objectives import DerivativeMode, OptimizationSpec
from foster_eom.domain.project import ProjectSpec
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.domain.topology import TopologySearchSpec
from foster_eom.gui.state import (
    EOMParams,
    MotionalBranchParams,
    ProjectState,
    SourceParams,
    TopologyParams,
)
from foster_eom.persistence.yaml_io import load_project, save_project


# ---------------------------------------------------------------------------
# Preset definitions (GUI policy — does NOT modify backend defaults)
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict[str, int]] = {
    "FAST": {
        "max_global_evaluations": 500,
        "polish_top_k": 1,
        "local_max_iterations": 20,
        "random_seed": 42,
    },
    "BALANCED": {
        "max_global_evaluations": 2_500,
        "polish_top_k": 2,
        "local_max_iterations": 100,
        "random_seed": 20260815,
    },
    "THOROUGH": {
        "max_global_evaluations": 50_000,
        "polish_top_k": 8,
        "local_max_iterations": 1_500,
        "random_seed": 20260815,
    },
}


def _compile_optimization_spec(state: ProjectState) -> OptimizationSpec:
    """Compile GUI preset + custom overrides into an explicit OptimizationSpec.

    Every budget field is set explicitly — no reliance on backend defaults.
    ``DerivativeMode.ANALYTICAL`` is always set (frozen GUI production path).
    """
    preset_name = state.optimization_preset.preset
    ow = state.objective_weights

    if preset_name == "CUSTOM":
        return OptimizationSpec(
            max_global_evaluations=state.optimization_preset.custom_max_global_evaluations,
            polish_top_k=state.optimization_preset.custom_polish_top_k,
            local_max_iterations=state.optimization_preset.custom_local_max_iterations,
            random_seed=20260815,
            local_derivative_mode=DerivativeMode.ANALYTICAL,
            objective_weight_gamma=ow.weight_gamma,
            objective_weight_voltage=ow.weight_voltage,
            objective_weight_loss=ow.weight_loss,
            objective_weight_complexity=ow.weight_complexity,
        )

    p = _PRESETS[preset_name]
    return OptimizationSpec(
        max_global_evaluations=p["max_global_evaluations"],
        polish_top_k=p["polish_top_k"],
        local_max_iterations=p["local_max_iterations"],
        random_seed=p["random_seed"],
        local_derivative_mode=DerivativeMode.ANALYTICAL,
        objective_weight_gamma=ow.weight_gamma,
        objective_weight_voltage=ow.weight_voltage,
        objective_weight_loss=ow.weight_loss,
        objective_weight_complexity=ow.weight_complexity,
    )


def state_to_spec(state: ProjectState) -> ProjectSpec:
    """Convert GUI ProjectState to backend ProjectSpec (Pydantic model)."""
    source = SourceSpec(
        mode=SourceMode(state.source.mode),
        thevenin_vrms=state.source.vth_rms,
        z_source_real_ohm=state.source.z_source_ohm,
        available_power_w=state.source.available_power_w,
    )

    branches = [
        MotionalBranch(rm_ohm=b.rm_ohm, lm_h=b.lm_h, cm_f=b.cm_f)
        for b in state.eom.motional_branches
    ]

    eom = EOMModelSpec(
        model_type=EOMModelType(state.eom.model_type),
        c0_f=state.eom.c0_f,
        rs_ohm=state.eom.rs_ohm,
        ls_h=state.eom.ls_h,
        g0_s=state.eom.g0_s,
        motional_branches=branches,
        data_file=state.eom.tabular_file,
        data_format=state.eom.tabular_format,
        validity_hz=state.eom.validity_hz,
    )

    # Build frequency targets with optional per-target voltage
    voltages = getattr(state, "voltage_targets_rms_v", [])
    freq_targets = []
    for i, f in enumerate(state.frequencies_hz):
        v = voltages[i] if i < len(voltages) else None
        freq_targets.append(FrequencyTarget(frequency_hz=f, voltage_target_rms_v=v))

    frequencies = FrequencyPlan(
        targets=freq_targets,
        sweep_f_min_hz=state.sweep_f_min_hz,
        sweep_f_max_hz=state.sweep_f_max_hz,
    )

    topology = TopologySearchSpec(
        branch1_cells_max=state.topology.n_cells_per_branch,
        branch2_cells_max=state.topology.n_cells_per_branch if state.topology.n_branches > 1 else 0,
        branch1_cells_min=1,
        branch2_cells_min=1 if state.topology.n_branches > 1 else 0,
        # Default pole settings handled by TopologySearchSpec defaults
    )

    optimization = _compile_optimization_spec(state)

    # Compile match constraints from GUI state
    mp = state.match_params
    matching = MatchConstraints(
        gamma_max=mp.gamma_max,
        resistance_min_ohm=mp.resistance_min_ohm,
        resistance_max_ohm=mp.resistance_max_ohm,
        max_abs_reactance_ohm=mp.max_abs_reactance_ohm,
    )

    # Compile stress constraints from GUI state
    sp = state.stress_params
    stress = StressConstraints(
        source_current_rms_max_a=sp.source_current_rms_max_a,
        off_target_eom_peak_rms_v=sp.off_target_eom_peak_rms_v,
        default_cap_peak_voltage_v=sp.default_cap_peak_voltage_v,
        default_ind_peak_current_a=sp.default_ind_peak_current_a,
    )

    # Compile component limits from GUI state
    cl = state.component_limits
    components = ComponentPolicy(
        continuous_limits=ContinuousLimits(
            l_min_h=cl.l_min_h,
            l_max_h=cl.l_max_h,
            c_min_f=cl.c_min_f,
            c_max_f=cl.c_max_f,
        )
    )

    return ProjectSpec(
        source=source,
        eom=eom,
        frequencies=frequencies,
        topology=topology,
        optimization=optimization,
        matching=matching,
        stress=stress,
        components=components,
    )


def spec_to_state(spec: ProjectSpec) -> ProjectState:
    """Reconstruct GUI ProjectState from a loaded ProjectSpec."""
    state = ProjectState()

    state.frequencies_hz = [t.frequency_hz for t in spec.frequencies.targets]
    state.sweep_f_min_hz = spec.frequencies.sweep_f_min_hz
    state.sweep_f_max_hz = spec.frequencies.sweep_f_max_hz

    state.source = SourceParams(
        mode=spec.source.mode.value,
        vth_rms=spec.source.thevenin_vrms,
        z_source_ohm=spec.source.z_source_real_ohm,
        available_power_w=spec.source.available_power_w,
    )

    branches = [
        MotionalBranchParams(rm_ohm=b.rm_ohm, lm_h=b.lm_h, cm_f=b.cm_f)
        for b in spec.eom.motional_branches
    ]
    state.eom = EOMParams(
        model_type=spec.eom.model_type.value,
        c0_f=spec.eom.c0_f,
        rs_ohm=spec.eom.rs_ohm,
        ls_h=spec.eom.ls_h,
        g0_s=spec.eom.g0_s,
        motional_branches=branches,
        tabular_file=spec.eom.data_file,
        tabular_format=spec.eom.data_format,
        validity_hz=spec.eom.validity_hz,
    )

    state.topology = TopologyParams(
        n_branches=2 if spec.topology.branch2_cells_max > 0 else 1,
        n_cells_per_branch=spec.topology.branch1_cells_max,
    )

    state.input_sha256 = state.compute_input_sha()
    return state


def _sidecar_path(base_path: str | Path) -> Path:
    p = Path(base_path)
    return p.with_name(p.name + ".gui.yaml")


def save_gui_project(state: ProjectState, path: str | Path) -> None:
    """Save ProjectSpec + GUI sidecar fields."""
    spec = state_to_spec(state)
    save_project(spec, path)

    # Write .gui.yaml sidecar
    sidecar_data = {
        "library_path": state.library_path,
        "library_sha": state.library_sha,
        "revision": state.revision,
        "input_sha256": state.input_sha256,
        "optimize_result_path": state.optimize_result_path,
        "optimize_revision": state.optimize_revision,
        "verify_result_path": state.verify_result_path,
        "verify_revision": state.verify_revision,
        "realization_result_path": state.realization_result_path,
        "realization_revision": state.realization_revision,
        "realization_library_sha": state.realization_library_sha,
        "robustness_result_path": state.robustness_result_path,
        "robustness_revision": state.robustness_revision,
        "robustness_library_sha": state.robustness_library_sha,
        "spice_result_path": state.spice_result_path,
        "spice_revision": state.spice_revision,
    }
    with open(_sidecar_path(path), "w", encoding="utf-8") as f:
        yaml.dump(sidecar_data, f)


def load_gui_project(path: str | Path) -> ProjectState:
    """Load ProjectSpec + GUI sidecar."""
    spec = load_project(path)
    state = spec_to_state(spec)

    sidecar = _sidecar_path(path)
    if sidecar.exists():
        with open(sidecar, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        state.library_path = data.get("library_path")
        state.library_sha = data.get("library_sha")
        state.revision = data.get("revision", state.revision)
        state.input_sha256 = data.get("input_sha256", state.input_sha256)
        state.optimize_result_path = data.get("optimize_result_path")
        state.optimize_revision = data.get("optimize_revision")
        state.verify_result_path = data.get("verify_result_path")
        state.verify_revision = data.get("verify_revision")
        state.realization_result_path = data.get("realization_result_path")
        state.realization_revision = data.get("realization_revision")
        state.realization_library_sha = data.get("realization_library_sha")
        state.robustness_result_path = data.get("robustness_result_path")
        state.robustness_revision = data.get("robustness_revision")
        state.robustness_library_sha = data.get("robustness_library_sha")
        state.spice_result_path = data.get("spice_result_path")
        state.spice_revision = data.get("spice_revision")

        # Invalidate results if inputs changed externally
        if state.input_sha256 != state.compute_input_sha():
            state.bump_revision()

    return state
